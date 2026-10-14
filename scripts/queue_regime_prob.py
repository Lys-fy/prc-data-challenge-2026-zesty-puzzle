#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Computes the nested out-of-fold regime probability q = P(r > 0), r = y - (MVT - AOBT_3), as a member feature
side table, using the fallback_classifier.py REGP recipe.
Inputs: <fold> positional arg (months39/28/410/janjul/612/511/prod), data/features_v9.parquet plus
durp/dreg/adsb/adsb2/adsb3 side tables, data/oof/fallback_regpl_<fold or rank>.npy; env NT (threads).
Output: data/external/queue_decomp_q_<tag>_features.parquet (MVT_ID_mvt, xq_q).

XQ: nested out-of-fold regime probability q = P(r > 0), r = y - (MVT - AOBT_3), as a MEMBER feature side table.
Recipe = fallback_classifier.py arm REGP exactly (RA3 set L, 393 features; LightGBM binary 127 leaves, lr 0.05, 800 rounds, ff 0.8,
bagging 0.8, l2 10, min_leaf 100; fitted on matched non-fb rows with |r| < 7200, weight min(|r|, 600)).
Nesting: for fold F (held months H) the training months T are split into seasonal pairs (m, m+6); every training row
gets q from a REGP model fitted on T minus its own pair (never sees its own label, never sees H). Validation rows get
fallback_regpl_<F> (REGP fitted on all of T), i.e. exactly the q the L2 stacker already uses. For 'prod': T = all 2025,
inner pairs (1,7)...(6,12), ranking rows get fallback_regpl_rank (all-2025 fit).
Writes data/external/queue_decomp_q_<tag>_features.parquet  [MVT_ID_mvt, xq_q]  (tag m39 | m28 | m410 | m17 | prod).
usage: queue_regime_prob.py <months39|months28|months410|janjul|months612|months511|prod>   env NT (threads, default 4)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, numpy as np, polars as pl, pandas as pd, lightgbm as lgb

NT = int(os.environ.get("NT", "4"))
os.environ.setdefault("POLARS_MAX_THREADS", str(NT))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train2 import cols_for, CAT_BASE
from sklearn.metrics import roc_auc_score

FOLD = sys.argv[1]
HOLD = {
    "months39": [3, 9],
    "months28": [2, 8],
    "months410": [4, 10],
    "janjul": [1, 7],
    "months612": [6, 12],
    "months511": [5, 11],
    "prod": [],
}[FOLD]
TAG = {
    "months39": "m39",
    "months28": "m28",
    "months410": "m410",
    "janjul": "m17",
    "months612": "m612",
    "months511": "m511",
    "prod": "prod",
}[FOLD]
DROP = "te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy"
FEATS = "base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,adsb,adsb2,adsb3"
SIDE = ["durp", "dreg", "adsb", "adsb2", "adsb3"]
t0 = time.time()
FP = f"{ROOT}/data/features_v9.parquet"
_schema = list(pl.read_parquet_schema(FP).keys())
_dummy = pl.DataFrame(schema={c: pl.Null for c in _schema})
_feats = list(CAT_BASE)
for g in FEATS.split(","):
    _feats += cols_for(_dummy, g)
_meta = ["MVT_ID_mvt", "y", "m", "is_rank", "fb", "mvt_aobt3"]
d = pl.read_parquet(FP, columns=[c for c in dict.fromkeys(_meta + _feats) if c in _schema])
for g in SIDE:
    d = d.join(pl.read_parquet(f"{ROOT}/data/external/{g}_features.parquet"), on="MVT_ID_mvt", how="left")
feats = list(CAT_BASE)
for g in FEATS.split(","):
    feats += cols_for(d, g)
feats = [f for f in feats if f in d.columns]
_dp = tuple(x for x in DROP.split(",") if x)
feats = [f for f in feats if not f.startswith(_dp)]
cats = [c for c in CAT_BASE if c in feats]
Xa = np.empty((d.height, len(feats)), dtype=np.float32)
for j0 in range(0, len(feats), 16):
    chunk = feats[j0 : j0 + 16]
    pdf = d.select(chunk).to_pandas()
    for j, c in enumerate(chunk):
        if c in cats:
            codes = pdf[c].astype("category").cat.codes.to_numpy().astype(np.float32)
            codes[codes < 0] = np.nan
            Xa[:, j0 + j] = codes
        else:
            Xa[:, j0 + j] = pdf[c].to_numpy(dtype=np.float64, na_value=np.nan).astype(np.float32, copy=False)
    del pdf
X = pd.DataFrame(Xa, columns=feats)
del Xa
y = d["y"].to_numpy()
m = d["m"].to_numpy()
is_rank = (d["is_rank"] == 1).to_numpy()
fb = d["fb"].to_numpy() == 1
a3 = d["mvt_aobt3"].to_numpy().astype(float)
ids = d["MVT_ID_mvt"].to_numpy()
del d
trn = (~is_rank) & ~np.isin(m, HOLD)
val = (~is_rank) & np.isin(m, HOLD)
print(f"[{time.time()-t0:.0f}s] {FOLD}: {len(feats)} feats, trn {trn.sum():,} val {val.sum():,}", flush=True)
r = y - a3
ok = np.isfinite(r) & ~fb & (np.abs(np.nan_to_num(r, nan=1e9)) < 7200)
prm = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    lambda_l2=10.0,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
    num_threads=NT,
    cat_smooth=10.0,
    seed=0,
)
q = np.full(len(y), np.nan)
Tm = sorted(set(m[trn].tolist()))
pairs = sorted({tuple(sorted({mm, ((mm + 5) % 12) + 1} & set(Tm))) for mm in Tm})
print(f"inner groups {pairs}", flush=True)
for gm in pairs:
    fit = np.where(trn & ok & ~np.isin(m, gm))[0]
    pr = np.where(trn & np.isin(m, gm))[0]
    b = lgb.train(
        prm,
        lgb.Dataset(
            X.iloc[fit], (r[fit] > 0).astype(int), weight=np.minimum(np.abs(r[fit]), 600.0), categorical_feature=cats
        ),
        800,
    )
    q[pr] = b.predict(X.iloc[pr])
    vv = pr[ok[pr]]
    print(
        f"[{time.time()-t0:.0f}s] group {gm}: fit {len(fit):,} pred {len(pr):,} AUC(clean matched) {roc_auc_score(r[vv] > 0, q[vv]):.4f} mean q {q[pr].mean():.4f}",
        flush=True,
    )
src = f"{ROOT}/data/oof/fallback_regpl_{'rank' if FOLD == 'prod' else FOLD}.npy"
Q = np.load(src)
tgt = np.where(is_rank if FOLD == "prod" else val)[0]
o = np.argsort(Q[0])
k = np.searchsorted(Q[0][o], ids[tgt].astype(float))
assert np.array_equal(
    Q[0][o][np.minimum(k, len(o) - 1)], ids[tgt].astype(float)
), f"{src} does not cover the target rows"
q[tgt] = Q[1][o][k]
vv = tgt[ok[tgt]]
if FOLD != "prod":
    print(
        f"outer q ({src}): AUC(clean matched val) {roc_auc_score(r[vv] > 0, q[vv]):.4f} mean {q[tgt].mean():.4f}",
        flush=True,
    )
else:
    print(f"rank q mean {q[tgt].mean():.4f}; training rows mean {q[trn].mean():.4f}", flush=True)
keep = np.isfinite(q)
assert keep[trn].all() and keep[tgt].all()
out = f"{ROOT}/data/external/queue_decomp_q_{TAG}_features.parquet"
pl.DataFrame({"MVT_ID_mvt": ids[keep], "xq_q": q[keep]}).write_parquet(out)
print(
    f"[{time.time()-t0:.0f}s] wrote {out}: {keep.sum():,} rows (train mean q {q[trn].mean():.4f}, target mean {q[tgt].mean():.4f})",
    flush=True,
)
