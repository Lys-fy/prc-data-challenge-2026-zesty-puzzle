#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Rebuilds the mechanism (fb) classifier (XF / L1): builds X exactly as train2.py does for the RA3 (LightGBM, set L)
or RAC2 (CatBoost, set C) recipe, fits classifier arms on the fold's training rows, and predicts its validation rows.
Inputs: positional <fold> <set L|C> <arms comma list>, env NT (threads, default 4). Writes
data/oof/fallback_classifier<ARM><SET>_<fold>.npy = [MVT_ID, p] (validation rows only).

Arm R0 = exact replica of train2.fit_cls (63 leaves, lr 0.05, min_leaf 50, l2 0, ff 0.99, cat_smooth 10, 400
rounds, stake (max(MVT-SCHED,0)/h)^2.42 + 0.5).
usage: fallback_classifier.py <fold> <set L|C> <arms comma list>   env NT (threads, default 4)
arms: R0 replica | A capacity+ES | B stake on |AOBT3-SCHED| (k 2.42) | B2 (k 2) | AB | C catboost | CB catboost + B stake
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, json, numpy as np, polars as pl, pandas as pd, lightgbm as lgb

NT = int(os.environ.get("NT", "4"))
os.environ.setdefault("POLARS_MAX_THREADS", str(NT))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train2 import cols_for, CAT_BASE
from sklearn.metrics import roc_auc_score, log_loss

FOLD, SET, ARMS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
HOLD = {"months39": [3, 9], "months28": [2, 8], "months410": [4, 10], "janjul": [1, 7], "rank": [1, 7]}[
    FOLD
]  # rank: train on all 2025 rows, predict the ranking rows
DROP_L = "te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy"
FEATS = {
    "L": "base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,adsb,adsb2,adsb3",
    "C": "base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,opdir,durp,dreg,adsb,adsb2",
}[SET]
DROP = {"L": DROP_L, "C": "te_fno_"}[SET]
SIDE = ["durp", "dreg", "adsb", "adsb2", "adsb3"]
t0 = time.time()
FP = f"{ROOT}/data/features_v9.parquet"
_schema = list(pl.read_parquet_schema(FP).keys())
_dummy = pl.DataFrame(schema={c: pl.Null for c in _schema})
_feats = list(CAT_BASE)
for g in FEATS.split(","):
    _feats += cols_for(_dummy, g)
_meta = [
    "MVT_ID_mvt",
    "y",
    "m",
    "is_rank",
    "ADEP_mvt",
    "fb",
    "fb_pru",
    "mvt_sched",
    "mvt_aobt3",
    "mvt_eobt1",
    "mvt_lobt",
    "ref_sr_p10",
    "unmatched",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "MVT_TIME_UTC_mvt",
]
d = pl.read_parquet(FP, columns=[c for c in dict.fromkeys(_meta + _feats) if c in _schema])
for g in SIDE:
    if g in FEATS.split(","):
        d = d.join(pl.read_parquet(f"{ROOT}/data/external/{g}_features.parquet"), on="MVT_ID_mvt", how="left")
feats = list(CAT_BASE)
for g in FEATS.split(","):
    feats += cols_for(d, g)
feats = [f for f in feats if f in d.columns]
_dp = tuple(x for x in DROP.split(",") if x)
feats = [f for f in feats if not f.startswith(_dp)]
cats = [c for c in CAT_BASE if c in feats]
if SET == "L":  # float32 block, categoricals as codes (train2 lgbm path with --f32)
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
else:  # train2 cat-model path: pandas frame with category dtype (the LightGBM classifier sees this frame)
    X = d.select(feats).to_pandas()
    for c in cats:
        X[c] = X[c].astype("category")
y = d["y"].to_numpy()
m = d["m"].to_numpy()
is_rank = (d["is_rank"] == 1).to_numpy()
fb = d["fb"].to_numpy() == 1
ms = d["mvt_sched"].to_numpy().astype(float)
a3 = d["mvt_aobt3"].to_numpy().astype(float)
ids = d["MVT_ID_mvt"].to_numpy()
del d
val = (~is_rank) & np.isin(m, HOLD)
trn = (~is_rank) & ~np.isin(m, HOLD)
if FOLD == "rank":
    val = is_rank.copy()
    trn = ~is_rank
fi = np.where(trn)[0]
vi = np.where(val)[0]
print(
    f"[{time.time()-t0:.0f}s] {FOLD} set {SET}: {len(feats)} feats, trn {len(fi):,} val {len(vi):,} fb rate {fb[fi].mean():.4f}",
    flush=True,
)
st_ms = (np.maximum(ms, 0) / 3600.0) ** 2.42 + 0.5
dB = np.where(
    np.isfinite(a3), np.abs(ms - np.nan_to_num(a3)), np.maximum(ms, 0)
)  # |AOBT_3 - SCHED| (= |(MVT-SCHED)-(MVT-AOBT_3)|), MVT-SCHED where AOBT_3 is null
st_B = (dB / 3600.0) ** 2.42 + 0.5
st_B2 = (dB / 3600.0) ** 2.0 + 0.5
BASE = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    lambda_l2=0.0,
    feature_fraction=0.99,
    verbose=-1,
    num_threads=NT,
    cat_smooth=10.0,
    seed=0,
)


def lgb_fit(st, prm, rounds):
    b = lgb.train(prm, lgb.Dataset(X.iloc[fi], fb[fi].astype(int), weight=st[fi], categorical_feature=cats), rounds)
    return b.predict(X.iloc[vi])


def lgb_es(st, prm):
    # inner month split of the TRAINING months: hold out the two training months nearest the fold's months (m+1 of each held month), pick rounds by stake-weighted log-loss, refit on all training rows with rounds x 1.1
    inner = [((h % 12) + 1) for h in HOLD]
    inner = [x if x not in HOLD else ((x % 12) + 1) for x in inner]
    itr = fi[~np.isin(m[fi], inner)]
    iva = fi[np.isin(m[fi], inner)]
    dtr = lgb.Dataset(X.iloc[itr], fb[itr].astype(int), weight=st[itr], categorical_feature=cats)
    dva = lgb.Dataset(X.iloc[iva], fb[iva].astype(int), weight=st[iva], categorical_feature=cats, reference=dtr)
    b = lgb.train(prm, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    n = int(b.best_iteration * 1.1)
    print(f"   ES inner months {inner}: best {b.best_iteration} -> refit {n}", flush=True)
    return lgb_fit(st, prm, n), n


def cat_fit(st):
    from catboost import CatBoostClassifier, Pool

    Xc = X.iloc[:, :].copy() if SET == "C" else X.copy()
    for c in cats:
        Xc[c] = (Xc[c].cat.codes if SET == "C" else Xc[c].fillna(-1)).astype(int).astype(str)
    inner = [((h % 12) + 1) for h in HOLD]
    inner = [x if x not in HOLD else ((x % 12) + 1) for x in inner]
    itr = fi[~np.isin(m[fi], inner)]
    iva = fi[np.isin(m[fi], inner)]
    prm = dict(
        loss_function="Logloss",
        depth=8,
        learning_rate=0.08,
        iterations=3000,
        l2_leaf_reg=3,
        border_count=128,
        thread_count=NT,
        random_seed=0,
        verbose=250,
        od_type="Iter",
        od_wait=100,
        one_hot_max_size=4,
        max_ctr_complexity=2,
    )
    mdl = CatBoostClassifier(**prm)
    mdl.fit(
        Pool(Xc.iloc[itr], fb[itr].astype(int), weight=st[itr], cat_features=cats),
        eval_set=Pool(Xc.iloc[iva], fb[iva].astype(int), weight=st[iva], cat_features=cats),
        use_best_model=True,
    )
    n = int(mdl.get_best_iteration() * 1.1) + 1
    print(f"   CAT inner best {mdl.get_best_iteration()} -> refit {n}", flush=True)
    prm.update(iterations=n)
    prm.pop("od_type")
    prm.pop("od_wait")
    mdl = CatBoostClassifier(**prm)
    mdl.fit(Pool(Xc.iloc[fi], fb[fi].astype(int), weight=st[fi], cat_features=cats))
    return mdl.predict_proba(Pool(Xc.iloc[vi], cat_features=cats))[:, 1], n


CAPA = dict(
    BASE,
    learning_rate=0.03,
    num_leaves=255,
    min_data_in_leaf=100,
    lambda_l2=10.0,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
)
log = {}
for arm in ARMS:
    t1 = time.time()
    extra = None
    if arm == "R0":
        p = lgb_fit(st_ms, BASE, 400)
    elif arm == "B":
        p = lgb_fit(st_B, BASE, 400)
    elif arm == "B2":
        p = lgb_fit(st_B2, BASE, 400)
    elif arm == "A":
        p, extra = lgb_es(st_ms, CAPA)
    elif arm == "AB":
        p, extra = lgb_es(st_B, CAPA)
    elif arm == "C":
        p, extra = cat_fit(st_ms)
    elif (
        arm == "D"
    ):  # B1: bagged random-forest-mode LightGBM, incumbent stake; pre-declared use = arithmetic mean of probabilities, killed if logit-error corr with the incumbent p >= 0.95
        prm = dict(
            BASE,
            boosting="rf",
            num_leaves=255,
            min_data_in_leaf=100,
            feature_fraction=0.5,
            bagging_fraction=0.632,
            bagging_freq=1,
            lambda_l2=10.0,
        )
        p = lgb_fit(st_ms, prm, 300)
        p0 = (
            lgb_fit(st_ms, BASE, 400)
            if not os.path.exists(f"{ROOT}/data/oof/fallback_clsr0{SET}_{FOLD}.npy")
            else np.load(f"{ROOT}/data/oof/fallback_clsr0{SET}_{FOLD}.npy")[1]
        )
        lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
        tgt = fb[vi].astype(float)
        e0 = tgt - p0
        e1 = tgt - p
        print(
            f"   D pre-check: residual corr {np.corrcoef(e0, e1)[0,1]:.4f} | logit corr {np.corrcoef(lg(p0), lg(p))[0,1]:.4f}",
            flush=True,
        )
    elif arm == "REGP":
        # L2: regime-sign classifier, label 1[r > 0] with r = y - (MVT-AOBT_3), fitted on matched non-fb training rows with |r| < 7200, weight min(|r|, 600); predicted on ALL val rows (no missingness pattern)
        r = y - a3
        ok = np.isfinite(r) & ~fb & (np.abs(np.nan_to_num(r, nan=1e9)) < 7200)
        fr = fi[ok[fi]]
        prm = dict(
            BASE,
            learning_rate=0.05,
            num_leaves=127,
            min_data_in_leaf=100,
            lambda_l2=10.0,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
        )
        b = lgb.train(
            prm,
            lgb.Dataset(
                X.iloc[fr], (r[fr] > 0).astype(int), weight=np.minimum(np.abs(r[fr]), 600.0), categorical_feature=cats
            ),
            800,
        )
        p = b.predict(X.iloc[vi])
        vv = ok[vi]
        np.save(f"{ROOT}/data/oof/fallback_regp{SET}_{FOLD}.npy", np.vstack([ids[vi].astype(float), p]))
        if FOLD != "rank":
            print(
                f"[{time.time()-t0:.0f}s] REGP AUC (clean matched val) {roc_auc_score(r[vi][vv] > 0, p[vv]):.4f}",
                flush=True,
            )
        else:
            print(f"[{time.time()-t0:.0f}s] REGP rank: {len(vi):,} rows, mean q {p.mean():.4f}", flush=True)
        continue
    elif arm == "CB":
        p, extra = cat_fit(st_B)
    else:
        raise SystemExit(arm)
    np.save(f"{ROOT}/data/oof/fallback_classifier{arm}{SET}_{FOLD}.npy", np.vstack([ids[vi].astype(float), p]))
    if FOLD == "rank":
        print(f"[{time.time()-t0:.0f}s] rank {arm}: mean p {p.mean():.4f} rounds {extra}", flush=True)
        continue
    auc = roc_auc_score(fb[vi], p)
    ll = log_loss(fb[vi], np.clip(p, 1e-7, 1 - 1e-7))
    llw = log_loss(fb[vi], np.clip(p, 1e-7, 1 - 1e-7), sample_weight=st_ms[vi])
    log[arm] = dict(auc=auc, ll=ll, llw=llw, rounds=extra, secs=round(time.time() - t1))
    print(
        f"[{time.time()-t0:.0f}s] {FOLD} {SET} {arm}: AUC {auc:.5f} logloss {ll:.5f} stake-wll {llw:.5f} rounds {extra} ({time.time()-t1:.0f}s)",
        flush=True,
    )
with open(f"{ROOT}/experiments/fallback_cls_log.jsonl", "a") as f:
    f.write(json.dumps(dict(fold=FOLD, set=SET, arms=log)) + "\n")
