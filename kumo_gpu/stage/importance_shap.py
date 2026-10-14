#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""SHAP-ranked column ordering for the engine's COLMODE=imp, parallel to the shipped gain ranking (kumo_gpu/data/km_importance_412_<fold>.tsv) but using LightGBM's exact
TreeSHAP (pred_contrib=True) mean |SHAP contribution| instead of split gain. Same fold-honest training protocol as that ranking (LightGBM
regressor fitted on a stratified subsample of the fold's TRAINING rows only, target y - mvt_aobt3, the 412 engine features, 400 rounds seed 0).
Writes kumo_gpu/data/km_importance_412_shap_<fold>.tsv (feature \\t mean|SHAP| \\t rank); read by cluster/km_engine.py importance() when
env SHAP_IMP=1 (OFF by default: existing km_importance_412_<fold>.tsv files and all other arms are untouched).
usage: python importance_shap.py [folds=janjul,months28] [NSUB=300000]   env PAYLOAD, PRC_THREADS, POOL"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, glob, numpy as np, polars as pl, lightgbm as lgb

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAYLOAD = os.environ.get("PAYLOAD", f"{B}/data/payload")
FEATS = open(f"{B}/data/km_features_412.txt").read().split()
assert len(FEATS) == 412
HOLD = {
    "janjul": [1, 7],
    "months28": [2, 8],
    "months39": [3, 9],
    "months410": [4, 10],
    "months511": [5, 11],
    "months612": [6, 12],
    "prod": None,
}
folds = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else ["janjul", "months28"]
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
NT = int(os.environ.get("PRC_THREADS", "6"))
rng = np.random.default_rng(0)
t0 = time.time()
print(f"importance_shap: folds {folds} NSUB {NSUB} threads {NT}", flush=True)
POOL = int(os.environ.get("POOL", "450000"))
D = []
files = sorted(glob.glob(f"{PAYLOAD}/member/km_*.parquet"))
ntot = sum(
    pl.scan_parquet(f)
    .filter((pl.col("meta__is_rank") != 1) & (pl.col("meta__fb_pru") != 1))
    .select(pl.len())
    .collect()
    .item()
    for f in files
)
frac0 = min(
    1.0, POOL / max(1, ntot)
)  # memory-light: keep a seeded random POOL-row pool across airports, read one airport at a time
for f in files:
    d = pl.read_parquet(
        f, columns=FEATS + ["meta__y", "meta__m", "meta__is_rank", "meta__fb_pru", "meta__mvt_aobt3", "meta__ADEP_mvt"]
    )
    d = d.filter((pl.col("meta__is_rank") != 1) & (pl.col("meta__fb_pru") != 1) & pl.col("meta__y").is_finite())
    if frac0 < 1.0:
        d = d.filter(pl.Series(rng.random(len(d)) < frac0))
    D.append(d)
D = pl.concat(D)
print(f"pool {len(D):,} of {ntot:,} clean 2025 rows (POOL {POOL}) | {time.time()-t0:.0f}s", flush=True)
X = D.select(FEATS).to_numpy()
y = (D["meta__y"] - D["meta__mvt_aobt3"]).to_numpy()
m = D["meta__m"].to_numpy()
apt = D["meta__ADEP_mvt"].to_numpy()
cell = np.array([f"{a}_{mm}" for a, mm in zip(apt, m)])
for fold in folds:
    t1 = time.time()
    keep = np.ones(len(y), bool) if HOLD[fold] is None else ~np.isin(m, HOLD[fold])
    idx = np.where(keep)[0]
    frac = min(1.0, NSUB / len(idx))
    pick = []
    for c in np.unique(cell[idx]):
        ci = idx[cell[idx] == c]
        pick.append(rng.choice(ci, max(1, int(round(frac * len(ci)))), replace=False))
    pick = np.concatenate(pick)
    dtrain = lgb.Dataset(X[pick], y[pick], feature_name=FEATS, free_raw_data=True)
    bst = lgb.train(
        dict(
            objective="regression",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=100,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            num_threads=NT,
            verbose=-1,
            seed=0,
        ),
        dtrain,
        num_boost_round=400,
    )
    contrib = bst.predict(X[pick], pred_contrib=True, num_threads=NT)[
        :, :-1
    ]  # exact TreeSHAP; last column is the bias/expected-value term
    shap_imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(-shap_imp)
    with open(f"{B}/data/km_importance_412_shap_{fold}.tsv", "w") as fh:
        fh.write(
            f"# LightGBM exact TreeSHAP mean|contrib| importance, fold {fold}: training months only ({'all 12' if HOLD[fold] is None else 'held ' + str(HOLD[fold])}), {len(pick):,} rows, 400 rounds, seed 0\n"
        )
        for r, j in enumerate(order):
            fh.write(f"{FEATS[j]}\t{shap_imp[j]:.4f}\t{r + 1}\n")
    top = ", ".join(FEATS[j] for j in order[:8])
    print(f"{fold}: rows {len(pick):,} | top-8 {top} | {time.time()-t1:.0f}s", flush=True)
print(f"IMPORTANCE_SHAP_DONE {time.time()-t0:.0f}s")
