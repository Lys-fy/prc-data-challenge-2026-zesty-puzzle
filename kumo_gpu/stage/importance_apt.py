#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Per-airport fold-honest column ordering for the engine's COLMODE=imp.
For every airport APT and fold: LightGBM gain on APT's clean TRAINING rows only (2025, held-out months excluded; prod = all 12 months), target
y - mvt_aobt3, the 412 engine features, the hyper-parameters of the shipped km_importance_412_<fold>.tsv ranking (lr 0.05, 63 leaves, min_data 100, ff 0.8, bagging 0.8/1,
lambda_l2 1, 400 rounds, seed 0); <= NSUB rows per airport-fold (seeded draw proportional over months). The held months' labels are never read.
Writes kumo_gpu/data/km_importance_412_<fold>_<APT>.tsv (feature \t gain \t rank), same format as the global files.
usage: python importance_apt.py [folds|all] [APT,APT,...|all] [NSUB=150000]   env PAYLOAD, PRC_THREADS (6)"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, numpy as np, polars as pl, lightgbm as lgb

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
APTS0 = "LTFM,EHAM,EGLL,LFPG,EDDF,LEMD,LEBL,EDDM,LIRF,LSZH".split(",")
a = sys.argv[1:]
folds = a[0].split(",") if a and a[0] != "all" else list(HOLD)
apts = a[1].split(",") if len(a) > 1 and a[1] != "all" else APTS0
NSUB = int(a[2]) if len(a) > 2 else 150000
NT = int(os.environ.get("PRC_THREADS", "6"))
t0 = time.time()
print(f"importance_apt: folds {folds} airports {apts} NSUB {NSUB} threads {NT}", flush=True)
for apt in apts:
    d = pl.read_parquet(
        f"{PAYLOAD}/member/km_{apt}.parquet",
        columns=FEATS + ["meta__y", "meta__m", "meta__is_rank", "meta__fb_pru", "meta__mvt_aobt3"],
    )
    d = d.filter((pl.col("meta__is_rank") != 1) & (pl.col("meta__fb_pru") != 1) & pl.col("meta__y").is_finite())
    X = d.select(FEATS).to_numpy().astype(np.float32)
    y = d["meta__y"].to_numpy().astype(float) - np.nan_to_num(d["meta__mvt_aobt3"].to_numpy().astype(float), nan=0.0)
    m = d["meta__m"].to_numpy()
    del d
    for fold in folds:
        t1 = time.time()
        rng = np.random.default_rng(0)
        idx = np.where(np.ones(len(y), bool) if HOLD[fold] is None else ~np.isin(m, HOLD[fold]))[0]
        frac = min(1.0, NSUB / len(idx))
        pick = np.concatenate(
            [
                rng.choice(ci, max(1, int(round(frac * len(ci)))), replace=False)
                for ci in (idx[m[idx] == c] for c in np.unique(m[idx]))
            ]
        )
        assert HOLD[fold] is None or not np.isin(m[pick], HOLD[fold]).any()
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
            lgb.Dataset(X[pick], y[pick], feature_name=FEATS, free_raw_data=True),
            num_boost_round=400,
        )
        gain = bst.feature_importance("gain")
        order = np.argsort(-gain, kind="stable")
        out = f"{B}/data/km_importance_412_{fold}_{apt}.tsv"
        with open(out + ".tmp", "w") as fh:
            fh.write(
                f"# LightGBM gain importance, airport {apt}, fold {fold}: training months only ({'all 12' if HOLD[fold] is None else 'held ' + str(HOLD[fold]) + ' excluded'}), {len(pick):,} rows, 400 rounds, seed 0\n"
            )
            for r, j in enumerate(order):
                fh.write(f"{FEATS[j]}\t{gain[j]:.1f}\t{r + 1}\n")
        os.replace(out + ".tmp", out)
        g = {}
        for f in (f"{B}/data/km_importance_412_{fold}.tsv",):
            if os.path.exists(f):
                g = [l.split("\t")[0] for l in open(f) if l.strip() and not l.startswith("#")][:50]
        ov = len(set(g) & set(FEATS[j] for j in order[:50])) if g else -1
        print(
            f"{apt} {fold}: rows {len(pick):,} | top-5 {', '.join(FEATS[j] for j in order[:5])} | gain>0 {int((gain > 0).sum())}/412 | top-50 overlap with global core {ov}/50 | {time.time() - t1:.0f}s",
            flush=True,
        )
print(f"IMPORTANCE_APT_DONE {time.time() - t0:.0f}s")
