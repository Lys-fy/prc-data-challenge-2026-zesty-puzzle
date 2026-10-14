#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Chain inputs for a v2 member ARM (env ARM, e.g. BIG): fold data/oof/big_<ARM>3_<fold>.npy = mean over seeds {big_<ARM>, big_<ARM>_s1, big_<ARM>_s2};
prod data/preds/big_<ARM>P_s{0,1,2}.parquet [MVT_ID_mvt, taxi]. Checks: finite, same ids, prod covers all 344,841 ranking rows; prints seed-to-seed RMS (info).
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, numpy as np, polars as pl

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARM = os.environ["ARM"]
SEEDS = [f"big_{ARM}", f"big_{ARM}_s1", f"big_{ARM}_s2"]
for f in ["janjul", "months28", "months39", "months410", "months511", "months612"]:
    A = [np.load(f"data/oof/{s}_{f}.npy") for s in SEEDS]
    o = [np.argsort(a[0]) for a in A]
    for a, oo in zip(A[1:], o[1:]):
        assert np.array_equal(a[0][oo], A[0][0][o[0]])
    P = np.vstack([a[1][oo] for a, oo in zip(A, o)])
    assert np.isfinite(P).all()
    np.save(f"data/oof/big_{ARM}3_{f}.npy", np.vstack([A[0][0][o[0]], P.mean(0), np.full(P.shape[1], np.nan)]))
    print(
        f"{f}: rows {P.shape[1]:,} seed RMS s0-s1 {np.sqrt(np.mean((P[0]-P[1])**2)):.1f} s0-s2 {np.sqrt(np.mean((P[0]-P[2])**2)):.1f}"
    )
if os.environ.get("NOPROD"):
    raise SystemExit(0)
rk = (
    pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank"])
    .filter(pl.col("is_rank") == 1)["MVT_ID_mvt"]
    .to_numpy()
)
for k, s in enumerate(SEEDS):
    a = np.load(f"data/oof/{s}_prod.npy")
    assert np.isfinite(a[1]).all() and np.array_equal(np.sort(a[0]), np.sort(rk)), s
    pl.DataFrame({"MVT_ID_mvt": a[0], "taxi": a[1]}).write_parquet(f"data/preds/big_{ARM}P_s{k}.parquet")
    print(f"prod {s}: {a.shape[1]:,} rows, mean {a[1].mean():.1f} -> data/preds/big_{ARM}P_s{k}.parquet")
