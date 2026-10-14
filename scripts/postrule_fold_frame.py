#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Post-rule fit frame: the 2025 fold rows of a 6-fold pool vector with their ranking-mix row weights, the frame
phase2/steps/p2_postrules.py fit reads (scratchpad/x4/X4_fold.parquet).
Rows  = the ids of <pool>_<fold>.npy over the 6 folds (each fold sorted by id, folds in FOLDS order).
ok    = labelled, NM-matched, not a ranking row, finite pool prediction.
w     = clock_lib.cell_weights per fold on the ok rows (cells apt|lnk|class|src, n_rank(cell) / n_fold(cell), fallbacks; 0 elsewhere).
r     = cf(pool) - y; fb = |MVT - SCHED - y| <= 60 s (block equals schedule); pocket = y - (MVT - AOBT_3) > 600 s.
Columns: clockresid_lib.frame + data/external/adsb5_features.parquet + fo, ok, w, r, fb, pocket, apt, lkc, obs, src.
ok and w depend only on the row set of the pool vector and on where it is finite, not on its values.
usage: postrule_fold_frame.py --pool <6-fold prefix, [id, pred] per fold> [--out scratchpad/x4/X4_fold.parquet]"""

import os, sys, glob, argparse, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K

FOLDS = ["months39", "months28", "months410", "janjul", "months612", "months511"]
ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--out", default="scratchpad/x4/X4_fold.parquet")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)


def ld(pref, t):
    z = np.load(sorted(glob.glob(f"{pref}_{t}.npy") or glob.glob(f"{pref}_{t}_*.npy"))[0])
    o = np.argsort(z[0])
    return z[0][o], z[1][o]


ids, p, fo = [], [], []
for t in FOLDS:
    i, v = ld(a.pool, t)
    ids.append(i)
    p.append(v)
    fo.append(np.full(len(i), t))
ids = np.concatenate(ids)
p = np.concatenate(p)
fo = np.concatenate(fo)
# row classes and weights
Fr = L.frame(ids)
y = Fr["y"].to_numpy().astype(float)
lo = Fr["mvt_lobt"].to_numpy().astype(float)
ok = np.isfinite(y) & (Fr["unmatched"].to_numpy() == 0) & (Fr["is_rank"].to_numpy() == 0) & np.isfinite(p)
r = L.cf(p, lo) - y
apt, lk, obs, sn = K.keys(Fr)
rk = K.keys(K.rank_frame())
ao = Fr["mvt_aobt3"].to_numpy().astype(float)
ms = Fr["mvt_sched"].to_numpy().astype(float)
fb = np.abs(np.nan_to_num(ms, nan=1e9) - y) <= 60
pocket = (y - np.nan_to_num(ao, nan=-1e9)) > 600
w = np.zeros(len(y))
for t in FOLDS:
    m = (fo == t) & ok
    w[m] = K.cell_weights(tuple(x[m] for x in (apt, lk, obs, sn)), rk)[0]
# frame
A5 = pl.read_parquet("data/external/adsb5_features.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
F = Fr.join(A5, on="MVT_ID_mvt", how="left", maintain_order="left")
apt, lk, obs, sn = K.keys(F)
F = F.with_columns(
    fo=pl.Series(fo),
    ok=pl.Series(ok),
    w=pl.Series(w),
    r=pl.Series(r),
    fb=pl.Series(fb),
    pocket=pl.Series(pocket),
    apt=pl.Series(apt),
    lkc=pl.Series(lk),
    obs=pl.Series(obs),
    src=pl.Series(sn),
)
F.write_parquet(a.out)
print(
    "fold frame",
    F.shape,
    "| ok rows",
    int(ok.sum()),
    "| weight sum per fold",
    " ".join(f"{t} {w[fo == t].sum():,.0f}" for t in FOLDS),
    "->",
    a.out,
)
