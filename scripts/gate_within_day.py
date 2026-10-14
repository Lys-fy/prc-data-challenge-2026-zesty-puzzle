#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Splits the difference arm - ref of two fold vectors into a within-day part (WD) and an airport-day-mean part (DAY), for arbitrary npy prefixes. usage: gate_within_day.py ARMPREFIX REFPREFIX OUTPREFIX folds_csv
writes OUTPREFIX_{WD,DAY}_<fold>.npy (= arm - mean_{apt x UTC day}(arm - ref) / ref + mean(...)), then scores both vs REF with gate_unsealed_folds (UNS only).
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, subprocess, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L

AP, RP, OP, FO = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")


def ld(p, t):
    import glob

    g = sorted(glob.glob(f"{p}_{t}.npy") or glob.glob(f"{p}_{t}_*.npy"))
    z = np.load(g[0])
    o = np.argsort(z[0])
    return z[0][o], z[1][o]


for t in FO:
    (ia, va), (ir, vr) = ld(AP, t), ld(RP, t)
    c = np.intersect1d(ia, ir)
    va = va[np.searchsorted(ia, c)]
    vr = vr[np.searchsorted(ir, c)]
    Fr = L.frame(c)
    key = (Fr["ADEP_mvt"].cast(pl.Utf8) + "|" + Fr["MVT_TIME_UTC_mvt"].dt.date().cast(pl.Utf8)).to_numpy()
    inv = np.unique(key, return_inverse=True)[1]
    dd = va - vr
    mu = (np.bincount(inv, dd) / np.bincount(inv))[inv]
    np.save(f"{OP}_WD_{t}.npy", np.vstack([c, va - mu]))
    np.save(f"{OP}_DAY_{t}.npy", np.vstack([c, vr + mu]))
subprocess.run(
    [sys.executable, "scripts/gate_unsealed_folds.py", f"WD=npy:{OP}_WD~npy:{RP}", f"DAY=npy:{OP}_DAY~npy:{RP}"],
    check=True,
)
