#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Jittered copies of a fold vector for clock_apply's augmentation path: <out>j<c>_<fold>.npy = [ID, cf(b + N(0, sd))], b = cf(pred).
usage: stage12_jitter.py <prefix> <sd> <ncopy>   (6 folds; rng seed 7)"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L

pre, sd, nc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(7)
FO = ["months39", "months28", "months410", "janjul", "months612", "months511"]
Z = {t: np.load(sorted(glob.glob(f"{pre}_{t}.npy"))[0]) for t in FO}
for t in FO:
    lo = L.frame(Z[t][0])["mvt_lobt"].to_numpy().astype(float)
    b = L.cf(Z[t][1], lo)
    for c in range(1, nc + 1):
        np.save(f"{pre}j{c}_{t}.npy", np.vstack([Z[t][0], L.cf(b + rng.normal(0, sd, len(b)), lo)]))
    print(t, len(b), flush=True)
