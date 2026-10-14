#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fixed-weight mix of fold vectors on the 4 original folds: cf(sum w_i v_i). usage: stage12_mix.py <out prefix> w1,w2,.. <prefix1> <prefix2> ..."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L

out, w, pr = sys.argv[1], [float(x) for x in sys.argv[2].split(",")], sys.argv[3:]
assert len(w) == len(pr) and abs(sum(w) - 1) < 1e-3
for t in ["months39", "months28", "months410", "janjul"]:
    V = []
    for p in pr:
        z = np.load(sorted(glob.glob(f"{p}_{t}.npy"))[0])
        o = np.argsort(z[0])
        V.append((z[0][o], z[1][o]))
    ids = V[0][0]
    for i, _ in V[1:]:
        assert np.array_equal(i, ids)
    lo = L.frame(ids)["mvt_lobt"].to_numpy().astype(float)
    np.save(f"{out}_{t}.npy", np.vstack([ids, L.cf(sum(wi * v for wi, (_, v) in zip(w, V)), lo)]))
    print(t, len(ids))
