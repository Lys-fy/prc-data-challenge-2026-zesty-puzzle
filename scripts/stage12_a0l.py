#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""A0+L fold vector from leave_eval's saved L_strict vectors and the stage-1 npz: A0L = where(LIRF, base, L_strict). usage: stage12_a0l.py <npz> <leave_stage tag> <out prefix>"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L

Z = np.load(sys.argv[1])
ids, fo, b = Z["ids"], Z["fo"], Z["b"]
lir = L.frame(ids)["ADEP_mvt"].cast(str).to_numpy().astype(str) == "LIRF"
for t in np.unique(fo):
    m = fo == t
    s = np.load(f"scratchpad/xlv/leave_stage_{sys.argv[2]}_L_strict_{t}.npy")
    assert np.array_equal(s[0], ids[m])
    np.save(f"{sys.argv[3]}_{t}.npy", np.vstack([ids[m], np.where(lir[m], b[m], s[1])]))
    print(t, m.sum())
