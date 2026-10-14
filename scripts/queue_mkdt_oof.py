#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fold OOF queue_eb_dt_dt_<fold>.npy (the dTK chain's OL) = queue_eb_ra3qa p/art + queue_decomp_ebqq taxi; the fold equivalent of queue_mkdt.py.
Rows [id, p*art+(1-p)*taxi_qq, p, taxi_qq, art]. Needs data/oof/queue_eb_ra3qa_<fold>.npy and data/oof/queue_decomp_ebqq_<fold>.npy.
usage: queue_mkdt_oof.py <fold> [...]   (folds: months39 months28 months410 janjul months511 months612)"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def ld(p, t):
    return np.load(sorted(glob.glob(f"data/oof/{p}_{t}.npy") or glob.glob(f"data/oof/{p}_{t}_*.npy"))[0])


for t in sys.argv[1:]:
    b = ld("queue_eb_ra3qa", t)
    c = ld("queue_decomp_ebqq", t)
    assert np.array_equal(b[0], c[0]) and np.allclose(b[4], c[4])
    o = np.vstack([b[0], b[2] * b[4] + (1 - b[2]) * c[3], b[2], c[3], b[4]])
    fn = f"data/oof/queue_eb_dt_dt_{t}.npy"
    assert not os.path.exists(fn), fn
    np.save(fn, o)
    print(t, o.shape, "mean pred", o[1].mean(), "vs queue_eb", b[1].mean())
