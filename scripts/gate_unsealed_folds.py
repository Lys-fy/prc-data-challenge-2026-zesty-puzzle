#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Unsealed-only view of scripts/sealed_score.py. Execs sealed_score.py's setup (loader, weights, sealed mask)
unchanged, then for each pair computes ONLY the UNS row set (~sealed): raw, raw w/o top-3, m = src_MW, m3 (rescaled to the full-fold weight
sum exactly as sealed_score) and the MW(uns) rule. Nothing about the sealed rows is computed or printed.
usage: gate_unsealed_folds.py LABEL=NEW~REF ...      (pair specs as month_weight_rescore / sealed_score)
Output lines:  [LABEL] <fold> UNS raw R [R3] m M m3 M3 | ...   and   [LABEL] mean m UNS X | RULE MW(uns) PASS|FAIL"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PAIRS = sys.argv[1:]
src = open(f"{ROOT}/scripts/sealed_score.py").read().split('print("## pairs:')[0]
sys.argv = [f"{ROOT}/scripts/sealed_score.py"]; H = {"__file__": f"{ROOT}/scripts/sealed_score.py", "__name__": "xl1d_uns"}; exec(src, H)
load, ev, cols, rule, FO = H["load"], H["ev"], H["cols"], H["rule"], H["FO"]
print("## UNS only: per fold raw [raw w/o top-3] m (src_MW, board s2, rescaled to the full-fold weight sum) m3", flush=True)
for p in PAIRS:
    lab, rest = p.split("=", 1); ns, rs = rest.split("~"); RU, folds = [], []
    for t in FO:
        a, b = load(ns, t), load(rs, t)
        if a is None or b is None: continue
        (i1, v1), (i0, v0) = a, b; common = np.intersect1d(i1, i0); v1 = v1[np.searchsorted(i1, common)]; v0 = v0[np.searchsorted(i0, common)]
        E = ev(t, common); ok = E["ok"]; y = E["y"][ok]
        assert np.isfinite(v1[ok]).all() and np.isfinite(v0[ok]).all(), (lab, t)
        d = (v1[ok] - y) ** 2 - (v0[ok] - y) ** 2; wm = E["wm"]; wd = wm * d; U = ~E["sealed"]
        r = np.array(cols(d, wd, U)); r[2:] *= wm.sum() / wm[U].sum(); RU.append(r); folds.append(t)
    if not RU: print(f"[{lab}] no folds", flush=True); continue
    RU = np.array(RU)
    print(f"[{lab}] " + " | ".join(f"{t} UNS raw {u[0]:+7.1f} [{u[1]:+7.1f}] m {u[2]:+7.1f} m3 {u[3]:+7.1f}" for t, u in zip(folds, RU)), flush=True)
    print(f"[{lab}] mean m UNS {RU[:, 2].mean():+.1f} | RULE MW(uns) {'PASS' if rule(RU, folds) else 'FAIL'} ({len(folds)} folds)", flush=True)
