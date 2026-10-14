#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Nested seed-MSEED stacked OOF of the chain meta (BASEC + L2 + EXTRA inputs, pairs protocol, clip +-150 s) on 4 or 6 folds
(env oof_build_folds); env OL / OC / ON / QOOF / EXTRA / CLOCK_OOF as in the stacker source.
Stacker source env oof_build_src (default scripts/stage12_stack.py); meta seed env MSEED (default 0).
Writes scratchpad/xof/oof_build_<NAME>STK_<fold>.npy = [MVT_ID, pred]."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("scratchpad/xof", exist_ok=True)
src = open(os.environ.get("oof_build_src", "scripts/stage12_stack.py")).read().split("variants = sys.argv")[0]
g = {"__file__": os.path.abspath("scripts/fallback_stack.py"), "__name__": "xf"}
exec(compile(src, "xf_stack", "exec"), g)
cols = g["BASEC"] + g["EXTRA"]["L2"] + sum((g["EXTRA"][x] for x in g["extra_inputs"]), [])
print(
    "folds",
    g["ALL"],
    "pairs",
    g["PAIRS"],
    "cols",
    cols,
    "clip",
    g["CLIP_LO"],
    g["CLIP_HI"],
    "METAFIT",
    g["METAFIT"],
    flush=True,
)
out = g["corr"](cols, int(os.environ.get("MSEED", "0")))
D = g["D"]
nm = os.environ["NAME"]
for t, v in out.items():
    np.save(f"scratchpad/xof/oof_build_{nm}STK_{t}.npy", np.vstack([D[t]["mid"], v]))
    msg = f"{nm} {t} n {len(v):,} mse {np.mean((v - D[t]['y']) ** 2):.1f}"
    ref = f"scratchpad/xu/unm_resid2_blkb_{t}.npy"
    if os.path.exists(ref):
        r = np.load(ref)
        o = np.argsort(r[0])
        k = np.searchsorted(r[0][o], D[t]["mid"])
        k = np.clip(k, 0, len(o) - 1)
        hit = r[0][o][k] == D[t]["mid"]
        msg += f" | vs unm_resid2_blkb: rows hit {hit.mean():.4f} max|diff| {np.abs(r[1][o][k][hit] - v[hit]).max():.4f} dMSE {np.mean((v[hit]-D[t]['y'][hit])**2 - (r[1][o][k][hit]-D[t]['y'][hit])**2):+.2f}"
    print(msg, flush=True)
