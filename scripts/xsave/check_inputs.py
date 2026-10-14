#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""List the row-level input files a member command reads (features + side tables of its --feats groups) and fail if any is still a
symlink out of the workspace (= not rebuilt for this period). usage: check_inputs.py WS FEATS FAM '<command>'   (ALLOW_LINK=1 to skip the failure)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, shlex

WS, FEATS, FAM, CMD = sys.argv[1:5]
t = shlex.split(CMD)
groups = t[t.index("--feats") + 1].split(",") if "--feats" in t else []
SIDE = {
    "durp": "durp",
    "dreg": "dreg",
    "adsb": "adsb",
    "adsb2": "adsb2",
    "adsb3": "adsb3",
    "xba": "xba",
    "xe": "xe_queue",
    "xqP": "queue_decomp_q_prod",
}
side = "--side-tables" in t or not ("nn_resid.py" in CMD)
files = [FEATS] + ([f"data/external/{SIDE[g]}_features.parquet" for g in groups if g in SIDE] if side else [])
if FAM.startswith("NBR_CLOCK"):
    files = [FEATS] + [
        f"data/external/{x}_features.parquet" for x in ("durp", "dreg", "adsb", "adsb2", "adsb3", "xba", "xe_queue")
    ]
bad = []
for f in files:
    p = os.path.join(WS, f) if not os.path.isabs(f) else f
    real = os.path.realpath(p)
    ok = os.path.exists(p)
    link = os.path.islink(p) and not real.startswith(os.path.realpath(WS) + os.sep)
    print(
        f"  input {f}: {'MISSING' if not ok else ('LINK->' + real if link else 'local')} {os.path.getsize(real) / 1e6 if ok else 0:.0f} MB"
    )
    if not ok or link:
        bad.append(f)
if bad and os.environ.get("ALLOW_LINK") != "1":
    sys.exit(f"inputs not rebuilt in {WS}: {bad} (ALLOW_LINK=1 to override)")
