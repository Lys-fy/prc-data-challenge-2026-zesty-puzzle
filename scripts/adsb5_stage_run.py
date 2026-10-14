#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run ANY clockresid_lib-based stage script (stage12_clock_apply.py, chain_clock_apply.py, leave_apply.py, leave_fit_tuned.py, ...) with the adsb5 side
table added to the clock inputs, like stage12_run.py does for the fold list. Env: adsb5_clock_adsb5=1 (default 1 here), oof_build_folds=4|6.
usage: adsb5_clock_adsb5=1 oof_build_folds=6 adsb5_stage_run.py <script.py> [args...]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import daystate_lib, clockresid_lib

if os.environ.get("oof_build_folds", "4") == "6":
    daystate_lib.FOLDS = daystate_lib.FOLDS + ["months612", "months511"]
    clockresid_lib.FOLDS = clockresid_lib.FOLDS + ["months612", "months511"]
if os.environ.get("adsb5_clock_adsb5", "1") == "1":
    clockresid_lib.ADSB = clockresid_lib.ADSB + [f"{ROOT}/data/external/adsb5_features.parquet"]
    clockresid_lib.TIMES = clockresid_lib.TIMES + ["adsb5_push", "adsb5_cusp", "adsb5_mv2"]
    _cc = clockresid_lib.clock_cols
    clockresid_lib.clock_cols = lambda F: _cc(F) + [
        c for c in F.columns if c.startswith("adsb5_") and F[c].dtype.is_numeric()
    ]
print("adsb5_stage_run folds", clockresid_lib.FOLDS, "adsb5", os.environ.get("adsb5_clock_adsb5", "1"), flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
