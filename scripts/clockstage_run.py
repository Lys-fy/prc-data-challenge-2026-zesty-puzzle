#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Runs a clockresid_lib-based stage script with the adsbL side table (longest-stop clocks) added to the clock inputs, through
pushstart_run.py (adsb6) and adsb5_stage_run.py (adsb5, 6 folds). Env: clockfold1_adsbl=1 (default) | 0 (then the run equals pushstart_run.py).
usage: clockfold1_adsbl=1 pushstart_adsb6=1 adsb5_clock_adsb5=1 oof_build_folds=6 clockstage_run.py <script.py> [args...]
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib

if os.environ.get("clockfold1_adsbl", "1") == "1":
    clockresid_lib.ADSB = clockresid_lib.ADSB + [f"{ROOT}/data/external/adsbL_features.parquet"]
    clockresid_lib.TIMES = clockresid_lib.TIMES + ["adsbL_stop_end", "adsbL_last_end"]
    _cc0 = clockresid_lib.clock_cols

    def _cc(F, _c=_cc0):
        return _c(F) + [c for c in F.columns if c.startswith("adsbL_") and F[c].dtype.is_numeric() and c not in _c(F)]

    clockresid_lib.clock_cols = _cc
print("clockstage_run adsbL", os.environ.get("clockfold1_adsbl", "1"), flush=True)
sys.argv = [f"{ROOT}/scripts/pushstart_run.py"] + sys.argv[1:]
runpy.run_path(f"{ROOT}/scripts/pushstart_run.py", run_name="__main__")
