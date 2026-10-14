#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run ANY clockresid_lib-based stage script with the adsb6 side table added to the clock inputs, ON TOP of adsb5_stage_run.py
(which adds adsb5 when adsb5_clock_adsb5=1 and the 6-fold list when oof_build_folds=6). Env: pushstart_adsb6=1 (default) | 0 (then identical to adsb5_stage_run.py).
usage: pushstart_adsb6=1 adsb5_clock_adsb5=1 oof_build_folds=6 pushstart_run.py <script.py> [args...]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib

if os.environ.get("pushstart_adsb6", "1") == "1":
    clockresid_lib.ADSB = clockresid_lib.ADSB + [f"{ROOT}/data/external/adsb6_features.parquet"]
    clockresid_lib.TIMES = clockresid_lib.TIMES + ["adsb6_push", "adsb6_pe", "adsb6_tx"]
    _cc0 = clockresid_lib.clock_cols

    def _cc(F, _c=_cc0):
        return _c(F) + [c for c in F.columns if c.startswith("adsb6_") and F[c].dtype.is_numeric() and c not in _c(F)]

    clockresid_lib.clock_cols = _cc
print("pushstart_run adsb6", os.environ.get("pushstart_adsb6", "1"), flush=True)
# hand over to adsb5_stage_run.py (it patches the fold list / adsb5 on the same module objects, then runs the script)
sys.argv = [f"{ROOT}/scripts/adsb5_stage_run.py"] + sys.argv[1:]
runpy.run_path(f"{ROOT}/scripts/adsb5_stage_run.py", run_name="__main__")
