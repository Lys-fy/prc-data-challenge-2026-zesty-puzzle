#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run an existing stage script (combo_treed_apply.py, combo_c4_apply.py, ...) with the fold list of daystate_lib / clockresid_lib set by env oof_build_folds
(4 = unchanged, 6 = + months612, months511), without editing the shared libraries. usage: stage12_run.py <script.py> [args...]
"""

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
print("stage12_run folds", daystate_lib.FOLDS, flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
