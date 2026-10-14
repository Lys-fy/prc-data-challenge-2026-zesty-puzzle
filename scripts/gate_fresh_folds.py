#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""gate_unsealed_folds.py exec'd unchanged except the fold list = the two chain folds no gate has ever read
(months511, months612; both are 6-fold-protocol OOF in every chain). Same UNS rows, src_MW weights, columns. Rule printed = gate_unsealed_folds.py's MW(uns)
on 2 folds (m < 0 on both, raw w/o top-3 < 0 on both, m3 < 0 on both). usage as gate_unsealed_folds.py."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(f"{ROOT}/scripts/gate_unsealed_folds.py").read()
a = 'load, ev, cols, rule, FO = H["load"], H["ev"], H["cols"], H["rule"], H["FO"]'
assert src.count(a) == 1
src = src.replace(a, a + '; FO = ["months511", "months612"]')
exec(
    compile(src, "xarch_fresh", "exec"), {"__file__": f"{ROOT}/scripts/gate_unsealed_folds.py", "__name__": "__main__"}
)
