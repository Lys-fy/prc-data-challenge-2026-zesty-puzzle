#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""scripts/final_combine.py run with FLOOR = 60 replaced by -1e9 (one asserted edit), for callers that apply the 60 s floor and the
LOBT bound after the combine (scripts/isolate_delta.py step 3). usage as final_combine.py."""

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(f"{ROOT}/scripts/final_combine.py").read()
a = "FLOOR = 60\n"
assert src.count(a) == 1
src = src.replace(a, "FLOOR = -1e9   # ISOLATE_SAMEHOST: floor enforced by the caller\n")
sys.argv[0] = f"{ROOT}/scripts/final_combine.py"
exec(
    compile(src, f"{ROOT}/scripts/final_combine.py", "exec"),
    {"__file__": f"{ROOT}/scripts/final_combine.py", "__name__": "__main__"},
)
