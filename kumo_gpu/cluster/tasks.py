#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Writes the unit lists used by the Slurm arrays and by run_box.sh. One line = one unit (see km_task.py).
usage: python tasks.py <outdir> [--skip-local]
  member.txt  7 folds (6 CV folds + prod) x 10 airports = 70 units  (--skip-local: drop janjul/months28, already on the build machine from the GPU workstation = 50 units)
  t2x.txt     2 chains (Kq, NNA3) x 4 folds x 10 airports = 80 units
  all.txt     member + t2x, largest units first (load balance for multi-GPU boxes)
Line index (0-based) = SLURM_ARRAY_TASK_ID."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys

APTS = [
    "LTFM",
    "EHAM",
    "EGLL",
    "LFPG",
    "EDDF",
    "LEMD",
    "LEBL",
    "EDDM",
    "LIRF",
    "LSZH",
]  # largest first (2025 rows per fold)
FOLDS = ["months39", "months28", "months410", "janjul", "months511", "months612", "prod"]
out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
fo = [f for f in FOLDS if not ("--skip-local" in sys.argv and f in ("janjul", "months28"))]
mem = [f"member:{f}:{a}" for a in APTS for f in fo]
t2x = [
    f"t2x:{c}:{f}:{a}" for a in APTS for c in ("Kq", "NNA3") for f in ("months39", "months28", "months410", "janjul")
]
for n, L in (("member", mem), ("t2x", t2x), ("all", mem + t2x)):
    open(f"{out}/{n}.txt", "w").write("\n".join(L) + "\n")
print(f"member {len(mem)} | t2x {len(t2x)} | all {len(mem) + len(t2x)} -> {out}/")
