#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Consistency check of phase2/params/constants_2025.json against the code: every constant's code literal must occur in the file named
next to it, and every fitted file that is part of the repository must exist. Reads files only; exit status 1 on any mismatch.
usage: python tools/check_constants.py [repo root]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import json, os, sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(os.path.join(R, "phase2/params/constants_2025.json")))
bad = 0
n = 0
for c in J["constants"]:
    for ref in c.get("code", []):
        n += 1
        p = os.path.join(R, ref["file"])
        ok = os.path.isfile(p) and ref["literal"] in open(p, encoding="utf-8").read()
        if not ok:
            bad += 1
            print(f"MISMATCH {c['name']}: {ref['literal']!r} not in {ref['file']}")
for f in J["fitted_files"]:
    if f["file"].startswith("phase2/params/"):
        n += 1
        if not os.path.isfile(os.path.join(R, f["file"])):
            bad += 1
            print(f"MISSING {f['file']}")
print(f"constants_2025.json: {len(J['constants'])} constants, {n} checks, {bad} problems")
sys.exit(1 if bad else 0)
