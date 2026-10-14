#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# trainer_diff.sh: prints `diff -u scripts/train2.py scripts/train2_<x>.py` for each copy of the member trainer (docs/PIPELINE.md §3).
# usage: bash tools/trainer_diff.sh [<x> ...]      (no argument: every copy)
cd "${PRC_ROOT:-$(dirname "$0")/..}" || exit 1
if [ $# -gt 0 ]; then set -- "${@/#/scripts/train2_}"; set -- "${@/%/.py}"; else set -- scripts/train2_*.py; fi
for f in "$@"; do
  echo "=== $f"
  diff -u scripts/train2.py "$f"
done
exit 0
