#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# KCAT8: collect KCAT members of $FOLDS (default janjul months28; KCAT8 needs all 6 folds) from $R (runW) into data/oof/KCAT_<fold>.npy
# (prod is NOT collected here: it uses the period's frames and is collected into the workspace by scripts/kumo_prod.sh kcat).
# usage: FOLDS="months39 months28 months410 months511 months612 janjul" bash scripts/kumo_kcat_collect.sh      (root = $PRC_ROOT, else the current directory)
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; NAME=${NAME:-KCAT}; R=${R:-scratchpad/kcat/runW}; T=scratchpad/kcat/collect2_$NAME; rm -rf $T; mkdir -p $T/results/member
FOLDS=${FOLDS:-janjul months28}; for f in $FOLDS; do n=$(ls $R/results/member/${NAME}_${f}_*.npz | wc -l); [ $n = 10 ] || { echo "$f: $n/10 members"; exit 2; }; ln -s $(readlink -f $R)/results/member/${NAME}_${f}_*.npz $T/results/member/; done
.venv/bin/python kumo_gpu/collect/collect.py $T
for f in $FOLDS; do md5sum data/oof/${NAME}_$f.npy; done
