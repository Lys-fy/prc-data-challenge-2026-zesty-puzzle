#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# KCATL16 bag = the Kumo engine aggregation (trimmed mean 0.2 over NINST 16) of KCAT8 instances 0-7 (as instances 0-7) and KCATL instances
# 0-7 (as instances 8-15), per airport. The two halves share contexts, columns and generators; only the target differs (raw vs log1p).
#   folds: aggregate the six 2025 folds and collect them into data/oof/KCATL16_<fold>.npy (scripts/kumo_kcat_collect.sh).
#   prod:  aggregate the period's ranking rows -> $KCATL16_RUN/results/member/KCATL16_prod_<APT>.npz (collected by scripts/kumo_prod.sh kcat).
# usage: bash scripts/kumo_kcat_logtarget_bag.sh folds|prod        Root = $PRC_ROOT, else the current directory.
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; R=$PWD; W=${1:?folds|prod}; kumo_engine_job=kcatl16_$W; source scripts/kumo_env.sh
K8=${KCAT_RUN:-$R/scratchpad/kcat/runW}; L=${KCATL_RUN:-$R/scratchpad/kcat/runL}; RB=${KCATL16_RUN:-$R/scratchpad/kcat/runL16}; V=${KCAT_CODES:-$R/scratchpad/kcat/codes}
E=$RB/results/engine/KCATL16; mkdir -p "$E"; U=$RB/units_$W.txt; : > "$U"
FOLDS="janjul months28 months39 months410 months511 months612"; [ "$W" = prod ] && FOLDS=prod
for f in $FOLDS; do for a in $APTS; do d=$E/${f}_$a; mkdir -p "$d"
  for i in 0 1 2 3 4 5 6 7; do
    ln -sf "$K8/results/engine/KCAT/${f}_$a/inst$i.npz" "$d/inst$i.npz"; ln -sf "$L/results/engine/KCATL/${f}_$a/inst$i.npz" "$d/inst$((i + 8)).npz"; done
  echo "agg:$f:$a" >> "$U"; done; done
export RUN=$RB NAME=KCATL16 NINST=16 CATMODE=cat SIZE=medium CAP=1000000 NCOL=100 COLMODE=imp CORE=50 EIN=1 CTXMODE=uniform SEED=0 AMP=fp16
if [ "$W" = prod ]; then
  PAYLOAD=${KCAT_PROD_PAYLOAD:?set KCAT_PROD_PAYLOAD (the period K250 payload)} CODES=$V/prod $KPY scripts/kumo_kcat_member.py --from-list "$U"
else
  CODES=$V/folds $KPY scripts/kumo_kcat_member.py --from-list "$U"
  NAME=KCATL16 R=$RB FOLDS="$FOLDS" bash scripts/kumo_kcat_collect.sh
fi
ls "$RB/results/member" | sed 's/_[A-Z]*\.npz//' | sort | uniq -c
echo KCATL16_BAG_DONE "$W"
