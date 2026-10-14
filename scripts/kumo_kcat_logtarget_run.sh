#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# KCATL member instances (log1p target; scripts/kumo_kcat_logtarget_member.py) on a CUDA GPU, 8 instances per airport.
#   folds: the six 2025 month folds (fold frames = $PAYLOAD of scripts/kumo_env.sh, fold codes $KCAT_CODES/folds); run once.
#   prod:  the period's ranking rows (the period's K250 payload, $KCAT_PROD_PAYLOAD, and its codes $KCAT_CODES/prod).
# Resumable: existing instance files are skipped. Output $KCATL_RUN/results/engine/KCATL/<fold>_<APT>/inst<i>.npz.
# usage: bash scripts/kumo_kcat_logtarget_run.sh folds|prod        SMOKE=1: CPU code-path smoke (janjul x LSZH, 2,000 rows) in a separate run dir.
# Root = $PRC_ROOT, else the current directory.
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; R=$PWD; W=${1:?folds|prod}; kumo_engine_job=kcatl_$W; source scripts/kumo_env.sh
V=${KCAT_CODES:-$R/scratchpad/kcat/codes}
export RUN=${KCATL_RUN:-$R/scratchpad/kcat/runL} NAME=KCATL CATMODE=cat SIZE=medium CAP=1000000 NCOL=100 COLMODE=imp CORE=50 NINST=8 EIN=1 CTXMODE=uniform SEED=0 AMP=fp16
FOLDS="janjul months28 months39 months410 months511 months612"; II="0 1 2 3 4 5 6 7"
if [ "${SMOKE:-0}" = 1 ]; then export RUN=${RUN}_smoke DEV=cpu SMOKE_ROWS=2000 CHUNK=512; rm -rf "$RUN"; FOLDS=janjul; APTS=LSZH; II=0; W=folds; fi
mkdir -p "$RUN"; U=""
if [ "$W" = folds ]; then
  for f in $FOLDS; do for a in $APTS; do for i in $II; do U="$U $f:$a:$i"; done; done; done
  CODES=$V/folds $KPY scripts/kumo_kcat_logtarget_member.py $U
else
  for a in $APTS; do for i in $II; do U="$U prod:$a:$i"; done; done
  PAYLOAD=${KCAT_PROD_PAYLOAD:?set KCAT_PROD_PAYLOAD (the period K250 payload)} CODES=$V/prod $KPY scripts/kumo_kcat_logtarget_member.py $U
fi
echo "KCATL_RUN_DONE $W $(find "$RUN/results/engine/KCATL" -name 'inst?.npz' | wc -l) instance files"
