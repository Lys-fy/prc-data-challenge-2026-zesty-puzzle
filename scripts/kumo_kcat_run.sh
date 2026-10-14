#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# KCAT8 widen on the GPU workstation: KCAT (K250 spec + C8 declared categoricals) on all 10 airports for ONE fold (arg: janjul|months28|...|prod).
# Folds use the fold frames (kumo_env.sh PAYLOAD = the kumo_gpu payload) + fold codes; prod uses the period's K250 frames + its codes
# (codes: scripts/kumo_kcat_codes.py). Output: $RUN/results/{engine/KCAT/<fold>_<APT>/inst*.npz, member/KCAT_<fold>_<APT>.npz}.
# Root = $PRC_ROOT (else the current directory); KCAT_RUN, KCAT_CODES, KCAT_PROD_PAYLOAD override the defaults below.
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; R=$PWD; F=${1:?fold}; kumo_engine_job=xkumo3w_$F; source scripts/kumo_env.sh
export RUN=${KCAT_RUN:-$R/scratchpad/kcat/runW} NAME=KCAT CATMODE=cat SIZE=medium CAP=1000000 NCOL=100 COLMODE=imp CORE=50 NINST=8 EIN=1 CTXMODE=uniform SEED=0 AMP=fp16
V=${KCAT_CODES:-$R/scratchpad/kcat/codes}
if [ "$F" = prod ]; then export PAYLOAD=${KCAT_PROD_PAYLOAD:?set KCAT_PROD_PAYLOAD (scripts/kumo_prod.sh kcat sets the workspace K250 payload)} CODES=$V/prod; else export CODES=$V/folds; fi
mkdir -p "$RUN"; L=$RUN/units_$F.txt; : > $L
for a in $APTS; do for i in 0 1 2 3 4 5 6 7; do echo "inst:$F:$a:$i" >> $L; done; echo "agg:$F:$a" >> $L; done
$KPY scripts/kumo_kcat_member.py --from-list $L
echo "kcat8_j3_done $F $(ls $RUN/results/member/KCAT_${F}_*.npz | wc -l)/10 members"
