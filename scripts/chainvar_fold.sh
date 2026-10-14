#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold side of one chain variant: 6-fold stack with the wide-leaf member and KM as extra inputs (scripts/chainvar_stk.sh) -> C4 / tree D ->
# clock stages (Huber clock stage with adsb6 + adsbL clocks, A0 + L) -> scratchpad/xau1/clockfold1_<PFX><C>_*. Every step is timed (/usr/bin/time: wall + peak RSS).
# usage: KM=kumo_member_screen_km PFX=kumo_member NT=10 bash scripts/chainvar_fold.sh Kq|NNA3        needs data/oof/<KM>_<fold>.npy on all 6 folds (chainvar_check.py km)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; C=$1; KM=${KM:?}; PFX=${PFX:?}; export KM PFX
TV="/usr/bin/time -f chain_hook_time_%C_wall=%es_maxrss=%MkB"
for f in months39 months28 months410 months511 months612 janjul; do ls data/oof/wideleaf_retune_a2_${f}_*.npy data/oof/${KM}_${f}.npy >/dev/null; done
t0=$(date +%s)
pru=1 NT=${NT:-4} $TV bash scripts/chainvar_stk.sh $C 2>&1
ls scratchpad/xhook/P13_${PFX}${C}xTD_janjul.npy scratchpad/xhook/P13_${PFX}${C}xTDj2_janjul.npy >/dev/null
t1=$(date +%s); echo "chain_hook_step stack+C4/treeD ${PFX}${C} $((t1-t0)) s"
clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=${NT:-4} $TV bash scripts/clockstage_fold.sh scratchpad/xhook/P13_${PFX}${C}xTD ${PFX}${C} 2>&1
ls scratchpad/xau1/clockfold1_${PFX}${C}_HA0L_*.npy scratchpad/xau1/clockfold1_${PFX}${C}_HuJ.npz >/dev/null
t2=$(date +%s); echo "chain_hook_step CLOCKFOLD1 stages ${PFX}${C} $((t2-t1)) s"; echo "chain_hook_step fold total ${PFX}${C} $((t2-t0)) s"
