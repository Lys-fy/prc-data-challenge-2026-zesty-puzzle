#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KSTK_CLOCK chain variant: the Kumo member KM is visible to the clock stages (Huber clock stage, A0 + leave stage) as the design column
# dd_kmx_taxi = base - KM taxi. The stack and C4 / tree D of the chain variant <PFX> (k250_stack in the final line) are reused (fold P13_<PFX><C>xTD, production
# PR_<PFX><C>x_XK_C4treeD); only the clock stage and A0 + L are refitted -> <PFX>C<C>.
# usage: KM=kumo_member_screen_km KMP=kumo_member_screen_kmp PFX=kumo_member NT=8 bash scripts/kumo_clock_chain.sh Kq|NNA3|gate       resumable
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; A=scratchpad/xau1; H=scratchpad/xhook; O=scratchpad/xchain; S=submissions; mkdir -p $O
KM=${KM:?}; KMP=${KMP:?}; PFX=${PFX:?}; NT=${NT:-8}; C=$1; export chain_fold_line_km=$KM chain_fold_line_kmp=$KMP
TV="/usr/bin/time -f clock_chain_fix_time_%C_wall=%es_maxrss=%MkB"; echo "CLOCK_CHAIN_FIX fix2 $PFX C=$C KM=$KM NT=$NT host $(hostname) $(date -u +%FT%TZ)"
if [ "$C" != gate ]; then NM=${PFX}C$C; t0=$(date +%s)
  if [ ! -f $A/clockfold1_${NM}_HA0L_janjul.npy ]; then
    ls $H/P13_${PFX}${C}xTD_janjul.npy $H/P13_${PFX}${C}xTDj2_janjul.npy >/dev/null
    clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=$NT $TV bash scripts/kumo_clock_stage.sh $H/P13_${PFX}${C}xTD $NM 2>&1
  fi; echo "clock_chain_fix_step fold $NM $(( $(date +%s) - t0 )) s"
  TD=$H/PR_${PFX}${C}x_XK_C4treeD.parquet; OUT=$H/PR_${NM}_XKAU1
  if [ -f $TD ] && [ ! -f ${OUT}_HA0L.parquet ]; then   # prod: same calls as chainvar_bundle_prod.sh prodstage (FORM=PC, A1, MS6 -> L seeds 6), runner swapped
    export pushstart_adsb6=1 clockfold1_adsbl=1 adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600 oof_build_folds=6 OMP_NUM_THREADS=$NT PYTHONHASHSEED=0
    $TV $P scripts/kumo_clock_run.py scripts/chain_clock_apply.py --target $TD --fit-oof $H/P13_${PFX}${C}xTD,$H/P13_${PFX}${C}xTDj1,$H/P13_${PFX}${C}xTDj2 --out ${OUT}_Hu6.parquet \
      --fset clkonly --shrink 1.0 --clip 300 --seeds 6 --wmode sqrt --objective huber --alpha 600 --nt $NT --excl-lirf --v53 stages/unmatched_rule.parquet 2>&1 | grep -E "kmx|capacity|wrote|clock_chain_fix_time"
    $TV $P scripts/kumo_clock_run.py scripts/leave_apply.py --target ${OUT}_Hu6.parquet --pre $TD --npz $A/clockfold1_${NM}_HuJ.npz --gnpz $A/clockfold1_${NM}_HuJ.npz \
      --out ${OUT}_HA0L.parquet --lirf-excl --arm L --seeds 6 --nt $NT 2>&1 | grep -E "kmx|wrote|clock_chain_fix_time"
    md5sum ${OUT}_HA0L.parquet; echo "clock_chain_fix_step prod $NM $(( $(date +%s) - t0 )) s"
  else echo "prod: TD $TD missing or output exists"; fi
else
  W5=0.3212,0.3065,0.3723; UM=scratchpad/xau3/um25_A1.npz; KREF=${KREF:-$PFX}   # GPU box: KREF=<PFX>G = the kumo_member stages redrawn on GPU
  [ -f $O/chain_fold_line_${PFX}_C_LE3B_janjul.npy ] || $P scripts/pru_daily_calib.py --pool $A/clockfold1_ldtk_ha0l,$A/clockfold1_${PFX}CKq_HA0L,$A/clockfold1_${PFX}CNNA3_HA0L --w $W5 --out $O/chain_fold_line_${PFX}_C_LE3 --um25 $UM | grep -E "INFO B " | head -2
  [ -f $O/chain_fold_line_${KREF}_w5_LE3B_janjul.npy ] || $P scripts/pru_daily_calib.py --pool $A/clockfold1_ldtk_ha0l,$A/clockfold1_${KREF}Kq_HA0L,$A/clockfold1_${KREF}NNA3_HA0L --w $W5 --out $O/chain_fold_line_${KREF}_w5_LE3 --um25 $UM | grep -E "INFO B " | head -2
  $P scripts/gate_unsealed_folds.py "GATE_fix2=npy:$O/chain_fold_line_${PFX}_C_LE3B~npy:$O/chain_fold_line_${KREF}_w5_LE3B" \
    "info_Kq_HA0L=npy:$A/clockfold1_${PFX}CKq_HA0L~npy:$A/clockfold1_${KREF}Kq_HA0L" "info_NNA3_HA0L=npy:$A/clockfold1_${PFX}CNNA3_HA0L~npy:$A/clockfold1_${KREF}NNA3_HA0L" \
    "info_Kq_Hu6=npy:$A/clockfold1_${PFX}CKq_Hu6~npy:$A/clockfold1_${KREF}Kq_Hu6" | grep -E "^\[" | tee $O/fix2_${PFX}.txt
  $P scripts/gate_within_day.py $O/chain_fold_line_${PFX}_C_LE3B $O/chain_fold_line_${KREF}_w5_LE3B $O/chain_fold_line_fix2_${PFX} months39,months28,months410,janjul | tail -4 | tee -a $O/fix2_${PFX}.txt
fi
echo clock_chain_fix_kmx_done
