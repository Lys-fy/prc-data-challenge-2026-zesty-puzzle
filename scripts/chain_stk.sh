#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold side of one chain: 6-fold stacked OOF of the meta stacker, then leave-one-fold-out C4 and tree D; one arm per call.
# usage: bash scripts/chain_stk.sh REG|BXK|CTL0|CTL1|SDs|SDu|KBdTK|KBKq|KBNNA3      outputs scratchpad/p13/P13_<N>STK_*, P13_<N>TD_* (+ jitter copies)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/p13; X=scratchpad/xof; N=$1; mkdir -p $D
export oof_build_folds=6 NT=${NT:-6} OMP_NUM_THREADS=${NT:-6} METAFIT=pairs OL=queue_eb_ra3qa OC=meta_oof_h_ctl ON=NNA2 QOOF=fallback_regpl CLOCK_OOF=lgb_adsb_linked_a EXTRA=lnk,xlk,xdk,xapt DARK_OOF=lgb_noclock_nc APT_MEMBER=lgb_airport_ltfm:LTFM,lgb_airport_lfpg:LFPG
export STACKSRC=scripts/chain_stack.py MSEED=0; REF=$X/oof_build_f6kdstk
RAW=te2_srh_te,mvt_eobt1,ref_sr_mean,mvt_aobt3,mvt_iobt,aobt3_minus_ref,te2_sr_te,mvt_sched,aobt3_eobt1,mvt_lobt
case $N in
  REG) ;;                                        # regression: must equal oof_build_f6kdstk (max|diff| 0)
  BXK) export EXTRA=$EXTRA,xl;;
  CTL0) export MSEED=1;;
  CTL1) export MSEED=1 OL=P13S1;;
  SDs) export OL=P13SDs;;
  SDu) export OL=P13SD;;
  REGPX) export OL=queue_eb_dt_dt EXTRA=$EXTRA,xpr; REF=$X/oof_build_pxdtkstk;;   # regression: must equal oof_build_pxdtkstk
  KBdTK) export OL=queue_eb_dt_dt EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW; REF=$X/oof_build_pbdtkstk;;
  KBKq) export OC=runway_queue_catq EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW; REF=$X/oof_build_pbkqstk;;
  KBNNA3) export ON=meta_oof_realmlp_nna3 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW; REF=$X/oof_build_pbnna3stk;;
  *) echo "N?"; exit 1;;
esac
NB=$N; if [ "${pru:-0}" = 1 ]; then export EXTRA=$EXTRA,xpr; N=${N}x; fi   # xpr form: EXTRA += xpr, output name <N>x
echo "P13 $N: OL=$OL OC=$OC ON=$ON EXTRA=$EXTRA MSEED=$MSEED"
NAME=$N REF=$REF $P scripts/chain_mkstk.py | cut -c1-160
[ "$N" = REG -o "$N" = REGPX ] && exit 0
daystate_root=${PRC_ROOT:-$PWD} daystate_oofdirs=${PRC_ROOT:-$PWD}/$D assembler_fl=$OL assembler_fc=$OC $P scripts/stage12_run.py scripts/combo_treed_apply.py --target none --pre none --out none \
  --fit-oof P13_${N}STK --gate --dry --save-oof $D/P13_${N}TD 2>&1 | grep -E "C4 \+ tree\[D\] vs|Error|Trace"
$P scripts/stage12_jitter.py $D/P13_${N}TD 25 2 >/dev/null
ls $D/P13_${N}TD_*.npy | wc -l
