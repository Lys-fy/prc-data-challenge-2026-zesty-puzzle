#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Stack of one pseudo-label chain (KBKq / KBNNA3 form, pru=1 required): the chain stack with the L member replaced by the pseudo-labelled
# twin (OL = pseudolabel_<ARM>), stack source scripts/chain_stack.py, outputs in scratchpad/xsyn as P13_XS<ARM><C>xSTK_* / P13_XS<ARM><C>xTD_* (+ jitter copies).
# usage: pru=1 bash scripts/pseudo_stk.sh PL3|B0 Kq|NNA3        REF = the base T2x stack scratchpad/p13/P13_KB<C>xSTK (info dMSE only)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/xsyn; X=scratchpad/p13; ARM=$1; C=$2; mkdir -p $D; export P13_D=$D
[ "${pru:-0}" = 1 ] || { echo "pseudo_stk.sh needs pru=1 (xpr input + x name; the wideleaf_fold.sh bug)"; exit 2; }
export oof_build_folds=6 NT=${NT:-4} OMP_NUM_THREADS=${NT:-4} METAFIT=pairs OL=pseudolabel_$ARM OC=meta_oof_h_ctl ON=NNA2 QOOF=fallback_regpl CLOCK_OOF=lgb_adsb_linked_a EXTRA=lnk,xlk,xdk,xapt DARK_OOF=lgb_noclock_nc APT_MEMBER=lgb_airport_ltfm:LTFM,lgb_airport_lfpg:LFPG
export STACKSRC=scripts/chain_stack.py MSEED=0
RAW=te2_srh_te,mvt_eobt1,ref_sr_mean,mvt_aobt3,mvt_iobt,aobt3_minus_ref,te2_sr_te,mvt_sched,aobt3_eobt1,mvt_lobt
case $C in
  Kq)   export OC=runway_queue_catq EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW; REF=$X/P13_KBKqxSTK;;
  NNA3) export ON=meta_oof_realmlp_nna3 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW; REF=$X/P13_KBNNA3xSTK;;
  *) echo "C?"; exit 1;;
esac
N=XS${ARM}${C}; export EXTRA=$EXTRA,xpr; N=${N}x
echo "PSEUDOLABEL stack $N: OL=$OL OC=$OC ON=$ON EXTRA=$EXTRA MSEED=$MSEED"
NAME=$N REF=$REF $P scripts/chain_mkstk.py | cut -c1-160
daystate_root=${PRC_ROOT:-$PWD} daystate_oofdirs=${PRC_ROOT:-$PWD}/$D assembler_fl=$OL assembler_fc=$OC $P scripts/stage12_run.py scripts/combo_treed_apply.py --target none --pre none --out none \
  --fit-oof P13_${N}STK --gate --dry --save-oof $D/P13_${N}TD 2>&1 | grep -E "C4 \+ tree\[D\] vs|Error|Trace"
$P scripts/stage12_jitter.py $D/P13_${N}TD 25 2 >/dev/null
ls $D/P13_${N}TD_*.npy | wc -l
