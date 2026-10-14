#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Stack of one chain variant (pru=1 required): the chain stack with the wide-leaf member as extra input xtd (EXTRA1_OOF = wideleaf_retune_a2) and
# the Kumo member KM as second extra input xtd2 (EXTRA2_OOF = <KM>@1, the [id, taxi] Kumo format); stack source scripts/chainvar_stack.py; outputs in scratchpad/xhook.
# usage: pru=1 KM=kumo_member_screen_km PFX=kumo_member bash scripts/chainvar_stk.sh Kq|NNA3      -> scratchpad/xhook/P13_<PFX><C>xSTK_*, P13_<PFX><C>xTD_* (+ jitter copies)
#        REG=1 (no xtd2) -> P13_<PFX>REG<C>x*: must equal the stack of scripts/wideleaf_stack.py (regression arm)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/xhook; C=$1; mkdir -p $D; export P13_D=$D EXTRA1_OOF=wideleaf_retune_a2
[ "${pru:-0}" = 1 ] || { echo "chainvar_stk.sh needs pru=1 (xpr input + x name; the wideleaf_fold.sh bug)"; exit 2; }
KM=${KM:?KM oof prefix, e.g. kumo_member_screen_km}; PFX=${PFX:?chain prefix, e.g. kumo_member}
export oof_build_folds=6 NT=${NT:-4} OMP_NUM_THREADS=${NT:-4} METAFIT=pairs OL=queue_eb_ra3qa OC=meta_oof_h_ctl ON=NNA2 QOOF=fallback_regpl CLOCK_OOF=lgb_adsb_linked_a EXTRA=lnk,xlk,xdk,xapt DARK_OOF=lgb_noclock_nc APT_MEMBER=lgb_airport_ltfm:LTFM,lgb_airport_lfpg:LFPG
export STACKSRC=${STACKSRC:-scripts/chainvar_stack.py} MSEED=0 EXTRA2_OOF=$KM@1
RAW=te2_srh_te,mvt_eobt1,ref_sr_mean,mvt_aobt3,mvt_iobt,aobt3_minus_ref,te2_sr_te,mvt_sched,aobt3_eobt1,mvt_lobt
X2=",xtd2"; NM=$PFX$C; if [ "${REG:-0}" = 1 ]; then X2=""; NM=${PFX}REG$C; fi
case $C in
  Kq)   export OC=runway_queue_catq EXTRA=$EXTRA,spr,raw,xtd$X2 audited_stack_raw=$RAW; REF=scratchpad/xbase/P13_XBKqxSTK;;
  NNA3) export ON=meta_oof_realmlp_nna3 EXTRA=$EXTRA,spr,raw,xtd$X2 audited_stack_raw=$RAW; REF=scratchpad/xbase/P13_XBNNA3xSTK;;
  *) echo "C?"; exit 1;;
esac
export EXTRA=$EXTRA,xpr; N=${NM}x
echo "CHAIN_HOOK stack $N: KM=$KM EXTRA1_OOF=$EXTRA1_OOF EXTRA2_OOF=$EXTRA2_OOF OL=$OL OC=$OC ON=$ON EXTRA=$EXTRA STACKSRC=$STACKSRC MSEED=$MSEED host $(hostname) $(date -u +%FT%TZ)"
NAME=$N REF=$REF $P scripts/chain_mkstk.py | cut -c1-200
[ "${STKONLY:-0}" = 1 ] && exit 0
daystate_root=${PRC_ROOT:-$PWD} daystate_oofdirs=${PRC_ROOT:-$PWD}/$D assembler_fl=$OL assembler_fc=$OC $P scripts/stage12_run.py scripts/combo_treed_apply.py --target none --pre none --out none \
  --fit-oof P13_${N}STK --gate --dry --save-oof $D/P13_${N}TD 2>&1 | grep -E "C4 \+ tree\[D\] vs|Error|Trace"
$P scripts/stage12_jitter.py $D/P13_${N}TD 25 2 >/dev/null
ls $D/P13_${N}TD_*.npy | wc -l
