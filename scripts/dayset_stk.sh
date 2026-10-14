#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Stack of the BIG_NOVELTY chain variant (pru=1 required): scripts/chainvar_stk.sh with an airport-day set-transformer member as a third extra input xtd3 (EXTRA3_OOF=<big>@1, env big, default big_ctx; production
# PXTD3=<BIG_PROD_PREFIX>:<NXDS>, default big_ctxp:3; the final line uses big_big3 / big_bigp, the BIG member; scripts/dayset_stack.py / dayset_assemble.py).
# Extra inputs: the wide-leaf member xtd (EXTRA1_OOF = wideleaf_retune_a2), the Kumo member KM as xtd2 (EXTRA2_OOF = <KM>@1) and BIG as xtd3; outputs in scratchpad/xhook.
# usage: pru=1 KM=kumo_member_screen_km PFX=kumo_member bash scripts/dayset_stk.sh Kq|NNA3      -> scratchpad/xhook/P13_<PFX><C>xSTK_*, P13_<PFX><C>xTD_* (+ jitter copies)
#        REG=1 (no xtd2) -> P13_<PFX>REG<C>x*: must equal the stack of scripts/wideleaf_stack.py (regression arm)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/xhook; C=$1; mkdir -p $D; export P13_D=$D EXTRA1_OOF=wideleaf_retune_a2
[ "${pru:-0}" = 1 ] || { echo "chainvar_stk.sh needs pru=1 (xpr input + x name; the wideleaf_fold.sh bug)"; exit 2; }
KM=${KM:?KM oof prefix, e.g. kumo_member_screen_km}; PFX=${PFX:?chain prefix, e.g. kumo_member}
export oof_build_folds=6 NT=${NT:-4} OMP_NUM_THREADS=${NT:-4} METAFIT=pairs OL=queue_eb_ra3qa OC=meta_oof_h_ctl ON=NNA2 QOOF=fallback_regpl CLOCK_OOF=lgb_adsb_linked_a EXTRA=lnk,xlk,xdk,xapt DARK_OOF=lgb_noclock_nc APT_MEMBER=lgb_airport_ltfm:LTFM,lgb_airport_lfpg:LFPG
export STACKSRC=${STACKSRC:-scripts/dayset_stack.py} MSEED=0 EXTRA2_OOF=$KM@1 EXTRA3_OOF=${big:-big_ctx}@1
RAW=te2_srh_te,mvt_eobt1,ref_sr_mean,mvt_aobt3,mvt_iobt,aobt3_minus_ref,te2_sr_te,mvt_sched,aobt3_eobt1,mvt_lobt
X2=",xtd2,xtd3"; NM=$PFX$C; if [ "${REG:-0}" = 1 ]; then X2=""; NM=${PFX}REG$C; fi
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
