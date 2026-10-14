#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Production side of one chain: pre (scripts/chain_assemble.py) -> C4 -> tree D -> clock stage (FORM PB|PC, A1 0|1) -> A0 + leave stage L.
# usage: FORM=PC A1=0 FBMODE=base bash scripts/chain_prod.sh <N> <stage name>   (N as chain_stk.sh; stage name = the chain_stage.sh out name whose npz is used)
# output scratchpad/p13/PR_<stage name>_HA0L.parquet (+ intermediates PR_<N>_{pre,C4,C4treeRAW,C4treeD}.parquet)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/p13; S=submissions; N=$1; SN=$2
export oof_build_folds=6 NT=${NT:-6} OMP_NUM_THREADS=${NT:-6} METAFIT=pairs OL=queue_eb_ra3qa OC=meta_oof_h_ctl ON=NNA2 QOOF=fallback_regpl CLOCK_OOF=lgb_adsb_linked_a EXTRA=lnk,xlk,xdk,xapt DARK_OOF=lgb_noclock_nc APT_MEMBER=lgb_airport_ltfm:LTFM,lgb_airport_lfpg:LFPG
export PLG=queue_eb_v47lqa NLGB=6 PCT=CATV17x NCAT=14 PNN=NNRA2 PXLK=lgb_adsb_linked_a NXLK=3 PXDK=lgb_noclock_nc NXDK=3 PXAPT=lgb_airport_ltfm:LTFM:3,lgb_airport_lfpg:LFPG:3
export BASE=stages/fallback_base.parquet FALLBACK=data/external/xba_fallback_ids.parquet FBMODE=${FBMODE:-base} PLGFB=runway_queue_v47lq NLGBFB=6 STACKSRC=scripts/chain_stack.py
RAW=te2_srh_te,mvt_eobt1,ref_sr_mean,mvt_aobt3,mvt_iobt,aobt3_minus_ref,te2_sr_te,mvt_sched,aobt3_eobt1,mvt_lobt
case $N in
  BXK) export EXTRA=$EXTRA,xl SL=queue_eb_v47lqa NSL=6;;
  SDs) export OL=P13SDs PLG=P13SDs NLGB=1;;
  SDu) export OL=P13SD PLG=P13SD NLGB=3;;
  KBdTK) export OL=queue_eb_dt_dt PLG=queue_decomp_ebdt NLGB=3 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW;;
  KBKq) export OC=runway_queue_catq PCT=runway_queue_catv17xq NCAT=6 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW;;
  KBNNA3) export ON=meta_oof_realmlp_nna3 PNN=meta_oof_realmlp_nnra3 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW;;
  *) echo "N?"; exit 1;;
esac
if [ "${pru:-0}" = 1 ]; then export EXTRA=$EXTRA,xpr; N=${N}x; fi   # xpr form
PRE=$D/PR_${N}_${FBMODE}_pre.parquet; C4=$D/PR_${N}_${FBMODE}_C4.parquet; TD=$D/PR_${N}_${FBMODE}_C4treeD.parquet
[ -f $PRE ] || { TAG=P13_$N ROUND=rint OUT=$PRE $P scripts/chain_assemble.py | grep -vE "^state" | tail -4; }
[ -f $C4 ] || daystate_root=${PRC_ROOT:-$PWD} daystate_oofdirs=${PRC_ROOT:-$PWD}/$D assembler_fl=$OL assembler_fc=$OC $P scripts/stage12_run.py scripts/combo_c4_apply.py --target $PRE --out $C4 --fit-oof P13_${N}STK \
  --feats PR5,PR15,PRT5,PRT15,PRX5,PRX15,PAX15,PRT15u,PRX5u,PRX15u,PAX15u --demean day --pscale --lgb $PLG --nlgb $NLGB --cat $PCT --ncat $NCAT --v53 stages/unmatched_rule.parquet | tail -1
[ -f $TD ] || daystate_root=${PRC_ROOT:-$PWD} daystate_oofdirs=${PRC_ROOT:-$PWD}/$D assembler_fl=$OL assembler_fc=$OC $P scripts/stage12_run.py scripts/combo_treed_apply.py --target $C4 --pre $PRE --fit-oof P13_${N}STK \
  --lgb $PLG --nlgb $NLGB --cat $PCT --ncat $NCAT --out $D/PR_${N}_${FBMODE}_C4treeRAW.parquet --out-d $TD | grep wrote
prodstage() {   # $1 = TD parquet, $2 = fold TD prefix, $3 = stage name (npz), $4 = out tag
  XA=""; [ "${A1:-0}" = 1 ] && XA="--excl-lirf"
  if [ "$FORM" = PC ]; then export adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600; else export adsb5_clock_adsb5=0; unset clock_tune_cleaves clock_tune_crounds || true; fi
  $P scripts/adsb5_stage_run.py scripts/chain_clock_apply.py --target $1 --fit-oof $2,${2}j1,${2}j2 --out $D/PR_$4_Hu6.parquet --fset clkonly --shrink 1.0 --clip 300 --seeds 6 --wmode sqrt --objective huber --alpha 600 --nt $NT $XA --v53 stages/unmatched_rule.parquet | grep -E "capacity|wrote"
  $P scripts/adsb5_stage_run.py scripts/leave_apply.py --target $D/PR_$4_Hu6.parquet --pre $1 --npz $D/P13_$3_HuJ.npz --gnpz $D/P13_$3_HuJ.npz --out $D/PR_$4_HA0L.parquet --lirf-excl --arm L --seeds 3 --nt $NT | grep wrote
  md5sum $D/PR_$4_HA0L.parquet
}
prodstage $TD $D/P13_${N}TD $SN ${SN}_${FBMODE}
