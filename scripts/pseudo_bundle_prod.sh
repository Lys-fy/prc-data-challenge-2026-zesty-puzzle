#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Production side of the pseudo-label chains pseudolabel_chainkq / PSEUDOLABEL_CHAIN_NNA3: the KBKq / KBNNA3 chains with the L member replaced
# by the pseudo-labelled twin (fold OOF pseudolabel_pl3, production seeds PLG=pseudolabel_pl3p88 x NLGB=2, teacher = the run's first-pass line),
# stack source scripts/chain_stack.py; fold stacks read from scratchpad/xsyn (P13_XSPL3<C>xSTK/TD), pre / C4 / TD written to scratchpad/xsyn,
# fold stage npz scratchpad/xau1/clockfold1_xspl3<C>_HuJ.npz. The production clock stage runs through scripts/clockstage_run.py (adsb6 + adsbL
# clocks, pushstart_adsb6=1 clockfold1_adsbl=1). The dTK chain is not rebuilt.
# usage: pru=1 FORM=PC A1=1 FBMODE=member NT=4 ARM=PSEUDOLABEL_CLOCKFOLD_ARM ARMIN=XS SEED6=1 MS6=1 pushstart_adsb6=1 clockfold1_adsbl=1 bash scripts/pseudo_bundle_prod.sh pseudolabel_chainkq|PSEUDOLABEL_CHAIN_NNA3 <fold stage name PSEUDOLABEL_CHAIN<C>>
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/xsyn; O=scratchpad/xsyn; OI=scratchpad/xsyn; NPZD=scratchpad/xau1; ARMIN=${ARMIN:?ARMIN tag}; export pushstart_adsb6=${pushstart_adsb6:-1} clockfold1_adsbl=${clockfold1_adsbl:-1}; S=submissions; N=$1; SN=$2; ARM=${ARM:?ARM tag}; mkdir -p $O
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
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
  pseudolabel_chainkq) export OL=pseudolabel_pl3 PLG=pseudolabel_pl3p88 NLGB=2 OC=runway_queue_catq PCT=runway_queue_catv17xq NCAT=6 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW;;
  PSEUDOLABEL_CHAIN_NNA3) export OL=pseudolabel_pl3 PLG=pseudolabel_pl3p88 NLGB=2 ON=meta_oof_realmlp_nna3 PNN=meta_oof_realmlp_nnra3 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW;;
  KBNNA3) export ON=meta_oof_realmlp_nna3 PNN=meta_oof_realmlp_nnra3 EXTRA=$EXTRA,spr,raw audited_stack_raw=$RAW;;
  *) echo "N?"; exit 1;;
esac
LSEEDS=3
if [ "${SEED6:-0}" = 1 ]; then export NXDK=6 PXAPT=lgb_airport_ltfm:LTFM:6,lgb_airport_lfpg:LFPG:6
  case $N in KBdTK) export PLG=sixseed_chain_ebdt6 NLGB=6;; KBNNA3|PSEUDOLABEL_CHAIN_NNA3) export PNN=sixseed_chain_nnra3x6;; esac; fi
if [ "${NN2x6:-0}" = 1 ] && [ "$PNN" = NNRA2 ]; then export PNN=sixseed_nnra2x6; fi
if [ "${CLOCK_SIX:-0}" = 1 ]; then export PXLK=sixseed_xlkx6 NXLK=6; fi
if [ "${MS6:-0}" = 1 ]; then export audited_stack_mseeds=0,1,2,3,4,5; LSEEDS=6; fi
echo "PSEUDOLABEL arm $ARM chain $N: OL=$OL PLG=$PLG x$NLGB PCT=$PCT x$NCAT PNN=$PNN ($(ls data/preds | grep -c "^${PNN}_s") files) PXLK=$PXLK x$NXLK PXDK=$PXDK x$NXDK PXAPT=$PXAPT MSEEDS=${audited_stack_mseeds:-default} LSEEDS=$LSEEDS"
if [ "${pru:-0}" = 1 ]; then export EXTRA=$EXTRA,xpr; N=${N}x; fi   # xpr form
PRE=$OI/PR_${N}_${ARMIN}_pre.parquet; C4=$OI/PR_${N}_${ARMIN}_C4.parquet; TD=$OI/PR_${N}_${ARMIN}_C4treeD.parquet
for f in data/preds/${PLG}_s0.parquet data/preds/${PLG}_s1.parquet $NPZD/clockfold1_${SN}_HuJ.npz $D/P13_${N}STK_janjul.npy $D/P13_${N}TD_janjul.npy; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
[ -f $PRE ] || { TAG=P13_$N ROUND=rint OUT=$PRE $P scripts/chain_assemble.py | grep -vE "^state" | tail -4; }
[ "${PREONLY:-0}" = 1 ] && exit 0   # probe: pre-stage only
[ -f $C4 ] || daystate_root=${PRC_ROOT:-$PWD} daystate_oofdirs=${PRC_ROOT:-$PWD}/$D assembler_fl=$OL assembler_fc=$OC $P scripts/stage12_run.py scripts/combo_c4_apply.py --target $PRE --out $C4 --fit-oof P13_${N}STK \
  --feats PR5,PR15,PRT5,PRT15,PRX5,PRX15,PAX15,PRT15u,PRX5u,PRX15u,PAX15u --demean day --pscale --lgb $PLG --nlgb $NLGB --cat $PCT --ncat $NCAT --v53 stages/unmatched_rule.parquet | tail -1
[ -f $TD ] || daystate_root=${PRC_ROOT:-$PWD} daystate_oofdirs=${PRC_ROOT:-$PWD}/$D assembler_fl=$OL assembler_fc=$OC $P scripts/stage12_run.py scripts/combo_treed_apply.py --target $C4 --pre $PRE --fit-oof P13_${N}STK \
  --lgb $PLG --nlgb $NLGB --cat $PCT --ncat $NCAT --out $O/PR_${N}_${ARM}_C4treeRAW.parquet --out-d $TD | grep wrote
prodstage() {   # $1 = TD parquet, $2 = fold TD prefix, $3 = stage name (npz), $4 = out tag
  XA=""; [ "${A1:-0}" = 1 ] && XA="--excl-lirf"
  if [ "$FORM" = PC ]; then export adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600; else export adsb5_clock_adsb5=0; unset clock_tune_cleaves clock_tune_crounds || true; fi
  $P scripts/clockstage_run.py scripts/chain_clock_apply.py --target $1 --fit-oof $2,${2}j1,${2}j2 --out $O/PR_$4_Hu6.parquet --fset clkonly --shrink 1.0 --clip 300 --seeds 6 --wmode sqrt --objective huber --alpha 600 --nt $NT $XA --v53 stages/unmatched_rule.parquet | grep -E "capacity|wrote"
  $P scripts/clockstage_run.py scripts/leave_apply.py --target $O/PR_$4_Hu6.parquet --pre $1 --npz $NPZD/clockfold1_$3_HuJ.npz --gnpz $NPZD/clockfold1_$3_HuJ.npz --out $O/PR_$4_HA0L.parquet --lirf-excl --arm L --seeds $LSEEDS --nt $NT | grep wrote
  md5sum $O/PR_$4_HA0L.parquet
}
prodstage $TD $D/P13_${N}TD $SN ${SN}_${ARM}
