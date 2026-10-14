#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold and production fits of the wide-leaf member and two clock-free twins.  usage: bash scripts/wideleaf_member.sh <arm> <fold> [threads]
#   A2s0 | A2s1  wideleaf_retune arm A2 (1023 leaves) of the queue_eb_ra3qa recipe, seed 0 / 1 -> data/oof/wideleaf_a2s<S>_<fold>  (train2_wideleaf.py; A2s0 on months28/janjul already exists as wideleaf_retune_a2)
#   NCt          clock-free member on the recipe target (resid_aobt3), the logged lgb_noclock_nct_janjul arguments -> data/oof/lgb_noclock_nct_<fold>  (train2_clockfree.py)
#   KDC          CatBoost clock-free twin of meta_oof_h_ctl (train2_catcfg.py, GPU via CAT_GPU=1) -> data/oof/wideleaf_kdc_<fold>
#   fold = janjul|months28|months39|months410|months511|months612|prod (prod: --predict-rank, env SEED)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; A=$1; F=$2; T=${3:-8}; SEED=${SEED:-0}
case $F in janjul) CV="--cv months:1,7";; months28) CV="--cv months:2,8";; months39) CV="--cv months:3,9";; months410) CV="--cv months:4,10";;
  months511) CV="--cv months:5,11";; months612) CV="--cv months:6,12";; prod) CV="--predict-rank";; *) echo "fold?"; exit 2;; esac
DROP=te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy
FEATS_L=base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,adsb,adsb2,adsb3,xba,xe
FEATS_NC=base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,xe
FEATS_CATNC=base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,opdir,durp,dreg
echo "WIDELEAF $A $F threads $T seed $SEED host $(hostname) arch $(uname -m)"
case $A in
  A2s[0-9])
    S=${A#A2s}; NAME=wideleaf_a2s${S}_$F; [ "$S" = 0 ] && NAME=wideleaf_retune_a2_$F; [ "$F" = prod ] && NAME=wideleaf_a2p_s$S   # seed 0 keeps the wideleaf_retune_a2 prefix (its m28/janjul files are wideleaf_retune's)
    /usr/bin/time -v .venv/bin/python scripts/train2_wideleaf.py --name $NAME --f32 --strategy mixture --mix-drop fbpru --tail-gate 0.5 --tail-gate-lobt --target resid_aobt3 --es-clean \
      --lr 0.01 --fixed-rounds 2000 --leaves 1023 --min-leaf 100 --l2 4.3 --ff 0.63 --cls-leaves 63 --cls-rounds 400 --cls-ff 0.99 --cls-weight stake --stake-power 2.42 --p-power 1.0 \
      --dayshift --dayshift-midnight --features data/features_v9.parquet --feats $FEATS_L --drop-cols $DROP --stake-floor 0.5 --seed $S $CV --threads $T --out experiments/wideleaf_results.jsonl ;;
  NCt)
    NAME=lgb_noclock_nct_$F; [ "$F" = prod ] && NAME=lgb_noclock_nct_s$SEED
    /usr/bin/time -v .venv/bin/python scripts/train2_clockfree.py --name $NAME --f32 --strategy mixture --mix-drop fbpru --tail-gate 0.5 --tail-gate-lobt --target resid_aobt3 --es-clean \
      --lr 0.01 --fixed-rounds 2000 --leaves 497 --min-leaf 100 --l2 4.3 --ff 0.63 --cls-leaves 63 --cls-rounds 400 --cls-ff 0.99 --cls-weight stake --stake-power 2.42 --p-power 1.0 \
      --dayshift --dayshift-midnight --features data/features_v9.parquet --feats $FEATS_NC --drop-cols $DROP --stake-floor 0.5 --seed $SEED $CV --threads $T --out experiments/wideleaf_results.jsonl ;;
  KDC)
    NAME=wideleaf_kdc_$F; [ "$F" = prod ] && NAME=wideleaf_kdc_s$SEED
    /usr/bin/time -v .venv/bin/python scripts/train2_catcfg.py --name $NAME --strategy mixture --mix-drop fb --tail-gate 0.5 --tail-gate-lobt --target resid_aobt3 --model cat \
      --cat-depth 10 --cat-lr 0.078 --cat-iters 6000 --fixed-rounds 4000 --cls-leaves 63 --cls-rounds 400 --cls-ff 0.99 --cls-weight stake --stake-power 2.42 --p-power 1.0 --stake-floor 0.05 \
      --dayshift --dayshift-midnight --features data/features_v9.parquet --feats $FEATS_CATNC --drop-cols te_fno_ --cat-gpu-ram 0.5 --seed $SEED $CV --threads $T --out experiments/wideleaf_results.jsonl ;;
  *) echo "arm?"; exit 2;;
esac
