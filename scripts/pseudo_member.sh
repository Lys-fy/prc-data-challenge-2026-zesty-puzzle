#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold and production fits of the pseudo-label member, recipe of queue_eb_ra3qa (lr 0.01 x 2000, 497 leaves, min_leaf 100, l2 4.3, ff 0.63, mixture +
# tail gate, seed 0) via scripts/train2_pseudo.py.  usage: bash scripts/pseudo_member.sh <arm> <fold> [threads]
#   B0    no pseudo rows (the same-host twin of queue_eb_ra3qa)                       -> data/oof/pseudolabel_b0_<fold>
#   PL3   + held months' matched rows pseudo-labelled by the CHAIN (v86 fold eq), w 0.3   -> data/oof/pseudolabel_pl3_<fold>
#   PL3B  the same rows labelled by the (B)-only fold teacher clockfold1_le3b      -> data/oof/pseudolabel_pl3b_<fold>
#   PL10  the same at w 1.0                                                        -> data/oof/pseudolabel_pl10_<fold>
#   PLB   placebo: the same rows labelled by the L member's OWN OOF, w 0.3          -> data/oof/pseudolabel_plb_<fold>
#   ORC   ceiling: the same rows with their TRUE labels, w 0.3                      -> data/oof/pseudolabel_orc_<fold>
#   fold = janjul|months28|months39|months410|months511|months612 | prod (--predict-rank; pseudo file scratchpad/xsyn/pseudo_prod_chain.parquet; env SEED)
# Pseudo files come from scripts/pseudo_labels.py (one row mask per fold, three label columns).
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; A=$1; F=$2; T=${3:-8}; SEED=${SEED:-0}
case $F in janjul) CV="--cv months:1,7";; months28) CV="--cv months:2,8";; months39) CV="--cv months:3,9";; months410) CV="--cv months:4,10";;
  months511) CV="--cv months:5,11";; months612) CV="--cv months:6,12";; prod) CV="--predict-rank";; *) echo "fold?"; exit 2;; esac
case $A in
  B0)   PS="";;
  PL3)  PS="--pseudo scratchpad/xsyn/pseudo_${F}_chain.parquet --pseudo-w 0.3"; [ "$F" = prod ] && PS="--pseudo scratchpad/xsyn/pseudo_prod_chain88.parquet --pseudo-w 0.3";;   # prod: teacher = the v88 combination line
  PL3B) PS="--pseudo scratchpad/xsyn/pseudo_${F}_chainB.parquet --pseudo-w 0.3";;   # (B)-only fold teacher clockfold1_le3b
  PL10) PS="--pseudo scratchpad/xsyn/pseudo_${F}_chain.parquet --pseudo-w 1.0";;
  PLB)  PS="--pseudo scratchpad/xsyn/pseudo_${F}_self.parquet --pseudo-w 0.3";;
  ORC)  PS="--pseudo scratchpad/xsyn/pseudo_${F}_oracle.parquet --pseudo-w 0.3";;
  *) echo "arm?"; exit 2;;
esac
NAME=pseudolabel_${A}_$F; [ "$F" = prod ] && NAME=pseudolabel_${A}P88_s$SEED
DROP=te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy
echo "PSEUDOLABEL $A $F threads $T seed $SEED host $(hostname) arch $(uname -m) pseudo '${PS}'"
/usr/bin/time -v .venv/bin/python scripts/train2_pseudo.py --name $NAME --f32 --strategy mixture --mix-drop fbpru --tail-gate 0.5 --tail-gate-lobt --target resid_aobt3 --es-clean \
  --lr 0.01 --fixed-rounds 2000 --leaves 497 --min-leaf 100 --l2 4.3 --ff 0.63 --cls-leaves 63 --cls-rounds 400 --cls-ff 0.99 --cls-weight stake --stake-power 2.42 --p-power 1.0 \
  --dayshift --dayshift-midnight --features data/features_v9.parquet --feats base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,adsb,adsb2,adsb3,xba,xe \
  --drop-cols $DROP --stake-floor 0.5 --seed $SEED $CV --threads $T $PS --out experiments/pseudolabel_results.jsonl
