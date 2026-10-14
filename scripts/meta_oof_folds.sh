#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold OOF of the stacking chains' meta inputs (OL / OC / ON of scripts/chain_prep.py), seed 0, one fold per call:
#   queue_eb_ra3qa (OL, all chains), meta_oof_h_ctl (OC, dTK and NNA3), NNA2 (ON, dTK and Kq), queue_decomp_ebqq -> queue_eb_dt_dt (OL, dTK), runway_queue_catq (OC, Kq), meta_oof_realmlp_nna3 (ON, NNA3).
# usage: bash scripts/meta_oof_folds.sh <fold> [names]     fold in months39 months28 months410 janjul months511 months612
# queue_eb_dt_dt needs every fold of queue_eb_ra3qa and queue_decomp_ebqq first: .venv/bin/python scripts/queue_mkdt_oof.py months39 months28 months410 janjul months511 months612
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; P=.venv/bin/python; TORCH_PY=${TORCH_PY:-$PWD/.venv-torch/bin/python}
T=$1; NAMES=${2:-queue_eb_ra3qa,meta_oof_h_ctl,NNA2,queue_decomp_ebqq,runway_queue_catq,meta_oof_realmlp_nna3}
case $T in months39) CV=3,9;; months28) CV=2,8;; months410) CV=4,10;; janjul) CV=1,7;; months511) CV=5,11;; months612) CV=6,12;; *) echo "unknown fold $T"; exit 2;; esac
F=base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx
DROP=te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy
S="--f32 --strategy mixture --mix-drop fbpru --tail-gate 0.5 --tail-gate-lobt --target resid_aobt3 --es-clean --lr 0.01 --fixed-rounds 2000 --leaves 497 --min-leaf 100 --l2 4.3 --ff 0.63 --cls-leaves 63 --cls-rounds 400 --cls-ff 0.99 --cls-weight stake --stake-power 2.42 --p-power 1.0 --dayshift --dayshift-midnight --features data/features_v9.parquet --drop-cols $DROP --stake-floor 0.5"
C="--strategy mixture --mix-drop fb --tail-gate 0.5 --tail-gate-lobt --target resid_aobt3 --model cat --cat-depth 10 --cat-lr 0.078 --fixed-rounds 4000 --cls-leaves 63 --cls-rounds 400 --cls-ff 0.99 --cls-weight stake --stake-power 2.42 --p-power 1.0 --dayshift --dayshift-midnight --features data/features_v9.parquet --drop-cols te_fno_ --threads 2 --cat-gpu-ram 0.75"
has(){ case ",$NAMES," in *",$1,"*) return 0;; esac; return 1; }
has queue_eb_ra3qa && $P scripts/train2.py --name queue_eb_ra3qa_$T $S --feats $F,durp,dreg,adsb,adsb2,adsb3,xba,xe --seed 0 --cv months:$CV --threads 4
has meta_oof_h_ctl && $P scripts/train2_catcfg.py --name meta_oof_h_ctl_$T $C --feats $F,opdir,durp,dreg,adsb,adsb2,adsb3 --seed 0 --cv months:$CV
has NNA2 && $TORCH_PY scripts/nn_resid.py --name NNA2 --cv months:$CV --features data/features_v9.parquet --feats $F,adsb,adsb2 --model realmlp --epochs 40 --seed 0
if has queue_decomp_ebqq; then
  case $T in months39) XQ=xq39;; months28) XQ=xq28;; months410) XQ=xq410;; janjul) XQ=xq17;; months511) XQ=xq511;; months612) XQ=xq612;; esac
  NT=10 $P scripts/queue_regime_prob.py $T
  $P scripts/train2_queue.py --name queue_decomp_ebqq_$T $S --feats $F,durp,dreg,adsb,adsb2,adsb3,xba,xe,$XQ --seed 0 --cv months:$CV --threads 10
fi
has runway_queue_catq && CAT_GPU=1 $P scripts/train2.py --name runway_queue_catq_$T $C --feats $F,opdir,durp,dreg,adsb,adsb2,adsb3,xe --seed 0 --cv months:$CV
has meta_oof_realmlp_nna3 && $TORCH_PY scripts/nn_resid.py --name meta_oof_realmlp_nna3 --cv months:$CV --features data/features_v9.parquet --side-tables --feats $F,durp,dreg,adsb,adsb2,adsb3,xba,xe --drop-cols $DROP --model realmlp --epochs 40 --seed 0
true
