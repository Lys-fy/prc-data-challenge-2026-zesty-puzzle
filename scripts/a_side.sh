#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# A1 + A2 of docs/REPRODUCE.md section 5: the 2025 member OOF and the chain fold sides, with the concrete names of the build.
# Every step writes a marker scratchpad/a_side/<step>.done, so a re-run continues where it stopped (rm the marker to redo a step).
# DRY=1 prints the commands of the selected steps without running them. Needs `make features` first; the 2025 constants (A3) follow
# with `make constants` (scripts/constants_2025.sh).
# Steps (default: all, in this order):
#   members    every family of phase2/manifest/members.tsv, every seed, 6 folds (--cv <fold> in place of --predict-rank)   -> data/oof/<name>_s<k>_<fold>.npy
#   meta       chain meta-input OOF + queue_eb_dt_dt (scripts/constants_2025.sh meta)
#   wideleaf   wide-leaf member wideleaf_a2s{0,1,2}, 6 folds
#   nbrclock   NBR_CLOCK MA / MBm, 6 folds
#   big        BIG set transformer, 3 seeds x 6 folds (torch env, GPU), then dayset_prep.py
#   k250       Kumo K250 (profile KSBWPc100): stage the payload, 6 folds x 10 airports x 8 instances (+ agg), collect (Kumo env, GPU)
#   kcat8      KCAT8: fold codes, 6 folds, collect (GPU)
#   kcatl      KCATL, KCATL16 bag, KCATLg tail guard (GPU)
#   chains     A2 dTK / Kq / NNA3: T2x stacks + clock stages (scratchpad/chain_jobs)
#   wideleafc  A2 wide-leaf chains XB (Kq, NNA3)
#   pseudo     pseudo-label fold teacher ((B) fold line of the three chains, LE3BA) -> pseudo_<fold>_chain files -> member pseudolabel_pl3, 6 folds
#   pseudoc    A2 pseudo-label chains XS (PL3; Kq, NNA3)
#   chainvar   A2 nbr_clock_twin_bm, NBR_CLOCK_TWIN, k250_stack, kcatlg_stack (Kq, NNA3)
#   kstk       KSTK gK on the 6 chain folds: dump the chain-stacker frames, Kumo (kstk_profiles.txt, GPU) into kumo_gpu/run_kstk_folds, collect gK
#   kstkc      A2 kstk_input chains (Kq, NNA3)
#   kclock     A2 KSTK_CLOCK chains (Kq, NNA3)
#   novelty    BIG novelty mask (label-free) and the A2 BIG_NOVELTY chains (Kq, NNA3)
# usage: [DRY=1] [NT=6] bash scripts/a_side.sh [step ...]       Kumo / torch steps use KUMO_PY / TORCH_PY (Makefile defaults)
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; export PRC_ROOT=$PWD; P=${PY:-.venv/bin/python}; O=scratchpad/a_side; mkdir -p $O
NT=${NT:-6}; export NT OMP_NUM_THREADS=$NT POLARS_MAX_THREADS=$NT
KPY=${KUMO_PY:-$PWD/.venv-kumo/bin/python}; TPY=${TORCH_PY:-$PWD/.venv-torch/bin/python}; export TORCH_PY=$TPY
FOLDS="months39 months28 months410 janjul months511 months612"; CH="Kq NNA3"
APTS="EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM"
STEPS=${*:-members meta wideleaf nbrclock big k250 kcat8 kcatl chains wideleafc pseudo pseudoc chainvar kstk kstkc kclock novelty}
log(){ echo "[$(date -u +%FT%TZ)] a_side: $*"; }
x(){ if [ "${DRY:-0}" = 1 ]; then echo "  $*"; else eval "$@"; fi; }      # one command line (env assignments allowed)
cv(){ case $1 in janjul) echo janjul;; months*) local m=${1#months}; case $m in 410) echo months:4,10;; 511) echo months:5,11;; 612) echo months:6,12;; *) echo "months:${m:0:1},${m:1}";; esac;; esac; }
units(){ for f in $FOLDS; do for a in $APTS; do for i in 0 1 2 3 4 5 6 7; do echo "inst:$f:$a:$i"; done; echo "agg:$f:$a"; done; done; }
for s in $STEPS; do
  [ -f $O/$s.done ] && { log "$s: done (marker)"; continue; }
  log "$s"
  case $s in
    members)   # name, seeds (s<a>-<b>), python, command with {S}; the two optional families have their own fold steps (wideleaf, pseudo)
      while IFS=$'\t' read -r name seeds _ _ _ py cmd _; do
        case $name in \#*|""|wideleaf_a2p|pseudolabel_pl3p88) continue;; esac
        a=${seeds#s}; for k in $(seq "${a%-*}" "${a#*-}"); do for f in $FOLDS; do
          c=$(echo "$cmd" | sed -e "s/{S}/$k/g" -e 's/ --predict-rank//; s/ --cv [^ ]*//'); x "$py $c --cv $(cv $f)"; done; done
      done < phase2/manifest/members.tsv;;
    meta)      x bash scripts/constants_2025.sh meta;;
    wideleaf)  for k in 0 1 2; do for f in $FOLDS; do x bash scripts/wideleaf_member.sh A2s$k $f $NT; done; done;;
    nbrclock)  x $P scripts/nbrclock_member.py MA,MBm ${FOLDS// /,};;
    big)       for f in $FOLDS; do for k in 0 1 2; do t=BIG; [ $k = 0 ] || t=BIG_s$k
                 x $TPY scripts/dayset_big_member.py --fold $f --tag $t --d 192 --layers 4 --heads 6 --epochs 16 --seed $k; done; done
               x ARM=BIG $P scripts/dayset_prep.py;;
    k250)      PL=${KUMO_PAYLOAD:-scratchpad/xclu/payload}; R=kumo_gpu/run_k250_folds
               x $P kumo_gpu/stage/stage_data.py $PL
               [ "${DRY:-0}" = 1 ] || { mkdir -p $R; units > $R/units.txt; }
               x PAYLOAD=$PL RUN=$R NAME=KSBWPc100 SIZE=medium CAP=1000000 NCOL=100 COLMODE=imp CORE=50 NINST=8 EIN=1 CTXMODE=uniform AMP=fp16 SEED=0 $KPY kumo_gpu/cluster/km_engine.py --from-list $R/units.txt
               x $P kumo_gpu/collect/collect.py $R;;
    kcat8)     PL=${KUMO_PAYLOAD:-scratchpad/xclu/payload}
               x $P scripts/kumo_kcat_codes.py data/features_v9.parquet $PL/member scratchpad/kcat/codes/folds
               for f in $FOLDS; do x bash scripts/kumo_kcat_run.sh $f; done
               x FOLDS=\"$FOLDS\" bash scripts/kumo_kcat_collect.sh;;
    kcatl)     x bash scripts/kumo_kcat_logtarget_run.sh folds; x bash scripts/kumo_kcat_logtarget_bag.sh folds
               x $P scripts/kumo_kcat_logtarget_tailguard.py $FOLDS;;
    chains)    for C in dTK Kq NNA3; do x bash scripts/chain_jobs/stack/j_T2x_$C.sh; done
               for j in j_L j_Kq j_NNA3; do x bash scripts/chain_jobs/clockstage/$j.sh; done;;
    wideleafc) for C in $CH; do x bash scripts/wideleaf_fold.sh $C; done;;
    pseudo)    A=scratchpad/xau1
               x $P scripts/pru_daily_calib.py --pool $A/clockfold1_ldtk_ha0l,$A/clockfold1_lkq_ha0l,$A/clockfold1_lnna3_ha0l --w 0.3212,0.3065,0.3723 --out $A/clockfold1_le3 --um25 scratchpad/xau3/um25_A1.npz
               x pseudolabel_fold_teacher=$A/clockfold1_le3BA pseudolabel_kind=chain $P scripts/pseudo_labels.py $FOLDS
               for f in $FOLDS; do x bash scripts/pseudo_member.sh PL3 $f $NT; done;;
    pseudoc)   for C in $CH; do x bash scripts/pseudo_fold.sh PL3 $C; done;;
    chainvar)  for C in $CH; do
                 x KM=nbr_clock_mbm PFX=nbr_clock_twin_bm bash scripts/chainvar_fold.sh $C
                 x KM=nbr_clock_ma PFX=NBR_CLOCK_TWIN bash scripts/chainvar_fold.sh $C
                 x KM=KSBWPc100 PFX=k250_stack bash scripts/chainvar_fold.sh $C
                 x KM=KCATLg PFX=kcatlg_stack bash scripts/chainvar_fold.sh $C; done;;
    kstk)      PL=${KUMO_PAYLOAD:-scratchpad/xclu/payload}; R=${KSTK_FOLD_RUN:-kumo_gpu/run_kstk_folds}; S6=kumo_gpu/data/payload/stk6
               x bash scripts/kstk_run.sh dump
               [ "${DRY:-0}" = 1 ] || { mkdir -p $R; units > $R/units.txt; }
               for C in $CH; do x PAYLOAD=$PL RUN=$R KSTK_DIR=$S6 KSTK_CHAIN=$C NAME=KSTK6$C SIZE=medium CAP=1000000 NCOL=1000 COLMODE=all NINST=8 EIN=1 CTXMODE=uniform AMP=fp16 SEED=0 KSTK_CLIP=1800 $KPY kumo_gpu/cluster/km_adapters_kstk.py --from-list $R/units.txt; done
               x bash scripts/kstk_run.sh gk $R;;
    kstkc)     for C in $CH; do x bash scripts/kstk_run.sh fold $C; done;;
    kclock)    for C in $CH; do x KM=KSBWPc100 KMP=KSBWPc100P PFX=k250_stack bash scripts/kumo_clock_chain.sh $C; done;;
    novelty)   x $P scripts/dayset_novelty.py
               for C in $CH; do x PFX=BIG_NOVELTY big=big_big3 BIG_PROD_PREFIX=big_bigp extra3_mask_ids=scratchpad/xds2/novelty_mask_ids.npy bash scripts/dayset_chain.sh fold $C; done;;
    *)         echo "unknown step $s"; exit 2;;
  esac
  [ "${DRY:-0}" = 1 ] || touch $O/$s.done
done
log "done: $STEPS"
