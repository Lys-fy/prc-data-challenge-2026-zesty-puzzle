#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# A3 of docs/REPRODUCE.md: refit the 2025 constants and the per-row fold tables the production run reads, from the A-side fold vectors.
# Every step writes a marker scratchpad/constants/<step>.done, so a re-run continues where it stopped (rm the marker to redo a step).
# Steps (default: all, in this order; the chain fold sides of A2 must exist before pool / postrules / tw1 / krs):
#   meta      chain meta-input OOF, 6 folds (scripts/meta_oof_folds.sh) + queue_eb_dt_dt (scripts/queue_mkdt_oof.py)          -> data/oof/      (run before A2)
#   pool      pool weights, global simplex on the 6-fold HA0 vectors (env POOL_CHAINS, default the five vectors of the shipped fit:
#             F6 HA0L, P12 dTK, P12 Kq, P12 RdLN, P12 NNA3 in scratchpad/xof; the F6 and RdLN slots get weight 0)          -> $O/pool_weights.json
#   postrules post-rule arms standfloor / offblock_pull: fold frame X4 + fit on the (B) fold vectors pru_pxxb                         -> $O/postrules_refit.json
#   tw1       TW1' twin shift: twin pairs, matched fold line table, shift fit                                        -> scratchpad/xquick/tw1p_shift.json
#   krs       KRS beta: v107 fold line at (B), post rules, beta per K250-spread tercile                              -> scratchpad/xchain2/v106K_K2_KRS_prod_coef.json
#   krsnl     KRS-NL: per-row fold table, then month-kernel beta + LightGBM residual model (mode A)                  -> scratchpad/xkrsnl/prod/{coef.json, gbm_A.txt}
#   exposed   unmatched-row context table (F4 context), ORD-share bounds 0.072 / 0.259, E1s clock cap 1,822 s       -> scratchpad/xhr/unm2025b.parquet
# pool and postrules write next to the shipped values (phase2/params/) for comparison; the other steps write the files the run reads.
# usage: [POOL_CHAINS="p1 p2 p3"] [NT=6] bash scripts/constants_2025.sh [step ...]
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; export PRC_ROOT=$PWD; P=${PY:-.venv/bin/python}; O=scratchpad/constants; mkdir -p $O
NT=${NT:-6}; export OMP_NUM_THREADS=$NT POLARS_MAX_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NT
FOLDS="months39 months28 months410 janjul months511 months612"
STEPS=${*:-meta pool postrules tw1 krs krsnl exposed}
log(){ echo "[$(date -u +%FT%TZ)] constants_2025: $*"; }
for s in $STEPS; do
  [ -f $O/$s.done ] && { log "$s: done (marker)"; continue; }
  log "$s"
  case $s in
    meta)      for t in $FOLDS; do bash scripts/meta_oof_folds.sh $t; done
               $P scripts/queue_mkdt_oof.py $FOLDS;;
    pool)      X=scratchpad/xof; PC=${POOL_CHAINS:-$X/oof_build_F6_HA0L $X/oof_build_P12dTKF6_HA0 $X/oof_build_P12KqF6_HA0 $X/oof_build_P12RdLNF6_HA0 $X/oof_build_P12NNA3F6_HA0}
               miss=$(for p in $PC; do ls ${p}_months39*.npy >/dev/null 2>&1 || echo "$p"; done)
               if [ -n "$miss" ]; then log "pool: skipped, fold vectors missing: $miss (set POOL_CHAINS, e.g. the A2 chains scratchpad/xau1/clockfold1_l{dtk,kq,nna3}_ha0l)"; continue; fi
               $P scripts/pool_weights_fit.py $O/pool_weights.json $PC
               echo "shipped: $(tr -d '\n ' < phase2/params/pool_weights.json)";;
    postrules) $P scripts/postrule_fold_frame.py --pool scratchpad/xpr/pru_pxxb --out scratchpad/x4/X4_fold.parquet
               $P phase2/steps/p2_postrules.py fit --fold scratchpad/xpr/pru_pxxb --x4 scratchpad/x4/X4_fold.parquet --out $O/postrules_refit.json
               echo "shipped (shifts rounded to 1 s): phase2/params/postrules_fixed.json";;
    tw1)       $P scripts/tw1_twins.py
               PY=$P bash scripts/error_atlas_line.sh
               $P scripts/tw1_rule_build.py;;
    krs)       NAME=v106K O=scratchpad/xchain2 NT=$NT bash scripts/krs_fold_line.sh;;
    krsnl)     $P scripts/months_rows_table.py scratchpad/xchain2/v106K scratchpad/xmonths/rows.parquet
               ROWS=scratchpad/xmonths/rows.parquet COEFDIR=scratchpad/xkrsnl/prod $P scripts/krs_nonlinear.py fit A;;
    exposed)   $P scripts/exposed_characterise.py
               $P scripts/exposed_context.py
               $P scripts/exposed_ord_bounds.py scratchpad/xhr/unm2025b.parquet
               $P scripts/e1s_clock_cap.py;;
    *)         echo "unknown step $s"; exit 2;;
  esac
  touch $O/$s.done
done
log "done: $STEPS"
