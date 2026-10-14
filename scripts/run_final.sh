#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Production run on the final ranking file, stage by stage (docs/REPRODUCE.md §6). Each stage runs on its own machine: copy the workspace
# phase2/work/<ws>/ between machines when the hardware changes. Every step writes a marker (phase2/work/<ws>/logs/<step>.done), so a
# re-run continues where it stopped. The 2025 side (member OOF, chain fold sides, 2025 constants; docs/REPRODUCE.md, part A) must exist first.
#   stage     hardware        what                                                                         output
#   inputs    CPU             workspace, OPDI, features, side tables, ADS-B clock extraction, constants     data/features_v9.parquet
#   members   CPU + CUDA GPU  production fit of every member seed on all 2025 rows (CatBoost, RealMLP, BIG on the GPU)   data/preds/
#   kumo      CUDA GPU        Kumo members K250 and KCATLg                                                  data/preds/KSBWPc100P_s0, KCATLgP_s0
#   line      CPU             FORM=v99 line: stacking chains, (B) calibration, unmatched / exposed rows by rule, post-rules   submissions/final.v99.parquet
#   kstk      CUDA GPU        KSTK (Kumo on the chain-stacker frames)                                       data/preds/KSTK6<C>P.npy
#   levers    CPU             chain-variant levers, TW1', KRS, KRS-NL (line A output)                       submissions/final.v111.parquet
#   bag       CPU             line B: copy of line A's workspace after the members, base line with the 24 + 16 seed bag (period <period>_bag)
#   assemble  CPU             scripts/assemble_final.sh on the two line outputs -> zesty-puzzle_final.parquet
# usage: bash scripts/run_final.sh [period [stage ...]]     period file phase2/periods/<period>.env (default: final); stages default: all, in order
#   THREADS (default 4) per member fit.
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; export PRC_ROOT=$PWD PYTHONHASHSEED=0 TORCH_PY=${TORCH_PY:-$PWD/.venv-torch/bin/python} KUMO_PY=${KUMO_PY:-$PWD/.venv-kumo/bin/python}
export PRU_DIR=${PRU_DIR:-data/external/pru_20261008}   # PRU csv dir read by pru_daily_calib.py ((B) calibration), relative to the workspace
PER=${1:-final}; shift || true; STAGES=${*:-inputs members kumo line kstk levers bag assemble}
wsname(){ local w; w=$(grep -E '^WS_NAME=' "phase2/periods/$1.env" | cut -d= -f2 || true); echo "${w:-$1}"; }
WSN=$(wsname "$PER"); WS=$PWD/phase2/work/$WSN; T=${THREADS:-4}; P=.venv/bin/python
BAG=${PER}_bag; WBN=$(wsname "$BAG"); WB=$PWD/phase2/work/$WBN
log(){ echo "[$(date -u +%FT%TZ)] run_final $PER: $*"; }
done_(){ [ -f "$WS/logs/$1.done" ]; }

stage_inputs(){
  log "2025 constants (docs/REPRODUCE.md A3; REFIT_CONSTANTS=1 refits them from the A-side fold vectors first)"
  [ "${REFIT_CONSTANTS:-0}" = 1 ] && bash scripts/constants_2025.sh tw1 krs krsnl exposed
  for f in phase2/params/postrules_fixed.json phase2/params/pool_weights.json phase2/params/krs_coef.json phase2/params/krsnl_coef.json \
           scratchpad/xkrsnl/prod/gbm_A.txt scratchpad/xhr/unm2025b.parquet; do   # the last two are built from competition data by make constants
    [ -f "$f" ] || { log "missing 2025 constant $f: run make constants (scripts/constants_2025.sh)"; exit 1; }; done
  log "TW1' shift TW1S=${TW1S:-1264 (default of p2_v107.sh)}"
  log "workspace and period inputs"
  bash phase2/run_pipeline.sh "$PER" ws
  bash phase2/steps/p2_final_inputs.sh "$PER" opdi
  bash phase2/run_pipeline.sh "$PER" np_features
  bash phase2/steps/p2_final_inputs.sh "$PER" side1 adsb_extract adsb_tables check
  bash phase2/run_pipeline.sh "$PER" np_side
  bash phase2/steps/p2_v107.sh "$PER" v107link
}
stage_members(){
  log "members: production fit of every seed on all 2025 rows"
  done_ members && return 0
  while IFS=$'\t' read -r fam seeds gpu rest; do case $fam in \#*|"") continue;; esac
    a=${seeds#s}; for s in $(seq "${a%-*}" "${a#*-}"); do WS=$WS CAT_GPU=$gpu bash scripts/xsave/final_refit.sh "$fam" "$s" "$T"; done
  done < scripts/xsave/members_xsave.tsv
  for A in nbr_clock_mbm nbr_clock_ma; do WS=$WS bash scripts/xsave/final_refit.sh $A 0 "$T"; done
  for s in 0 1 2; do WS=$WS CAT_GPU=1 bash scripts/xsave/final_refit.sh big_big $s "$T"; done
  ( cd "$WS" && "$PRC_ROOT/$P" - <<'PY'
import numpy as np, polars as pl                     # BIG prod seeds (final_refit.sh big_big) -> the chain's member files data/preds/big_bigp_s<k>.parquet
for k in range(3):
    a = np.load(f"data/oof/big_BIG_s{k}_prod.npy"); assert np.isfinite(a[1]).all()
    pl.DataFrame({"MVT_ID_mvt": a[0], "taxi": a[1]}).write_parquet(f"data/preds/big_bigp_s{k}.parquet"); print("BIG prod seed", k, a.shape[1], "rows")
PY
  )
  MARK=1 bash scripts/members_check.sh "$WSN"; touch "$WS/logs/members.done"
}
stage_kumo(){ log "Kumo members K250 and KCATLg (GPU)"; bash scripts/kumo_prod.sh "$WSN" k250stage k250 k250collect kcat; }
stage_line(){ log "FORM=v99 line (line A)"; bash phase2/run_pipeline.sh "$PER"; }
stage_kstk(){ log "KSTK (GPU)"; bash scripts/kumo_prod.sh "$WSN" kstkdump kstkbundle kstk kstkgk; }
stage_levers(){ log "chain-variant levers, TW1', KRS, KRS-NL -> submissions/final.v111.parquet"; bash phase2/steps/p2_v107.sh "$PER"; }
stage_bag(){
  log "line B ($BAG): copy of line A's workspace after the members, base line with the seed bag of $BAG.env"
  [ -f "phase2/periods/$BAG.env" ] || { log "no phase2/periods/$BAG.env"; exit 2; }
  if [ ! -d "$WB" ]; then
    cp -a "$WS" "$WB"; rm -rf "$WB/submissions" "$WB/stages"; mkdir -p "$WB/submissions" "$WB/stages"
    ( cd "$WB/logs" && rm -f np_derive.done pre.done pre_v87.done prod_*.done v99prep.done kumo_f13.done kumo_f4.done xb_*.done iso_*.done xs_*.done \
        m_pseudolabel_pl3p88.done post*.done compare*.done )   # every step of the base line after the members and the linked fold sides
  fi
  bash phase2/run_pipeline.sh "$BAG"
}
stage_assemble(){
  log "final assembly from line A ($WSN) and line B ($WBN)"
  local S=$WS/scratchpad/final_inputs O=$WS/scratchpad/final_assembly; mkdir -p "$S"
  for x in "kcatlg_line $WS/submissions/final.v111.parquet" "bag_line $WB/submissions/final.v99.parquet" \
           "bag_classes $WB/scratchpad/p2/U.parquet" "features_final $WS/data/features_v9.parquet"; do set -- $x; ln -sfn "$2" "$S/$1.parquet"; done
  [ -e "$O/zesty-puzzle_final.parquet" ] || STAGES=$S CHECK=${CHECK:-0} bash scripts/assemble_final.sh "$O"
  md5sum "$O/zesty-puzzle_final.parquet"; log "done: $O/zesty-puzzle_final.parquet"
}
for st in $STAGES; do
  case $st in inputs|members|kumo|line|kstk|levers|bag|assemble) stage_$st;; *) echo "unknown stage $st"; exit 2;; esac
done
