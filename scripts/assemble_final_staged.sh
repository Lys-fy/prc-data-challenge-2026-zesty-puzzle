#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Staged assembly of the submitted final file (docs/REPRODUCE.md §6b). The submitted file was assembled from the saved outputs of four
# workspace runs of the same line, not in the single pass of scripts/run_final.sh. No step reads an earlier submission; every unmatched and
# exposed row of all four months is priced by the label-free rules.
#   S1 schedule-copy relink on the final ranking file          rule_schedule_copy_lirf_relink.py   -> schedule_copy/final_dryrun.parquet
#   S2 rule-priced rows from the 40-seed bag line               rulebag_rows_merge.py               -> S2_rulebag.parquet
#   S3 KCAT8 in the v102 slot, as a delta (K250 -> KCAT8)       staged_delta_apply.py               -> S3_kcat8.parquet
#   S4 schedule-copy rule at LIRF                               rule_schedule_copy_lirf_apply.py    -> S4_schedule_copy.parquet
#   S5 KCATLg in the v102 slot, as a delta (KCAT8 -> KCATLg)    staged_delta_apply.py --changed-only -> zesty-puzzle_final.parquet
#   S6 public-period check file (Jan/Jul, public ids)           public_period_extract.py            -> public_janjul.parquet
# usage: STAGES=<stage dir> bash scripts/assemble_final_staged.sh <out dir>
#   stage dir (saved line outputs, submissions/final.v111.parquet or post.v99 of each workspace; manifest/final_staged_inputs.md5):
#     base_line.parquet    K250 in the v102 slot (V102ARM=k250_stack), default seed bag      bag_line.parquet     FORM=v99 line with FZ_L / FZ_C (24 + 16 seeds)
#     bag_classes.parquet  row classes of the bag workspace (scratchpad/p2/U.parquet)  kcat8_line.parquet   KCAT8 in the v102 slot
#     kcatlg_line.parquet  KCATLg in the v102 slot (V102ARM=kcatlg_stack)                     features_final.parquet  data/features_v9.parquet of the final workspace
#   root ($PRC_ROOT): data/raw/training_*.parquet, data/final/final_ranking.parquet, data/features_v9.parquet (public period, for S6).
#   CHECK=0 skips the md5 checks against manifest/final_staged_{inputs,outputs}.md5. Nothing is overwritten.
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; export PRC_ROOT=$PWD POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}
S=${STAGES:?set STAGES to the stage dir}; O=${1:?out dir}; P=${PY:-$PWD/.venv/bin/python}; mkdir -p "$O"
log(){ echo "[$(date -u +%FT%TZ)] assemble_final_staged: $*"; }
[ "${CHECK:-1}" = 1 ] && { log "stage inputs vs manifest/final_staged_inputs.md5"; ( cd "$S" && md5sum -c "$PRC_ROOT/manifest/final_staged_inputs.md5" ); }
log "S1 schedule-copy relink";  $P scripts/rule_schedule_copy_lirf_relink.py data/final/final_ranking.parquet "$O/schedule_copy"
log "S2 rule-priced rows";      $P scripts/rulebag_rows_merge.py "$S/base_line.parquet" "$S/bag_line.parquet" "$S/bag_classes.parquet" "$O/S2_rulebag.parquet"
log "S3 KCAT8 delta";           $P scripts/staged_delta_apply.py "$O/S2_rulebag.parquet" "$S/base_line.parquet" "$S/kcat8_line.parquet" "$S/features_final.parquet" "$O/S3_kcat8.parquet"
log "S4 schedule-copy rule";    $P scripts/rule_schedule_copy_lirf_apply.py "$O/S3_kcat8.parquet" "$O/schedule_copy/final_dryrun.parquet" "$O/S4_schedule_copy.parquet"
log "S5 KCATLg delta";          $P scripts/staged_delta_apply.py "$O/S4_schedule_copy.parquet" "$S/kcat8_line.parquet" "$S/kcatlg_line.parquet" "$S/features_final.parquet" \
                                   "$O/zesty-puzzle_final.parquet" --changed-only --protect "$S/bag_classes.parquet" "$O/schedule_copy/final_dryrun.parquet"
$P scripts/validate_submission.py "$O/zesty-puzzle_final.parquet" --features "$S/features_final.parquet"
log "S6 public-period check file"; $P scripts/public_period_extract.py "$O/zesty-puzzle_final.parquet" data/features_v9.parquet "$O/public_janjul.parquet"
[ "${CHECK:-1}" = 1 ] && { log "outputs vs manifest/final_staged_outputs.md5"; ( cd "$O" && md5sum -c "$PRC_ROOT/manifest/final_staged_outputs.md5" ); }
md5sum "$O/zesty-puzzle_final.parquet" "$O/public_janjul.parquet"; log "done"
