#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Final assembly of the submitted file from the two line outputs (docs/REPRODUCE.md §6, stage F):
#   F1 schedule-copy relink on the final ranking file                    rule_schedule_copy_lirf_relink.py -> schedule_copy/final_dryrun.parquet
#   F2 rule-priced rows (unmatched LIRF, exposed) from the bag line      rulebag_rows_merge.py             -> F2_merge.parquet
#   F3 schedule-copy rule at LIRF (69 rows)                              rule_schedule_copy_lirf_apply.py  -> zesty-puzzle_final.parquet
#   F4 public-period check file (Jan/Jul rows on the public ids)         public_period_extract.py          -> public_janjul.parquet
# usage: STAGES=<stage dir> bash scripts/assemble_final.sh <out dir>
#   stage dir (manifest/final_inputs.md5):
#     kcatlg_line.parquet     submissions/final.v111.parquet of the KCATLg line (phase2/periods/final.env)
#     bag_line.parquet        submissions/final.v99.parquet of the bag line (phase2/periods/final_bag.env)
#     bag_classes.parquet     scratchpad/p2/U.parquet of the bag line (row classes matched / unm_LIRF / exposed / unm_other)
#     features_final.parquet  data/features_v9.parquet of the KCATLg line (LOBT bound of the validator)
#   root ($PRC_ROOT): data/raw/training_*.parquet, data/final/final_ranking.parquet, data/features_v9.parquet (public period, for F4).
#   CHECK=0 skips the md5 checks against manifest/final_{inputs,outputs}.md5. Nothing is overwritten.
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; export PRC_ROOT=$PWD POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-4}
S=${STAGES:?set STAGES to the stage dir}; O=${1:?out dir}; P=${PY:-$PWD/.venv/bin/python}; mkdir -p "$O"
log(){ echo "[$(date -u +%FT%TZ)] assemble_final: $*"; }
[ "${CHECK:-1}" = 1 ] && { log "stage inputs vs manifest/final_inputs.md5"; ( cd "$S" && md5sum -c "$PRC_ROOT/manifest/final_inputs.md5" ); }
log "F1 schedule-copy relink";  $P scripts/rule_schedule_copy_lirf_relink.py data/final/final_ranking.parquet "$O/schedule_copy"
log "F2 rule-priced rows";      $P scripts/rulebag_rows_merge.py "$S/kcatlg_line.parquet" "$S/bag_line.parquet" "$S/bag_classes.parquet" "$O/F2_merge.parquet"
log "F3 schedule-copy rule";    $P scripts/rule_schedule_copy_lirf_apply.py "$O/F2_merge.parquet" "$O/schedule_copy/final_dryrun.parquet" "$O/zesty-puzzle_final.parquet"
$P scripts/validate_submission.py "$O/zesty-puzzle_final.parquet" --features "$S/features_final.parquet"
log "F4 public-period check file"; $P scripts/public_period_extract.py "$O/zesty-puzzle_final.parquet" data/features_v9.parquet "$O/public_janjul.parquet"
[ "${CHECK:-1}" = 1 ] && { log "outputs vs manifest/final_outputs.md5"; ( cd "$O" && md5sum -c "$PRC_ROOT/manifest/final_outputs.md5" ); }
md5sum "$O/zesty-puzzle_final.parquet" "$O/public_janjul.parquet"; log "done"
