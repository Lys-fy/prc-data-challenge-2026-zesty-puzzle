#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_CLUSTER_SETUP: submit smoke/parity -> (afterok) member array + t2x array. Run on the cluster login node from the bundle root.
# usage: bash jobs/submit_all.sh [--skip-local] [--only member|t2x]
set -euo pipefail; B=$(cd "$(dirname "$0")/.." && pwd); export BUNDLE=$B; source "$B/jobs/config.env"
grep -q "__PARTITION__\|__ACCOUNT__\|__GPUTYPE__" "$B/jobs/config.env" && { echo "fill the placeholders in jobs/config.env first"; exit 2; }
mkdir -p "$RUN/logs" "$RUN/lists" "$RUN/results"
SKIP=""; ONLY=""
while [ $# -gt 0 ]; do case $1 in --skip-local) SKIP=--skip-local;; --only) ONLY=$2; shift;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
"$ENVDIR/bin/python" "$B/cluster/tasks.py" "$RUN/lists" $SKIP
COMMON=(-p "$SBATCH_PARTITION" -A "$SBATCH_ACCOUNT" $GPU_OPTS $QOS_OPTS --cpus-per-task="$CPUS" --mem="$MEM")
S=$(sbatch --parsable "${COMMON[@]}" -t "$TIME_SMOKE" -o "$RUN/logs/smoke_%j.out" --export=ALL,BUNDLE="$B" "$B/jobs/km_smoke.sbatch"); echo "smoke+parity $S"
for L in member t2x; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$L" ] && continue
  N=$(wc -l < "$RUN/lists/$L.txt"); T=TIME_${L^^}
  J=$(sbatch --parsable "${COMMON[@]}" -t "${!T}" --dependency=afterok:$S --array=0-$((N - 1))%$MAX_PAR \
      -o "$RUN/logs/${L}_%A_%a.out" --export=ALL,BUNDLE="$B",LIST="$RUN/lists/$L.txt" "$B/jobs/km_array.sbatch"); echo "$L array $J ($N units)"
done
echo "status: bash jobs/status.sh"
