#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_CLUSTER_SETUP: NO-SLURM runner for a plain SSH GPU box (1..8 GPUs). Run inside tmux/screen.
# usage: bash cluster/run_box.sh [all|member|t2x] [--skip-local]      (from the bundle root, after env/setup_env.sh)
# One worker per visible GPU; workers claim units from the list with an atomic mkdir lock; finished units (result file exists) are skipped,
# so after a crash / reboot / spot preemption just run the same command again (stale locks are cleared at start).
# Order: parity first (gate), then the list. PARITY must say PASS before the long run is worth paying for.
set -euo pipefail; B=$(cd "$(dirname "$0")/.." && pwd); cd "$B"
export PAYLOAD=${PAYLOAD:-$B/data/payload} RUN=${RUN:-$B/run} HF_HOME=${HF_HOME:-$B/data/payload/vendor/hf} HF_HUB_OFFLINE=1; PY=${ENVDIR:-$B/.venv-kumo}/bin/python
WHAT=${1:-all}; mkdir -p "$RUN/logs" "$RUN/lists"; "$PY" cluster/tasks.py "$RUN/lists" ${2:-} >/dev/null
rm -rf "$RUN/locks"   # no worker is running when this script starts
if ! grep -qs "PARITY PASS" "$RUN/parity_${PROFILE:-m20e8}.txt"; then
  PRC_THREADS=4 "$PY" cluster/km_task.py parity 2>&1 | tee "$RUN/parity_${PROFILE:-m20e8}.txt"
  grep -q "PARITY PASS" "$RUN/parity_${PROFILE:-m20e8}.txt" || { echo "PARITY not PASS -> stop"; exit 1; }
fi
NG=$(nvidia-smi -L | wc -l); NC=$(nproc); T=$(( NC / NG > 8 ? 8 : (NC / NG < 1 ? 1 : NC / NG) ))
echo "box $(hostname): $NG GPU(s), $NC cpus -> $NG workers x $T threads | list $WHAT ($(wc -l < "$RUN/lists/$WHAT.txt") units) | $(date -u)"
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g PRC_THREADS=$T nohup "$PY" cluster/km_task.py --from-list "$RUN/lists/$WHAT.txt" --claim > "$RUN/logs/worker_gpu$g.log" 2>&1 &
done
wait; bash jobs/status.sh; echo "RUN_BOX_DONE $(date -u)  -> next: copy run/results back and collect on the build machine (docs/kumo_gpu.md, Running step 3)"
