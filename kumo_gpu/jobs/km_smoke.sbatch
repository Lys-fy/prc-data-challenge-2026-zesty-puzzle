#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_CLUSTER_SETUP: GPU self-check + PARITY (janjul x LSZH recomputed vs the GPU workstation output) + rows/s projection. Gate for the arrays (afterok).
#SBATCH --job-name=xclu-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
source "${BUNDLE:?}/jobs/config.env"; cd "$BUNDLE"
echo "xclu smoke on $(hostname) $(date -u)"; nvidia-smi || true
CUDA_FLAVOR=${CUDA_FLAVOR:-} SKIP_VENV=1 ENV_PY="$ENVDIR/bin/python" OFFLINE=1 bash env/setup_env.sh     # weights sha + GPU smoke (no network)
export PRC_THREADS=4
"$ENVDIR/bin/python" cluster/km_task.py parity | tee "$RUN/parity_${PROFILE:-m20e8}.txt"
grep -q "PARITY PASS" "$RUN/parity_${PROFILE:-m20e8}.txt" || { echo "PARITY not PASS: arrays will not start (afterok)."; exit 1; }
