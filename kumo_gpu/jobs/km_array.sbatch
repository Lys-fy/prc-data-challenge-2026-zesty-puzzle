#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_CLUSTER_SETUP Slurm array: one array task = STRIDE consecutive units of $LIST (member.txt / t2x.txt / all.txt, from cluster/tasks.py).
# Submitted by submit_all.sh (it fills partition/account/gres/time from config.env); can also be submitted by hand:
#   sbatch -p <part> -A <acct> --gres=gpu:<type>:1 -t 02:00:00 --array=0-69%8 --export=ALL,LIST=$RUN/lists/member.txt jobs/km_array.sbatch
# Resume: a unit whose result file exists is skipped, so re-submitting the same array (or `scontrol requeue`) only runs the missing units.
#SBATCH --job-name=xclu-km
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
set -euo pipefail
source "${BUNDLE:?}/jobs/config.env"
STRIDE=${STRIDE:-1}; i0=$(( SLURM_ARRAY_TASK_ID * STRIDE ))
mapfile -t U < <(sed -n "$(( i0 + 1 )),$(( i0 + STRIDE ))p" "${LIST:?}")
trap 'echo "USR1 (time limit / preemption): requeue"; scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" || true' USR1
echo "xclu array ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID} on $(hostname) | units ${U[*]} | $(date -u)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "$BUNDLE"; export PRC_THREADS=${SLURM_CPUS_PER_TASK:-4} MEMLOG=${MEMLOG:-$RUN/logs/mem_${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}.log}   # KUMO_ENGINE: memory line per minute + host guard
"$ENVDIR/bin/python" cluster/km_task.py "${U[@]}" &
wait $!
