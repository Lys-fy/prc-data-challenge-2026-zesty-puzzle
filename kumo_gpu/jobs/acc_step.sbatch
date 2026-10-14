#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_ENGINE acceptance: generic one-GPU step. Submitted by acceptance/run_acceptance.sh (MODE=slurm) with --export=ALL,BUNDLE=...,CMDFILE=<script written by run_acceptance.sh> (a file, because --export splits on commas).
#SBATCH --job-name=xkfm-acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -uo pipefail; source "${BUNDLE:?}/jobs/config.env"; cd "$BUNDLE"
echo "acc step on $(hostname) $(date -u) | $(cat "${CMDFILE:?}")"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
export MEMLOG=${RUN}/logs/mem_${SLURM_JOB_ID}.log PRC_THREADS=${SLURM_CPUS_PER_TASK:-4}
bash "$CMDFILE"
