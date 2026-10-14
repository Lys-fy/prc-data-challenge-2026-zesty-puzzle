#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_CLUSTER_SETUP OPTIONAL: KUMO_ICL_LIRF fine-tuned Kumo member for one fold: fine-tune + infer. Only after an coordinator GO.
#   sbatch ... --export=ALL,FOLD=months28 jobs/km_ft.sbatch    (stage 1 = months28 and janjul)
#SBATCH --job-name=xclu-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -euo pipefail
source "${BUNDLE:?}/jobs/config.env"; cd "$BUNDLE"; export PRC_THREADS=8 SIZE=${SIZE:-small}
CK="$RUN/artifacts/xtfm2/ft_${SIZE}_${FOLD:?}.pt"; OUT="$RUN/data/oof/kumo_icl_lirf_ft_${FOLD}.npy"
[ -f "$RUN/ft_done_$FOLD" ] || "$ENVDIR/bin/python" cluster/km_ft.py ft "$FOLD"      # (no mid-run resume: 600 steps; re-run restarts the fine-tune)
touch "$RUN/ft_done_$FOLD"
[ -f "$OUT" ] || "$ENVDIR/bin/python" cluster/km_ft.py infer "$FOLD"
ls -la "$CK" "$OUT"
