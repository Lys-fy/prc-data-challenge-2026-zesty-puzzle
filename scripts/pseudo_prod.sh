#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# PSEUDOLABEL: production member seed <SEED> of the pseudo-labelled L twin: 2025 rows + matched 2026 ranking rows labelled by the period's first-pass line (teacher; scripts/pseudo_labels.py prod)
# (scratchpad/xsyn/pseudo_prod_chain88.parquet, same mask p < .05 / teacher < 7,200 s and weight .3 as the folds) -> data/preds/pseudolabel_pl3p88_s<SEED>.parquet
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; S=$1; T=${2:-8}
ls scratchpad/xsyn/pseudo_prod_chain88.parquet >/dev/null
SEED=$S bash scripts/pseudo_member.sh PL3 prod $T
ls -la data/preds/pseudolabel_pl3p88_s$S.parquet
