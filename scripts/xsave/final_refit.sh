#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# SAVE_MODEL: FINAL-table prod refit of one member seed, rank-only + saved model. Runs the members_xsave.tsv command unchanged except:
#   --features <FEATS>, --name <PFX><FAM>_s<S> (PFX default empty = the shipped name; in a phase-2 workspace outputs land in WS/data/preds),
#   --threads <T>, --out <WS>/scratchpad/xsave/final_fit.jsonl, + --rank-only --save-model <WS>/artifacts/xsave/<FAM>_s<S>.
# usage: WS=<project or workspace root> FEATS=data/features_v9.parquet PFX= bash scripts/xsave/final_refit.sh FAM SEED [THREADS]
#   FAM = any line of scripts/xsave/members_xsave.tsv, or nbr_clock_mbm / nbr_clock_ma.  GPU families (CATV17x, runway_queue_catv17xq, CATV9x) need CAT_GPU=1.
set -euo pipefail; LST=$(cd "$(dirname "$0")" && pwd)/members_xsave.tsv; WS=${WS:-${PRC_ROOT:-$PWD}}; cd "$WS"; FEATS=${FEATS:-data/features_v9.parquet}; PFX=${PFX:-}
G=$(cd "$WS/../.." 2>/dev/null && pwd)/lib/guard; if [ -f "$G/sitecustomize.py" ] && [ -d "$WS/manifest" ]; then export P2_ROOT=$(pwd) PYTHONPATH=$G${PYTHONPATH:+:$PYTHONPATH} P2_LOG=${P2_LOG:-$(pwd)/logs/xsave_ledger.jsonl}; mkdir -p logs; fi   # phase-2 copy-on-write guard: a symlinked output is replaced, never written through
FAM=$1; S=$2; T=${3:-4}; TM=$(command -v gtime || echo /usr/bin/time); mkdir -p artifacts/xsave scratchpad/xsave/logs; D=artifacts/xsave/${FAM}_s${S}
echo "SAVE_MODEL final refit $FAM s$S root $WS feats $FEATS host $(hostname) $(uname -m) threads $T $(date -u +%FT%TZ)"
if [[ $FAM == nbr_clock_* ]]; then A=${FAM#nbr_clock_}; .venv/bin/python "$(dirname "$LST")/check_inputs.py" "$WS" "$FEATS" "$FAM" "" || exit 3; [ "${DRY:-0}" = 1 ] && exit 0
  PRC_THREADS=$T $TM -v .venv/bin/python scripts/nbrclock_member.py $A prod --save-model $D --pred-name ${PFX}nbr_clock_${A}P_s0 2> scratchpad/xsave/logs/final_${FAM}.time | grep -v "Warning" | tail -4
elif [ $FAM = big_big ]; then .venv/bin/python "$(dirname "$LST")/check_inputs.py" "$WS" "$FEATS" "$FAM" "" || exit 3; [ "${DRY:-0}" = 1 ] && exit 0   # BIG set transformer (GPU: CAT_GPU=1); output WS/data/oof/big_${PFX}BIG_s<S>_prod.npy
  PRC_THREADS=$T $TM -v ${TORCH_PY:-.venv-torch/bin/python} scripts/dayset_big_member.py --fold prod --tag ${PFX}BIG_s$S --d 192 --layers 4 --heads 6 --epochs 16 --seed $S --features $FEATS --save-model $D 2> scratchpad/xsave/logs/final_${FAM}_s${S}.time | grep -v Warning | tail -4
else
  L=$(awk -F'\t' -v f="$FAM" '$1==f' "$LST"); [ -n "$L" ] || { echo "no member $FAM"; exit 2; }
  PY=$(echo "$L" | cut -f6); export TORCH_PY=${TORCH_PY:-$PWD/.venv-torch/bin/python}; PY=$(eval echo "$PY"); CMD=$(echo "$L" | cut -f7 | sed "s/{S}/$S/g")
  CMD=$(echo "$CMD" | sed -E "s#--name ${FAM}_s${S}( |\$)#--name ${PFX}${FAM}_s${S}\1#; s#--threads [0-9]+#--threads $T#; s#--features [^ ]+#--features $FEATS#")
  .venv/bin/python "$(dirname "$LST")/check_inputs.py" "$WS" "$FEATS" "$FAM" "$CMD" || exit 3
  echo "CMD: $PY $CMD --out scratchpad/xsave/final_fit.jsonl --rank-only --save-model $D"; [ "${DRY:-0}" = 1 ] && exit 0
  $TM -v $PY $CMD --out scratchpad/xsave/final_fit.jsonl --rank-only --save-model $D 2> scratchpad/xsave/logs/final_${FAM}_s${S}.time | grep -v Warning | tail -4
fi
grep -h "Elapsed\|Maximum resident" scratchpad/xsave/logs/final_${FAM}_s${S}.time scratchpad/xsave/logs/final_${FAM}.time 2>/dev/null || true; du -sh $D; echo "save_model_final_done $FAM s$S $(date -u +%FT%TZ)"
