#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Kumo Tabular members on a period's ranking rows, inside a phase2 workspace (GPU steps need CUDA; nothing is trained: the context is the model).
#   usage: bash scripts/kumo_prod.sh <ws_name> <step> ...        root = $PRC_ROOT, else the current directory
#   k250stage   (CPU, ~25 GB)  kumo_gpu/stage/stage_data.py on the workspace -> WS/scratchpad/kumo/k250_payload (member/km_<APT>.parquet + vendor/)
#   k250        (GPU)          K250 = KSBWPc100 spec (medium, the airport's whole 2025 pool as context, 100 importance columns, 8 instances, fp16, seed 0)
#   k250collect (CPU)          collect.py into the workspace: data/preds/KSBWPc100P_s0.parquet + data/oof/KSBWPc100_prod.npy; marker m_K250
#   kcat        (GPU + CPU)    KCAT8 codes + prod instances, KCATL prod instances, the KCATL16 bag, both collected into the workspace,
#                              then the tail guard -> data/oof/KCATLg_prod.npy + data/preds/KCATLgP_s0.parquet; marker m_KCATLg
#   kstkdump    (CPU)          the chain-stacker prod frames stk_<C>_prod.npz (needs pre / pre_v87, all members and K250)
#   kstkbundle  (CPU)          stk6 dir = the 2025 fold frames (A side, kumo_gpu/data/payload/stk6) + the new prod frames (+ MD5SUMS)
#   kstk        (GPU)          KSTK6Kq / KSTK6NNA3 prod instances (km_adapters_kstk.py, 8 instances per airport)
#   kstkgk      (CPU)          gK = the 2025 fold instances (A side, $KSTK_FOLD_RUN) + these prod instances -> data/preds/KSTK6<C>P.npy; marker m_KSTK
set -euo pipefail; cd "${PRC_ROOT:-$PWD}"; REPO=$PWD; P=$REPO/.venv/bin/python
WSN=${1:?ws}; shift; WS=$REPO/phase2/work/$WSN; K=$WS/scratchpad/kumo; PL=$K/k250_payload; mkdir -p "$K"
kumo_engine_job=kumo_$WSN; source scripts/kumo_env.sh; KSTK_FOLD_RUN=${KSTK_FOLD_RUN:-$REPO/kumo_gpu/run_kstk_folds}
log(){ echo "[$(date -u +%FT%TZ)] kumo_prod $WSN $*"; }
units(){ for a in $APTS; do for i in 0 1 2 3 4 5 6 7; do echo "inst:prod:$a:$i"; done; echo "agg:prod:$a"; done; }
collect(){   # $1 = run dir, $2 = member name: link its 10 prod member files and collect them into the workspace (rows checked vs the ranking rows)
  local T=$K/collect_$2; rm -rf "$T"; mkdir -p "$T/results/member"; ln -s "$(readlink -f "$1")"/results/member/$2_prod_*.npz "$T/results/member/"
  [ "$(ls "$T/results/member" | wc -l)" = 10 ] || { echo "$2 prod incomplete"; exit 2; }
  PRC_ROOT=$WS $P kumo_gpu/collect/collect.py "$T" --replace | tail -n 3; }
for ST in "$@"; do log "$ST"; case $ST in
  k250stage)
    mkdir -p "$WS/artifacts"; [ -e "$WS/artifacts/xkumo" ] || ln -s "$(readlink -f "$REPO/artifacts/xkumo")" "$WS/artifacts/xkumo"   # Kumo weights (env/README.md)
    rm -rf "$PL"; PRC_ROOT=$WS $P kumo_gpu/stage/stage_data.py "$PL";;
  k250)
    units > "$K/units_k250.txt"
    PAYLOAD=$PL RUN=$K/k250_run NAME=KSBWPc100 SIZE=medium CAP=1000000 NCOL=100 COLMODE=imp CORE=50 NINST=8 EIN=1 CTXMODE=uniform AMP=fp16 SEED=0 \
      $KPY kumo_gpu/cluster/km_engine.py --from-list "$K/units_k250.txt";;
  k250collect)
    for f in "$WS/data/preds/KSBWPc100P_s0.parquet" "$WS/data/oof/KSBWPc100_prod.npy"; do [ -L "$f" ] && rm -f "$f"; done
    PRC_ROOT=$WS $P kumo_gpu/collect/collect.py "$K/k250_run"
    $P - "$WS" <<'PY'
import sys, numpy as np, polars as pl
ws = sys.argv[1]; z = np.load(f"{ws}/data/oof/KSBWPc100_prod.npy"); K = pl.read_parquet(f"{ws}/data/preds/KSBWPc100P_s0.parquet")
F = pl.read_parquet(f"{ws}/data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank"]).filter(pl.col("is_rank") == 1)
assert z.shape[0] > 2 and np.isfinite(K["taxi"].to_numpy()).all() and set(K["MVT_ID_mvt"].to_list()) == set(F["MVT_ID_mvt"].to_list()), "K250 prod check"
print("K250 prod", z.shape, "| rank rows", F.height, "| spread finite share", float(np.isfinite(z[2]).mean()))
PY
    touch "$WS/logs/m_K250.done";;
  kcat)
    C=$K/kcat_codes; $P scripts/kumo_kcat_codes.py "$WS/data/features_v9.parquet" "$PL/member" "$C/prod"
    export KCAT_PROD_PAYLOAD=$PL KCAT_CODES=$C KCAT_RUN=$K/kcat_runW KCATL_RUN=$K/kcat_runL KCATL16_RUN=$K/kcat_runL16
    bash scripts/kumo_kcat_run.sh prod; bash scripts/kumo_kcat_logtarget_run.sh prod; bash scripts/kumo_kcat_logtarget_bag.sh prod
    collect "$KCAT_RUN" KCAT; collect "$KCATL16_RUN" KCATL16
    $P scripts/kumo_kcat_logtarget_tailguard.py prod "$WS"; touch "$WS/logs/m_KCATLg.done";;
  kstkdump)
    ( cd "$WS"; export P2_ROOT=$WS PRC_ROOT=$WS PYTHONPATH=$REPO/phase2/lib/guard P2_STRICT=0 P2_NRANK=$(grep -E '^NRANK=' "$REPO/phase2/periods/$WSN.env" | cut -d= -f2) MODE=$(grep -E '^MODE=' "$REPO/phase2/periods/$WSN.env" | cut -d= -f2)
      mkdir -p kumo_gpu/data/payload/stk6; rm -f kumo_gpu/data/payload/stk6/stk_*_prod.npz
      for f in "$REPO"/kumo_gpu/data/payload/stk6/stk_*.npz; do case $f in *_prod.npz) ;; *) ln -sf "$(readlink -f "$f")" kumo_gpu/data/payload/stk6/;; esac; done
      NT=${NT:-6} bash scripts/kstk_run.sh dump );;
  kstkbundle)
    S6=$K/stk6; rm -rf "$S6"; mkdir -p "$S6"
    for f in "$REPO"/kumo_gpu/data/payload/stk6/stk_*.npz; do case $f in *_prod.npz) ;; *) ln -s "$(readlink -f "$f")" "$S6/";; esac; done
    for C in Kq NNA3; do ln -s "$(readlink -f "$WS/kumo_gpu/data/payload/stk6/stk_${C}_prod.npz")" "$S6/"; done
    ( cd "$S6" && md5sum -- *.npz > MD5SUMS );;
  kstk)
    S6=$K/stk6; ( cd "$S6" && md5sum -c --quiet MD5SUMS )
    export PAYLOAD=$PL RUN=$K/kstk_run SIZE=medium CAP=1000000 NCOL=1000 COLMODE=all NINST=8 EIN=1 CTXMODE=uniform AMP=fp16 SEED=0 KSTK_CLIP=1800 KSTK_DIR=$S6
    KSTK_CHAIN=Kq NAME=KSTK6Kq $KPY kumo_gpu/cluster/km_adapters_kstk.py --selftest --real
    units > "$K/units_kstk.txt"
    for C in Kq NNA3; do KSTK_CHAIN=$C NAME=KSTK6$C $KPY kumo_gpu/cluster/km_adapters_kstk.py --from-list "$K/units_kstk.txt"; done;;
  kstkgk)   # merged run dir: the 2025 fold instance dirs (A side) + this period's prod instance dirs
    MR=$K/kstk_merged; rm -rf "$MR"
    for C in Kq NNA3; do E=$MR/results/engine/KSTK6$C; mkdir -p "$E"
      for d in "$KSTK_FOLD_RUN"/results/engine/KSTK6$C/*/; do b=$(basename "$d"); case $b in prod_*) ;; *) ln -s "$(readlink -f "$d")" "$E/$b";; esac; done
      for d in "$K"/kstk_run/results/engine/KSTK6$C/prod_*/; do ln -s "$(readlink -f "$d")" "$E/$(basename "$d")"; done
      echo "KSTK6$C: $(ls "$E" | grep -vc '^prod_') fold dirs, $(ls "$E" | grep -c '^prod_') prod dirs"; done
    for C in Kq NNA3; do f=$WS/data/preds/KSTK6${C}P.npy; [ -L "$f" ] && rm -f "$f"; done
    ( cd "$WS"; KSTK_DIR=$K/stk6 $P scripts/kstk_gk.py "$MR" Kq NNA3 )
    touch "$WS/logs/m_KSTK.done";;
  *) echo "unknown step $ST"; exit 2;;
esac; log "$ST done"; done
