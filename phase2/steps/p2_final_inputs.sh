#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# FINAL_PERIOD: MODE=new period inputs for the final ranking file, inside the phase2 workspace (run after `run_pipeline.sh <period> ws`,
# before `run_pipeline.sh <period> np_features`). Uses the prefetched external data; only the competition file is new.
#   usage: bash phase2/steps/p2_final_inputs.sh <period> [step ...]     steps (default all): ext_link opdi [np_features] side1 adsb_x_JJ adsb_x_FJ adsb_tables check
#   ext_link      prefetched adsb.lol day files (scratchpad/xfinal/adsblol_2026/apt_*), the METAR table to 8 Oct (metar_10apt_ext), the PRU csv dir
#                 (PRU_DIR, default data/external/pru_20261008) and the OPDI month files (OPDI_DIR, fetched beforehand by scripts/fetch_opdi.py)
#                 linked into the workspace's data/external (links only; the project's data/ is never written)
#   adsb_extract  every ranking day (from data/raw/ranking.parquet): pull if missing, then the clock extractors adsb_extract, adsb_extract2,
#                 adsb_extract3, adsb_segment_extract (clk5), adsb_pushstart_extract (clkP / adsb6), adsb_longstop_extract (clkL / adsbL); NPAR days in parallel
#   adsb_tables   p2_period_inputs.py --build-adsb (adsb, adsb2, adsb3, adsb5, adsb6, adsbL side tables from all extracted days)
#   check         p2_period_inputs.py --check (coverage of PRU / ADS-B / METAR / OPDI per day and airport)
set -euo pipefail
P2=$(cd "$(dirname "$0")/.." && pwd); REPO=${PRC_ROOT:-$(dirname "$P2")}; PERIOD=${1:?usage: p2_final_inputs.sh <period> [step ...]}; shift
set -a; source "$P2/periods/$PERIOD.env"; set +a
WS=$P2/work/${WS_NAME:-$PERIOD}; PY=$REPO/.venv/bin/python; NPAR=${NPAR:-6}
PRU_DIR=${PRU_DIR:-data/external/pru_20261008}; OPDI_DIR=${OPDI_DIR:-$REPO/scratchpad/xfinal/opdi_2026}
log(){ echo "[$(date +%T)] $*"; }
run(){ local s=$1; shift; [ -f "$WS/logs/$s.done" ] && { log "skip $s (done)"; return 0; }
       log "step $s"; local t0; t0=$(date +%s)
       ( cd "$WS" && "$@" ) > "$WS/logs/$s.log" 2>&1 || { log "FAILED $s (see $WS/logs/$s.log)"; tail -20 "$WS/logs/$s.log"; exit 1; }
       printf "%s\t%s\t%s\t%s\n" "$s" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u +%FT%TZ)" "$(( $(date +%s) - t0 ))" >> "$WS/logs/times.tsv"
       touch "$WS/logs/$s.done"; log "done $s ($(( $(date +%s) - t0 )) s)"; }
step_ext_link(){
  local n=0; for f in "$REPO"/scratchpad/xfinal/adsblol_2026/apt_*.parquet; do [ -e "$f" ] || continue
    t=data/external/adsblol/$(basename $f); [ -e $t ] || { ln -s "$(readlink -f $f)" $t; n=$((n+1)); }; done; echo "adsb.lol day files linked: $n"
  rm -f data/external/metar_10apt.parquet; ln -s "$(readlink -f $REPO/scratchpad/xfinal/metar_10apt_ext.parquet)" data/external/metar_10apt.parquet
  ls -la data/external/metar_10apt.parquet; ls -d "$REPO/$PRU_DIR" >/dev/null
  [ -e "$PRU_DIR" ] || ln -s "$(readlink -f $REPO/$PRU_DIR)" "$PRU_DIR"
  local k=0; if [ -d "$OPDI_DIR" ]; then for f in "$OPDI_DIR"/*.parquet; do t=data/external/opdi/$(basename $f); [ -e $t ] || { ln -s "$(readlink -f $f)" $t; k=$((k+1)); }; done; fi
  echo "OPDI files linked: $k (from $OPDI_DIR)"; ls data/external/opdi | sed -n '/flight_list_2026/p'
}
step_adsb_extract(){   # optional $1 = regex on the day (e.g. '-0[17]-' for Jan/Jul only); steps adsb_x_JJ / adsb_x_FJ = the two halves
  env -u PYTHONPATH $PY -c "
import polars as pl; R = pl.read_parquet('data/raw/ranking.parquet', columns=['MVT_TIME_UTC_mvt'])
d = R.select(pl.col('MVT_TIME_UTC_mvt').dt.replace_time_zone(None).dt.date().unique().sort())['MVT_TIME_UTC_mvt'].cast(pl.Utf8).to_list()
import re, sys; d = [x for x in d if re.search(sys.argv[1], x)]
open('scratchpad/xfinal/rank_days.txt', 'w').write('\n'.join(d) + '\n'); print(len(d), 'ranking days', d[0], '..', d[-1])" "${1:-.}"
  cat > scratchpad/xfinal/extract_day.sh <<'EOF'
#!/usr/bin/env bash
# one day: pull (skipped if the apt file exists) -> clk -> clk2 -> clk3 -> clk5 (adsb5_clock) -> clkP (pushstart) -> clkL (longest stop); NP=1 inside (days run in parallel)
set -uo pipefail; D=$1; P=.venv/bin/python; t0=$(date +%s)
for s in adsb_pull.py adsb_extract.py adsb_extract2.py adsb_extract3.py adsb_segment_extract.py adsb_pushstart_extract.py adsb_longstop_extract.py; do
  NP=1 $P scripts/$s $D > scratchpad/xfinal/extract_logs/${D}_$s.log 2>&1 || { echo "DAY $D FAILED in $s"; exit 1; }; done
echo "DAY $D ok $(( $(date +%s) - t0 )) s"
EOF
  mkdir -p scratchpad/xfinal/extract_logs; chmod +x scratchpad/xfinal/extract_day.sh
  # the workspace's clock files of the ranking days start as LINKS to the shared extraction (keyed to the MVT_IDs of another ranking file;
  # adsb_longstop_extract skips existing outputs): drop those links so every ranking day is re-extracted against this period's ranking file
  # (local files are never removed)
  local k=0; while read D; do T=${D//-/}; for f in data/external/adsblol/clk*_$T.parquet; do [ -L "$f" ] && { rm -f "$f"; k=$((k+1)); }; done; done < scratchpad/xfinal/rank_days.txt
  echo "dropped $k linked clock files of ranking days"
  xargs -a scratchpad/xfinal/rank_days.txt -n 1 -P $NPAR bash scratchpad/xfinal/extract_day.sh | tee -a scratchpad/xfinal/extract_days.txt
  ! grep -q FAILED scratchpad/xfinal/extract_days.txt
}
step_opdi(){ $PY scripts/build_opdi_features.py; }      # OPDI per-departure table for the period's rows (build_features.py joins it: run BEFORE np_features)
step_side1(){   # per-row side tables of the 'make features' step that run_pipeline's np_side does not rebuild (it rebuilds dreg, xba fallback ids, q, XQ)
  $PY scripts/build_xba_features.py; $PY scripts/build_xe_queue_features.py; $PY scripts/build_durp_features.py
}
step_adsb_tables(){ $PY "$P2/steps/p2_period_inputs.py" --build-adsb; }
step_check(){ $PY "$P2/steps/p2_period_inputs.py" --check; }
export P2_REPO=$REPO P2_ROOT=$WS PRC_ROOT=$WS P2_LOG=$WS/logs/ledger.jsonl P2_STRICT=0 PYTHONPATH=$P2/lib/guard POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage P2_NRANK=${NRANK:-344841} PRU_DIR
mkdir -p "$WS/scratchpad/xfinal"
for s in ${*:-ext_link opdi side1 adsb_x_JJ adsb_x_FJ adsb_tables check}; do
  case $s in ext_link|opdi|side1|adsb_extract|adsb_tables|check) run $s step_$s;; adsb_x_JJ) run $s step_adsb_extract '-0[17]-';; adsb_x_FJ) run $s step_adsb_extract '-0[26]-';;
    *) echo "unknown step $s"; exit 2;; esac
done
