#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# The adsb.lol pipeline in its dependency order, from the public archives to the six side tables used by the
# final model (docs/DATA_SOURCES.md §2 row 1; ODbL 1.0, "Contains information from adsb.lol, made available under the ODbL").
#   pull      adsb_pull.py          archive (tag planes-readsb-prod-0, fallback -staging-0) -> data/external/adsblol/apt_YYYYMMDD.parquet
#   extract   adsb_extract.py       per-departure link + clocks                               -> clk_YYYYMMDD.parquet
#   standpos  build_stand_pos.py    stand positions from the clk days in manifest/standpos_days.txt (the 305 production days) -> stand_pos.parquet
#   extract2  adsb_extract2.py      stand-relative clocks (needs stand_pos)                    -> clk2_*
#   extract3  adsb_extract3.py      third-generation segment features (needs stand_pos)        -> clk3_*
#   extract5  adsb_segment_extract.py        push-back clocks (adsb5_clock)                           -> clk5_*
#   extract6  adsb_pushstart_extract.py        push-START clocks with heading semantics (pushstart)  -> clkP_*
#   extractL  adsb_longstop_extract.py       longest-stop clock + trace-start class (adsbL)     -> clkL_*
#   tables    build_adsb{,2,3,5,6,L}_features.py -> data/external/adsb{,2,3,5,6,L}_features.parquet (need data/features_v9.parquet)
# adsb_pull.py and adsb_longstop_extract.py skip days whose output exists; the other extractors rewrite their day files (deterministic).
# The production pull (19-21 Sep 2026) got 410 of the 427 days; the other 17 are missing from prod-0/staging-0 and stay NULL
# (docs/known_issues.md 4.3). Days pulled from other release tags are not part of the final model.
# usage: scripts/adsb_all.sh [--days FILE] [--dry-run] [step ...]      (default: all steps; env NP = processes for extract5/6/L, default 4)
set -euo pipefail
ROOT="${PRC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT"
PY="${PY:-.venv/bin/python}"; NP="${NP:-4}"; export NP
DAYS=""; DRY=0; STEPS=()
while [ $# -gt 0 ]; do case "$1" in --days) DAYS="$2"; shift 2;; --dry-run) DRY=1; shift;; *) STEPS+=("$1"); shift;; esac; done
[ ${#STEPS[@]} -eq 0 ] && STEPS=(pull extract standpos extract2 extract3 extract5 extract6 extractL tables)
if [ -z "$DAYS" ]; then     # 2025 (training) + January and July 2026 (ranking)
  DAYS=$(mktemp); "$PY" - "$DAYS" <<"PY"
import sys, datetime as dt
out = []
for lo, hi in (("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-01-31"), ("2026-07-01", "2026-07-31")):
    d, e = dt.date.fromisoformat(lo), dt.date.fromisoformat(hi)
    while d <= e: out.append(d.isoformat()); d += dt.timedelta(days=1)
open(sys.argv[1], "w").write("\n".join(out) + "\n")
PY
fi
# days that have an archive (the later steps only run on those)
PULLED=$(mktemp); for d in $(cat "$DAYS"); do [ -f "data/external/adsblol/apt_${d//-/}.parquet" ] && echo "$d"; done > "$PULLED" || true
run(){ echo "[$(date +%T)] $*"; [ "$DRY" = 1 ] || "$@"; }
for s in "${STEPS[@]}"; do
  case "$s" in
    pull)     run "$PY" scripts/adsb_pull.py --queue "$DAYS"
              for d in $(cat "$DAYS"); do [ -f "data/external/adsblol/apt_${d//-/}.parquet" ] && echo "$d"; done > "$PULLED" || true
              echo "pulled days: $(wc -l < "$PULLED")";;
    extract)  run xargs -a "$PULLED" -n 20 "$PY" scripts/adsb_extract.py;;
    standpos) SPD="${STANDPOS_DAYS-manifest/standpos_days.txt}"   # production day set (305 days); STANDPOS_DAYS= (empty) = all days
              if [ -n "$SPD" ]; then run env STANDPOS_DAYS="$SPD" "$PY" scripts/build_stand_pos.py; else run "$PY" scripts/build_stand_pos.py; fi;;
    extract2) run xargs -a "$PULLED" -n 20 "$PY" scripts/adsb_extract2.py;;
    extract3) run xargs -a "$PULLED" -n 20 "$PY" scripts/adsb_extract3.py;;
    extract5) run "$PY" scripts/adsb_segment_extract.py --days "$PULLED";;
    extract6) run "$PY" scripts/adsb_pushstart_extract.py --days "$PULLED";;
    extractL) run "$PY" scripts/adsb_longstop_extract.py --days "$PULLED";;
    tables)   test -f data/features_v9.parquet || { echo "tables need data/features_v9.parquet (make features)"; exit 1; }
              for t in adsb adsb2 adsb3 adsb5 adsb6 adsbL; do run "$PY" "scripts/build_${t}_features.py"; done;;
    *) echo "unknown step $s"; exit 1;;
  esac
done
