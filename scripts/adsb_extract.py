#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Per-departure ADS-B clocks from a day's airport-box points (adsb_pull.py output).
Link: (airport, callsign) from the trace's flight field, ground->air transition within +-300 s of MVT_TIME.
Ground segment: contiguous 'ground' points before that transition (gaps < 30 min), cut at a readsb leg-start flag.
Stationary runs by displacement (< 40 m from the run anchor, >= 60 s). Clocks (epoch s):
  t_first_seen  first point of the segment            t_first_move  end of the first stationary run IF it starts the segment
  t_hold_end    end of the last stationary run         t_tko         first airborne point
Flags: started_stationary, n_runs, n_ground, span, first_gs (derived), first_run_len, run0_lat/lon (for stand centroids).
usage: adsb_extract.py YYYY-MM-DD [...]  -> data/external/adsblol/clk_YYYYMMDD.parquet
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


def movements(day):
    fs = (
        [f"{ROOT}/data/raw/ranking.parquet"]
        if day.startswith("2026")
        else [f for f in glob.glob(f"{ROOT}/data/raw/training_*.parquet") if day[:7] + "-01_" in f]
    )
    if not fs:
        raise SystemExit(f"no raw movement file for {day[:7]}")
    R = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
    if "PHASE_mvt" in R.columns:
        R = R.filter(pl.col("PHASE_mvt") == "DEP")
    cols = [
        "MVT_ID_mvt",
        pl.col("ADEP_mvt").alias("apt"),
        pl.coalesce(pl.col("CALLSIGN_flt"), pl.col("FLIGHT_mvt")).str.strip_chars().alias("cs"),
        pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").alias("tko"),
        pl.col("STAND_mvt").alias("stand"),
    ]
    return R.filter(pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.date().cast(pl.Utf8) == day).select(cols)


def extract(day):
    tag = day.replace("-", "")
    out = f"{BASE}/clk_{tag}.parquet"
    src = f"{BASE}/apt_{tag}.parquet"
    if not os.path.exists(src):
        print(f"{day}: no points file")
        return
    P = pl.read_parquet(src)
    D = movements(day)
    G = {k: v for k, v in P.group_by(["icao", "apt"])}
    bycs = {}
    for (icao, apt), g in G.items():
        for c in g["cs"].drop_nulls().unique().to_list():
            bycs.setdefault((apt, c), []).append(icao)
    res = []
    for r in D.iter_rows(named=True):
        best = None
        for icao in bycs.get((r["apt"], r["cs"]), []):
            g = G[(icao, r["apt"])]
            t = g["t"].to_numpy()
            gr = g["ground"].to_numpy()
            la = g["lat"].to_numpy()
            lo = g["lon"].to_numpy()
            fl = g["flags"].to_numpy() if "flags" in g.columns else np.zeros(len(t), dtype=int)
            air = np.where((gr == 0) & (np.abs(t - r["tko"]) < 300))[0]
            if len(air) == 0:
                continue
            k = air[0]
            j = k
            while j > 0 and gr[j - 1] == 1 and t[j] - t[j - 1] < 1800 and not (fl[j] & 2):
                j -= 1
            if k - j < 2:
                continue
            st, sla, slo = t[j:k], la[j:k], lo[j:k]
            runs = []
            i = 0
            while i < len(st):
                i2 = i
                while i2 + 1 < len(st) and hav(sla[i], slo[i], sla[i2 + 1], slo[i2 + 1]) < 40.0:
                    i2 += 1
                if st[i2] - st[i] >= 60.0:
                    runs.append((i, i2))
                i = i2 + 1
            started_stat = bool(runs and runs[0][0] == 0)
            t_first_move = float(st[runs[0][1]]) if started_stat else np.nan
            d01 = hav(sla[0], slo[0], sla[1], slo[1])
            gs0 = d01 / max(st[1] - st[0], 1.0)
            cand = dict(
                MVT_ID_mvt=r["MVT_ID_mvt"],
                icao=icao,
                t_first_seen=float(st[0]),
                t_first_move=t_first_move,
                t_hold_end=float(st[runs[-1][1]]) if runs else np.nan,
                t_tko=float(t[k]),
                started_stationary=started_stat,
                n_runs=len(runs),
                n_ground=int(k - j),
                span=float(st[-1] - st[0]),
                first_gs=float(gs0),
                first_run_len=float(st[runs[0][1]] - st[runs[0][0]]) if runs else np.nan,
                run0_lat=float(np.median(sla[runs[0][0] : runs[0][1] + 1])) if started_stat else np.nan,
                run0_lon=float(np.median(slo[runs[0][0] : runs[0][1] + 1])) if started_stat else np.nan,
                hold_total=float(sum(st[b] - st[a] for a, b in runs[1:])) if len(runs) > 1 else 0.0,
            )
            if best is None or abs(cand["t_tko"] - r["tko"]) < abs(best["t_tko"] - r["tko"]):
                best = cand
        if best:
            res.append(best)
    if res:
        C = D.join(pl.DataFrame(res), on="MVT_ID_mvt", how="left")
    else:  # day with no usable points (e.g. adsb.lol published nothing): keep the schema so downstream code works
        C = D.with_columns(
            icao=pl.lit(None, dtype=pl.Utf8),
            t_first_seen=pl.lit(None, dtype=pl.Float64),
            t_first_move=pl.lit(None, dtype=pl.Float64),
            t_hold_end=pl.lit(None, dtype=pl.Float64),
            t_tko=pl.lit(None, dtype=pl.Float64),
            started_stationary=pl.lit(None, dtype=pl.Boolean),
            n_runs=pl.lit(None, dtype=pl.Int64),
            n_ground=pl.lit(None, dtype=pl.Int64),
            span=pl.lit(None, dtype=pl.Float64),
            first_gs=pl.lit(None, dtype=pl.Float64),
            first_run_len=pl.lit(None, dtype=pl.Float64),
            run0_lat=pl.lit(None, dtype=pl.Float64),
            run0_lon=pl.lit(None, dtype=pl.Float64),
            hold_total=pl.lit(None, dtype=pl.Float64),
        )
    C.write_parquet(out)
    h = C.filter(pl.col("icao").is_not_null())
    s = h.filter(pl.col("started_stationary").fill_null(False))
    print(
        f"{day}: departures {len(D):,} linked {len(h):,} ({100*len(h)/max(len(D),1):.0f}%) started-stationary {len(s):,} ({100*len(s)/max(len(D),1):.1f}%) | per apt linked/stationary: "
        + " ".join(
            f"{a} {int((h['apt']==a).sum())}/{int((s['apt']==a).sum())}" for a in sorted(set(D["apt"].to_list()))
        ),
        flush=True,
    )


if __name__ == "__main__":
    for day in sys.argv[1:]:
        try:
            extract(day)
        except Exception as e:
            print(f"{day}: EXTRACT FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
