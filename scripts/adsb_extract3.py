#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Third-generation ADS-B trace features (groups P1 + P2): for every departure linked by
the v1 rule (same link as adsb_extract.py: (airport, callsign), ground->air transition within +-300 s of MVT_TIME), from
the ground segment before take-off:
  P1  t_mv_gs    first time the readsb ground speed is >= GS_MOVE kt on two consecutive points after the own-stand period
      t_mv_pos   same with the position-derived speed >= 1.5 m/s          gs0 / v0  speed at the first point (kt / m/s)
  P2  lat0, lon0 position of the first point   d_tko0  distance (m) of the first point to the take-off point
      path_m     path length of the segment    span_s  its duration    mean_v  path/span    ppm  points per minute
      d_first_stand  distance of the first point to the own stand (NaN without a learned stand position)
All clocks are epoch seconds; build_adsb3_features.py makes them MVT_TIME-relative. Label-free (no BLOCK, no y, no fb).
Writes clk3_YYYYMMDD.parquet. usage: adsb_extract3.py YYYY-MM-DD [...]   (~40 s per day)"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adsb_extract import hav, movements, BASE

GS_MOVE = 4.0
SP = pl.read_parquet(f"{BASE}/stand_pos.parquet").filter(pl.col("n") >= 5)
SPD = {(a, s): (la, lo) for a, s, la, lo in zip(SP["apt"], SP["stand"], SP["lat"], SP["lon"])}
COLS = [
    "t_mv_gs",
    "t_mv_pos",
    "gs0",
    "v0",
    "lat0",
    "lon0",
    "d_tko0",
    "path_m",
    "span_s",
    "mean_v",
    "ppm",
    "d_first_stand",
]


def extract3(day):
    tag = day.replace("-", "")
    src = f"{BASE}/apt_{tag}.parquet"
    out = f"{BASE}/clk3_{tag}.parquet"
    if not os.path.exists(src):
        print(f"{day}: no points file")
        return
    P = pl.read_parquet(src)
    D = movements(day)
    G = {k: v for k, v in P.group_by(["icao", "apt"])} if len(P) else {}
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
            gs = g["gs"].to_numpy().astype(float)
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
            st, sla, slo, sgs = t[j:k], la[j:k], lo[j:k], np.nan_to_num(gs[j:k], nan=0.0)
            step = hav(sla[:-1], slo[:-1], sla[1:], slo[1:])
            v = step / np.maximum(np.diff(st), 1.0)
            path = float(step.sum())
            span = float(st[-1] - st[0])
            d_tko = hav(sla, slo, la[k], lo[k])
            key = (r["apt"], r["stand"])
            dstand = hav(sla, slo, *SPD[key]) if key in SPD else None
            at = np.where(dstand <= 60)[0] if dstand is not None else np.array([], dtype=int)
            i0 = int(at[-1]) if len(at) else 0
            mv = mv_pos = np.nan
            for i in range(i0, len(st) - 1):
                if sgs[i] >= GS_MOVE and sgs[i + 1] >= GS_MOVE:
                    mv = float(st[i])
                    break
            for i in range(i0, len(v) - 1):
                if v[i] >= 1.5 and v[i + 1] >= 1.5:
                    mv_pos = float(st[i])
                    break
            cand = dict(
                MVT_ID_mvt=r["MVT_ID_mvt"],
                t_tko3=float(t[k]),
                t_mv_gs=mv,
                t_mv_pos=mv_pos,
                gs0=float(sgs[0]),
                v0=float(v[0]),
                lat0=float(sla[0]),
                lon0=float(slo[0]),
                d_tko0=float(d_tko[0]),
                path_m=path,
                span_s=span,
                mean_v=path / max(span, 1.0),
                ppm=60.0 * len(st) / max(span, 1.0),
                d_first_stand=float(dstand[0]) if dstand is not None else np.nan,
            )
            if best is None or abs(cand["t_tko3"] - r["tko"]) < abs(best["t_tko3"] - r["tko"]):
                best = cand
        if best:
            res.append(best)
    if res:
        C = D.join(pl.DataFrame(res), on="MVT_ID_mvt", how="left")
    else:
        C = D.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in ["t_tko3"] + COLS])
    C.write_parquet(out)
    print(
        f"{day}: departures {len(D):,} linked {len(res):,} gs-move clock {sum(1 for x in res if np.isfinite(x['t_mv_gs'])):,}",
        flush=True,
    )


if __name__ == "__main__":
    for day in sys.argv[1:]:
        try:
            extract3(day)
        except Exception as e:
            print(f"{day}: EXTRACT3 FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
