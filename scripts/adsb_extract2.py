#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Second-generation ADS-B clocks using the learned stand positions (build_stand_pos.py). On top of adsb_extract.py's
clocks, for each linked departure whose STAND has a learned position (n >= 5):
  t_at_stand   last ground point within R_AT (60 m) of the own stand before take-off        (NaN if never seen there)
  t_leave      first point after t_at_stand farther than R_LEAVE (120 m) from the stand     (pushback/taxi start)
  d_first      distance (m) of the segment's first point from the own stand
  n_at_stand   number of points within R_AT of the stand; stand_seen = n_at_stand > 0
  arr_flag     the ground segment spans > 3 h (arrival taxi-in probably included)
Writes clk2_YYYYMMDD.parquet (all v1 columns + these). usage: adsb_extract2.py YYYY-MM-DD [...]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adsb_extract import hav, movements, BASE

R_AT, R_LEAVE = 60.0, 120.0
SP = pl.read_parquet(f"{BASE}/stand_pos.parquet").filter(pl.col("n") >= 5)
SPD = {(a, s): (la, lo) for a, s, la, lo in zip(SP["apt"], SP["stand"], SP["lat"], SP["lon"])}


def extract2(day):
    tag = day.replace("-", "")
    src = f"{BASE}/apt_{tag}.parquet"
    v1 = f"{BASE}/clk_{tag}.parquet"
    if not (os.path.exists(src) and os.path.exists(v1)):
        print(f"{day}: missing inputs")
        return
    P = pl.read_parquet(src)
    C1 = pl.read_parquet(v1)
    G = {k: v for k, v in P.group_by(["icao", "apt"])} if len(P) else {}
    rows = []
    for r in C1.filter(pl.col("icao").is_not_null()).iter_rows(named=True):
        key = (r["apt"], r["stand"])
        if key not in SPD or (r["icao"], r["apt"]) not in G:
            continue
        sla0, slo0 = SPD[key]
        g = G[(r["icao"], r["apt"])]
        t = g["t"].to_numpy()
        gr = g["ground"].to_numpy()
        la = g["lat"].to_numpy()
        lo = g["lon"].to_numpy()
        tko = r["t_tko"]
        k = int(np.searchsorted(t, tko))
        j = k
        while j > 0 and gr[j - 1] == 1 and t[j] - t[j - 1] < 1800:
            j -= 1
        if k - j < 1:
            continue
        st, dist = t[j:k], hav(la[j:k], lo[j:k], sla0, slo0)
        at = np.where(dist <= R_AT)[0]
        t_at = float(st[at[-1]]) if len(at) else np.nan
        t_leave = np.nan
        if len(at):
            after = np.where((st > t_at) & (dist > R_LEAVE))[0]
            t_leave = float(st[after[0]]) if len(after) else np.nan
        rows.append(
            dict(
                MVT_ID_mvt=r["MVT_ID_mvt"],
                t_at_stand=t_at,
                t_leave=t_leave,
                d_first=float(dist[0]),
                n_at_stand=int(len(at)),
                arr_flag=bool(st[-1] - st[0] > 10800),
            )
        )
    C2 = C1.join(pl.DataFrame(rows), on="MVT_ID_mvt", how="left") if rows else C1
    C2.write_parquet(f"{BASE}/clk2_{tag}.parquet")
    seen = C2.filter(pl.col("n_at_stand").fill_null(0) > 0) if rows else C2.head(0)
    print(
        f"{day}: linked {int(C2['icao'].is_not_null().sum()):,} | own stand seen {len(seen):,} ({100*len(seen)/max(len(C2),1):.1f}% of departures) vs stationary-start {int(C2['started_stationary'].fill_null(False).sum()):,}",
        flush=True,
    )


if __name__ == "__main__":
    for day in sys.argv[1:]:
        try:
            extract2(day)
        except Exception as e:
            print(f"{day}: EXTRACT2 FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
