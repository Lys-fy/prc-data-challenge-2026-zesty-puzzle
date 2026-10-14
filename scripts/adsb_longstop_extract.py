#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Longest-stop clocks (clkL_*): segment-summary clocks for every departure linked by adsb_extract.py (icao + t_first_seen + t_tko from clk_*.parquet).
Ground segment = the points of that airframe in [t_first_seen, t_tko] (the same segment adsb_extract used). Stationary runs by the adsb_extract rule
(< 40 m from the run's first point, >= 60 s). Written per row (epoch s unless stated):
  t_stop_start / t_stop_end / stop_len   the LONGEST stationary run          d0      distance first fix -> that run's anchor (m)
  t_last_end                              end of the last run >= 300 s        n300    number of runs >= 300 s
  mv_pre                                  share of points before the longest run's end with gs > 3 kt
  start_cls                               0 leg-flag / 1 trace start / 2 hole >= 30 min / 3 previous landing (airborne point right before) / 4 midnight
usage: adsb_longstop_extract.py YYYY-MM-DD [...] | --days file   (env NP processes, default 4)   -> data/external/adsblol/clkL_YYYYMMDD.parquet
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl, multiprocessing as mp

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


def one(t, gr, la, lo, gs, fl, t0, t1):
    jj = np.where(t == t0)[0]
    kk = np.where(t == t1)[0]
    if not len(jj) or not len(kk):
        return None
    j, k = int(jj[0]), int(kk[0])
    if k <= j:
        return None
    st, sla, slo, sgs = t[j:k], la[j:k], lo[j:k], gs[j:k]
    i = 0
    runs = []
    while i < len(st):
        i2 = i
        while i2 + 1 < len(st) and hav(sla[i], slo[i], sla[i2 + 1], slo[i2 + 1]) < 40.0:
            i2 += 1
        if st[i2] - st[i] >= 60.0:
            runs.append((i, i2))
        i = i2 + 1
    if j == 0:
        sc = 1
    elif fl[j] & 2:
        sc = 0
    elif t[j] - t[j - 1] >= 1800:
        sc = 2
    elif gr[j - 1] == 0:
        sc = 3
    else:
        sc = 0
    if (t0 % 86400) < 30:
        sc = 4
    out = dict(start_cls=sc, n_runs=len(runs), n300=int(sum(1 for a, b in runs if st[b] - st[a] >= 300)))
    if runs:
        a, b = max(runs, key=lambda ab: st[ab[1]] - st[ab[0]])
        out.update(
            t_stop_start=float(st[a]),
            t_stop_end=float(st[b]),
            stop_len=float(st[b] - st[a]),
            d0=float(hav(sla[0], slo[0], sla[a], slo[a])),
            mv_pre=float(np.mean(sgs[: b + 1] > 3.0)) if b >= 0 else np.nan,
        )
        long_ = [(a2, b2) for a2, b2 in runs if st[b2] - st[a2] >= 300]
        out["t_last_end"] = float(st[long_[-1][1]]) if long_ else np.nan
    else:
        out.update(t_stop_start=np.nan, t_stop_end=np.nan, stop_len=np.nan, d0=np.nan, mv_pre=np.nan, t_last_end=np.nan)
    return out


def extract(day):
    tag = day.replace("-", "")
    out = f"{BASE}/clkL_{tag}.parquet"
    src = f"{BASE}/apt_{tag}.parquet"
    v1 = f"{BASE}/clk_{tag}.parquet"
    if os.path.exists(out):
        return f"{day}: exists"
    if not (os.path.exists(src) and os.path.exists(v1)):
        return f"{day}: missing inputs"
    C1 = pl.read_parquet(v1, columns=["MVT_ID_mvt", "apt", "icao", "t_first_seen", "t_tko"]).filter(
        pl.col("icao").is_not_null()
    )
    P = pl.scan_parquet(src)
    names = P.collect_schema().names()
    P = (
        P.select(["icao", "apt", "t", "lat", "lon", "ground", "gs"] + (["flags"] if "flags" in names else []))
        .collect()
        .sort(["icao", "t"])
    )
    if "flags" not in P.columns:
        P = P.with_columns(flags=pl.lit(0, dtype=pl.Int32))
    G = {k: v for k, v in P.group_by(["icao", "apt"])} if len(P) else {}
    rows = []
    for r in C1.iter_rows(named=True):
        g = G.get((r["icao"], r["apt"]))
        if g is None:
            continue
        res = one(
            g["t"].to_numpy(),
            g["ground"].to_numpy(),
            g["lat"].to_numpy(),
            g["lon"].to_numpy(),
            np.nan_to_num(g["gs"].to_numpy().astype(float)),
            g["flags"].fill_null(0).to_numpy(),
            r["t_first_seen"],
            r["t_tko"],
        )
        if res:
            res["MVT_ID_mvt"] = r["MVT_ID_mvt"]
            rows.append(res)
    D = pl.DataFrame(rows) if rows else pl.DataFrame({"MVT_ID_mvt": pl.Series([], dtype=pl.Float64)})
    C = C1.select(["MVT_ID_mvt"]).join(D, on="MVT_ID_mvt", how="left")
    C.write_parquet(out)
    return f"{day}: linked {len(C1):,} with a run {int(D['n_runs'].gt(0).sum()) if rows else 0:,}"


if __name__ == "__main__":
    a = sys.argv[1:]
    days = open(a[1]).read().split() if a[0] == "--days" else a
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "4"))) as pool:
        for m in pool.imap_unordered(extract, days):
            print(m, flush=True)
