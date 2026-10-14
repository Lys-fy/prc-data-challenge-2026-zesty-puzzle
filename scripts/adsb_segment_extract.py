#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Alternative off-block clock from positions (label-free). For every departure linked by adsb_extract.py
(icao + t_tko from clk_YYYYMMDD.parquet), ground segment = contiguous ground points before t_tko (gaps < 30 min, no leg-flag
cut, max 4 h). All speeds from POSITIONS (readsb gs is not used).
  park run        stationary run: points within R_P (12 m) of the run's median anchor, >= MIN_PARK_S (10 s) and >= 2 points;
                  the run is ended by TWO consecutive points beyond R_P (jitter-robust)
  S-run           the park run that starts the segment (aircraft observed parked when first seen)
  T-run           the EARLIEST park run (>= 30 s) whose anchor is within 80 m of the learned own stand (stand_pos, n >= 5)
  t_push          end of the park run (last point inside R_P) — S-run if present, else T-run; push_src 1 = S, 2 = T, 0 = none
  t_push_i        interpolated crossing of R_P between the last inside and the first outside point
  t_mv2           traffic-library style movement start: first time after the park run (or segment start) at which the
                  position speed over >= 10 s windows exceeds 2 kt continuously for >= 30 s
  pushback        cusp test on the path after t_push (10 m resampled steps, first 300 m): bearing change > 120 deg -> push_cusp = 1,
                  t_cusp = time of the cusp, t_fwd = first time after the cusp 20 m away from the cusp point.
                  push_trk: median |reported track - movement bearing| over the first 30 m of motion > 120 deg (nose-reported
                  heading while reversing) -> 1
  park_len, park_n, d_park_stand (m, anchor to learned stand), first_moving (segment's first 2 points already > 1 m/s)
Writes clk5_YYYYMMDD.parquet.  usage: [NP=4] adsb_segment_extract.py [--apts EDDM,LEMD] YYYY-MM-DD [...] | --days file
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl
import multiprocessing as mp

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
R_P, MIN_PARK_S, R_STAND = 12.0, 10.0, 80.0
SP = pl.read_parquet(f"{BASE}/stand_pos.parquet").filter(pl.col("n") >= 5)
SPD = {(a, s): (la, lo) for a, s, la, lo in zip(SP["apt"], SP["stand"], SP["lat"], SP["lon"])}
APTS = (
    os.environ.get("adsb5_clock_apts", "").split(",") if os.environ.get("adsb5_clock_apts") else None
)  # spawn-safe (polars deadlocks under fork)


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return (
        np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))
        % 360
    )


def adiff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def park_runs(st, la, lo):
    """stationary runs (i0, i1, anchor_lat, anchor_lon); run ends at two consecutive points beyond R_P from the anchor"""
    runs = []
    i = 0
    n = len(st)
    while i < n - 1:
        i1 = i
        ala, alo = la[i], lo[i]
        while i1 + 1 < n:
            d1 = hav(ala, alo, la[i1 + 1], lo[i1 + 1])
            if d1 <= R_P:
                i1 += 1
            elif i1 + 2 < n and hav(ala, alo, la[i1 + 2], lo[i1 + 2]) <= R_P:
                i1 += 2  # single jitter point
            else:
                break
            if i1 - i < 20:
                ala, alo = np.median(la[i : i1 + 1]), np.median(lo[i : i1 + 1])
        if st[i1] - st[i] >= MIN_PARK_S and i1 > i:
            runs.append((i, i1, float(ala), float(alo)))
        i = i1 + 1
    return runs


def one(t, gr, la, lo, trk, tko_ad, stand):
    k = int(np.searchsorted(t, tko_ad))
    if k >= len(t) or k < 2:
        return None
    j = k
    while j > 0 and gr[j - 1] == 1 and t[j] - t[j - 1] < 1800 and t[k] - t[j - 1] < 14400:
        j -= 1
    if k - j < 2:
        return None
    st, sla, slo, strk = t[j:k], la[j:k], lo[j:k], trk[j:k]
    runs = park_runs(st, sla, slo)
    S = runs[0] if runs and runs[0][0] == 0 else None
    T = None
    dstand = np.nan
    if stand is not None:
        for (
            r
        ) in (
            runs
        ):  # EARLIEST such run (the last one is usually the post-pushback engine-start stop, smoke test: median -240 s)
            if st[r[1]] - st[r[0]] >= 30 and hav(r[2], r[3], *stand) <= R_STAND:
                T = r
                break
    R = S if S is not None else T
    out = dict(
        push_src=0 if R is None else (1 if S is not None else 2),
        t_push=np.nan,
        t_push_i=np.nan,
        t_mv2=np.nan,
        push_cusp=0,
        push_trk=-1,
        t_cusp=np.nan,
        t_fwd=np.nan,
        park_len=np.nan,
        park_n=0,
        d_park_stand=np.nan,
        n_seg5=int(k - j),
        t_seg0=float(st[0]),
        first_moving=int(hav(sla[0], slo[0], sla[1], slo[1]) / max(st[1] - st[0], 1.0) > 1.0),
    )
    i0 = 0
    if R is not None:
        a, b, ala, alo = R
        i0 = b
        out.update(t_push=float(st[b]), park_len=float(st[b] - st[a]), park_n=int(b - a + 1))
        if stand is not None:
            out["d_park_stand"] = float(hav(ala, alo, *stand))
        if b + 1 < len(st):
            d0 = hav(ala, alo, sla[b], slo[b])
            d1 = hav(ala, alo, sla[b + 1], slo[b + 1])
            f = np.clip((R_P - d0) / max(d1 - d0, 1e-6), 0, 1)
            out["t_push_i"] = float(st[b] + f * (st[b + 1] - st[b]))
        # pushback signature on the path after the park run
        idx = [b]
        cum = 0.0
        for q in range(b + 1, len(st)):
            dd = hav(sla[idx[-1]], slo[idx[-1]], sla[q], slo[q])
            if dd >= 10.0:
                idx.append(q)
                cum += dd
            if cum > 300 or st[q] - st[b] > 1200:
                break
        if len(idx) >= 3:
            brg = bearing(sla[idx[:-1]], slo[idx[:-1]], sla[idx[1:]], slo[idx[1:]])
            ch = adiff(brg[1:], brg[:-1])
            c = np.where(ch > 120)[0]
            if len(c):
                qc = idx[c[0] + 1]
                out.update(push_cusp=1, t_cusp=float(st[qc]))
                far = np.where((st > st[qc]) & (hav(sla[qc], slo[qc], sla, slo) > 20))[0]
                if len(far):
                    out["t_fwd"] = float(st[far[0]])
            # reported track vs movement bearing over the first 30 m
            q30 = [q for q in range(b + 1, len(st)) if hav(ala, alo, sla[q], slo[q]) <= 40][:20]
            if len(q30) >= 2:
                mb = bearing(ala, alo, sla[q30], slo[q30])
                tk = strk[q30]
                okk = np.isfinite(tk) & (hav(ala, alo, sla[q30], slo[q30]) > 5)
                if okk.sum() >= 1:
                    out["push_trk"] = int(np.median(adiff(tk[okk], mb[okk])) > 120)
    # traffic-style: speed over >= 10 s windows > 2 kt continuously for >= 30 s, from the park run end (or segment start)
    n = len(st)
    v = np.full(n, np.nan)
    jj = 0
    for i in range(i0, n):
        jj = max(jj, i + 1)
        while jj < n and st[jj] - st[i] < 10:
            jj += 1
        if jj >= n:
            break
        v[i] = hav(sla[i], slo[i], sla[jj], slo[jj]) / (st[jj] - st[i])
    fast = v > 1.029
    for i in range(i0, n):
        if not fast[i]:
            continue
        e = i
        while e + 1 < n and fast[e + 1]:
            e += 1
        if st[e] - st[i] >= 30 or (e + 1 < n and np.isnan(v[e + 1]) and st[-1] - st[i] >= 30):
            out["t_mv2"] = float(st[i])
            break
    return out


def extract(day):
    tag = day.replace("-", "")
    out = f"{BASE}/clk5_{tag}.parquet"
    src = f"{BASE}/apt_{tag}.parquet"
    v1 = f"{BASE}/clk_{tag}.parquet"
    if not (os.path.exists(src) and os.path.exists(v1)):
        return f"{day}: missing inputs"
    C1 = pl.read_parquet(v1, columns=["MVT_ID_mvt", "apt", "stand", "icao", "t_tko"]).filter(
        pl.col("icao").is_not_null()
    )
    if APTS:
        C1 = C1.filter(pl.col("apt").is_in(APTS))
    P = pl.scan_parquet(src)
    P = P.select(
        ["icao", "apt", "t", "lat", "lon", "ground"] + (["track"] if "track" in P.collect_schema().names() else [])
    )  # early days: no track column
    if APTS:
        P = P.filter(pl.col("apt").is_in(APTS))
    P = P.collect().sort(["icao", "t"])
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
            (g["track"].to_numpy().astype(float) if "track" in g.columns else np.full(g.height, np.nan)),
            r["t_tko"] - 0.5,
            SPD.get((r["apt"], r["stand"])),
        )
        if res:
            res["MVT_ID_mvt"] = r["MVT_ID_mvt"]
            rows.append(res)
    D = pl.DataFrame(rows) if rows else pl.DataFrame({"MVT_ID_mvt": pl.Series([], dtype=pl.Float64)})
    C = C1.select(["MVT_ID_mvt"]).join(D, on="MVT_ID_mvt", how="left")
    C.write_parquet(out)
    return f"{day}: linked {len(C1):,} with park run {int((D['push_src'] > 0).sum()) if rows else 0:,}"


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--apts":
        os.environ["adsb5_clock_apts"] = a[1]
        APTS = a[1].split(",")
        a = a[2:]
    days = open(a[1]).read().split() if a[0] == "--days" else a
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "4"))) as pool:
        for m in pool.imap_unordered(extract, days):
            print(m, flush=True)
