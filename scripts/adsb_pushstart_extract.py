#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Pushback-START clock from positions + heading semantics (label-free).
Scout facts (27 Sep): readsb stores the HEADING (nose) in the ground 'track' field, not the motion direction; pushback = motion
opposite to heading at 1.6-2.8 kt; ~half of EGLL pushes are first seen mid-push; parked positions jump 20-48 m (multipath).
Same link and ground segment as adsb_segment_extract.py (icao + t_tko from clk_*.parquet; contiguous ground points, gaps < 30 min, <= 4 h).
Positions in a local metric frame. Motion bearing ALWAYS from successive positions (>= 5 m steps); 'track' used only as heading.
  jump-robust park run (from the segment start): anchor = running median of inside points; a point is inside if <= R_IN (15 m);
      an excursion (points > R_IN) is a JUMP and the run continues if the trace comes back inside within J_S (120 s) and the excursion
      stays < R_JUMP (60 m); the run ends at the last inside point after which it does not come back (a real departure).
  t_ps     push start = end of the start park run (last inside point); t_ps_x = back-extrapolated start: first outside point minus
           (its distance from the anchor / the speed of the next 10-60 m of motion), bounded to [t_last_inside, t_first_outside]
  rev      motion after the run is opposite to the heading: median |heading - motion bearing| over the first 10-60 m > 120 deg
  mid-push (no start park run): the first motion (first 5-60 m) is reverse and slow (< 3 m/s) -> t_mp = first fix, t_mp_x = first fix
           minus (distance first fix -> learned stand <= 120 m) / push speed (clip 0.4-2 m/s)
  t_pe     end of the reverse motion (last point of the first reverse stretch); t_tx = taxi start: first point after t_pe whose motion
           is forward (|heading - bearing| < 60) and that is >= 15 m from the push-end position; hold = t_tx - t_pe
  src      1 parked -> reverse (push), 2 parked -> forward/unknown heading, 3 first seen mid-push (reverse), 4 first seen moving forward,
           5 first seen moving, heading unknown, 0 other
  t_best   push start where one exists: src 1/2 -> t_ps_x, src 3 -> t_mp_x
Writes clkP_YYYYMMDD.parquet.  usage: [NP=4] adsb_pushstart_extract.py YYYY-MM-DD [...] | --days file"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl
import multiprocessing as mp

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
R_IN, R_JUMP, J_S, MIN_PARK_S = 15.0, 60.0, 120.0, 10.0
SP = pl.read_parquet(f"{BASE}/stand_pos.parquet").filter(pl.col("n") >= 5)
SPD = {(a, s): (la, lo) for a, s, la, lo in zip(SP["apt"], SP["stand"], SP["lat"], SP["lon"])}


def adiff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def xy(la, lo, la0, lo0):
    return (lo - lo0) * np.cos(np.radians(la0)) * 111320.0, (la - la0) * 110540.0


def brg(x0, y0, x1, y1):
    return np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360


def park_start(t, x, y):
    """jump-robust park run from index 0. returns (i_last_inside, ax, ay, n_jumps) or None"""
    n = len(t)
    ins = [0]
    ax, ay = x[0], y[0]
    i = 1
    nj = 0
    while i < n:
        d = np.hypot(x[i] - ax, y[i] - ay)
        if d <= R_IN:
            ins.append(i)
            if len(ins) <= 30:
                ax, ay = np.median(x[ins]), np.median(y[ins])
            i += 1
            continue
        # excursion: look ahead for a return inside within J_S s while staying < R_JUMP
        k = i
        back = None
        while k < n and t[k] - t[i] <= J_S:
            dk = np.hypot(x[k] - ax, y[k] - ay)
            if dk > R_JUMP:
                break
            if dk <= R_IN:
                back = k
                break
            k += 1
        if back is None:
            break
        nj += 1
        i = back
    il = ins[-1]
    if t[il] - t[0] < MIN_PARK_S or il == 0:
        return None
    return il, float(ax), float(ay), nj


def motion(t, x, y, hd, i0, px, py, dmin=10.0, dmax=60.0):
    """motion after index i0 relative to point (px,py): points at distance (dmin, dmax]; returns (rev_share, n_hd, v, idx)"""
    d = np.hypot(x - px, y - py)
    q = [j for j in range(i0 + 1, len(t)) if d[j] > dmin]
    q = [j for j in q if d[j] <= dmax and t[j] - t[i0] < 600][:15]
    if not q:
        return np.nan, 0, np.nan, q
    b = brg(px, py, x[q], y[q])
    h = hd[q]
    ok = np.isfinite(h)
    rv = float(np.mean(adiff(h[ok], b[ok]) > 120)) if ok.sum() else np.nan
    j = q[-1]
    v = (d[j] - d[i0]) / max(t[j] - t[i0], 1.0)
    return rv, int(ok.sum()), float(v), q


def one(t, gr, la, lo, hd, tko_ad, stand):
    k = int(np.searchsorted(t, tko_ad))
    if k >= len(t) or k < 2:
        return None
    j = k
    while j > 0 and gr[j - 1] == 1 and t[j] - t[j - 1] < 1800 and t[k] - t[j - 1] < 14400:
        j -= 1
    if k - j < 2:
        return None
    st, x, y, h = t[j:k], *xy(la[j:k], lo[j:k], la[j], lo[j]), hd[j:k]
    sx, sy = xy(np.array([stand[0]]), np.array([stand[1]]), la[j], lo[j]) if stand is not None else (None, None)
    out = dict(
        src=0,
        t_seg0=float(st[0]),
        t_ps=np.nan,
        t_ps_x=np.nan,
        t_mp=np.nan,
        t_mp_x=np.nan,
        t_pe=np.nan,
        t_tx=np.nan,
        t_best=np.nan,
        rev=np.nan,
        n_hd=0,
        v_push=np.nan,
        n_jump=0,
        park_len=np.nan,
        d_stand=np.nan,
        n_seg=int(k - j),
    )
    P = park_start(st, x, y)
    i_end = None
    if P is not None:
        il, ax, ay, nj = P
        out.update(t_ps=float(st[il]), n_jump=nj, park_len=float(st[il] - st[0]))
        if sx is not None:
            out["d_stand"] = float(np.hypot(ax - sx[0], ay - sy[0]))
        rv, nh, v, q = motion(st, x, y, h, il, ax, ay)
        out.update(rev=rv, n_hd=nh)
        if il + 1 < len(st):
            d1 = np.hypot(x[il + 1] - ax, y[il + 1] - ay)
            vv = np.clip(v if np.isfinite(v) and v > 0 else 1.0, 0.4, 3.0)
            out["t_ps_x"] = float(np.clip(st[il + 1] - d1 / vv, st[il], st[il + 1]))
            out["v_push"] = float(v) if np.isfinite(v) else np.nan
        out["src"] = 1 if (np.isfinite(rv) and rv > 0.5) else 2
        out["t_best"] = out["t_ps_x"] if np.isfinite(out["t_ps_x"]) else out["t_ps"]
        i_end = il
        px, py = ax, ay
    else:
        # first seen moving: motion bearing over the first 5-60 m from the first fix
        rv, nh, v, q = motion(st, x, y, h, 0, x[0], y[0], dmin=5.0)
        out.update(rev=rv, n_hd=nh, v_push=v)
        if sx is not None:
            out["d_stand"] = float(np.hypot(x[0] - sx[0], y[0] - sy[0]))
        if np.isfinite(rv) and rv > 0.5 and np.isfinite(v) and v < 3.0:
            out["src"] = 3
            out["t_mp"] = float(st[0])
            vv = float(np.clip(v, 0.4, 2.0))
            ds = out["d_stand"] if np.isfinite(out["d_stand"]) and out["d_stand"] <= 120 else 0.0
            out["t_mp_x"] = float(st[0] - ds / vv)
            out["t_best"] = out["t_mp_x"]
            i_end = 0
            px, py = x[0], y[0]
        elif np.isfinite(rv):
            out["src"] = 4
        elif q:
            out["src"] = 5
    # push end and taxi start (only after a reverse start)
    if i_end is not None and out["src"] in (1, 3):
        stp = [m for m in range(i_end + 1, len(st)) if np.hypot(x[m] - x[m - 1], y[m] - y[m - 1]) >= 3.0]
        pe = None
        for m in stp:
            b = brg(x[m - 1], y[m - 1], x[m], y[m])
            if np.isfinite(h[m]) and adiff(h[m], b) < 60:
                break
            pe = m
            if st[m] - st[i_end] > 900:
                break
        if pe is not None:
            out["t_pe"] = float(st[pe])
            for m in range(pe + 1, len(st)):
                if np.hypot(x[m] - x[pe], y[m] - y[pe]) >= 15 and np.isfinite(h[m]):
                    b = (
                        brg(x[m - 1], y[m - 1], x[m], y[m])
                        if np.hypot(x[m] - x[m - 1], y[m] - y[m - 1]) >= 3
                        else brg(x[pe], y[pe], x[m], y[m])
                    )
                    if adiff(h[m], b) < 60:
                        out["t_tx"] = float(st[m - 1])
                        break
    return out


def extract(day):
    tag = day.replace("-", "")
    out = f"{BASE}/clkP_{tag}.parquet"
    src = f"{BASE}/apt_{tag}.parquet"
    v1 = f"{BASE}/clk_{tag}.parquet"
    if not (os.path.exists(src) and os.path.exists(v1)):
        return f"{day}: missing inputs"
    C1 = pl.read_parquet(v1, columns=["MVT_ID_mvt", "apt", "stand", "icao", "t_tko"]).filter(
        pl.col("icao").is_not_null()
    )
    P = pl.scan_parquet(src)
    P = P.select(
        ["icao", "apt", "t", "lat", "lon", "ground"] + (["track"] if "track" in P.collect_schema().names() else [])
    )
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
    D = (
        pl.DataFrame(rows, infer_schema_length=None)
        if rows
        else pl.DataFrame({"MVT_ID_mvt": pl.Series([], dtype=pl.Float64)})
    )
    C = C1.select(["MVT_ID_mvt"]).join(D, on="MVT_ID_mvt", how="left")
    C.write_parquet(out)
    vc = D["src"].value_counts().sort("src") if rows else None
    return f"{day}: linked {len(C1):,} src " + (
        " ".join(f"{a}:{b}" for a, b in vc.iter_rows()) if vc is not None else "-"
    )


if __name__ == "__main__":
    a = sys.argv[1:]
    days = open(a[1]).read().split() if a[0] == "--days" else a
    with mp.get_context("spawn").Pool(int(os.environ.get("NP", "4"))) as pool:
        for m in pool.imap_unordered(extract, days):
            print(m, flush=True)
