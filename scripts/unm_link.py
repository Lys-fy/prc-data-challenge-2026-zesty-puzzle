# SPDX-License-Identifier: GPL-3.0-only
"""Links unmatched departures to adsb.lol traces by candidate callsign (a label-free IATA->ICAO map plus the raw
flight number), verified by a ground-to-air transition within 60 s of MVT_TIME; ambiguous or no-match rows are
left unlinked. Records pre-takeoff ground clocks (first_seen, first_move, hold_end, at_stand, land_prev, ...) for
linked rows, and the callsign-free adsbU hex for every row.
Inputs: optional NPROC (positional, worker count, default 4); scratchpad/xur2/{csmap,frame}.parquet,
data/external/openflights/airlines.dat, data/external/adsblol/{apt_*.parquet, stand_pos.parquet, clkU_*.parquet}.
Output: scratchpad/xur2/link.parquet.

Callsign link of unmatched departures: links UNMATCHED departures to adsb.lol traces through an IATA->ICAO callsign map, with an identity check.
Candidate callsigns for a row's FLIGHT_mvt at airport A:
  (1) matched rows' (A, FLIGHT_mvt) -> CALLSIGN_flt pairs (scratchpad/xur2/csmap.parquet; label-free, both years), any airport as fallback;
  (2) OpenFlights airlines.dat: IATA designator -> ICAO designator(s) + the numeric part (with/without leading zeros, trailing letter kept);
  (3) FLIGHT_mvt itself (ICAO-style numbers; production already links most of these).
Identity check: a trace (hex) broadcasting one of the candidate callsigns at A must have a ground->air transition within +-60 s of MVT_TIME;
exactly one such (hex, transition) -> link, else none (ambiguous counted). Clocks from the ground segment before take-off (as adsb_unm_clocks):
  land_prev (the same hex's last landing at the airport within 12 h: arrival side), first_seen (segment start, gaps < 1800 s), first_move (end of the first >= 60 s stationary run if it starts the segment), hold_end (end of the last run),
  at_stand (last point within 60 m of the stand centroid), n_ground, span, d_first_stand. All as MVT - t (s).
Also records the adsbU (callsign-free) hex for the same row, to measure agreement.
usage: unm_link.py [NPROC]  -> scratchpad/xur2/link.parquet"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, re, csv, numpy as np, polars as pl
import multiprocessing as mp

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + "")
os.chdir(ROOT)
BASE = "data/external/adsblol"
TOL = 60.0


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


# ---- maps
MAP = pl.read_parquet("scratchpad/xur2/csmap.parquet")
m_apt = {}
m_any = {}
for a, f, c, n in MAP.select("apt", "fl", "cs", "n").iter_rows():
    m_apt.setdefault((a, f), []).append(c)
    m_any.setdefault(f, []).append(c)
OF = {}
with open("data/external/openflights/airlines.dat", encoding="utf-8", errors="ignore") as fh:
    for r in csv.reader(fh):
        if len(r) < 8:
            continue
        iata, icao, active = r[3].strip(), r[4].strip(), r[7].strip()
        if re.fullmatch(r"[A-Z0-9]{2}", iata or "") and re.fullmatch(r"[A-Z]{3}", icao or ""):
            OF.setdefault(iata, []).append((0 if active == "Y" else 1, icao))
OF = {k: [c for _, c in sorted(v)] for k, v in OF.items()}


def cands(apt, fl):
    out = {}
    for c in m_apt.get((apt, fl), []):
        out.setdefault(c, "map_apt")
    for c in m_any.get(fl, []):
        out.setdefault(c, "map_any")
    m = re.fullmatch(r"([A-Z0-9]{2})([0-9]{1,4})([A-Z]?)", fl)
    if m and m.group(1) in OF:
        num = m.group(2).lstrip("0") or "0"
        for ic in OF[m.group(1)]:
            for c in (ic + num + m.group(3), ic + m.group(2) + m.group(3)):
                out.setdefault(c, "openflights")
    out.setdefault(fl, "self")
    return out


SP = pl.read_parquet(f"{BASE}/stand_pos.parquet").filter(pl.col("n") >= 5)
SPD = {(a, s): (la, lo) for a, s, la, lo in zip(SP["apt"], SP["stand"], SP["lat"], SP["lon"])}
F = pl.read_parquet(
    "scratchpad/xur2/frame.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "fl", "STAND_mvt", "MVT_TIME_UTC_mvt", "cls"]
).with_columns(
    tko=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64),
    day=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y%m%d"),
    stand=pl.col("STAND_mvt").cast(pl.Utf8),
)
Uh = {}
for f in glob.glob(f"{BASE}/clkU_*.parquet"):
    for i, h in pl.read_parquet(f, columns=["MVT_ID_mvt", "icao"]).iter_rows():
        Uh[float(i)] = h


def clocks(g, k, stand_xy):
    t = g["t"].to_numpy()
    gr = g["ground"].to_numpy()
    la = g["lat"].to_numpy()
    lo = g["lon"].to_numpy()
    j = k
    while j > 0 and gr[j - 1] == 1 and t[j] - t[j - 1] < 1800:
        j -= 1
    o = dict(n_ground=int(k - j))
    ld = (
        np.where((gr[1 : j + 1] == 1) & (gr[:j] == 0))[0] + 1
    )  # arrival side: the same hex's last landing (air->ground) before this ground segment ends
    ld = ld[(t[ld] < t[k]) & (t[ld] > t[k] - 43200)]
    if len(ld):
        o["t_land_prev"] = float(t[ld[-1]])
    if k - j < 2:
        return o
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
    o.update(
        t_first_seen=float(st[0]),
        t_first_move=float(st[runs[0][1]]) if runs and runs[0][0] == 0 else np.nan,
        t_hold_end=float(st[runs[-1][1]]) if runs else np.nan,
        n_runs=len(runs),
        span=float(st[-1] - st[0]),
    )
    if stand_xy:
        d = hav(sla, slo, stand_xy[0], stand_xy[1])
        at = np.where(d <= 60.0)[0]
        o["d_first_stand"] = float(d[0])
        if len(at):
            o["t_at_stand"] = float(st[at[-1]])
    return o


def run(day):
    src = f"{BASE}/apt_{day}.parquet"
    D = F.filter(pl.col("day") == day)
    if not os.path.exists(src) or D.height == 0:
        return [], {}
    have = set(pl.read_parquet_schema(src))
    P = pl.read_parquet(
        src, columns=[c for c in ["icao", "apt", "t", "lat", "lon", "ground", "cs", "type", "reg"] if c in have]
    )
    for c in ("type", "reg"):
        if c not in P.columns:
            P = P.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
    P = P.with_columns(cs=pl.col("cs").str.strip_chars())
    rows = []
    st = {}
    for r in D.iter_rows(named=True):
        a = r["ADEP_mvt"]
        cd = cands(a, r["fl"] or "")
        Pa = P.filter((pl.col("apt") == a) & pl.col("cs").is_in(list(cd)))
        hits = []
        for ic, g in Pa.group_by("icao"):
            ic = ic[0] if isinstance(ic, tuple) else ic
            g2 = P.filter((pl.col("apt") == a) & (pl.col("icao") == ic)).sort(
                "t"
            )  # the hex's whole trace at the airport (callsign may appear late)
            t = g2["t"].to_numpy()
            gr = g2["ground"].to_numpy()
            for k in np.where((gr[1:] == 0) & (gr[:-1] == 1))[0] + 1:
                if abs(t[k] - r["tko"]) <= TOL:
                    css = [c for c in g.get_column("cs").unique().to_list() if c in cd]
                    hits.append(
                        (ic, int(k), g2, cd[css[0]] if css else "?", css[0] if css else None, float(t[k] - r["tko"]))
                    )
        s = st.setdefault(a, [0, 0, 0])
        s[0] += 1
        u = dict(MVT_ID_mvt=r["MVT_ID_mvt"], apt=a, day=day, n_hits=len(hits), adsbU_hex=Uh.get(float(r["MVT_ID_mvt"])))
        if len(hits) == 1:
            ic, k, g2, srcm, cs, dt = hits[0]
            s[1] += 1
            ty = g2["type"].drop_nulls()
            rg = g2["reg"].drop_nulls()
            u.update(
                hex=ic, cs=cs, map_src=srcm, dt_tko=dt, type=ty[0] if len(ty) else None, reg=rg[0] if len(rg) else None
            )
            u.update(clocks(g2, k, SPD.get((a, r["stand"])) if r["stand"] else None))
        elif len(hits) > 1:
            s[2] += 1
        rows.append(u)
    return rows, st


if __name__ == "__main__":
    days = sorted(F["day"].unique().to_list())
    npr = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = []
    tot = {}
    with mp.get_context("spawn").Pool(npr) as pool:  # spawn: polars deadlocks under fork
        for rows, st in pool.imap_unordered(run, days):
            out += rows
            for a, v in st.items():
                tot[a] = [x + y for x, y in zip(tot.get(a, [0, 0, 0]), v)]
    print("airport: rows / linked / ambiguous", {a: v for a, v in sorted(tot.items())})
    L = pl.DataFrame(out, infer_schema_length=None)
    for c in ["t_first_seen", "t_first_move", "t_hold_end", "t_at_stand", "t_land_prev"]:
        if c not in L.columns:
            L = L.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    L.write_parquet("scratchpad/xur2/link.parquet")
    print("wrote", L.height)
