#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""adsb.lol globe_history day puller (ODbL 1.0, attribution: "Contains information from adsb.lol, made available
under the Open Database License"). Per day: download the release parts, extract traces, keep every point inside the
ten airport boxes (± 0.07 lat / ± 0.10 lon) -> data/external/adsblol/apt_YYYYMMDD.parquet (~10 MB), delete the rest.
Resumable: a day with an existing parquet is skipped. usage: adsb_pull.py YYYY-MM-DD [YYYY-MM-DD ...]  |  --queue file
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, json, gzip, time, shutil, subprocess, numpy as np, polars as pl
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
APT = {
    "EDDF": (50.0333, 8.5706),
    "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785),
    "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479),
    "LIRF": (41.8003, 12.2389),
    "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
DLAT, DLON = 0.07, 0.10


def scan(path):
    try:
        raw = open(path, "rb").read()
        d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    except Exception:
        return []
    t0 = d["timestamp"]
    icao = d["icao"]
    typ = d.get("t")
    reg = d.get("r")
    out = []
    cs = None
    for p in d["trace"]:
        lat, lon = p[1], p[2]
        if lat is None:
            continue
        ex = p[8] if len(p) > 8 else None
        if isinstance(ex, dict) and ex.get("flight"):
            cs = ex["flight"].strip()
        for a, (la, lo) in APT.items():
            if abs(lat - la) < DLAT and abs(lon - lo) < DLON:
                out.append(
                    (
                        icao,
                        typ,
                        reg,
                        a,
                        t0 + p[0],
                        lat,
                        lon,
                        1 if p[3] == "ground" else 0,
                        (float(p[3]) if isinstance(p[3], (int, float)) else np.nan),
                        float(p[4]) if p[4] is not None else np.nan,
                        float(p[5]) if p[5] is not None else np.nan,
                        int(p[6]) if p[6] is not None else 0,
                        cs,
                    )
                )
                break
    return out


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def pull(day):
    tag = day.replace("-", "")
    out = f"{BASE}/apt_{tag}.parquet"
    if os.path.exists(out):
        print(f"{day}: exists, skip", flush=True)
        return
    yr = day[:4]
    t0 = time.time()
    parts = []
    for inst in ("prod-0", "staging-0"):
        rel = f"v{day.replace('-', '.')}-planes-readsb-{inst}"
        parts = []
        if inst != "prod-0":
            print(f"{day}: prod release missing, trying {inst}", flush=True)
        for p in ("aa", "ab", "ac", "ad"):
            f = f"{BASE}/{rel}.tar.{p}"
            if not os.path.exists(f) or os.path.getsize(f) < 1000:
                ok = False
                for attempt in range(4):
                    r = sh(
                        f"curl -sfL --retry 3 -o {f} https://github.com/adsblol/globe_history_{yr}/releases/download/{rel}/{rel}.tar.{p}"
                    )
                    if r.returncode == 0 and os.path.getsize(f) > 1000:
                        ok = True
                        break
                    if r.returncode == 22:
                        break  # 404: no such part
                    time.sleep(30)
                if not ok:
                    if os.path.exists(f):
                        os.remove(f)
                    break
            parts.append(f)
        if parts:
            break
    if not parts:
        print(f"{day}: NO RELEASE FOUND", flush=True)
        return
    d = f"{BASE}/d{tag}"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    r = sh(f"cat {' '.join(parts)} | tar -xf - -C {d} ./traces 2>&1")
    files = glob.glob(f"{d}/traces/*/*.json")
    if len(files) < 1000:
        print(f"{day}: extraction failed ({len(files)} files) {r.stdout[:200]}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
        return
    with Pool(max(2, os.cpu_count() - 2)) as pool:
        rows = [x for rs in pool.imap_unordered(scan, files, chunksize=64) for x in rs]
    SCHEMA = {
        "icao": pl.Utf8,
        "type": pl.Utf8,
        "reg": pl.Utf8,
        "apt": pl.Utf8,
        "t": pl.Float64,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "ground": pl.Int8,
        "alt": pl.Float64,
        "gs": pl.Float64,
        "track": pl.Float64,
        "flags": pl.Int32,
        "cs": pl.Utf8,
    }
    P = pl.DataFrame(rows, schema=SCHEMA, orient="row", strict=False).sort(
        ["icao", "t"]
    )  # explicit dtypes: inference from leading None rows crashed on the first str (19 Sep, 8 h lost)
    P.write_parquet(out)
    shutil.rmtree(d, ignore_errors=True)
    for f in parts:
        os.remove(f)
    print(
        f"{day}: {len(files):,} traces -> {len(P):,} airport points / {P['icao'].n_unique():,} airframes in {time.time()-t0:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    days = open(sys.argv[2]).read().split() if sys.argv[1] == "--queue" else sys.argv[1:]
    for day in days:
        try:
            pull(day)
        except Exception as e:
            print(f"{day}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            import shutil as _sh

            _sh.rmtree(f"{BASE}/d{day.replace('-', '')}", ignore_errors=True)
