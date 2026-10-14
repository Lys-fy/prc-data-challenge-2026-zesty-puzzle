#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Download the OPDI v0.0.2 files read by scripts/build_opdi_features.py (docs/DATA_SOURCES.md §2 row 3).
EUROCONTROL Open Performance Data Initiative, https://www.opdi.aero/ ; reuse with attribution, non-commercial.
  flight_list_YYYYMM.parquet                     monthly, for the months of the training and ranking periods
  flight_events_YYYYMMDD_YYYYMMDD.parquet        10-day chunks on a fixed grid anchored 2024-12-26, every chunk that overlaps a period
from https://www.eurocontrol.int/performance/data/download/OPDI/v002/{flight_list,flight_events}/ into data/external/opdi/ (~15 GB).
A chunk the publisher does not have (HTTP 404; e.g. flight_events_20251002_20251012, the known 2-11 Oct 2025 OPDI gap) is recorded in
MANIFEST.tsv as missing and skipped. The production pull (7 Sep 2026) had these files plus two February 2026 event chunks
(20260209_20260219, 20260219_20260301) that no training or ranking departure falls into.
usage: fetch_opdi.py [--out data/external/opdi] [--periods 2025-01-01:2025-12-31,2026-01-01:2026-01-31,2026-07-01:2026-07-31]
                     [--check]   (--check: HEAD only, compare Content-Length with the local files, download nothing)"""

import argparse, datetime as dt, hashlib, os, subprocess, sys, time, urllib.error, urllib.request

ROOT = os.environ.get("PRC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{ROOT}/data/external/opdi")
ap.add_argument("--periods", default="2025-01-01:2025-12-31,2026-01-01:2026-01-31,2026-07-01:2026-07-31")
ap.add_argument("--check", action="store_true")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()
P = [tuple(dt.date.fromisoformat(x) for x in p.split(":")) for p in a.periods.split(",")]
months = sorted(
    {(d.year, d.month) for lo, hi in P for d in (lo + dt.timedelta(days=i) for i in range((hi - lo).days + 1))}
)
names = [f"flight_list/flight_list_{y}{m:02d}.parquet" for y, m in months]
g = dt.date(2024, 12, 26)
while g <= max(hi for _, hi in P):
    e = g + dt.timedelta(days=10)
    if any(g <= hi and e > lo for lo, hi in P):  # chunk overlaps a period
        names.append(f"flight_events/flight_events_{g:%Y%m%d}_{e:%Y%m%d}.parquet")
    g = e
if a.list:
    print("\n".join(names))
    sys.exit(0)


def req(url, method="GET"):
    return urllib.request.Request(
        url, method=method, headers={"User-Agent": "prc-data-challenge-2026 reproduction (research)"}
    )


os.makedirs(a.out, exist_ok=True)
rows = []
bad = 0
for n in names:
    url = f"{BASE}/{n}"
    dst = f"{a.out}/{n.split('/')[1]}"
    try:
        with urllib.request.urlopen(req(url, "HEAD"), timeout=60) as r:
            size = int(r.headers.get("Content-Length", -1))
    except urllib.error.HTTPError as e:
        rows.append((url, "missing", e.code, ""))
        print("MISSING at source", n, e.code)
        continue
    if a.check:
        loc = os.path.getsize(dst) if os.path.exists(dst) else -1
        ok = loc == size
        bad += not ok
        rows.append((url, size, loc, "ok" if ok else "DIFF"))
        print(("ok  " if ok else "DIFF"), n, size, loc)
        continue
    if not (os.path.exists(dst) and os.path.getsize(dst) == size):
        # curl, not urllib: urllib's read(amt) returned a silently truncated file on a dropped connection (28 Sep test: 11.9 of 33.9 MB).
        # The size is checked against the server's Content-Length; resume with -C - and retry, fail closed.
        for k in range(5):
            subprocess.run(
                [
                    "curl",
                    "-sfL",
                    "--retry",
                    "5",
                    "-C",
                    "-",
                    "-A",
                    "prc-data-challenge-2026 reproduction (research)",
                    "-o",
                    dst + ".part",
                    url,
                ]
            )
            if os.path.exists(dst + ".part") and os.path.getsize(dst + ".part") == size:
                break
            time.sleep(15)
        else:
            sys.exit(
                f"FAILED {url}: {os.path.getsize(dst + '.part') if os.path.exists(dst + '.part') else 0} of {size} bytes"
            )
        os.replace(dst + ".part", dst)
    h = hashlib.md5()
    [h.update(b) for b in iter(lambda f=open(dst, "rb"): f.read(1 << 24), b"")]
    rows.append((url, size, h.hexdigest(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print(n, size, rows[-1][2])
with open(f"{a.out}/MANIFEST{'_check' if a.check else ''}.tsv", "w") as f:
    f.write("url\tbytes\tmd5_or_local\tstatus_or_utc\n")
    [f.write("\t".join(map(str, r)) + "\n") for r in rows]
print(
    f"{len(names)} files expected, {sum(1 for r in rows if r[1] == 'missing')} missing at source"
    + (f", {bad} size mismatches" if a.check else "")
)
sys.exit(1 if bad else 0)
