#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Download the EUROCONTROL Performance Review Unit csv files used by the model (docs/DATA_SOURCES.md §2 row 2).
Licence: "This data is published by EUROCONTROL for information purposes. It may be copied in whole or in part, provided that
EUROCONTROL is mentioned as the source and it is not used for commercial purposes" (https://ansperformance.eu/about/disclaimer/).
Files: {all_pre_departure_delays, atc_pre_departure_delays, atfm_slot_adherence, airport_traffic, taxi_out_additional_time} x {2025, 2026}
from https://www.eurocontrol.int/performance/data/download/csv/<stem>_<year>.csv, into <out>/ plus <out>/MANIFEST.tsv (url, bytes, md5,
UTC download time). The production run used the 24/25 Sep 2026 downloads (data through 31 Aug 2026); the 2026 files grow monthly,
rows up to 31 Jul 2026 are what the model reads. Read by scripts/pru_build.py (env PRU_DIR) and pru_daily_calib.py.
usage: fetch_pru.py [--out data/external/pru_YYYYMMDD] [--years 2025,2026] [--force]"""

import argparse, hashlib, os, sys, time, urllib.request

ROOT = os.environ.get("PRC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://www.eurocontrol.int/performance/data/download/csv"
STEMS = [
    "all_pre_departure_delays",
    "atc_pre_departure_delays",
    "atfm_slot_adherence",
    "airport_traffic",
    "taxi_out_additional_time",
]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{ROOT}/data/external/pru_{time.strftime('%Y%m%d', time.gmtime())}")
ap.add_argument("--years", default="2025,2026")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rows = []
for stem in STEMS:
    for y in a.years.split(","):
        fn = f"{stem}_{y}.csv"
        dst = f"{a.out}/{fn}"
        url = f"{BASE}/{fn}"
        if os.path.exists(dst) and not a.force:
            print("exists", dst)
        else:
            for k in range(3):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "prc-data-challenge-2026 reproduction (research)"}
                    )
                    with urllib.request.urlopen(req, timeout=300) as r:
                        n = int(r.headers.get("Content-Length", -1))
                        body = r.read()  # read() without amt raises IncompleteRead on truncation
                    if n >= 0 and len(body) != n:
                        raise IOError(f"truncated: {len(body)} of {n} bytes")
                    open(dst + ".part", "wb").write(body)
                    os.replace(dst + ".part", dst)
                    break
                except Exception as e:
                    print(f"retry {k + 1} {url}: {e!r}", file=sys.stderr)
                    time.sleep(10)
            else:
                sys.exit(f"FAILED {url}")
        b = open(dst, "rb").read()
        rows.append((url, len(b), hashlib.md5(b).hexdigest(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        print(fn, len(b), rows[-1][2])
with open(f"{a.out}/MANIFEST.tsv", "w") as f:
    f.write("url\tbytes\tmd5\tchecked_utc\n")
    [f.write("\t".join(map(str, r)) + "\n") for r in rows]
print("Source: EUROCONTROL Performance Review Unit (ansperformance.eu). Wrote", a.out)
