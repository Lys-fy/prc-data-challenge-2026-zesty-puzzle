#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Download METAR observations (routine + specials) for the 10 challenge airports from the
Iowa Environmental Mesonet ASOS archive and cache the raw CSVs under data/external/metar_raw/.

One request per station-year chunk (bounded by START/END), 1 request/second, cached: a chunk
whose CSV already exists (and has > 1 line) is skipped.  Re-run to fill gaps.

Usage:
  uv run scripts/download_metar.py            # download all missing chunks
  uv run scripts/download_metar.py --force    # re-download everything
"""

import argparse, sys, time
from datetime import date
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "metar_raw"
STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
START, END = date(2024, 12, 1), date(2026, 8, 1)  # END exclusive-ish (IEM day2 is inclusive)
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def chunks():
    """(station, d1, d2) per station-year; IEM treats day2 as inclusive so END is fine."""
    for st in STATIONS:
        d1 = START
        while d1 < END:
            d2 = min(date(d1.year + 1, 1, 1), END)
            yield st, d1, d2
            d1 = d2


def fetch(st, d1, d2):
    params = [
        ("station", st),
        ("data", "all"),
        ("year1", d1.year),
        ("month1", d1.month),
        ("day1", d1.day),
        ("year2", d2.year),
        ("month2", d2.month),
        ("day2", d2.day),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("report_type", "3"),
        ("report_type", "4"),
        ("missing", "empty"),
        ("trace", "empty"),
    ]
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=600)
            r.raise_for_status()
            txt = r.text
            if not txt.startswith("station,valid"):
                raise RuntimeError("unexpected response head: %r" % txt[:200])
            return txt
        except Exception as e:  # noqa
            print(f"  attempt {attempt+1} failed for {st} {d1}..{d2}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    failed = []
    for st, d1, d2 in chunks():
        out = RAW / f"{st}_{d1:%Y%m%d}_{d2:%Y%m%d}.csv"
        if out.exists() and out.stat().st_size > 200 and not a.force:
            print(f"cached  {out.name}  ({out.stat().st_size/1e6:.1f} MB)")
            continue
        t0 = time.time()
        txt = fetch(st, d1, d2)
        if txt is None:
            failed.append(out.name)
            continue
        out.write_text(txt)
        n = txt.count("\n") - 1
        print(f"fetched {out.name}  rows={n}  {time.time()-t0:.0f}s", flush=True)
        time.sleep(1.0)
    if failed:
        print("FAILED:", failed, file=sys.stderr)
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()
