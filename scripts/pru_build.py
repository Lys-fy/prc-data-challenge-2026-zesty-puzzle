#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Build the daily per-airport PRU table
from the csv files of a PRU download directory: env PRU_DIR (the Makefile sets data/external/pru_20261008, the 8 Oct 2026 download), unset:
data/external/pru_20260924 (the 24 Sep 2026 download); the 24 Sep, 25 Sep and 8 Oct downloads give the same table (docs/DATA_SOURCES.md row 2).
Output data/external/xpr_pru_daily.parquet: ADEP_mvt, day (UTC date = PRU FLT_DATE), allpre_min (DLY_ALL_PRE_2), n_ifr (FLT_DEP_IFR_2),
atcpre_min (DLY_ATC_PRE_2), n_dep1, n_reg, n_late, n_early, n_arr_ifr. Prints the missingness of every column on 2025 and on Jan/Jul 2026."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, polars as pl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); S = os.environ.get("PRU_DIR") or f"{ROOT}/data/external/pru_20260924"   # PRU_DIR = a fetch_pru.py output dir
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
def rd(stem, cols):
    fs = sorted(glob.glob(f"{S}/{stem}_202[56].csv")); assert len(fs) == 2, fs
    d = pl.concat([pl.read_csv(f, infer_schema_length=0).select(["FLT_DATE", "APT_ICAO"] + list(cols)) for f in fs])
    d = d.filter(pl.col("APT_ICAO").is_in(APT)).with_columns(day=pl.col("FLT_DATE").str.slice(0, 10).str.to_date("%Y-%m-%d"),
        **{v: pl.col(k).cast(pl.Float64, strict=False) for k, v in cols.items()}).rename({"APT_ICAO": "ADEP_mvt"})
    d = d.select(["ADEP_mvt", "day"] + list(cols.values())); assert d.select("ADEP_mvt", "day").is_duplicated().sum() == 0, stem; return d
A = rd("all_pre_departure_delays", {"DLY_ALL_PRE_2": "allpre_min", "FLT_DEP_IFR_2": "n_ifr", "FLT_DEP_1": "n_dep1"})
B = rd("atc_pre_departure_delays", {"DLY_ATC_PRE_2": "atcpre_min", "FLT_DEP_IFR_2": "n_ifr_atc"})
C = rd("atfm_slot_adherence", {"FLT_DEP_REG_1": "n_reg", "FLT_DEP_OUT_LATE_1": "n_late", "FLT_DEP_OUT_EARLY_1": "n_early", "FLT_DEP_1": "n_dep1_sa"})
T = rd("airport_traffic", {"FLT_ARR_IFR_2": "n_arr_ifr", "FLT_DEP_IFR_2": "n_ifr_tr"})
days = pl.DataFrame({"day": pl.date_range(pl.date(2025, 1, 1), pl.date(2026, 7, 31), "1d", eager=True)})
G = pl.DataFrame({"ADEP_mvt": APT}).join(days, how="cross")
for x in (A, B, C, T): G = G.join(x, on=["ADEP_mvt", "day"], how="left")
G = G.sort("ADEP_mvt", "day"); os.makedirs(f"{ROOT}/data/external", exist_ok=True); G.write_parquet(os.environ.get("pru_out") or f"{ROOT}/data/external/xpr_pru_daily.parquet")
print("rows", G.height, "| last day with allpre per airport:", G.filter(pl.col("allpre_min").is_not_null()).group_by("ADEP_mvt").agg(pl.col("day").max()).sort("ADEP_mvt").rows())
for nm, f in (("2025", pl.col("day").dt.year() == 2025), ("JanJul2026", (pl.col("day").dt.year() == 2026) & pl.col("day").dt.month().is_in([1, 7]))):
    s = G.filter(f); print(f"missing on {nm} ({s.height} airport-days):", {c: s[c].null_count() for c in G.columns[2:]})
    print(f"  per airport null allpre/n_ifr/atcpre/n_reg: " + " ".join(f"{a} {r['a']}/{r['b']}/{r['c']}/{r['d']}" for a, r in ((a, s.filter(pl.col('ADEP_mvt') == a).select(a=pl.col('allpre_min').null_count(), b=pl.col('n_ifr').null_count(), c=pl.col('atcpre_min').null_count(), d=pl.col('n_reg').null_count()).row(0, named=True)) for a in APT)))
print("n_ifr vs n_ifr_atc / n_ifr_tr equal share:", G.select(a=(pl.col("n_ifr") == pl.col("n_ifr_atc")).mean(), b=(pl.col("n_ifr") == pl.col("n_ifr_tr")).mean()).row(0))
