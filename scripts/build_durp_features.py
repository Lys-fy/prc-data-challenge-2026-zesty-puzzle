#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the durp side table (prefix durp_), the pre-departure subset of the dur feature family, directly from the raw
training/ranking parquet files in data/raw/. Writes --out (default data/external/durp_features.parquet), 8 columns keyed by MVT_ID_mvt.

Exactly the 8 columns of the duration feature table that are known before off-block, renamed
dur_* -> durp_* (checked frame-equal, same row order, against that file). The post-departure columns (actual_air, arr_delay,
ratio, air_minus_planned) are left out on purpose; this script computes the 8 columns independently so it does not depend
on dur_features.parquet.
usage: build_durp_features.py [--out data/external/durp_features.parquet]"""

import argparse, glob, os, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.environ.get("PRC_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=f"{ROOT}/data/external/durp_features.parquet", help="output features parquet")
a = ap.parse_args()
cols = [
    "MVT_ID_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "EOBT_1_flt",
    "LOBT_flt",
    "IOBT_flt",
    "AOBT_3_flt",
    "ARVT_1_flt",
    "ADES_flt",
    "ADES_FILED_flt",
    "ADES_mvt",
]
files = sorted(glob.glob(f"{ROOT}/data/raw/training_*.parquet")) + [f"{ROOT}/data/raw/ranking.parquet"]
R = pl.concat(
    [pl.scan_parquet(f).select(cols).filter(pl.col("PHASE_mvt") == "DEP").collect() for f in files],
    how="diagonal_relaxed",
)
s = lambda x, y: (pl.col(x) - pl.col(y)).dt.total_seconds().cast(pl.Float32)
F = R.select(
    "MVT_ID_mvt",
    s("ARVT_1_flt", "EOBT_1_flt").alias("durp_planned_block"),
    s("LOBT_flt", "EOBT_1_flt").alias("durp_lobt_minus_eobt"),
    s("IOBT_flt", "EOBT_1_flt").alias("durp_iobt_minus_eobt"),
    s("AOBT_3_flt", "LOBT_flt").alias("durp_aobt_minus_lobt"),
    s("LOBT_flt", "SCHED_TIME_UTC_mvt").alias("durp_lobt_minus_sched"),
    (pl.col("ADES_flt") != pl.col("ADES_mvt")).cast(pl.Float32).alias("durp_ades_changed"),
    (pl.col("ADES_FILED_flt") != pl.col("ADES_flt")).cast(pl.Float32).alias("durp_filed_changed"),
    s("ARVT_1_flt", "MVT_TIME_UTC_mvt").alias("durp_plan_arr_minus_mvt"),
)
F.write_parquet(a.out)
print("wrote", a.out, F.shape)
