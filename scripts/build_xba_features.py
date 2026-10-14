#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the xba side table: AOBT_3-anchored stand arrival linkage, label-free with respect to departures (arrivals
are labelled in both 2025 and the 2026 ranking file, so every column is observable at inference). Reads
data/features_v9.parquet and the raw training/ranking parquet files in data/raw/; writes
data/external/xba_features.parquet keyed by MVT_ID_mvt.

For each departure with AOBT_3 on stand S at airport A:
  xba_since     AOBT_3 - in-block of the last arrival on (A, S) at or before AOBT_3
  xba_next      in-block of the first arrival on (A, S) after AOBT_3 - AOBT_3   (a later arrival bounds the true off-block)
  xba_prev_tin  that previous arrival's taxi-in minus the (A, S) median taxi-in
  xba_prev_ibs  that previous arrival's in-block - its SCHED
  xba_next_tin  the next arrival's taxi-in anomaly
The existing `turn` group anchors on MVT (take-off), so when another aircraft blocks in between AOBT_3 and MVT it
sees that aircraft, not the one before AOBT_3."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

R = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + "")
raw = pl.concat(
    [
        pl.read_parquet(
            f,
            columns=[
                "MVT_ID_mvt",
                "PHASE_mvt",
                "ADES_mvt",
                "STAND_mvt",
                "MVT_TIME_UTC_mvt",
                "BLOCK_TIME_UTC_mvt",
                "SCHED_TIME_UTC_mvt",
            ],
        )
        for f in sorted(glob.glob(f"{R}/data/raw/training_*.parquet")) + [f"{R}/data/raw/ranking.parquet"]
    ],
    how="vertical_relaxed",
)
arr = (
    raw.filter(
        (pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null() & pl.col("STAND_mvt").is_not_null()
    )
    .with_columns(
        pl.col("ADES_mvt").alias("ADEP_mvt"),
        pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").alias("t_ib"),
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("MVT_TIME_UTC_mvt")).dt.total_seconds().alias("tin"),
        (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("ibs"),
    )
    .filter(pl.col("tin").is_between(0, 7200))
)
arr = (
    arr.with_columns((pl.col("tin") - pl.col("tin").median().over(["ADEP_mvt", "STAND_mvt"])).alias("tin_anom"))
    .select("ADEP_mvt", "STAND_mvt", "t_ib", "tin_anom", "ibs")
    .sort("t_ib")
)
D = pl.read_parquet(
    f"{R}/data/features_v9.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "AOBT_3_flt"]
).with_columns(pl.col("AOBT_3_flt").dt.epoch("s").alias("t_a3"))
Dk = D.filter(pl.col("t_a3").is_not_null() & pl.col("STAND_mvt").is_not_null()).sort("t_a3")
p = Dk.join_asof(arr, left_on="t_a3", right_on="t_ib", by=["ADEP_mvt", "STAND_mvt"], strategy="backward").select(
    "MVT_ID_mvt",
    (pl.col("t_a3") - pl.col("t_ib")).alias("xba_since"),
    pl.col("tin_anom").alias("xba_prev_tin"),
    pl.col("ibs").alias("xba_prev_ibs"),
)
n = Dk.join_asof(
    arr, left_on="t_a3", right_on="t_ib", by=["ADEP_mvt", "STAND_mvt"], strategy="forward", allow_exact_matches=False
).select("MVT_ID_mvt", (pl.col("t_ib") - pl.col("t_a3")).alias("xba_next"), pl.col("tin_anom").alias("xba_next_tin"))
T = D.select("MVT_ID_mvt").join(p, on="MVT_ID_mvt", how="left").join(n, on="MVT_ID_mvt", how="left")
T = T.with_columns([pl.col(c).cast(pl.Float32) for c in T.columns if c.startswith("xba_")])
T.write_parquet(f"{R}/data/external/xba_features.parquet")
print(T.shape)
print(T.describe())
