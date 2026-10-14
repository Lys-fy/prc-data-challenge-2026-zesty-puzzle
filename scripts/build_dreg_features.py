#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds label-free airport delay-regime features from the departures surrounding each row (MVT-SCHED, known for every row in 2025 and 2026).
Reads --features (default data/features_v9.parquet); writes --out (default data/external/dreg_features.parquet), keyed by MVT_ID_mvt.

Rolling windows of +-30 / +-60 / +-120 min around each departure's take-off time, per airport: count, median delay, share > 30 min,
share > 2 h, unmatched share (own row included; windows hold dozens of rows). Day-level (UTC date) median delay and share > 2 h.
Ratio of the 60-min median to the airport-month median (transductive within each month)."""

import os, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--features", default=f"{ROOT}/data/features_v9.parquet", help="input features parquet")
_ap.add_argument("--out", default=f"{ROOT}/data/external/dreg_features.parquet", help="output features parquet")
_a = _ap.parse_args()
D = pl.read_parquet(
    _a.features, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "mvt_sched", "unmatched", "is_rank", "m"]
)
# year is taken from the timestamp (any batch), not from is_rank
D = D.with_columns(
    pl.col("mvt_sched").cast(pl.Float64),
    (pl.col("unmatched") == 1).cast(pl.Float64).alias("unm"),
    (pl.col("mvt_sched") > 1800).cast(pl.Float64).alias("late30"),
    (pl.col("mvt_sched") > 7200).cast(pl.Float64).alias("late2h"),
    pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
    pl.col("MVT_TIME_UTC_mvt").dt.year().alias("yr"),
).sort("ADEP_mvt", "MVT_TIME_UTC_mvt")
out = D.select("MVT_ID_mvt")
for w in (30, 60, 120):
    r = D.rolling(index_column="MVT_TIME_UTC_mvt", period=f"{2*w}m", offset=f"-{w}m", group_by="ADEP_mvt").agg(
        pl.col("MVT_ID_mvt").first().alias("_id"),
        pl.len().alias(f"dreg_n{w}"),
        pl.col("mvt_sched").median().alias(f"dreg_med{w}"),
        pl.col("mvt_sched").quantile(0.75).alias(f"dreg_p75_{w}"),
        pl.col("late30").mean().alias(f"dreg_late30_{w}"),
        pl.col("late2h").mean().alias(f"dreg_late2h_{w}"),
        pl.col("unm").mean().alias(f"dreg_unm{w}"),
    )
    # rolling() returns one row per input row in order; attach by position
    assert r.height == D.height, "rolling output must have one row per input row (positional attachment)"
    r = r.with_columns(D["MVT_ID_mvt"].alias("MVT_ID_mvt")).drop("_id", "ADEP_mvt", "MVT_TIME_UTC_mvt")
    out = out.join(r, on="MVT_ID_mvt", how="left")
    print(f"window {w} done")
day = D.group_by("ADEP_mvt", "day").agg(
    pl.col("mvt_sched").median().alias("dreg_day_med"),
    pl.col("late2h").mean().alias("dreg_day_late2h"),
    pl.col("unm").mean().alias("dreg_day_unm"),
    pl.len().alias("dreg_day_n"),
)
mon = D.group_by("ADEP_mvt", "yr", "m").agg(pl.col("mvt_sched").median().alias("_mon_med"))
X = (
    D.select("MVT_ID_mvt", "ADEP_mvt", "day", "yr", "m")
    .join(day, on=["ADEP_mvt", "day"], how="left")
    .join(mon, on=["ADEP_mvt", "yr", "m"], how="left")
)
out = out.join(X.drop("ADEP_mvt", "day", "yr", "m"), on="MVT_ID_mvt", how="left")
out = out.with_columns(
    (pl.col("dreg_med60") - pl.col("_mon_med")).alias("dreg_med60_excess"),
    (pl.col("dreg_day_med") - pl.col("_mon_med")).alias("dreg_day_excess"),
).drop("_mon_med")
out = out.with_columns([pl.col(c).cast(pl.Float32) for c in out.columns if c != "MVT_ID_mvt"])
out.write_parquet(_a.out)
print(out.shape, out.columns)
chk = (
    out.join(D.select("MVT_ID_mvt", "yr", "m"), on="MVT_ID_mvt")
    .filter(pl.col("m").is_in([1, 7]))
    .group_by("yr", "m")
    .agg(
        pl.col("dreg_med60").mean().round(0),
        pl.col("dreg_late2h_60").mean().round(3),
        pl.col("dreg_unm60").mean().round(3),
        pl.col("dreg_n60").mean().round(1),
        pl.col("dreg_med60").is_null().mean().round(3).alias("null"),
    )
    .sort("m", "yr")
)
print(chk)
