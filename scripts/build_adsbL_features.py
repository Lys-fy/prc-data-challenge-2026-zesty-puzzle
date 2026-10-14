#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Side table `adsbL` (prefix adsbL_, longest-stop clocks) from adsb_longstop_extract.py outputs (clkL_*.parquet), MVT_TIME-relative, label-free.
  adsbL_stop_end / adsbL_stop_start = MVT - end / start of the LONGEST stationary run of the segment; adsbL_stop_len = its length
  adsbL_pre  = t_stop_end - t_first_seen (time between the first fix and the end of the longest stop; ~0 when the segment starts with it)
  adsbL_d0   = distance first fix -> longest-run anchor (m);  adsbL_last_end = MVT - end of the last run >= 300 s;  adsbL_n300, adsbL_n_runs
  adsbL_mv_pre = share of points before the longest stop's end with gs > 3 kt;  adsbL_start_cls (0 leg / 1 trace / 2 hole / 3 landing / 4 midnight)
The other side tables (adsb ... adsb6) are not modified. usage: build_adsbL_features.py"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
fs = sorted(glob.glob(f"{BASE}/clkL_*.parquet"))
print(f"{len(fs)} day files")
C = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed").unique("MVT_ID_mvt", keep="first")
K = pl.concat(
    [pl.read_parquet(f, columns=["MVT_ID_mvt", "t_first_seen"]) for f in sorted(glob.glob(f"{BASE}/clk_*.parquet"))],
    how="diagonal_relaxed",
).unique("MVT_ID_mvt", keep="first")
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).with_columns(
    tko=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)
)
X = (
    C.with_columns(pl.col("MVT_ID_mvt").cast(F["MVT_ID_mvt"].dtype))
    .join(F.select(["MVT_ID_mvt", "tko"]), on="MVT_ID_mvt", how="inner")
    .join(K.with_columns(pl.col("MVT_ID_mvt").cast(F["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")
    .with_columns(
        adsbL_stop_end=pl.col("tko") - pl.col("t_stop_end"),
        adsbL_stop_start=pl.col("tko") - pl.col("t_stop_start"),
        adsbL_stop_len=pl.col("stop_len"),
        adsbL_pre=pl.col("t_stop_end") - pl.col("t_first_seen"),
        adsbL_d0=pl.col("d0"),
        adsbL_last_end=pl.col("tko") - pl.col("t_last_end"),
        adsbL_n300=pl.col("n300"),
        adsbL_n_runs=pl.col("n_runs"),
        adsbL_mv_pre=pl.col("mv_pre"),
        adsbL_start_cls=pl.col("start_cls"),
    )
    .select(
        ["MVT_ID_mvt"]
        + [
            f"adsbL_{c}"
            for c in [
                "stop_end",
                "stop_start",
                "stop_len",
                "pre",
                "d0",
                "last_end",
                "n300",
                "n_runs",
                "mv_pre",
                "start_cls",
            ]
        ]
    )
)
X = X.with_columns([pl.col(c).cast(pl.Float32).fill_nan(None) for c in X.columns if c != "MVT_ID_mvt"])
X.write_parquet(f"{ROOT}/data/external/adsbL_features.parquet")
print(
    f"rows {len(X):,} with a longest stop {int(X['adsbL_stop_end'].is_not_null().sum()):,} n300>0 {int((X['adsbL_n300'] > 0).sum()):,}"
)
print(X.describe())
