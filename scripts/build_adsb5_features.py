#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Side table `adsb5` (prefix adsb5_, adsb5_clock) from adsb_segment_extract.py outputs (clk5_*.parquet), MVT_TIME-relative, label-free.
The other side tables (adsb/adsb2/adsb3/adsb4) are not modified.
  adsb5_push   = MVT - t_push   when the segment STARTS with a park run (aircraft seen parked; 12 m radius, >= 10 s) = pushback start
  adsb5_push_i = same, interpolated crossing of 12 m
  adsb5_pushT  = MVT - end of the earliest >= 30 s park run within 80 m of the learned stand when the segment does not start parked
                 (mostly the post-pushback engine-start stop; kept separate)
  adsb5_cusp   = MVT - t_cusp (end of pushback: path reversal > 120 deg), adsb5_fwd = MVT - forward-taxi start after the cusp
  adsb5_mv2    = MVT - traffic-style movement start (> 2 kt for >= 30 s, position speed)
  adsb5_src (0 none / 1 S / 2 T), adsb5_park_len, adsb5_dstand (anchor to learned stand, m), adsb5_cuspf, adsb5_trk, adsb5_first_moving
usage: build_adsb5_features.py"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
fs = sorted(glob.glob(f"{BASE}/clk5_*.parquet"))
print(f"{len(fs)} day files")
C = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed").unique("MVT_ID_mvt", keep="first")
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).with_columns(
    tko=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)
)
X = (
    C.with_columns(pl.col("MVT_ID_mvt").cast(F["MVT_ID_mvt"].dtype))
    .join(F.select(["MVT_ID_mvt", "tko"]), on="MVT_ID_mvt", how="inner")
    .with_columns(
        adsb5_push=pl.when(pl.col("push_src") == 1).then(pl.col("tko") - pl.col("t_push")),
        adsb5_push_i=pl.when(pl.col("push_src") == 1).then(pl.col("tko") - pl.col("t_push_i")),
        adsb5_pushT=pl.when(pl.col("push_src") == 2).then(pl.col("tko") - pl.col("t_push")),
        adsb5_cusp=pl.col("tko") - pl.col("t_cusp"),
        adsb5_fwd=pl.col("tko") - pl.col("t_fwd"),
        adsb5_mv2=pl.col("tko") - pl.col("t_mv2"),
        adsb5_src=pl.col("push_src"),
        adsb5_park_len=pl.col("park_len"),
        adsb5_dstand=pl.col("d_park_stand"),
        adsb5_cuspf=pl.col("push_cusp"),
        adsb5_trk=pl.col("push_trk"),
        adsb5_first_moving=pl.col("first_moving"),
    )
    .select(
        ["MVT_ID_mvt"]
        + [
            f"adsb5_{c}"
            for c in [
                "push",
                "push_i",
                "pushT",
                "cusp",
                "fwd",
                "mv2",
                "src",
                "park_len",
                "dstand",
                "cuspf",
                "trk",
                "first_moving",
            ]
        ]
    )
)
X = X.with_columns([pl.col(c).cast(pl.Float32).fill_nan(None) for c in X.columns if c != "MVT_ID_mvt"])
X.write_parquet(f"{ROOT}/data/external/adsb5_features.parquet")
print(
    f"rows {len(X):,} with S-push {int(X['adsb5_push'].is_not_null().sum()):,} cusp {int(X['adsb5_cusp'].is_not_null().sum()):,}"
)
print(X.describe())
