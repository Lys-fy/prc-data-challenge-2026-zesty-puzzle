#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Second-generation ADS-B clock features (side table `adsb2`, prefix adsb2_) from adsb_extract2.py outputs, all
relative to MVT_TIME / AOBT_3: adsb2_at = MVT - t_at_stand, adsb2_leave = MVT - t_leave, adsb2_at_aobt, adsb2_leave_aobt,
adsb2_d_first (m from own stand at first observation), adsb2_n_at, adsb2_seen (own stand seen), adsb2_arr (segment > 3 h).
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
fs = sorted(glob.glob(f"{BASE}/clk2_*.parquet"))
print(f"{len(fs)} day files")
C = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
for c in ("t_at_stand", "t_leave", "d_first", "n_at_stand", "arr_flag"):
    if c not in C.columns:
        C = C.with_columns(pl.lit(None).alias(c))
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "mvt_aobt3"])
X = (
    C.join(F, on="MVT_ID_mvt", how="left")
    .with_columns(
        adsb2_at=(pl.col("tko") - pl.col("t_at_stand")),
        adsb2_leave=(pl.col("tko") - pl.col("t_leave")),
        adsb2_d_first=pl.col("d_first"),
        adsb2_n_at=pl.col("n_at_stand"),
        adsb2_seen=(pl.col("n_at_stand").fill_null(0) > 0).cast(pl.Int8),
        adsb2_arr=pl.col("arr_flag").cast(pl.Int8),
    )
    .with_columns(
        adsb2_at_aobt=pl.col("adsb2_at") - pl.col("mvt_aobt3"),
        adsb2_leave_aobt=pl.col("adsb2_leave") - pl.col("mvt_aobt3"),
    )
    .select(
        [
            "MVT_ID_mvt",
            "adsb2_at",
            "adsb2_leave",
            "adsb2_at_aobt",
            "adsb2_leave_aobt",
            "adsb2_d_first",
            "adsb2_n_at",
            "adsb2_seen",
            "adsb2_arr",
        ]
    )
)
X = X.unique(subset=["MVT_ID_mvt"], keep="first").with_columns(
    [pl.col(c).cast(pl.Float32).fill_nan(None) for c in X.columns if c != "MVT_ID_mvt"]
)
X.write_parquet(f"{ROOT}/data/external/adsb2_features.parquet")
print(f"rows {len(X):,} own-stand seen {int(X['adsb2_seen'].sum()):,}")
print(X.select(["adsb2_at", "adsb2_leave", "adsb2_d_first"]).describe())
