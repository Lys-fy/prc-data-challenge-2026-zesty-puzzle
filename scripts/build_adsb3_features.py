#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Side table `adsb3` (prefix adsb3_) from adsb_extract3.py outputs, all MVT_TIME-relative or geometric, label-free:
  adsb3_mv_gs = MVT - t_mv_gs, adsb3_mv_pos = MVT - t_mv_pos, adsb3_mv_gs_aobt = adsb3_mv_gs - mvt_aobt3,
  adsb3_gs0, adsb3_v0, adsb3_lat0, adsb3_lon0, adsb3_d_tko0, adsb3_path, adsb3_span, adsb3_mean_v, adsb3_ppm, adsb3_d_first_stand.
Register in train2.py: add "adsb3": None to the group dict (line ~19), "adsb3": ("adsb3_",) to the prefix map (line ~26)
and ("adsb3", ROOT / "data/external/adsb3_features.parquet") to the side-table list (line ~134). usage: build_adsb3_features.py
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
fs = sorted(glob.glob(f"{BASE}/clk3_*.parquet"))
print(f"{len(fs)} day files")
C = pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "mvt_aobt3"])
X = (
    C.join(F, on="MVT_ID_mvt", how="left")
    .with_columns(
        adsb3_mv_gs=pl.col("tko") - pl.col("t_mv_gs"),
        adsb3_mv_pos=pl.col("tko") - pl.col("t_mv_pos"),
        adsb3_gs0=pl.col("gs0"),
        adsb3_v0=pl.col("v0"),
        adsb3_lat0=pl.col("lat0"),
        adsb3_lon0=pl.col("lon0"),
        adsb3_d_tko0=pl.col("d_tko0"),
        adsb3_path=pl.col("path_m"),
        adsb3_span=pl.col("span_s"),
        adsb3_mean_v=pl.col("mean_v"),
        adsb3_ppm=pl.col("ppm"),
        adsb3_d_first_stand=pl.col("d_first_stand"),
    )
    .with_columns(adsb3_mv_gs_aobt=pl.col("adsb3_mv_gs") - pl.col("mvt_aobt3"))
    .select(
        ["MVT_ID_mvt"]
        + [
            c
            for c in [
                "adsb3_mv_gs",
                "adsb3_mv_pos",
                "adsb3_mv_gs_aobt",
                "adsb3_gs0",
                "adsb3_v0",
                "adsb3_lat0",
                "adsb3_lon0",
                "adsb3_d_tko0",
                "adsb3_path",
                "adsb3_span",
                "adsb3_mean_v",
                "adsb3_ppm",
                "adsb3_d_first_stand",
            ]
        ]
    )
)
X = X.unique(subset=["MVT_ID_mvt"], keep="first").with_columns(
    [pl.col(c).cast(pl.Float32).fill_nan(None) for c in X.columns if c != "MVT_ID_mvt"]
)
X.write_parquet(f"{ROOT}/data/external/adsb3_features.parquet")
print(f"rows {len(X):,} with gs-move clock {int(X['adsb3_mv_gs'].is_not_null().sum()):,}")
print(X.select(["adsb3_mv_gs", "adsb3_d_tko0", "adsb3_ppm"]).describe())
