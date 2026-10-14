#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""ADS-B clock features (side table `adsb`, prefix adsb_) from adsb_extract.py outputs. All RELATIVE to the row's own
timestamps, label-free: MVT_TIME is given for ranking rows; AOBT_3 via mvt_aobt3 = MVT - AOBT_3 from features_v9.
  adsb_linked          trace found (1/0)                     adsb_stat      segment starts stationary at a gate (1/0)
  adsb_lb              MVT - t_first_seen  (taxi lower bound)  adsb_move      MVT - t_first_move (physical taxi; NaN unless stat)
  adsb_lb_aobt         adsb_lb - mvt_aobt3 (first seen relative to AOBT_3: <0 = seen before AOBT_3)
  adsb_move_aobt       adsb_move - mvt_aobt3                 adsb_hold      MVT - t_hold_end
  adsb_n_ground, adsb_span, adsb_first_gs, adsb_run0_len, adsb_hold_total, adsb_n_runs, adsb_tko_err (t_tko - MVT)
Rows on days without a pulled archive get NULL everywhere (distinguished from 'day pulled, no trace' by adsb_linked=0).
usage: build_adsb_features.py
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
fs = sorted(glob.glob(f"{BASE}/clk_*.parquet"))
print(f"{len(fs)} day files")
C = pl.concat(
    [pl.read_parquet(f).with_columns(pl.lit(os.path.basename(f)[4:12]).alias("day")) for f in fs],
    how="diagonal_relaxed",
)
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "mvt_aobt3"])
X = (
    C.join(F, on="MVT_ID_mvt", how="left")
    .with_columns(
        adsb_linked=pl.col("icao").is_not_null().cast(pl.Int8),
        adsb_stat=pl.col("started_stationary").fill_null(False).cast(pl.Int8),
        adsb_lb=(pl.col("tko") - pl.col("t_first_seen")),
        adsb_move=(pl.col("tko") - pl.col("t_first_move")),
        adsb_hold=(pl.col("tko") - pl.col("t_hold_end")),
        adsb_n_ground=pl.col("n_ground"),
        adsb_span=pl.col("span"),
        adsb_first_gs=pl.col("first_gs"),
        adsb_run0_len=pl.col("first_run_len"),
        adsb_hold_total=pl.col("hold_total"),
        adsb_n_runs=pl.col("n_runs"),
        adsb_tko_err=(pl.col("t_tko") - pl.col("tko")),
    )
    .with_columns(
        adsb_lb_aobt=pl.col("adsb_lb") - pl.col("mvt_aobt3"), adsb_move_aobt=pl.col("adsb_move") - pl.col("mvt_aobt3")
    )
    .select(
        ["MVT_ID_mvt"]
        + [
            c
            for c in [
                "adsb_linked",
                "adsb_stat",
                "adsb_lb",
                "adsb_move",
                "adsb_hold",
                "adsb_lb_aobt",
                "adsb_move_aobt",
                "adsb_n_ground",
                "adsb_span",
                "adsb_first_gs",
                "adsb_run0_len",
                "adsb_hold_total",
                "adsb_n_runs",
                "adsb_tko_err",
            ]
        ]
    )
)
X = X.unique(subset=["MVT_ID_mvt"], keep="first").with_columns(
    [pl.col(c).cast(pl.Float32).fill_nan(None) for c in X.columns if c != "MVT_ID_mvt"]
)
X.write_parquet(f"{ROOT}/data/external/adsb_features.parquet")
print(f"rows {len(X):,} linked {int(X['adsb_linked'].sum()):,} stationary-start {int(X['adsb_stat'].sum()):,}")
print(X.select(["adsb_lb", "adsb_move", "adsb_lb_aobt", "adsb_move_aobt"]).describe())
