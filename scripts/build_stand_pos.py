#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Stand positions learned from ADS-B: per (airport, STAND_mvt) the median lat/lon of the first stationary run of
departures that started stationary (adsb_extract outputs, all days), with counts and spread (MAD in metres).
-> data/external/adsblol/stand_pos.parquet. usage: build_stand_pos.py"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
fs = sorted(glob.glob(f"{BASE}/clk_*.parquet"))
# The production stand_pos.parquet (20 Sep 2026 16:10) is built from 305 of the 410 clk days. STANDPOS_DAYS=<file of YYYY-MM-DD> restricts the input to such a list (manifest/standpos_days.txt reproduces it exactly).
if os.environ.get("STANDPOS_DAYS"):
    keep = {d.strip().replace("-", "") for d in open(os.environ["STANDPOS_DAYS"]) if d.strip()}
    fs = [f for f in fs if os.path.basename(f)[4:12] in keep]
    assert len(fs) == len(keep), f"{len(keep) - len(fs)} listed days have no clk file"
C = pl.concat(
    [
        pl.read_parquet(f, columns=["apt", "stand", "started_stationary", "run0_lat", "run0_lon", "first_run_len"])
        for f in fs
    ],
    how="diagonal_relaxed",
)
C = C.filter(
    pl.col("started_stationary").fill_null(False)
    & pl.col("run0_lat").is_not_null()
    & pl.col("run0_lat").is_not_nan()
    & pl.col("stand").is_not_null()
    & (pl.col("first_run_len") >= 120)
)
S = C.group_by(["apt", "stand"]).agg(
    n=pl.len(),
    lat=pl.col("run0_lat").median(),
    lon=pl.col("run0_lon").median(),
    lat_q1=pl.col("run0_lat").quantile(0.25),
    lat_q3=pl.col("run0_lat").quantile(0.75),
    lon_q1=pl.col("run0_lon").quantile(0.25),
    lon_q3=pl.col("run0_lon").quantile(0.75),
)
S = S.with_columns(
    spread_m=((pl.col("lat_q3") - pl.col("lat_q1")) * 111000)
    .pow(2)
    .add(((pl.col("lon_q3") - pl.col("lon_q1")) * 111000 * np.cos(np.radians(47))).pow(2))
    .sqrt()
).select("apt", "stand", "n", "lat", "lon", "spread_m")
S = S.sort(["apt", "stand"])  # deterministic row order (group_by order is arbitrary; consumers key by apt, stand)
S.write_parquet(f"{BASE}/stand_pos.parquet")
print(f"{len(fs)} days, {len(C):,} stationary-start departures with a stand -> {len(S):,} (airport, stand) positions")
print(
    S.group_by("apt")
    .agg(stands=pl.len(), n=pl.col("n").sum(), good=(pl.col("n") >= 5).sum(), med_spread=pl.col("spread_m").median())
    .sort("apt")
)
print(
    "share of stationary-start departures whose stand has n>=5:",
    float(C.join(S.filter(pl.col("n") >= 5).select("apt", "stand"), on=["apt", "stand"], how="inner").height / len(C)),
)
