#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Side table `adsb6` (prefix adsb6_, pushstart) from adsb_pushstart_extract.py outputs (clkP_*.parquet), MVT_TIME-relative, label-free.
Old tables (adsb..adsb5) untouched.
  adsb6_push   = MVT - push start (t_best: parked start -> back-extrapolated park-run end; first seen mid-push -> first fix minus stand distance / speed)
  adsb6_push1  = same, only src 1 (parked, then reverse motion vs heading = seen pushback)
  adsb6_push3  = same, only src 3 (first seen mid-push)
  adsb6_pe     = MVT - end of the reverse motion (push end)      adsb6_tx = MVT - taxi start (first forward motion >= 15 m past the push end)
  adsb6_hold   = t_tx - t_pe (standstill between push end and taxi start)
  adsb6_src (0..5, see adsb_pushstart_extract.py), adsb6_rev (share of reverse-motion points), adsb6_vpush, adsb6_njump, adsb6_parklen, adsb6_dstand
usage: build_adsb6_features.py"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = f"{ROOT}/data/external/adsblol"
fs = sorted(glob.glob(f"{BASE}/clkP_*.parquet"))
print(f"{len(fs)} day files")
C = (
    pl.concat([pl.read_parquet(f) for f in fs], how="diagonal_relaxed")
    .unique("MVT_ID_mvt", keep="first")
    .filter(pl.col("src").is_not_null())
)
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).with_columns(
    tko=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)
)
X = (
    C.with_columns(pl.col("MVT_ID_mvt").cast(F["MVT_ID_mvt"].dtype))
    .join(F.select(["MVT_ID_mvt", "tko"]), on="MVT_ID_mvt", how="inner")
    .with_columns(
        adsb6_push=pl.col("tko") - pl.col("t_best"),
        adsb6_push1=pl.when(pl.col("src") == 1).then(pl.col("tko") - pl.col("t_best")),
        adsb6_push3=pl.when(pl.col("src") == 3).then(pl.col("tko") - pl.col("t_best")),
        adsb6_pe=pl.col("tko") - pl.col("t_pe"),
        adsb6_tx=pl.col("tko") - pl.col("t_tx"),
        adsb6_hold=pl.col("t_tx") - pl.col("t_pe"),
        adsb6_src=pl.col("src"),
        adsb6_rev=pl.col("rev"),
        adsb6_vpush=pl.col("v_push"),
        adsb6_njump=pl.col("n_jump"),
        adsb6_parklen=pl.col("park_len"),
        adsb6_dstand=pl.col("d_stand"),
    )
    .select(
        ["MVT_ID_mvt"]
        + [
            f"adsb6_{c}"
            for c in ["push", "push1", "push3", "pe", "tx", "hold", "src", "rev", "vpush", "njump", "parklen", "dstand"]
        ]
    )
)
X = X.with_columns([pl.col(c).cast(pl.Float32).fill_nan(None) for c in X.columns if c != "MVT_ID_mvt"])
X.write_parquet(f"{ROOT}/data/external/adsb6_features.parquet")
print(
    f"rows {len(X):,} push {int(X['adsb6_push'].is_not_null().sum()):,} push1 {int(X['adsb6_push1'].is_not_null().sum()):,} push3 {int(X['adsb6_push3'].is_not_null().sum()):,}"
)
print(X.describe())
