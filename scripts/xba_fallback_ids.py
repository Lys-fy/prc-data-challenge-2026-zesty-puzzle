#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Flags matched ranking departures whose xba stand-join values are stale because the join reaches across a gap in
the ranking file's date coverage: xba_since > 30 d, xba_next > 30 d, or xba_next is null. On these rows assembly
uses the xba-free member (V47L) instead of xba_fallback_v47la.
Takes no arguments. Inputs: data/external/xba_features.parquet, data/features_v9.parquet.
Output: data/external/xba_fallback_ids.parquet (MVT_ID_mvt, reason).

xba edge fallback: ranking rows whose AOBT_3-anchored stand joins reach across the holes of the 2026
ranking file (no Feb-Jun / Aug 2026 arrivals), so their xba values are stale. Declared rule, label-free:
  xba_since > 30 d  |  xba_next > 30 d  |  xba_next is null   (either 2026 month; 4,266 rows)
(matched ranking departures only). On these rows assembly uses the xba-free member (V47L) instead of xba_fallback_v47la.
Writes data/external/xba_fallback_ids.parquet (MVT_ID_mvt, reason)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D30 = 30 * 86400
T = pl.read_parquet(f"{ROOT}/data/external/xba_features.parquet")
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "m", "unmatched"])
J = F.join(T, on="MVT_ID_mvt").filter((pl.col("is_rank") == 1) & (pl.col("unmatched") == 0))
J = J.with_columns(
    pl.when(pl.col("xba_since") > D30)
    .then(pl.lit("since_stale"))
    .when(pl.col("xba_next") > D30)
    .then(pl.lit("next_stale"))
    .when(pl.col("xba_next").is_null())
    .then(pl.lit("next_null"))
    .otherwise(None)
    .alias("reason")
)
out = J.filter(pl.col("reason").is_not_null()).select("MVT_ID_mvt", "reason")
out.write_parquet(f"{ROOT}/data/external/xba_fallback_ids.parquet")
print(out.height, "rows ->", f"{ROOT}/data/external/xba_fallback_ids.parquet")
print(out.group_by("reason").len())
