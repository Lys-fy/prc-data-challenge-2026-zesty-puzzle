#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Feature NaN-rate shift between 2025 training rows and the 2026 ranking rows (and Jan vs Jul 2026).
A feature that is rarely missing in training but often missing in 2026 is extrapolated by every model, and CatBoost
(nan_mode=Min) and LightGBM (learned missing-direction) extrapolate differently -- a candidate for why a blend that
wins every 2025 fold loses on the board."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sch = pl.read_parquet_schema(f"{ROOT}/data/features_v9.parquet")
num = [
    c
    for c, t in sch.items()
    if t in (pl.Float32, pl.Float64, pl.Int64, pl.Int32, pl.Int8, pl.Int16) and c not in ("is_rank", "m")
]
D = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["is_rank", "m"] + num)
tr = D.filter(pl.col("is_rank") == 0)
rk = D.filter(pl.col("is_rank") == 1)
rj = rk.filter(pl.col("m") == 1)
ru = rk.filter(pl.col("m") == 7)
rows = []
for c in num:
    a = tr[c].null_count() / tr.height
    b = rk[c].null_count() / rk.height
    if abs(b - a) > 0.02:
        rows.append((c, a, b, rj[c].null_count() / rj.height, ru[c].null_count() / ru.height))
rows.sort(key=lambda r: -abs(r[2] - r[1]))
print(f"features with |NaN rate 2026 - 2025| > 2 pp: {len(rows)} of {len(num)}")
print(f"  {'feature':28s} {'2025':>6s} {'2026':>6s} {'diff':>7s} {'Jan26':>6s} {'Jul26':>6s}")
for c, a, b, j, u in rows[:40]:
    print(f"  {c:28s} {a:6.3f} {b:6.3f} {b-a:+7.3f} {j:6.3f} {u:6.3f}")
