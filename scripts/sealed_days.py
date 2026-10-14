#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fixed, label-free sealed split of the 2025 gate folds: 20 % of the airport-days are held out of every gate decision.
Unit = airport-day (ADEP_mvt, UTC date of MVT_TIME_UTC_mvt; the same day key as clock_lib.dbin). Universe = every (airport, day) that has
at least one non-ranking 2025 row in data/features_v9.parquet. Strata = airport x month. In each stratum the days are sorted by date and
permuted with numpy default_rng(SEED + stratum index) (strata sorted by airport, month); the first round(0.20 * n_days) are sealed.
Reads ONLY the calendar columns ADEP_mvt, MVT_TIME_UTC_mvt, is_rank -- no label, no prediction, no result.
Output data/sealed_days.parquet: ADEP_mvt, day, month, sealed (bool). Prints counts per stratum and the md5 of the file.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, hashlib, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SEED = 20260929
SHARE = 0.20
OUT = "data/sealed_days.parquet"
G = pl.read_parquet("data/features_v9.parquet", columns=["ADEP_mvt", "MVT_TIME_UTC_mvt", "is_rank"])
G = G.filter((pl.col("is_rank") == 0) & (pl.col("MVT_TIME_UTC_mvt").dt.year() == 2025))
D = (
    G.select(pl.col("ADEP_mvt").cast(pl.Utf8), day=pl.col("MVT_TIME_UTC_mvt").dt.date())
    .unique()
    .with_columns(month=pl.col("day").dt.month())
    .sort("ADEP_mvt", "day")
)
strata = D.select("ADEP_mvt", "month").unique().sort("ADEP_mvt", "month")
parts = []
for i, (apt, mon) in enumerate(strata.iter_rows()):
    S = D.filter((pl.col("ADEP_mvt") == apt) & (pl.col("month") == mon)).sort("day")
    n = S.height
    k = int(round(SHARE * n))
    perm = np.random.default_rng(SEED + i).permutation(n)
    flag = np.zeros(n, bool)
    flag[perm[:k]] = True
    parts.append(S.with_columns(sealed=pl.Series(flag)))
out = pl.concat(parts).sort("ADEP_mvt", "day")
out.write_parquet(OUT, compression="zstd", statistics=False)
md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
print(
    f"seed {SEED} share {SHARE} strata {strata.height} airport-days {out.height:,} sealed {int(out['sealed'].sum()):,} ({100 * out['sealed'].mean():.1f} %)"
)
print(out.group_by("month").agg(days=pl.len(), sealed=pl.col("sealed").sum()).sort("month"))
print(out.group_by("ADEP_mvt").agg(days=pl.len(), sealed=pl.col("sealed").sum()).sort("ADEP_mvt"))
# content hash (independent of parquet writer version): sorted sealed keys
keys = "\n".join(f"{a}|{d}" for a, d in out.filter(pl.col("sealed")).select("ADEP_mvt", "day").iter_rows())
print(f"file md5 {md5}\ncontent md5 (sorted sealed 'APT|YYYY-MM-DD' lines) {hashlib.md5(keys.encode()).hexdigest()}")
