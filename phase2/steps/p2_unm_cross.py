#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The label-free 'cross' flag of the period's unmatched ranking departures, the only column Kumo F4 (kumo_icl_lirf.py) reads
from scratchpad/xhr/unm2026.parquet. Same definition as scripts/exposed_characterise.py (which also needs historical submission files and is therefore
not run here): cross = local calendar day of MVT_TIME - local calendar day of SCHED_TIME at the departure airport (DST-aware).
Writes scratchpad/xhr/unm2026.parquet (MVT_ID_mvt, cls, ADEP_mvt, cross) for the period; the file name is the one kumo_icl_lirf.py reads.
usage: p2_unm_cross.py      (cwd = workspace)"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
TZ = {
    "LIRF": "Europe/Rome",
    "EHAM": "Europe/Amsterdam",
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "LFPG": "Europe/Paris",
    "LSZH": "Europe/Zurich",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "EGLL": "Europe/London",
    "LTFM": "Europe/Istanbul",
}


def utc_off(col):
    return pl.coalesce(
        [
            pl.when(pl.col("ADEP_mvt") == a).then(
                (
                    pl.col(col).dt.convert_time_zone(z).dt.replace_time_zone(None)
                    - pl.col(col).dt.replace_time_zone(None)
                ).dt.total_seconds()
            )
            for a, z in TZ.items()
        ]
    )


def loc_date(col):
    return (pl.col(col).dt.epoch("s") + utc_off(col)) // 86400


R = pl.read_parquet(
    "data/raw/ranking.parquet",
    columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"],
).filter(pl.col("PHASE_mvt") == "DEP")
R = R.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").cast(pl.Utf8))
F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched", "mvt_sched"]).filter(
    (pl.col("is_rank") == 1) & (pl.col("unmatched") == 1)
)
U = F.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(R, on="MVT_ID_mvt", how="left")
assert U["MVT_TIME_UTC_mvt"].null_count() == 0, "unmatched ranking rows without a ranking-file row"
U = U.with_columns(
    cls=pl.when(pl.col("mvt_sched").fill_null(0) > 28800)
    .then(pl.lit("exposed"))
    .when(pl.col("ADEP_mvt") == "LIRF")
    .then(pl.lit("unm_LIRF"))
    .otherwise(pl.lit("unm_other")),
    cross=(loc_date("MVT_TIME_UTC_mvt") - loc_date("SCHED_TIME_UTC_mvt")).cast(pl.Int32),
)
os.makedirs("scratchpad/xhr", exist_ok=True)
p = "scratchpad/xhr/unm2026.parquet"
if os.path.islink(p):
    os.unlink(p)
U.select("MVT_ID_mvt", "cls", "ADEP_mvt", "cross").write_parquet(p)
print(
    f"unmatched ranking rows {U.height}: by class {dict(U.group_by('cls').len().rows())}; cross != 0 {int((U['cross'] != 0).sum())} -> {p}"
)
