#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Identifies the TW1' class of ranking rows by a label-free rule on the raw movement files: at LFPG, a DEP record
whose only paired record (same airport + FLIGHT_mvt, |dMVT| <= 1800 s, pair size 2, |dMVT| <= 300 s) is NM-matched
is in the class. Inputs: RANKING.parquet, training_*.parquet (found next to it), FEATURES.parquet, optional
--expect IDS.parquet (asserted equal to the rule's result). Output: OUT.parquet with column MVT_ID_mvt.

TW1' class BY RULE from the raw movement files (label-free, no ordering), for any ranking file.
Pair construction = scripts/tw1_twins.py (same airport + FLIGHT_mvt, DEP records chained when |dMVT| <= 1800 s; training + ranking files together),
selector = scripts/tw1_rule_build.py /: LFPG, non-null FLIGHT_mvt, pair of exactly 2 records, |dMVT| <= 300 s, exactly ONE NM-matched
record (FLIGHT_ID_mvt not null) -> that record. Output: the ranking-row ids (column MVT_ID_mvt).
usage: p2_tw1_rule.py RANKING.parquet FEATURES.parquet OUT.parquet [--expect IDS.parquet]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
RANK, FEAT, OUT = sys.argv[1:4]
EXP = sys.argv[sys.argv.index("--expect") + 1] if "--expect" in sys.argv else None
RC = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]
tr = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(RANK)), "training_*.parquet")))
raw = pl.concat(
    [pl.read_parquet(f, columns=RC).with_columns(rank=pl.lit(0, pl.Int8)) for f in tr]
    + [pl.read_parquet(RANK, columns=RC).with_columns(rank=pl.lit(1, pl.Int8))],
    how="diagonal_relaxed",
)
dep = (
    raw.filter(pl.col("PHASE_mvt") == "DEP")
    .with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Int64),
        mvt_s=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"),
        matched=pl.col("FLIGHT_ID_mvt").is_not_null(),
    )
    .sort(["ADEP_mvt", "FLIGHT_mvt", "mvt_s", "MVT_ID_mvt"])
)
G = ["ADEP_mvt", "FLIGHT_mvt"]
dep = dep.with_columns(
    dprev=(pl.col("mvt_s") - pl.col("mvt_s").shift(1).over(G)),
    dnext=(pl.col("mvt_s").shift(-1).over(G) - pl.col("mvt_s")),
)
dep = dep.with_columns(newpair=~(pl.col("dprev").abs() <= 1800).fill_null(False)).with_columns(
    pair=pl.col("newpair").cast(pl.Int64).cum_sum()
)
tw = dep.filter((pl.col("dprev").abs() <= 1800) | (pl.col("dnext").abs() <= 1800)).with_columns(
    pair_n=pl.len().over("pair")
)
T = tw.filter(
    pl.col("FLIGHT_mvt").is_not_null()
    & (pl.col("pair_n") == 2)
    & ((pl.col("dprev").abs() <= 300) | (pl.col("dnext").abs() <= 300))
)
T = T.with_columns(nm=pl.col("matched").cast(pl.Int32).sum().over("rank", "pair")).with_columns(
    sel=pl.col("matched") & (pl.col("nm") == 1)
)
S = T.filter((pl.col("rank") == 1) & (pl.col("ADEP_mvt") == "LFPG") & pl.col("sel")).select("MVT_ID_mvt")
# only ids that are ranking rows of the features table (matched rows; the shift applies to NM-matched records)
F = (
    pl.read_parquet(FEAT, columns=["MVT_ID_mvt", "is_rank", "unmatched"])
    .filter(pl.col("is_rank") == 1)
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
)
S = S.join(F, on="MVT_ID_mvt", how="inner")
print(
    f"TW1' class (rule): {S.height} ranking rows, unmatched among them {int((S['unmatched'] == 1).sum())}; "
    f"2025 class rows (info): {T.filter((pl.col('rank') == 0) & (pl.col('ADEP_mvt') == 'LFPG') & pl.col('sel')).height}"
)
S.select("MVT_ID_mvt").write_parquet(OUT)
if EXP:
    e = set(pl.read_parquet(EXP)["MVT_ID_mvt"].cast(pl.Int64).to_list())
    s = set(S["MVT_ID_mvt"].to_list())
    print(
        f"vs {EXP}: rule {len(s)}, reference {len(e)}, common {len(s & e)}, only rule {sorted(s - e)[:10]}, only reference {sorted(e - s)[:10]}"
    )
    assert s == e, "TW1' rule != reference ids"
    print("TW1' RULE == REFERENCE IDS")
