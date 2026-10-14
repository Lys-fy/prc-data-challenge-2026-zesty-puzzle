# SPDX-License-Identifier: GPL-3.0-only
"""'Twin' departure records = same airport + callsign, two DEP rows with |dMVT| <= 30 min (one physical take-off recorded twice).
Lists the pairs side by side (2025 with labels, 2026 label-free) and writes scratchpad/xbig2/twins.parquet (both years). Read-only on data/.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
os.makedirs("scratchpad/xbig2", exist_ok=True)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(330)
pl.Config.set_tbl_rows(90)
pl.Config.set_fmt_str_lengths(20)
RC = [
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "FLIGHT_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "STAND_mvt",
    "RUNWAY_mvt",
    "AIRCRAFT_TYPE_mvt",
    "LOBT_flt",
    "AOBT_3_flt",
    "EOBT_1_flt",
    "CALLSIGN_flt",
    "AIRCRAFT_OPERATOR_flt",
]
raw = pl.concat(
    [
        pl.read_parquet(f, columns=RC).with_columns(rank=pl.lit(0, pl.Int8))
        for f in sorted(glob.glob("data/raw/training_*.parquet"))
    ]
    + [pl.read_parquet("data/raw/ranking.parquet", columns=RC).with_columns(rank=pl.lit(1, pl.Int8))],
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
    pair_n=pl.len().over("pair"), pos=pl.col("MVT_ID_mvt").rank("ordinal").over("pair")
)
tw = tw.with_columns(
    blk=pl.col("BLOCK_TIME_UTC_mvt").dt.strftime("%m-%d %H:%M:%S"),
    sch=pl.col("SCHED_TIME_UTC_mvt").dt.strftime("%m-%d %H:%M"),
    mvt=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%m-%d %H:%M:%S"),
    aobt=pl.col("AOBT_3_flt").dt.strftime("%H:%M:%S"),
    lobt=pl.col("LOBT_flt").dt.strftime("%H:%M"),
    y=(pl.col("MVT_TIME_UTC_mvt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds(),
    ms=(pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
    g=(pl.col("AOBT_3_flt") - pl.col("BLOCK_TIME_UTC_mvt")).dt.total_seconds(),
)
print(
    "pairs 2025",
    tw.filter(pl.col("rank") == 0).select(pl.col("pair").n_unique()).item(),
    "2026",
    tw.filter(pl.col("rank") == 1).select(pl.col("pair").n_unique()).item(),
    "| pair sizes",
    tw.group_by("rank", "pair_n").len().sort("rank", "pair_n").rows(),
)
C = [
    "pair",
    "pos",
    "MVT_ID_mvt",
    "rank",
    "ADEP_mvt",
    "FLIGHT_mvt",
    "ADES_mvt",
    "STAND_mvt",
    "RUNWAY_mvt",
    "matched",
    "FLIGHT_ID_mvt",
    "sch",
    "blk",
    "mvt",
    "aobt",
    "lobt",
    "ms",
    "y",
    "g",
]
print("--- 2025 LFPG pairs (first 70 rows) ---")
print(tw.filter((pl.col("rank") == 0) & (pl.col("ADEP_mvt") == "LFPG")).select(C).head(70))
print("--- 2025 non-LFPG pairs ---")
print(tw.filter((pl.col("rank") == 0) & (pl.col("ADEP_mvt") != "LFPG")).select(C).head(40))
P = (
    tw.filter(pl.col("rank") == 0)
    .group_by("pair")
    .agg(
        apt=pl.col("ADEP_mvt").first(),
        n=pl.len(),
        n_m=pl.col("matched").sum(),
        y_m=pl.col("y").filter(pl.col("matched")).mean(),
        y_u=pl.col("y").filter(~pl.col("matched")).mean(),
        ymin=pl.col("y").min(),
        ymax=pl.col("y").max(),
        same_blk=(pl.col("BLOCK_TIME_UTC_mvt").n_unique() == 1),
        same_stand=(pl.col("STAND_mvt").n_unique() == 1),
        same_sched=(pl.col("SCHED_TIME_UTC_mvt").n_unique() == 1),
        y_idfirst=pl.col("y").sort_by("MVT_ID_mvt").first(),
        y_idlast=pl.col("y").sort_by("MVT_ID_mvt").last(),
        dblk=(pl.col("BLOCK_TIME_UTC_mvt").max() - pl.col("BLOCK_TIME_UTC_mvt").min()).dt.total_seconds(),
    )
)
print("pair summary by airport x (n, n matched):")
print(
    P.group_by("apt", "n", "n_m")
    .agg(
        pairs=pl.len(),
        y_matched_mean=pl.col("y_m").mean(),
        y_unm_mean=pl.col("y_u").mean(),
        ymax_med=pl.col("ymax").median(),
        ymin_med=pl.col("ymin").median(),
        same_blk=pl.col("same_blk").mean(),
        dblk_med=pl.col("dblk").median(),
        same_stand=pl.col("same_stand").mean(),
        same_sched=pl.col("same_sched").mean(),
        y_idfirst_med=pl.col("y_idfirst").median(),
        y_idlast_med=pl.col("y_idlast").median(),
    )
    .sort("apt", "n", "n_m")
)
print("--- 2026 pairs (label-free) ---")
print(tw.filter(pl.col("rank") == 1).select([c for c in C if c not in ("blk", "y", "g")]).head(60))
tw.select(
    "MVT_ID_mvt", "pair", "pos", "pair_n", "rank", "ADEP_mvt", "FLIGHT_mvt", "matched", "y", "ms", "g", "dprev", "dnext"
).write_parquet("scratchpad/xbig2/twins.parquet")
print("written twins.parquet", tw.height)
