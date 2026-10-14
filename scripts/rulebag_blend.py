# SPDX-License-Identifier: GPL-3.0-only
"""Frozen-blend production p / taxi / art for all unmatched ranking rows (from data/preds), for the unm_LIRF gate. Runs on the build machine."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
os.makedirs("scratchpad/xhr2", exist_ok=True)
V25 = os.environ.get(
    "FZ_L", ",".join([f"V25L_s{i}" for i in range(10)] + [f"V25Lb_s{i}" for i in range(10, 15)])
).split(",")
CAT9 = os.environ.get("FZ_C", "CATV9x_s0,CATV9x_s1").split(",")  # seed lists: period file
C = ["TAXITIME_SEC_mvt", "p_raw", "art", "taxi", "q_ds"]


def blend(names):
    acc = None
    for n in names:
        d = (
            pl.read_parquet(f"data/preds/{n}.parquet")
            .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
            .sort("MVT_ID_mvt")
            .select("MVT_ID_mvt", *C)
        )
        acc = d if acc is None else acc.with_columns([(pl.col(c) + d[c]).alias(c) for c in C])
    return acc.with_columns([(pl.col(c) / len(names)).alias(c) for c in C])


A = blend(V25)
B = blend(CAT9)
F = A.join(B, on="MVT_ID_mvt", suffix="_c")
F = F.select(
    "MVT_ID_mvt",
    fz=(0.5 * pl.col("TAXITIME_SEC_mvt") + 0.5 * pl.col("TAXITIME_SEC_mvt_c")).clip(lower_bound=60),
    fzr_p=0.5 * pl.col("p_raw") + 0.5 * pl.col("p_raw_c"),
    fzr_art=0.5 * pl.col("art") + 0.5 * pl.col("art_c"),
    fzr_taxi=0.5 * pl.col("taxi") + 0.5 * pl.col("taxi_c"),
    fzr_q=0.5 * pl.col("q_ds") + 0.5 * pl.col("q_ds_c"),
    fzL_p=pl.col("p_raw"),
    fzC_p=pl.col("p_raw_c"),
    fzL_taxi=pl.col("taxi"),
    fzC_taxi=pl.col("taxi_c"),
)
U = (
    pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched", "ADEP_mvt", "mvt_sched"])
    .filter((pl.col("is_rank") == 1) & (pl.col("unmatched") == 1))
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
)
V62 = (
    pl.read_parquet("stages/unmatched_rule.parquet")
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    .rename({"TAXITIME_SEC_mvt": "v62"})
)
O = U.join(F, on="MVT_ID_mvt", how="left").join(V62, on="MVT_ID_mvt", how="left")
print(O.shape)
print(
    O.filter(pl.col("ADEP_mvt") == "LIRF").select(
        (pl.col("fz").round() - pl.col("v62")).abs().max().alias("max_abs_fz_minus_v62"), pl.len()
    )
)
print(O.filter(pl.col("ADEP_mvt") == "LIRF").head(5))
O.write_parquet("scratchpad/xhr2/fzr_rank.parquet")
print("done")
