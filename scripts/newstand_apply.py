#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""NEWSTAND_DELTA: new-stand neighbour delta on top of a bundle.
usage: newstand_apply.py <base.parquet> <ids.parquet> <out.parquet> <predsdir>   (preds newstand_delta_p{U,N}_s{0,1}.parquet, ids from newstand_map.py prod N)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, numpy as np, polars as pl

base, idf, out, pdir = sys.argv[1:5]
B = pl.read_parquet(base)
ids = pl.read_parquet(idf).filter(pl.col("NB_STAND").is_not_null())
ds = []
for s in (0, 1):
    u = pl.read_parquet(f"{pdir}/newstand_delta_pu_s{s}.parquet")
    n = pl.read_parquet(f"{pdir}/newstand_delta_pn_s{s}.parquet")
    pc = "TAXITIME_SEC_mvt"
    ds.append(
        u.select("MVT_ID_mvt", pl.col(pc).alias("u"))
        .join(n.select("MVT_ID_mvt", pl.col(pc).alias("n")), on="MVT_ID_mvt")
        .select("MVT_ID_mvt", (pl.col("n") - pl.col("u")).alias(f"d{s}"))
    )
D = ds[0].join(ds[1], on="MVT_ID_mvt").with_columns(((pl.col("d0") + pl.col("d1")) / 2).alias("delta"))
U = pl.read_parquet(f"{pdir}/unm_ids.parquet")  # MVT_ID_mvt of unmatched ranking rows (excluded)
T = ids.join(U, on="MVT_ID_mvt", how="anti").join(D, on="MVT_ID_mvt")
print(
    "rows changed",
    T.height,
    "delta mean",
    T["delta"].mean(),
    "abs mean",
    T["delta"].abs().mean(),
    "seed agreement corr",
    np.corrcoef(T["d0"], T["d1"])[0, 1],
)
print(
    T.group_by("ADEP_mvt", "STAND_mvt")
    .agg(pl.len(), pl.col("delta").mean().round(1))
    .sort("ADEP_mvt", "STAND_mvt")
    .rows()
)
L = (
    pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "mvt_lobt"])
    .filter(pl.col("is_rank") == 1)
    .drop("is_rank")
)
# same output constraints as every bundle (validate_submission.py): >= 60 s and the LOBT law |pred - mvt_lobt| <= 3606 s
O = B.join(T.select("MVT_ID_mvt", "delta"), on="MVT_ID_mvt", how="left").join(L, on="MVT_ID_mvt", how="left")
O = O.with_columns((pl.col("TAXITIME_SEC_mvt") + pl.col("delta")).round(0).alias("_v"))
O = O.with_columns(
    pl.when(pl.col("mvt_lobt").is_not_null())
    .then(pl.col("_v").clip(pl.col("mvt_lobt") - 3606, pl.col("mvt_lobt") + 3606))
    .otherwise(pl.col("_v"))
    .clip(60, None)
    .alias("_v")
)
O = O.with_columns(
    pl.when(pl.col("delta").is_not_null())
    .then(pl.col("_v").cast(pl.Int32))
    .otherwise(pl.col("TAXITIME_SEC_mvt"))
    .alias("TAXITIME_SEC_mvt")
).select(B.columns)
assert O.height == B.height and (O["MVT_ID_mvt"] == B["MVT_ID_mvt"]).all()
print("rows differing from base", int((O["TAXITIME_SEC_mvt"] != B["TAXITIME_SEC_mvt"]).sum()))
O.write_parquet(out)
print("wrote", out)
