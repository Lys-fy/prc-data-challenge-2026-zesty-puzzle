#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Pools N chain prediction parquets into one by a weighted mean, rounded, on matched rows; rows where all chains
agree keep chain 1's value exactly. Weights: env W="w1,w2,..." (normalised to sum 1; default equal). Asserts the
unmatched rows equal REF in every chain and in the output.
Inputs: REF, OUT, A B [C ...] (positional parquet paths), data/features_v9.parquet, env W. Output: OUT parquet.

Pool of N chain vectors: rint(weighted mean of the chains) on matched rows; rows where all chains agree keep chain 1 exactly.
Weights: env W="w1,w2,..." (normalised to sum 1; default equal). Asserts that the unmatched rows equal REF in every chain and in the output.
usage: pool_mix.py REF OUT A B [C ...]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, hashlib, numpy as np, polars as pl

REF, OUT, CH = sys.argv[1], sys.argv[2], sys.argv[3:]
r = pl.read_parquet(REF)
col = [c for c in r.columns if c != "MVT_ID_mvt"][0]
V = []
for c in CH:
    x = pl.read_parquet(c)
    assert x["MVT_ID_mvt"].equals(r["MVT_ID_mvt"]), c
    V.append(x[col].to_numpy().astype(float))
F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched"]).filter(
    pl.col("is_rank") == 1
)
u = (
    r.select("MVT_ID_mvt")
    .join(F.with_columns(pl.col("MVT_ID_mvt").cast(r["MVT_ID_mvt"].dtype)), on="MVT_ID_mvt", how="left")["unmatched"]
    .fill_null(1)
    .to_numpy()
    == 1
)
vr = r[col].to_numpy().astype(float)
for c, v in zip(CH, V):
    assert np.array_equal(v[u], vr[u]), f"unmatched rows differ: {c}"
import os

WT = np.array([float(x) for x in os.environ["W"].split(",")]) if os.environ.get("W") else np.full(len(CH), 1 / len(CH))
assert len(WT) == len(CH) and (WT >= 0).all()
WT = WT / WT.sum()
print("weights", np.round(WT, 4))
V = np.vstack(V)
same = np.all(V == V[0], axis=0)
v = np.where(u | same, V[0], np.rint(WT @ V))
out = r.with_columns(pl.Series(col, v).cast(r[col].dtype))
out.write_parquet(OUT)
assert np.array_equal(out[col].to_numpy()[u], r[col].to_numpy()[u])
d = v - V[0]
print(
    f"chains {len(CH)}; rows {len(v):,} unmatched {u.sum():,} byte-identical to REF; changed vs chain 1 {(d != 0).sum():,}, mean {d.mean():+.2f} RMS {np.sqrt((d**2).mean()):.1f} max {np.abs(d).max():.0f}"
)
print("md5", hashlib.md5(open(OUT, "rb").read()).hexdigest())
