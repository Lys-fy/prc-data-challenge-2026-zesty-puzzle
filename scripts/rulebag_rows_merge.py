#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Seed-bag merge of the rule-priced rows (staged assembly of the submitted file, docs/REPRODUCE.md §6b, step S2).
<out> = <base line> with ONLY the rows of class "exposed" / "unm_LIRF" (row classes of the bag workspace, scratchpad/p2/U.parquet) taken
from the bag line: the same FORM=v99 line run with the longer seed lists FZ_L / FZ_C of the period file (24 LightGBM + 16 CatBoost seeds),
output at post.v99. Every other row stays byte-identical, in the base order. Refuses to overwrite its output.
usage: rulebag_rows_merge.py <base line parquet> <bag line parquet> <bag U.parquet> <out>"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import hashlib, os, sys, numpy as np, polars as pl

base_p, bag_p, u_p, out_p = sys.argv[1:5]
assert not os.path.exists(out_p), f"refusing: {out_p} exists"
v = pl.read_parquet(base_p)
f = pl.read_parquet(bag_p)
U = pl.read_parquet(u_p).select(mid=pl.col("MVT_ID_mvt").cast(pl.Float64), cls="cls")
J = (
    v.with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
    .join(
        f.select(mid=pl.col("MVT_ID_mvt").cast(pl.Float64), f=pl.col("TAXITIME_SEC_mvt")),
        on="mid",
        how="left",
        maintain_order="left",
    )
    .join(U, on="mid", how="left", maintain_order="left")
)
assert J.height == v.height == f.height and J["f"].null_count() == 0 and J["cls"].null_count() == 0
fro = J["cls"].is_in(["exposed", "unm_LIRF"]).to_numpy()
a = J["TAXITIME_SEC_mvt"].to_numpy()
b = np.where(fro, J["f"].to_numpy(), a).astype(a.dtype)
o = v.with_columns(pl.Series("TAXITIME_SEC_mvt", b).cast(v.schema["TAXITIME_SEC_mvt"]))
assert o.schema == v.schema and o["MVT_ID_mvt"].equals(v["MVT_ID_mvt"])
d = b.astype(float) - a.astype(float)
cls = J["cls"].to_numpy()
for c in ["unm_LIRF", "exposed"]:
    x = cls == c
    print(
        f"{c}: n {x.sum()} changed {(d[x] != 0).sum()} RMS {np.sqrt(np.mean(d[x] ** 2)):.0f} s mean {d[x].mean():+.0f}"
    )
assert (d[~fro] == 0).all(), "a row outside the two rule classes changed"
tmp = out_p + ".tmp"
o.write_parquet(tmp)
os.link(tmp, out_p)
os.unlink(tmp)  # os.link fails if out_p appeared meanwhile
print(
    f"wrote {out_p}: rows {o.height:,}, all-row RMS {np.sqrt(np.mean(d ** 2)):.2f} s, md5 {hashlib.md5(open(out_p, 'rb').read()).hexdigest()}"
)
