#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Staged member swap as a delta (staged assembly of the submitted file, docs/REPRODUCE.md §6b, steps S3 and S5).
  d = <to line> - <from line>, per MVT_ID_mvt: two runs of the same line that differ only in the member of the v102 slot
      (S3: K250 -> KCAT8, S5: KCAT8 -> KCATLg).
  <out> = <base> + d; the LOBT law of the period's features table (|pred - mvt_lobt| <= 3,606 s, rows with mvt_lobt only), rint,
          floor 60 s, the dtype of the base. Default: on every row. --changed-only: on the rows with d != 0; the other rows stay byte-identical.
  --protect <U.parquet> <final_dryrun.parquet>: assert d == 0 on every rule-priced row (U cls != matched) and on the schedule-copy rule rows
          (the chain never prices them).
usage: staged_delta_apply.py <base> <from line> <to line> <features_v9.parquet> <out> [--changed-only] [--protect <U> <final_dryrun>]
Refuses to overwrite its output."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, hashlib, numpy as np, polars as pl

a = sys.argv[1:]
ONLY = "--changed-only" in a
a = [x for x in a if x != "--changed-only"]
PROT = None
if "--protect" in a:
    i = a.index("--protect")
    PROT = a[i + 1 : i + 3]
    del a[i : i + 3]
base_p, from_p, to_p, feat_p, out_p = a
assert not os.path.exists(out_p), f"refusing: {out_p} exists"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
rd = lambda p: pl.read_parquet(p).select(
    mid=pl.col("MVT_ID_mvt").cast(pl.Float64), g=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)
)
B = pl.read_parquet(base_p)
D = rd(from_p).join(rd(to_p), on="mid", suffix="_k").with_columns(d=pl.col("g_k") - pl.col("g"))
assert D.height == B.height, (D.height, B.height)
dd = dict(zip(D["mid"].to_list(), D["d"].to_list()))
print(
    f"delta {os.path.basename(to_p)} - {os.path.basename(from_p)}: {int((D['d'] != 0).sum()):,} rows changed, RMS {np.sqrt(np.mean(D['d'].to_numpy() ** 2)):.2f} s, max |d| {D['d'].abs().max():.0f}"
)
if PROT:
    U = pl.read_parquet(PROT[0]).select(mid=pl.col("MVT_ID_mvt").cast(pl.Float64), cls="cls")
    prot = (
        U.filter(pl.col("cls") != "matched")["mid"].to_list()
        + pl.read_parquet(PROT[1])["MVT_ID_mvt"].cast(pl.Float64).to_list()
    )
    assert all(dd.get(i, 0.0) == 0.0 for i in prot), "delta touches rule-priced / schedule-copy rows"
    print(f"rule-priced + schedule-copy rows {len(set(prot)):,}: delta 0 on all")
mid = B["MVT_ID_mvt"].cast(pl.Float64).to_list()
g0 = B["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
d = np.array([dd.get(i, 0.0) for i in mid])
g = g0 + d
F = pl.read_parquet(feat_p, columns=["MVT_ID_mvt", "is_rank", "mvt_lobt"]).filter(pl.col("is_rank") == 1)
lo = dict(zip(F["MVT_ID_mvt"].cast(pl.Float64).to_list(), F["mvt_lobt"].cast(pl.Float64).to_list()))
m = (d != 0) if ONLY else np.ones(len(d), bool)
l = np.array([lo.get(i) if lo.get(i) is not None else np.nan for i in mid], dtype=float)
v = m & ~np.isnan(l) & (np.abs(g - l) > 3606.5)
g[v] = np.clip(g[v], l[v] - 3606, l[v] + 3606)
g = np.where(m, np.maximum(np.rint(g), 60), g0)
o = B.with_columns(pl.Series("TAXITIME_SEC_mvt", g).cast(B.schema["TAXITIME_SEC_mvt"]))
assert o.height == B.height and o["TAXITIME_SEC_mvt"].null_count() == 0 and o["MVT_ID_mvt"].equals(B["MVT_ID_mvt"])
tmp = out_p + ".tmp"
o.write_parquet(tmp)
os.link(tmp, out_p)
os.unlink(tmp)  # os.link fails if out_p appeared meanwhile
print(
    f"wrote {out_p}: rows {o.height:,} | LOBT clips {int(v.sum())} | rows changed vs base {int((o['TAXITIME_SEC_mvt'] != B['TAXITIME_SEC_mvt']).sum()):,} | md5 {md5(out_p)}"
)
