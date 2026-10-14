#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Composes a base prediction file with one or more isolate prediction files into one output, by the v94 rule:
out = round(base + sum_i (isolate_i - base)), floored at 60 s. Each isolate's delta must be zero on unmatched rows.
Inputs: --base <file>, --out <file>, one or more isolate files (positional); data/features_v9.parquet for the
unmatched flags. Output: --out parquet, same schema as --base.

the v94 composition rule inside a workspace. v94 = v88 + (v89 - v88) + (v91 - v88) (final_combine.py add --allow-overlap):
the WIDELEAF chains (v89) and the PSEUDOLABEL chains (v91) are two isolates on the same base, each built by the full post line with its own Kq / NNA3 chains;
their deltas are summed on the rows both touch. Here all three files are built in the workspace by the same post line (no uploaded file is read):
  out = round(base + sum_i (iso_i - base)),  floor 60 s (final_combine's floor), unmatched rows asserted unchanged (isolates touch matched rows).
usage: p2_compose.py --base <file> --out <file> <iso1> [<iso2> ...]     (cwd = workspace)"""

import argparse, os, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", required=True, help="base prediction parquet")
ap.add_argument("--out", required=True, help="output parquet path")
ap.add_argument("iso", nargs="+", help="one or more isolate prediction parquet files")
a = ap.parse_args()
c = "TAXITIME_SEC_mvt"
B = pl.read_parquet(a.base)
b = B[c].to_numpy().astype(np.float64)
tot = np.zeros_like(b)
touch = np.zeros(len(b), np.int32)
F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched"]).filter(
    pl.col("is_rank") == 1
)
u = (
    B.select("MVT_ID_mvt")
    .join(
        F.with_columns(pl.col("MVT_ID_mvt").cast(B.schema["MVT_ID_mvt"])),
        on="MVT_ID_mvt",
        how="left",
        maintain_order="left",
    )["unmatched"]
    .fill_null(1)
    .to_numpy()
    == 1
)
for p in a.iso:
    I = pl.read_parquet(p)
    assert I["MVT_ID_mvt"].equals(B["MVT_ID_mvt"]), p
    d = I[c].to_numpy().astype(np.float64) - b
    assert (d[u] == 0).all(), f"{p}: unmatched rows differ from the base"
    tot += d
    touch += d != 0
    print(f"  + {p}: rows {int((d != 0).sum()):,}, RMS {np.sqrt((d ** 2).mean()):.2f} s, mean {d.mean():+.3f}")
o = np.maximum(np.round(b + tot), 60)
res = B.with_columns(pl.Series(c, o.astype(np.int64)).cast(B.schema[c]))
assert res.schema == B.schema
res.write_parquet(a.out)
dd = o - b
print(
    f"composed {a.out}: rows changed {int((dd != 0).sum()):,} (touched by >1 isolate {int((touch > 1).sum()):,}), RMS vs base {np.sqrt((dd ** 2).mean()):.2f} s"
)
