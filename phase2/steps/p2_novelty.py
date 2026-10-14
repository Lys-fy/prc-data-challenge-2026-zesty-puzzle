#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the novelty mask for a new ranking period, combining the shipped 2025 mask with a freshly computed ranking-period
mask: mask = (shipped mask intersect 2025 ids) union (newly flagged intersect ranking ids). Inputs: SHIPPED_MASK.npy
(positional arg 1), scripts/dayset_novelty.py (run for its label-free mask logic), data/features_v9.parquet.
Output: OUT_MASK.npy (positional arg 2).

BIG_NOVELTY novelty mask for a new ranking period (cwd = workspace). Runs scripts/dayset_novelty.py's label-free mask part unchanged
(the §6c(i) diagnostic needs the big_ctx prod members and is skipped), then keeps the 2025 part of the shipped mask (argv[1]) so that the linked
fold side stays exactly the one the chains were fitted with: mask = (shipped mask ∩ 2025 ids) ∪ (new flagged ∩ ranking ids).
usage: p2_novelty.py SHIPPED_MASK.npy OUT_MASK.npy"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, numpy as np, polars as pl

SHIP, OUT = sys.argv[1:3]
src = open("scripts/dayset_novelty.py").read()
src = src.split("# §6c (i) re-run")[0]
g = {"__file__": "scripts/dayset_novelty.py", "__name__": "xds2_novelty_mask"}
exec(compile(src, "dayset_novelty.py", "exec"), g)
new = np.load("scratchpad/xds2/novelty_mask_ids.npy")
old = np.load(SHIP)
F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank"])
r = set(F.filter(pl.col("is_rank") == 1)["MVT_ID_mvt"].to_list())
t = set(F.filter(pl.col("is_rank") == 0)["MVT_ID_mvt"].to_list())
o25 = np.array([i for i in old if i in t])
n25 = np.array([i for i in new if i in t])
nrk = np.array([i for i in new if i in r])
print(
    f"2025 part: shipped {len(o25):,} vs recomputed {len(n25):,} (common {len(set(o25) & set(n25)):,}) -> shipped kept | ranking part: {len(nrk):,} rows flagged"
)
np.save(OUT, np.concatenate([o25, nrk]))
print("wrote", OUT, len(o25) + len(nrk))
