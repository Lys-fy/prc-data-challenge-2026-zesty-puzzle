#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Applies a callsign-link lever arm to unmatched 'unm_other' ranking rows: on the rows the arm covers, value =
max(rint(cand), 60); every other unmatched row keeps its U2 (or U) value.
Inputs: <arm> (positional, selects scratchpad/xur2/ARM_<arm>.npz, written by unm_link_characterise.py ->
unm_link.py -> unm_link_lever.py), scratchpad/p2/U.parquet, scratchpad/p2/U2.parquet.
Output: scratchpad/p2/<out>.parquet (MVT_ID_mvt, u2), <out> given as the second positional arg (UA1 or U3).

Callsign-link levers of the unmatched rows (scripts/unm_link_lever.py arms) applied by rule, as a rule on the period's own rows
and the V3 arm (v84). Inputs: scratchpad/xur2/ARM_<arm>.npz written in this workspace by unm_link_characterise.py -> unm_link.py -> unm_link_lever.py
(its ranking reference is the U2 rule through the v76_p13T2x slot). Rule (= the two build scripts): on the ranking rows the arm covers
(isr & uo: unm_other), value = max(rint(cand), 60); every other unmatched row keeps its value.
Writes scratchpad/p2/<out>.parquet (MVT_ID_mvt, u2) = U2 overrides + the arm's rows.   usage: p2_unm_link_apply.py <arm> <out: UA1 | U3>
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
arm, out = sys.argv[1], sys.argv[2]
z = np.load(f"scratchpad/xur2/ARM_{arm}.npz", allow_pickle=True)
r = z["isr"] & z["uo"]
A = pl.DataFrame({"mid": z["ids"][r].astype(float), "a": np.maximum(np.rint(z["cand"][r]), 60).astype(np.int32)})
U = pl.read_parquet("scratchpad/p2/U.parquet").with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
U2 = (
    pl.read_parquet("scratchpad/p2/U2.parquet")
    .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
    .select("mid", "u2")
)
J = U.filter(pl.col("unmatched") == 1).join(U2, on="mid", how="left").join(A, on="mid", how="left")
assert J.filter(pl.col("a").is_not_null())["cls"].eq("unm_other").all(), "arm rows outside unm_other"
J = J.with_columns(base=pl.coalesce("u2", "u").cast(pl.Float64)).with_columns(
    new=pl.coalesce(pl.col("a").cast(pl.Float64), "base")
)
d = (J["new"] - J["base"]).to_numpy()
assert (np.abs(d) <= 7200).all()
J.filter(pl.col("new") != pl.col("u")).select("MVT_ID_mvt", pl.col("new").cast(pl.Int32).alias("u2")).write_parquet(
    f"scratchpad/p2/{out}.parquet"
)
print(
    f"UNM_LINK2 {arm}: arm rows {A.height}, unmatched rows changed vs U2 {(d != 0).sum()} (mean {d[d != 0].mean() if (d != 0).any() else 0:+.1f}, RMS over changed {np.sqrt((d[d != 0]**2).mean()) if (d != 0).any() else 0:.1f}, max {np.abs(d).max():.0f}) -> scratchpad/p2/{out}.parquet"
)
