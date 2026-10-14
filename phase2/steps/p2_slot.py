#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds one submission-shaped stage file (e.g. stages/unmatched_rule.parquet) from by-rule parts, ascending by
MVT_ID_mvt: unmatched rows come from scratchpad/p2/U(2/A1/3).parquet, matched rows from --matched or a C1
placeholder. Reads no submission file.
Inputs: --out, --unmatched (U/U2/UA1/U3/...), --matched (optional); scratchpad/p2/U.parquet and
data/features_v9.parquet. Output: --out parquet, schema {MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32}.

phase2_ws: compose a submission-shaped stage file inside a workspace from by-rule parts (stages/unmatched_rule.parquet,
stages/fallback_base.parquet, stages/unmatched_resid.parquet, stages/unmatched_link_a1.parquet). The stage scripts read these files by
fixed names; this script builds them BY RULE in the workspace and reads no submission file.

  --unmatched U   unmatched rows = scratchpad/p2/U.parquet column u (C1 / frozen blend rules)
  --unmatched U2  unmatched rows = U with the unm_resid model applied (scratchpad/p2/U2.parquet column u2); likewise UA1 (+ callsign-link arm A1), U3 (+ callsign-link arm V3)
  --matched <parquet>   matched rows from this file (same schema); default: the C1 rule on every row (a placeholder that satisfies the
                        submission law; stage scripts read a stage file's MATCHED rows only where noted, e.g. the fallback rows of fallback_base)
Row order: ascending MVT_ID_mvt (the stage scripts assert equal id order between files).
Schema: {MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32}.
usage: p2_slot.py --out stages/unmatched_rule.parquet --unmatched U [--matched <file>]"""

import os, argparse, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True, help="output stage parquet path")
ap.add_argument("--unmatched", default="U", help="unmatched-row source: U, U2, UA1, U3, ... or a parquet path")
ap.add_argument("--matched", default="", help="matched-row source parquet; default: C1 placeholder")
a = ap.parse_args()
U = pl.read_parquet("scratchpad/p2/U.parquet").with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64)).sort("mid")
if a.unmatched != "U":  # U2 / UA1 / U3 ...: overrides (column u2) on top of U, e.g. scratchpad/p2/U2.parquet
    f = a.unmatched if a.unmatched.endswith(".parquet") else f"scratchpad/p2/{a.unmatched}.parquet"
    U2 = pl.read_parquet(f).with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64)).select("mid", "u2")
    U = (
        U.join(U2, on="mid", how="left", maintain_order="left")
        .with_columns(u=pl.when(pl.col("unmatched") == 1).then(pl.coalesce("u2", "u")).otherwise(pl.col("u")))
        .drop("u2")
    )
order = U.select("mid")  # sorted by id: every stage file is written in ascending MVT_ID order
J = order.join(U.select("mid", "unmatched", "u", "c1"), on="mid", how="left", maintain_order="left")
val = np.maximum(np.rint(J["c1"].to_numpy().astype(float)), 60.0)  # placeholder for matched rows
if a.matched:
    M = (
        pl.read_parquet(a.matched)
        .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
        .select("mid", pl.col("TAXITIME_SEC_mvt").alias("mv"))
    )
    mv = order.join(M, on="mid", how="left", maintain_order="left")["mv"].to_numpy().astype(float)
    assert np.isfinite(mv).all(), "matched file misses ids"
    val = mv
um = J["unmatched"].to_numpy() == 1
val = np.where(um, J["u"].to_numpy().astype(float), val)
assert np.isfinite(val).all()
lo = (
    order.join(
        pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "mvt_lobt"])
        .filter(pl.col("is_rank") == 1)
        .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
        .select("mid", "mvt_lobt"),
        on="mid",
        how="left",
        maintain_order="left",
    )["mvt_lobt"]
    .to_numpy()
    .astype(float)
)
if not a.matched:  # placeholder must obey the LOBT law like every submission
    val = np.where(
        um | np.isnan(lo), val, np.maximum(np.clip(val, np.nan_to_num(lo) - 3606, np.nan_to_num(lo) + 3606), 60.0)
    )
out = pl.DataFrame(
    {"MVT_ID_mvt": order["mid"], "TAXITIME_SEC_mvt": pl.Series(val.astype(np.int32))},
    schema={"MVT_ID_mvt": pl.Float64, "TAXITIME_SEC_mvt": pl.Int32},
)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
out.write_parquet(a.out)
print(
    f"stage file {a.out}: rows {out.height:,}, unmatched {int(um.sum()):,} from {a.unmatched}, matched from {a.matched or 'C1 placeholder'}"
)
