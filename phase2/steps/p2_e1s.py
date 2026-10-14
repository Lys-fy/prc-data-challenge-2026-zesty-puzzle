#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Applies the E1s at-stand ADS-B clock rule to exposed, non-LIRF unmatched ranking rows: replaces TAXITIME_SEC_mvt
with max(rint(MVT_TIME - t_at_stand), 60) when that value is <= MAXCLK (env var, default 1822 s); other rows keep
their value. Inputs: --in prediction parquet, data/features_v9.parquet, scratchpad/xur2/link.parquet (built by
unm_link.py). Outputs: --out parquet (same schema as --in) and --table (the E1s rows applied).

E1s: the own at-stand ADS-B clock rule for exposed non-LIRF rows, as a period-generic rule.
Scope: ranking departures of class 'exposed' (unmatched, MVT - SCHED > 8 h) at airports other than LIRF that have a callsign link (scripts/unm_link.py) with an
at-stand clock (scratchpad/xur2/link.parquet, built in the workspace by unm_link.py in step pre_v87).
Rule: value = max(rint(MVT_TIME - t_at_stand), 60) if that value <= MAXCLK (default 1822 s = the largest at-stand clock on the 2025 rows,
i.e. the 2025 support; docs/METHOD.md §8: the cut was set after the 2026 clocks had been seen, and the rule failed its pre-declared criterion (b)).
Every other row keeps its value. Reads no label. Behind the driver flag E1S=0/1 (default 0, pending a decision; phase2/README.md §5).
usage: p2_e1s.py --in <file> --out <file> [--table scratchpad/p2/E1s.parquet]     (cwd = workspace)"""

import os, argparse, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="inp", required=True, help="input prediction parquet")
ap.add_argument("--out", required=True, help="output parquet path")
ap.add_argument("--table", default="scratchpad/p2/E1s.parquet", help="path to write the E1s rows that were applied")
a = ap.parse_args()
MAXCLK = float(os.environ.get("MAXCLK", "1822"))
F = pl.read_parquet(
    "data/features_v9.parquet",
    columns=["MVT_ID_mvt", "is_rank", "unmatched", "mvt_sched", "ADEP_mvt", "MVT_TIME_UTC_mvt"],
).filter(pl.col("is_rank") == 1)
F = F.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").cast(pl.Utf8))
E = F.filter(
    (pl.col("unmatched") == 1) & (pl.col("mvt_sched").fill_null(0) > 28800) & (pl.col("ADEP_mvt") != "LIRF")
)  # unm_resid_lib class 'exposed', non-LIRF
LK = (
    pl.read_parquet("scratchpad/xur2/link.parquet")
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    .select("MVT_ID_mvt", "hex", "t_at_stand")
)
E = E.join(LK, on="MVT_ID_mvt", how="left").with_columns(
    tk=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)
)
E = E.with_columns(clk=pl.col("tk") - pl.col("t_at_stand"))
Q = E.filter(pl.col("clk").is_not_null())
Q = Q.with_columns(new=pl.Series(np.maximum(np.rint(Q["clk"].to_numpy()), 60)))  # max(rint(MVT - t_at_stand), 60)
T = Q.filter(pl.col("new") <= MAXCLK).select(
    pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col("new").cast(pl.Int64).alias("e1s")
)
print(
    f"E1s: non-LIRF exposed ranking rows {E.height}, linked {int(E['hex'].is_not_null().sum())}, with an at-stand clock {Q.height}, "
    f"<= MAXCLK {MAXCLK:.0f} s: {T.height} (cut {Q.height - T.height})"
)
os.makedirs(os.path.dirname(a.table), exist_ok=True)
T.write_parquet(a.table)
B = pl.read_parquet(a.inp)
c = "TAXITIME_SEC_mvt"
pos = {v: i for i, v in enumerate(B["MVT_ID_mvt"].to_numpy().astype(float))}
out = B[c].to_numpy().copy()
ix = np.array([pos[v] for v in T["MVT_ID_mvt"].to_numpy()], dtype=np.int64)
old = out[ix].astype(float)
out[ix] = T["e1s"].to_numpy().astype(out.dtype)
d = out[ix].astype(float) - old
assert (np.abs(d) <= 86400).all()
res = B.with_columns(pl.Series(c, out).cast(B.schema[c]))
assert res.schema == B.schema
res.write_parquet(a.out)
print(
    f"E1s applied: rows changed {int((d != 0).sum())}, mean {d[d != 0].mean() if (d != 0).any() else 0:+.0f}, max |d| {np.abs(d).max() if len(d) else 0:.0f} -> {a.out}"
)
