#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Applies the label-free px_safe rule to exposed LIRF rows: moves TAXITIME_SEC_mvt to a frozen-blend-derived
candidate value when a break-even test on three label scenarios (SUB/ORD/DS) favours the move; other rows are
unchanged. Inputs: --in prediction parquet, --u (per-row cls/cross/fzr_* features). Outputs: --out parquet
(same schema as --in) and --ledger (per-row decision detail for exposed LIRF rows).

PX-safe rule for exposed LIRF rows (px_safe), computed label-free from the features of the period's own rows; the logic is that of
scripts/exposed_mechanisms.py Part 3 '2026 ledger'.

Scope: exposed LIRF rows (unmatched, MVT-SCHED > 8 h, ADEP = LIRF). Per row, from the frozen-blend decomposition only:
  px  = rint(frozen art) if the row crosses a local midnight between SCHED and MVT, else rint(MVT - SCHED); floor 60
  label scenarios (no labels used): SUB y = MS | ORD y = 1,000 s | DS y = 86,400 + (MS > 24 h ? MS - 86,400 + 1,000 : 1,000)
  dS, dO, dD = (px - y)^2 - (v - y)^2 per scenario, v = the current (frozen) value
  gNon = -((1 - q) dS + q dD) for crossers (q = frozen day-shift share), -dS otherwise; break-even ORD prob. be = gNon / (gNon + dO)
  move iff |px - v| >= 1 and gNon > 0 and be >= 2 * ub, ub = 2025 upper bound of the ORD share (crossers 0.072, non-crossers 0.259)
usage: p2_exposed_scenarios.py --in <file> --out <file> [--u scratchpad/p2/U.parquet] [--ledger <parquet>]"""

import os, argparse, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
UB_CROSS, UB_NONCROSS = 0.072, 0.259
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="inp", required=True, help="input prediction parquet")
ap.add_argument("--out", required=True, help="output parquet path")
ap.add_argument("--u", default="scratchpad/p2/U.parquet", help="per-row cls/cross/fzr_* feature parquet")
ap.add_argument(
    "--ledger", default="scratchpad/p2/xhr_ledger.parquet", help="path to write the per-row decision ledger"
)
a = ap.parse_args()
B = pl.read_parquet(a.inp)
col = "TAXITIME_SEC_mvt"
U = pl.read_parquet(a.u).with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64)).drop("MVT_ID_mvt")
J = B.with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64)).join(U, on="mid", how="left", maintain_order="left")
assert J.height == B.height and J["cls"].null_count() == 0
v0 = B[col].to_numpy()
sc = ((J["cls"] == "exposed") & (J["ADEP_mvt"] == "LIRF")).to_numpy()
ms = J["mvt_sched"].to_numpy().astype(float)
cr = J["cross"].to_numpy().astype(bool)
q = J["fzr_q"].to_numpy().astype(float)
art = J["fzr_art"].to_numpy().astype(float)
v = v0.astype(float)
px = np.maximum(np.rint(np.where(cr, art, ms)), 60.0)
yS, yO, yD = ms, np.full_like(ms, 1000.0), 86400.0 + np.where(ms > 86400, ms - 86400 + 1000, 1000.0)
dS = (px - yS) ** 2 - (v - yS) ** 2
dO = (px - yO) ** 2 - (v - yO) ** 2
dD = (px - yD) ** 2 - (v - yD) ** 2
mv = np.abs(px - v) >= 1
with np.errstate(invalid="ignore", divide="ignore"):
    gNon = -np.where(cr, (1 - q) * dS + q * dD, dS)
    be = np.where(gNon > 0, gNon / np.maximum(gNon + dO, 1e-9), 0.0)
ub = np.where(cr, UB_CROSS, UB_NONCROSS)
g1 = sc & mv & (be >= 2 * ub) & (gNon > 0)
out = v0.copy()
out[g1] = np.maximum(px[g1], 60).astype(v0.dtype)
o = B.with_columns(pl.Series(col, out).cast(B.schema[col]))
assert o.schema == B.schema and o["MVT_ID_mvt"].equals(B["MVT_ID_mvt"])
o.write_parquet(a.out)
L = (
    J.filter(pl.Series(sc))
    .select("MVT_ID_mvt", "FLIGHT_mvt", "mvt_sched", "cross", "fzr_p", "fzr_q", "fzr_art")
    .with_columns(v=v[sc], px=px[sc], moved=mv[sc], breakeven_ORD=be[sc], G1_pass=g1[sc], new=out[sc])
)
L.write_parquet(a.ledger)
print(
    f"px_safe PX-safe: exposed LIRF rows {int(sc.sum())}, crossers {int((sc & cr).sum())}; moved {int(g1.sum())} (non-crossers -> MS {int((g1 & ~cr).sum())}, crossers -> art {int((g1 & cr).sum())}); moves: {sorted((out[g1].astype(int) - v0[g1].astype(int)).tolist())}"
)
