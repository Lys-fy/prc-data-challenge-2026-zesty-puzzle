# SPDX-License-Identifier: GPL-3.0-only
"""The clock cap MAXCLK of the E1s rule (phase2/steps/p2_e1s.py, default 1,822 s): the largest at-stand clock on the 2025 rows the rule covers.
Rows: 2025 training departures of class 'exposed' (unmatched, MVT - SCHED > 8 h) at airports other than LIRF that have a callsign link with an
at-stand clock in scratchpad/xur2/link.parquet (scripts/unm_link.py). Clock = max(rint(MVT_TIME - t_at_stand), 60), the value E1s would assign.
Cap = the maximum of that clock over these rows (the 2025 support of the rule). Also prints the label check of the same rows (y - clock).
usage: e1s_clock_cap.py   (cwd = repository root or PRC_ROOT)"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, json, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
SHIPPED = 1822.0
F = pl.read_parquet(
    "data/features_v9.parquet",
    columns=["MVT_ID_mvt", "is_rank", "unmatched", "mvt_sched", "ADEP_mvt", "MVT_TIME_UTC_mvt", "y"],
).filter(pl.col("is_rank") == 0)
F = F.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("ADEP_mvt").cast(pl.Utf8))
E = F.filter(
    (pl.col("unmatched") == 1) & (pl.col("mvt_sched").fill_null(0) > 28800) & (pl.col("ADEP_mvt") != "LIRF")
)  # class 'exposed', non-LIRF, as in p2_e1s.py
LK = (
    pl.read_parquet("scratchpad/xur2/link.parquet")
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    .select("MVT_ID_mvt", "hex", "t_at_stand")
)
E = E.join(LK, on="MVT_ID_mvt", how="left").with_columns(
    tk=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)
)
Q = E.with_columns(clk=pl.col("tk") - pl.col("t_at_stand")).filter(pl.col("clk").is_not_null())
clk = np.maximum(np.rint(Q["clk"].to_numpy()), 60)
y = Q["y"].to_numpy().astype(float)
e = y - clk
cap = float(clk.max()) if len(clk) else float("nan")
print(
    f"2025 non-LIRF exposed rows {E.height}, linked {int(E['hex'].is_not_null().sum())}, with an at-stand clock {Q.height}; clock range {clk.min():.0f}-{cap:.0f} s"
)
print(
    "label check e = y - clock: min %.0f median %.0f max %.0f, rms %.0f"
    % (e.min(), np.median(e), e.max(), np.sqrt((e**2).mean()))
)
print(f"MAXCLK = {cap:.0f} s (shipped {SHIPPED:.0f}, |diff| {abs(cap - SHIPPED):.0f})")
os.makedirs("scratchpad/xur2", exist_ok=True)
json.dump(dict(n=Q.height, maxclk=cap), open("scratchpad/xur2/e1s_clock_cap.json", "w"))
print("wrote scratchpad/xur2/e1s_clock_cap.json")
