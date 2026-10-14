#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""phase2_ws: layer-0 reproducibility. Compare a features table rebuilt from the raw files by the current
scripts/build_features.py with the shared data/features_v9.parquet (the table every member and stage was fitted on).
Per column: identical / max |diff| / rows differing / null-pattern differences; rows matched by MVT_ID_mvt.
usage: p2_features_check.py <rebuilt.parquet> <reference.parquet> [--json out.json]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
a, b = sys.argv[1], sys.argv[2]
js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
sa, sb = pl.read_parquet_schema(a), pl.read_parquet_schema(b)
only_a, only_b = [c for c in sa if c not in sb], [c for c in sb if c not in sa]
common = [c for c in sb if c in sa]
ia = pl.read_parquet(a, columns=["MVT_ID_mvt"]).with_row_index("ia")
ib = pl.read_parquet(b, columns=["MVT_ID_mvt"]).with_row_index("ib")
J = ib.join(ia, on="MVT_ID_mvt", how="left")
miss = int(J["ia"].null_count())
print(
    f"rows: rebuilt {ia.height:,} reference {ib.height:,}; reference rows missing in rebuilt {miss:,}; columns only rebuilt {only_a}, only reference {only_b}"
)
J = J.filter(pl.col("ia").is_not_null())
ka, kb = J["ia"].to_numpy(), J["ib"].to_numpy()
res = {}
bad = []
for i in range(0, len(common), 40):
    cols = [c for c in common[i : i + 40] if c != "MVT_ID_mvt"]
    A = pl.read_parquet(a, columns=cols)[ka]
    B = pl.read_parquet(b, columns=cols)[kb]
    for c in cols:
        x, y = A[c], B[c]
        nd = int((x.is_null() != y.is_null()).sum())
        if x.dtype.is_numeric() and y.dtype.is_numeric():
            xv = x.cast(pl.Float64).to_numpy()
            yv = y.cast(pl.Float64).to_numpy()
            m = ~np.isnan(xv) & ~np.isnan(yv)
            d = np.abs(xv[m] - yv[m])
            diff = int((d > 1e-9 * np.maximum(1, np.abs(yv[m]))).sum())
            mx = float(d.max()) if d.size else 0.0
        else:
            eq = (x.cast(pl.Utf8) == y.cast(pl.Utf8)).fill_null(True)
            diff = int((~eq).sum())
            mx = None
        res[c] = {"null_pattern_diff": nd, "rows_diff": diff, "max_abs": mx, "dtype": [str(x.dtype), str(y.dtype)]}
        if nd or diff:
            bad.append(c)
print(f"columns compared {len(res)}; identical {len(res) - len(bad)}; differing {len(bad)}")
for c in bad[:80]:
    r = res[c]
    print(
        f"   {c:40s} rows diff {r['rows_diff']:9,} null-pattern diff {r['null_pattern_diff']:9,} max|d| {r['max_abs']}"
    )
if js:
    json.dump(
        {"only_rebuilt": only_a, "only_reference": only_b, "missing_rows": miss, "columns": res},
        open(js, "w"),
        indent=1,
    )
