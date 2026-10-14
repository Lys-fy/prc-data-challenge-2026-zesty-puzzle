#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Public-period check file from the final file (docs/REPRODUCE.md §6b, step S6): the Jan/Jul rows of the final file on the 344,841
ids of the public ranking file (features table of the public period, is_rank == 1), joined on MVT_ID_mvt. The final ranking file re-jitters
the movement times by up to 12 s, so mvt_lobt moves too: the public file's LOBT law (|pred - mvt_lobt| <= 3,606 s, public mvt_lobt) is
applied (rows outside are clipped to the bound and counted), rint, floor 60. Then scripts/validate_submission.py on the public features.
usage: public_period_extract.py <final file> <public features_v9.parquet> <out public_janjul.parquet>      Refuses to overwrite.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, hashlib, subprocess, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
fin_p, feat_p, o = sys.argv[1:4]
assert not os.path.exists(o), f"refusing: {o} exists"
c = "TAXITIME_SEC_mvt"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
A = pl.read_parquet(fin_p)
print(f"{fin_p}: {A.height:,} rows, md5 {md5(fin_p)}")
F = pl.read_parquet(feat_p, columns=["MVT_ID_mvt", "is_rank", "unmatched", "ADEP_mvt"]).filter(pl.col("is_rank") == 1)
ids = F.select("MVT_ID_mvt")
out = ids.join(A, on="MVT_ID_mvt", how="left").select("MVT_ID_mvt", pl.col(c).cast(pl.Int32))
assert out.height == F.height and out[c].null_count() == 0, (out.height, out[c].null_count())
L = out.join(
    pl.read_parquet(feat_p, columns=["MVT_ID_mvt", "is_rank", "mvt_lobt"])
    .filter(pl.col("is_rank") == 1)
    .select("MVT_ID_mvt", "mvt_lobt"),
    on="MVT_ID_mvt",
    how="left",
    maintain_order="left",
)
y = L[c].to_numpy().astype(float)
lo = L["mvt_lobt"].to_numpy().astype(float)
h = ~np.isnan(lo)
yc = y.copy()
yc[h] = np.clip(y[h], lo[h] - 3606, lo[h] + 3606)
yc = np.maximum(np.rint(yc), 60)
print(f"LOBT law of the public file: {int((yc != y).sum())} rows clipped (max move {np.abs(yc - y).max():.0f} s)")
out = out.with_columns(pl.Series(c, yc).cast(pl.Int32))
os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
out.write_parquet(o)
print(f"wrote {o}: {out.height:,} rows, md5 {md5(o)}")
v = subprocess.run(
    [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_submission.py"),
        o,
        "--features",
        feat_p,
    ],
    capture_output=True,
    text=True,
)
print(v.stdout.strip()[-600:], v.stderr.strip()[-300:])
assert v.returncode == 0
