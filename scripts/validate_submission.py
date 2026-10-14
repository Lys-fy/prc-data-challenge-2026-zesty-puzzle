#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fail-closed check of a submission file against the ranking rows of a features table; exit status 1 on any failure.
Inputs: <candidate.parquet> [reference.parquet] [--features <features_v9.parquet>] (default $PRC_ROOT/data/features_v9.parquet).
Output: one OK line (rows, mean, min, max) and, with a reference file, a diff summary.

Checks: schema {MVT_ID_mvt: Float64, TAXITIME_SEC_mvt: Int32}, exactly the ranking ids of the features table (is_rank == 1), no nulls,
no value below 60 s, LOBT law (|pred - mvt_lobt| <= 3606 where mvt_lobt exists). The expected row count comes from the features table
(P2_NRANK, when set, must agree with it). File names are free; a name other than the upload forms zesty-puzzle_final.parquet /
zesty-puzzle_vN.parquet (integer N) only prints a note.
usage: validate_submission.py <candidate.parquet> [reference.parquet] [--features <features_v9.parquet>]"""

import os, re, sys, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR = 60  # s, competition floor
LOBT_BOUND = 3606.5  # s, |pred - mvt_lobt| above this is a violation (LOBT law 3,600 s + 6 s tolerance)


def check(d, F, nrank=None, feat="features"):
    """Assert the checks above. d: submission frame; F: features rows with is_rank == 1 (MVT_ID_mvt, mvt_lobt);
    nrank: P2_NRANK value or None. Returns (n, y) = number of ranking rows and the predictions in F's row order."""
    assert d.schema == {"MVT_ID_mvt": pl.Float64, "TAXITIME_SEC_mvt": pl.Int32}, d.schema
    n = F.height
    if nrank:
        assert int(nrank) == n, f"P2_NRANK {nrank} != {n} ranking rows in {feat}"
    assert len(d) == n and d["MVT_ID_mvt"].n_unique() == n, (len(d), n)
    j = F.join(d, on="MVT_ID_mvt", how="inner")
    assert len(j) == n, "id set differs from the ranking set"
    y = j["TAXITIME_SEC_mvt"].to_numpy()
    lo = j["mvt_lobt"].to_numpy().astype(float)
    assert (
        d["TAXITIME_SEC_mvt"].null_count() == 0 and np.isfinite(y).all() and y.min() >= FLOOR
    ), "nulls or values below the 60 s floor"
    viol = int((~np.isnan(lo) & (np.abs(y - lo) > LOBT_BOUND)).sum())
    assert viol == 0, f"{viol} LOBT-law violations"
    return n, y


def main(argv):
    if argv[:1] in (["-h"], ["--help"]):
        sys.exit(print(__doc__))
    a = list(argv)
    FEAT = f"{ROOT}/data/features_v9.parquet"
    if "--features" in a:
        i = a.index("--features")
        FEAT = a[i + 1]
        del a[i : i + 2]
    cand = a[0]
    ref = a[1] if len(a) > 1 else None
    name = os.path.basename(cand)
    if not re.fullmatch(r"zesty-puzzle_(v\d+|final)\.parquet", name):
        print(f"note: {name} is not an upload name (zesty-puzzle_final.parquet / zesty-puzzle_vN.parquet)")
    d = pl.read_parquet(cand)
    F = pl.read_parquet(FEAT, columns=["MVT_ID_mvt", "is_rank", "mvt_lobt"]).filter(pl.col("is_rank") == 1)
    n, y = check(d, F, os.environ.get("P2_NRANK"), FEAT)
    print(f"OK {name}: {n:,} rows, mean {y.mean():.1f} s, min {y.min()}, max {y.max()}")
    if ref:
        r = pl.read_parquet(ref).rename({"TAXITIME_SEC_mvt": "ref"})
        jj = d.join(r, on="MVT_ID_mvt")
        dd = (jj["TAXITIME_SEC_mvt"] - jj["ref"]).to_numpy().astype(float)
        print(
            f"   vs {os.path.basename(ref)}: rows changed {int((dd != 0).sum()):,}, RMS change {np.sqrt((dd**2).mean()):.1f} s,"
            f" mean shift {dd.mean():+.2f} s, max |change| {np.abs(dd).max():.0f} s"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
