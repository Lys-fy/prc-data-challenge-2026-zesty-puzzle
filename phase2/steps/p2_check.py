#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Validates or compares TAXITIME_SEC_mvt prediction parquet files against the period's ranking id set
(data/features_v9.parquet). `validate <file>` checks schema, id completeness, no nulls, minimum 60 s, and the LOBT
law. `compare <file> <reference> [label] [--json out.json]` reports row-level diffs (overall, per class, per
airport) and byte-identity between the two files. Inputs: the file(s) named on the command line plus
data/features_v9.parquet. Output: stdout report, and with --json a JSON file of the compare result.

phase2_ws: validation and comparison helpers (period-agnostic version of validate_submission.py).
p2_check.py validate <file>                 schema {Float64, Int32}, exactly the period's ranking ids, no nulls, >= 60 s, LOBT law
p2_check.py compare <file> <reference> [label] [--json out.json]
      rows changed / RMS / mean / max |diff| overall and per class (matched, unm_other, unm_LIRF, exposed) and per airport;
      byte-identity (md5) of the two files."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, hashlib, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


def ranking():
    F = pl.read_parquet(
        "data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched", "ADEP_mvt", "mvt_sched", "mvt_lobt"]
    ).filter(pl.col("is_rank") == 1)
    ms0 = pl.col("mvt_sched").fill_null(0)
    return F.with_columns(
        mid=pl.col("MVT_ID_mvt").cast(pl.Float64),
        ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8),
        cls=pl.when(pl.col("unmatched") == 0)
        .then(pl.lit("matched"))
        .when(ms0 > 28800)
        .then(pl.lit("exposed"))
        .when(pl.col("ADEP_mvt") == "LIRF")
        .then(pl.lit("unm_LIRF"))
        .otherwise(pl.lit("unm_other")),
    ).select("mid", "cls", "ADEP_mvt", "mvt_lobt")


def validate(f):
    d = pl.read_parquet(f)
    assert d.schema == {"MVT_ID_mvt": pl.Float64, "TAXITIME_SEC_mvt": pl.Int32}, d.schema
    R = ranking()
    assert d.height == R.height and d["MVT_ID_mvt"].n_unique() == R.height, (d.height, R.height)
    j = R.join(d.with_columns(mid=pl.col("MVT_ID_mvt")), on="mid", how="inner")
    assert j.height == R.height, "id set differs from the ranking set"
    y = j["TAXITIME_SEC_mvt"].to_numpy()
    lo = j["mvt_lobt"].to_numpy().astype(float)
    assert d["TAXITIME_SEC_mvt"].null_count() == 0 and y.min() >= 60, "nulls or values below 60 s"
    viol = int((~np.isnan(lo) & (np.abs(y - lo) > 3606.5)).sum())
    assert viol == 0, f"{viol} LOBT-law violations"
    print(
        f"VALID {f}: {d.height:,} rows, mean {y.mean():.1f} s, min {y.min()}, max {y.max()}, md5 {hashlib.md5(open(f, 'rb').read()).hexdigest()}"
    )


def compare(f, r, label="", js=None):
    R = ranking()
    A = (
        pl.read_parquet(f)
        .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
        .select("mid", a=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    )
    B = (
        pl.read_parquet(r)
        .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
        .select("mid", b=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    )
    J = R.join(A, on="mid", how="left").join(B, on="mid", how="left")
    assert J["a"].null_count() == 0 and J["b"].null_count() == 0
    J = J.with_columns(d=pl.col("a") - pl.col("b"))
    same = hashlib.md5(open(f, "rb").read()).hexdigest() == hashlib.md5(open(r, "rb").read()).hexdigest()

    def st(D):
        d = D["d"].to_numpy()
        return {
            "n": int(len(d)),
            "changed": int((d != 0).sum()),
            "rms": float(np.sqrt((d**2).mean())) if len(d) else 0.0,
            "mean": float(d.mean()) if len(d) else 0.0,
            "max": float(np.abs(d).max()) if len(d) else 0.0,
        }

    out = {
        "file": f,
        "ref": r,
        "byte_identical": same,
        "all": st(J),
        "by_class": {c: st(J.filter(pl.col("cls") == c)) for c in ("matched", "unm_other", "unm_LIRF", "exposed")},
        "by_airport": {a_: st(J.filter(pl.col("ADEP_mvt") == a_)) for a_ in sorted(J["ADEP_mvt"].unique().to_list())},
    }
    s = out["all"]
    print(
        f"COMPARE {label or os.path.basename(f)} vs {os.path.basename(r)}: byte-identical {same} | rows changed {s['changed']:,} of {s['n']:,} | RMS {s['rms']:.4f} s | mean {s['mean']:+.4f} | max {s['max']:.0f}"
    )
    for c, v in out["by_class"].items():
        print(f"   {c:9s} n {v['n']:7,} changed {v['changed']:6,} RMS {v['rms']:8.4f} max {v['max']:6.0f}")
    if js:
        json.dump(out, open(js, "w"), indent=1)
    return out


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "validate":
        validate(sys.argv[2])
    else:
        js = None
        args = sys.argv[2:]
        if "--json" in args:
            i = args.index("--json")
            js = args[i + 1]
            args = args[:i] + args[i + 2 :]
        compare(args[0], args[1], args[2] if len(args) > 2 else "", js)
