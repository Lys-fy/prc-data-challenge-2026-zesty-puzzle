# SPDX-License-Identifier: GPL-3.0-only
"""The two ORD-share bounds of the scenario-safe rule on exposed LIRF rows (phase2/steps/p2_exposed_scenarios.py UB_CROSS, UB_NONCROSS).
Rows: the 2025 exposed LIRF departures of the context table (cls == 'exposed', ADEP = LIRF: unmatched, MVT - SCHED > 8 h), split by crossing
(the local calendar day of MVT differs from that of SCHED, cross >= 1). Per class: k = rows with an ORD label (60 <= y <= 5,400 s and not SUB/DS),
n = rows; bound = one-sided 95 % exact binomial (Clopper-Pearson) upper limit of the ORD share, i.e. the p with P(X <= k | n, p) = 0.05
(for k = 0 this is 1 - 0.05^(1/n)). The rule moves a row only if its break-even ORD probability is at least twice this bound.
usage: exposed_ord_bounds.py [context table, default scratchpad/xhr/unm2025b.parquet]   (cwd = repository root or PRC_ROOT)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, math, json, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
SHIPPED = {"cross": 0.072, "noncross": 0.259}


def cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def upper(k, n, alpha=0.05):  # bisection on P(X <= k | n, p) = alpha; exact for any k < n
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


f = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/xhr/unm2025b.parquet"
X = (
    pl.read_parquet(f, columns=["cls", "ADEP_mvt", "cross", "regime"])
    .filter((pl.col("cls") == "exposed") & (pl.col("ADEP_mvt") == "LIRF"))
    .with_columns(cr=pl.col("cross") >= 1)
)
print("2025 exposed LIRF rows by crossing x regime:", sorted(X.group_by("cr", "regime").len().rows()))
out = {}
for tag, cr in (("cross", True), ("noncross", False), ("all", None)):
    s = X if cr is None else X.filter(pl.col("cr") == cr)
    n = s.height
    k = int((s["regime"] == "ORD").sum())
    out[tag] = dict(n=n, k=k, ub=upper(k, n))
    print(
        f"{tag:9s} ORD {k}/{n} -> 95 % upper bound {out[tag]['ub']:.5f}"
        + (
            f" (shipped {SHIPPED[tag]}, |diff| after rounding to 3 decimals {abs(round(out[tag]['ub'], 3) - SHIPPED[tag]):.3g})"
            if tag in SHIPPED
            else ""
        )
    )
os.makedirs("scratchpad/xhr", exist_ok=True)
json.dump(out, open("scratchpad/xhr/exposed_ord_bounds.json", "w"), indent=1)
print("wrote scratchpad/xhr/exposed_ord_bounds.json")
