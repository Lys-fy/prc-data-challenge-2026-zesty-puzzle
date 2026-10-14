# SPDX-License-Identifier: GPL-3.0-only
"""Command line: `evaluate` on 2025 month-pair folds, `predict` a ranking file into a submission.

    python -m baseline.run evaluate --raw-dir data/raw --out data/baseline
    python -m baseline.run predict  --raw-dir data/raw --ranking data/final/final_ranking.parquet --out data/baseline"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import data, features, model, rule
from .config import DEFAULT_FOLDS, FOLDS


def rmse(err: np.ndarray) -> float:
    """Root mean squared error, NaN for an empty vector."""
    return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")


def with_rule(m: model.Model, dep: pl.DataFrame, base: np.ndarray, links: pl.DataFrame) -> tuple[np.ndarray, int]:
    """The schedule-copy rule on top of the model prediction (it runs last, as in the full pipeline)."""
    return rule.apply(dep, base, links, m.off_med.get(rule.AIRPORT, 0.0))


def evaluate(args: argparse.Namespace) -> list[dict]:
    """Fit without the two held-out months, score them; RMSE overall, matched, unmatched, and without the rule."""
    rows = data.load(data.training_files(args.raw_dir))
    dep, links = features.build(rows), rule.link(rows)
    results = []
    for name in args.folds.split(","):
        t0, held = time.time(), pl.col("month").is_in(list(FOLDS[name]))
        test = dep.filter(held)
        m = model.fit(dep.filter(~held), args.threads)
        base, y = model.predict(m, test), test["TAXITIME_SEC_mvt"].to_numpy()
        pred, n_rule = with_rule(m, test, base, links)
        err, no_rule, matched = pred - y, base - y, test["matched"].to_numpy().astype(bool)
        results.append({"fold": name, "rows": test.height, "rmse": rmse(err), "rmse_matched": rmse(err[matched]),
                        "rmse_matched_anchor_only": rmse((features.anchor(test, m.fallback) - y)[matched]),
                        "rmse_unmatched": rmse(err[~matched]), "rmse_unmatched_no_rule": rmse(no_rule[~matched]),
                        "unmatched_rows": int((~matched).sum()), "rule_rows": n_rule, "seconds": round(time.time() - t0, 1)})
        print(json.dumps(results[-1]), flush=True)
    return results


def predict(args: argparse.Namespace) -> pl.DataFrame:
    """Fit on all 2025 departures and write the submission for the ranking file (MVT_ID_mvt Float64, TAXITIME_SEC_mvt Int32)."""
    m = model.fit(features.build(data.load(data.training_files(args.raw_dir))), args.threads)
    rows = data.load([args.ranking])
    dep = features.build(rows)
    pred, n_rule = with_rule(m, dep, model.predict(m, dep), rule.link(rows))
    sub = dep.select(pl.col("MVT_ID_mvt").cast(pl.Float64), TAXITIME_SEC_mvt=pl.Series(np.rint(pred)).cast(pl.Int32))
    assert sub.height == sub["MVT_ID_mvt"].n_unique() and sub["TAXITIME_SEC_mvt"].min() >= 60
    print(f"ranking departures {sub.height:,}; schedule-copy rule rows {n_rule}; mean {sub['TAXITIME_SEC_mvt'].mean():.1f} s")
    return sub


def main() -> None:
    """Parse the command line and run one sub-command; outputs go to --out (keep it outside the repository or in data/)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["evaluate", "predict"])
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="folder with training_*.parquet")
    p.add_argument("--ranking", type=Path, default=Path("data/final/final_ranking.parquet"), help="ranking file to predict")
    p.add_argument("--out", type=Path, default=Path("data/baseline"), help="output folder (git-ignored under data/)")
    p.add_argument("--folds", default=DEFAULT_FOLDS, help=f"comma list of {','.join(FOLDS)}")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.command == "evaluate":
        (args.out / "folds.json").write_text(json.dumps(evaluate(args), indent=1))
    else:
        predict(args).write_parquet(args.out / f"baseline_{args.ranking.stem}.parquet")
    print(f"{args.command} done in {time.time() - t0:.0f} s; outputs in {args.out}")


if __name__ == "__main__":
    main()
