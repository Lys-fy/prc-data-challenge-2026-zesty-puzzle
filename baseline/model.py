# SPDX-License-Identifier: GPL-3.0-only
"""One LightGBM model on the residual y - anchor, with the output bounds of docs/METHOD.md."""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl

from . import features as F
from .config import FLOOR_S, LGB_PARAMS, LOBT_BOUND_S, NUM_ROUNDS


@dataclass
class Model:
    """A fitted booster plus what it needs to encode new rows: category codes, anchor fallbacks and the rule's off_med."""
    booster: lgb.Booster
    vocab: dict[str, dict[str, int]]
    fallback: dict[str, float]
    off_med: dict[str, float]


def matrix(dep: pl.DataFrame, vocab: dict[str, dict[str, int]]) -> np.ndarray:
    """Feature matrix: categories as integer codes (unseen or null -> missing), numbers as float."""
    cats = [pl.col(c).cast(pl.String).replace_strict(v, default=None, return_dtype=pl.Int32) for c, v in vocab.items()]
    return dep.select(*cats, *F.NUM_COLS).cast(pl.Float64).to_numpy()


def airport_medians(values: pl.Expr, dep: pl.DataFrame) -> dict[str, float]:
    """Per-airport median of an expression over the given departures."""
    return dict(dep.group_by("ADEP_mvt").agg(values.median()).iter_rows())


def fit(train: pl.DataFrame, threads: int) -> Model:
    """Fit on labelled departures; the target is the residual to the anchor (METHOD §1)."""
    y = pl.col("TAXITIME_SEC_mvt").cast(pl.Float64)
    fallback = airport_medians(y, train.filter(~pl.col("matched").cast(pl.Boolean)))
    off_med = airport_medians(y - pl.col("mvt_aobt3"), train.filter(pl.col("mvt_aobt3").is_not_null()))
    vocab = {c: {v: i for i, v in enumerate(train[c].cast(pl.String).drop_nulls().unique().sort())} for c in F.CAT_COLS}
    resid = train["TAXITIME_SEC_mvt"].to_numpy() - F.anchor(train, fallback)
    data = lgb.Dataset(matrix(train, vocab), resid, categorical_feature=list(range(len(F.CAT_COLS))), free_raw_data=True)
    booster = lgb.train({**LGB_PARAMS, "num_threads": threads}, data, num_boost_round=NUM_ROUNDS)
    return Model(booster, vocab, fallback, off_med)


def predict(model: Model, dep: pl.DataFrame) -> np.ndarray:
    """Anchor + predicted residual, kept within LOBT +- 3,606 s where LOBT exists, floored at 60 s."""
    pred = F.anchor(dep, model.fallback) + model.booster.predict(matrix(dep, model.vocab))
    lobt = dep["mvt_lobt"].fill_null(np.nan).to_numpy()
    pred = np.where(np.isnan(lobt), pred, np.clip(pred, lobt - LOBT_BOUND_S, lobt + LOBT_BOUND_S))
    return np.maximum(pred, FLOOR_S)
