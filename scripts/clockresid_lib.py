#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""clock_resid_stage ): shared code for the clock-residual second stage on ADS-B-LINKED matched rows.
frame(ids)       features_v9 covariates + adsb/adsb2/adsb3 columns for the given MVT_IDs (fold or ranking rows)
design(Fr, base) numeric design matrix for a feature set ('clkonly' / 'full'), with dd_<clock> = base - clock
fit_predict      LightGBM (X1's fixed settings) on clip(y - base, +-1800), several seeds averaged
apply_corr       cf(base + s * clip(g, +-c)) on linked rows, base elsewhere
The airport code is a FIXED sorted mapping (fit and apply use the same codes)."""

import os, numpy as np, polars as pl, lightgbm as lgb

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FOLDS = ["months39", "months28", "months410", "janjul"]
PAIRS = FOLDS[:3]
TIMES = ["adsb_lb", "adsb_move", "adsb_hold", "adsb2_at", "adsb2_leave", "adsb3_mv_gs", "adsb3_mv_pos"]
BASECOV = [
    "MVT_ID_mvt",
    "is_rank",
    "y",
    "ADEP_mvt",
    "unmatched",
    "mvt_lobt",
    "mvt_aobt3",
    "mvt_eobt1",
    "mvt_sched",
    "hour",
    "MVT_TIME_UTC_mvt",
    "LOBT_flt",
]
ADSB = [
    f"{ROOT}/data/external/adsb_features.parquet",
    f"{ROOT}/data/external/adsb2_features.parquet",
    f"{ROOT}/data/external/adsb3_features.parquet",
]
PRM = dict(
    objective="l2",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=500,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
)


def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606))
    return np.maximum(v, 60.0)


def frame(ids):
    """one row per id (order of ids), covariates + all ADS-B columns; lk = adsb_linked (NULL -> 0)"""
    Q = pl.DataFrame({"MVT_ID_mvt": np.asarray(ids, dtype=float)})
    G = pl.scan_parquet(f"{ROOT}/data/features_v9.parquet").select(BASECOV).collect()
    G = G.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
    F = Q.join(G, on="MVT_ID_mvt", how="left", maintain_order="left")
    for t in ADSB:
        A = pl.read_parquet(t)
        A = A.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
        F = F.join(A, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert F.height == len(ids)
    return F.with_columns(lk=pl.col("adsb_linked").fill_null(0).cast(pl.Float64))


def clock_cols(F):
    return [
        c
        for c in F.columns
        if c.startswith(("adsb_", "adsb2_", "adsb3_")) and c != "adsb_linked" and F[c].dtype.is_numeric()
    ]


def design(F, base, fset, clk=None):
    """base: numpy array aligned with F. Returns float32 matrix and the column names."""
    clk = clk if clk is not None else clock_cols(F)
    F = F.with_columns(
        base=pl.Series(np.asarray(base, dtype=float)),
        apt=pl.col("ADEP_mvt").cast(pl.Utf8).replace_strict(APTS, list(range(len(APTS))), default=-1).cast(pl.Float64),
    )
    dd = [c for c in TIMES if c in F.columns]
    F = F.with_columns([(pl.col("base") - pl.col(c)).alias("dd_" + c) for c in dd])
    cc = clk + ["dd_" + c for c in dd]
    cols = {
        "full": ["base", "mvt_aobt3", "mvt_lobt", "mvt_eobt1", "mvt_sched", "hour", "apt"] + cc,
        "clkonly": ["base", "mvt_aobt3", "apt"] + cc,
        "ctrl": ["base", "mvt_aobt3", "mvt_lobt", "mvt_eobt1", "mvt_sched", "hour", "apt"],
    }[fset]
    return F.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy().astype(np.float32), cols


def fit_predict(Xtr, rtr, Xte, cols, seeds=(0, 1), nt=4, rounds=400):
    g = np.zeros(len(Xte))
    imp = np.zeros(len(cols))
    for s in seeds:
        m = lgb.train(
            dict(PRM, seed=s, num_threads=nt),
            lgb.Dataset(Xtr, rtr, categorical_feature=[cols.index("apt")], free_raw_data=True),
            rounds,
        )
        g += m.predict(Xte, num_threads=nt) / len(seeds)
        imp += m.feature_importance("gain")
    return g, imp


def apply_corr(base, g, lk, lo, s, c):
    return np.where(lk, cf(base + s * np.clip(g, -c, c), lo), base)
