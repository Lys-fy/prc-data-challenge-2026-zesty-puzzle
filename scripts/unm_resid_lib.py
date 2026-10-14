# SPDX-License-Identifier: GPL-3.0-only
"""Shared loaders for the unmatched-row study. 2025 unmatched rows (features_v9 unmatched==1, is_rank==0), fold by month."""

import os, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + "")
os.chdir(ROOT)
FOLDS = {
    "months39": (3, 9),
    "months28": (2, 8),
    "months410": (4, 10),
    "janjul": (1, 7),
    "months511": (5, 11),
    "months612": (6, 12),
}
GATE = ["months39", "months28", "months410", "janjul"]
SELF = ["months511", "months612"]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
RN = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)


def find(M, f):
    pats = [f"data/oof/{M}_{f}_*.npy", f"data/oof/{M}_{f}.npy"] + (
        [f"data/oof/{M}_months17.npy", f"data/oof/{M}_janjul.npy"] if f == "janjul" else []
    )
    for p in pats:
        g = sorted(glob.glob(p))
        if g:
            return g[0]
    raise FileNotFoundError(M + " " + f)


def cls_expr():
    ms = pl.col("mvt_sched").fill_null(0)
    return (
        pl.when(ms > 28800)
        .then(pl.lit("exposed"))
        .when(pl.col("ADEP_mvt") == "LIRF")
        .then(pl.lit("unm_LIRF"))
        .otherwise(pl.lit("unm_other"))
        .alias("cls")
    )


def frame(rank=False):
    F = pl.read_parquet(
        "data/features_v9.parquet",
        columns=["MVT_ID_mvt", "is_rank", "unmatched", "ADEP_mvt", "y", "mvt_sched", "MVT_TIME_UTC_mvt", "mvt_lobt"],
    ).filter((pl.col("unmatched") == 1) & (pl.col("is_rank") == (1 if rank else 0)))
    A = pl.read_parquet("data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"])
    F = F.join(A, on="MVT_ID_mvt", how="left").with_columns(
        cls_expr(), lk=pl.col("adsb_linked").fill_null(0).cast(pl.Int8), mon=pl.col("MVT_TIME_UTC_mvt").dt.month()
    )
    inv = {m: f for f, ms in FOLDS.items() for m in ms}
    F = F.with_columns(fold=pl.col("mon").replace_strict(inv, default="none"))
    return F.sort("MVT_ID_mvt")


def member(F, M, row=1):
    """fold OOF of member M (row index) aligned to F; NaN where missing"""
    out = np.full(F.height, np.nan)
    ids = F["MVT_ID_mvt"].to_numpy().astype(float)
    pos = {v: i for i, v in enumerate(ids)}
    for f in FOLDS:
        try:
            a = np.load(find(M, f))
        except FileNotFoundError:
            continue
        if row >= a.shape[0]:
            continue
        ix = np.array([pos.get(i, -1) for i in a[0]])
        k = ix >= 0
        fm = F["fold"].to_numpy()[ix[k]] == f
        out[ix[k][fm]] = a[row][k][fm]
    return out


def rank_weights(F, R):
    """ranking-mix weights for fold rows of class unm_other: n_rank(apt,lk)/n_fold(apt,lk) per fold (fallback apt)"""
    w = np.zeros(F.height)
    fo = F["fold"].to_numpy()
    a = F["ADEP_mvt"].to_numpy()
    l = F["lk"].to_numpy()
    c = F["cls"].to_numpy()
    Ro = R.filter(pl.col("cls") == "unm_other")
    ra = Ro["ADEP_mvt"].to_numpy()
    rl = Ro["lk"].to_numpy()
    for f in FOLDS:
        s = (fo == f) & (c == "unm_other")
        for ap in np.unique(ra):
            for lk in (0, 1):
                nr = np.sum((ra == ap) & (rl == lk))
                if nr == 0:
                    continue
                k = s & (a == ap) & (l == lk)
                if k.sum() < 20:
                    k = s & (a == ap)
                w[k] += nr / k.sum()
    return w
