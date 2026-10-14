#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Per-row meta inputs from the PRU daily table (data/external/xpr_pru_daily.parquet, pru_build.py) and a
base prediction. Identical code for the 2025 fold side and the 2026 ranking side (parity by construction):
  day key   = UTC date of predicted AOBT = ATOT - pred; a flight enters day d's sums and gets features only if month(ATOT) == month(d)
  pred      = the meta's base v41 where given (finite), else the per-airport median label of all 2025 clean matched rows (one constant table)
  xP  = PRU DLY_ALL_PRE_2*60 / FLT_DEP_IFR_2 (s per IFR departure)
  xSh = our sum of max(0, ATOT - pred - STD) / our departures of that day
  xG  = clip((PRU sum - our sum) / max(our predicted-delayed count, 5), +-900)
  xR  = clip(PRU sum / max(our sum, 60), 0, 5)
  xdl = clip(ATOT - pred - STD, +-7200) (this row's predicted pre-departure delay)
  xGd = xG * Phi(xdl / 120)
  xlate = FLT_DEP_OUT_LATE_1 / max(FLT_DEP_REG_1, 1);  xreg = FLT_DEP_REG_1 / FLT_DEP_1 (slot adherence file)
(ATC pre-departure delay is not used: its per-airport missingness differs between 2025 and Jan/Jul 2026)"""

import os, math, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["xP", "xSh", "xG", "xR", "xdl", "xGd", "xlate", "xreg"]
_F = None


def frame():
    global _F
    if _F is None:
        F = pl.read_parquet(
            f"{ROOT}/data/features_v9.parquet",
            columns=[
                "MVT_ID_mvt",
                "ADEP_mvt",
                "MVT_TIME_UTC_mvt",
                "SCHED_TIME_UTC_mvt",
                "is_rank",
                "unmatched",
                "fb",
                "y",
            ],
        )
        F = F.with_columns(
            ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8),
            atot=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64),
            std=pl.col("SCHED_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64),
            mid=pl.col("MVT_ID_mvt").cast(pl.Float64),
        )
        _F = F
    return _F


def medians():
    F = frame()
    g = (
        F.filter(
            (pl.col("is_rank") == 0) & (pl.col("unmatched") == 0) & (pl.col("fb") == 0) & pl.col("y").is_not_null()
        )
        .group_by("ADEP_mvt")
        .agg(pl.col("y").median())
    )
    return dict(g.iter_rows())


def feats(side, mid, pred):
    """side: 0 = all 2025 departures, 1 = all ranking departures. mid/pred: base prediction for (a subset of) the side's rows.
    Returns a polars frame MVT_ID (float) + COLS for every departure of the side."""
    F = frame().filter(pl.col("is_rank") == side)
    med = medians()
    P = pl.DataFrame({"mid": np.asarray(mid, float), "pb": np.asarray(pred, float)}).filter(pl.col("pb").is_finite())
    F = F.join(P, on="mid", how="left").with_columns(
        p=pl.when(pl.col("pb").is_not_null())
        .then(pl.col("pb"))
        .otherwise(pl.col("ADEP_mvt").replace_strict(med, return_dtype=pl.Float64))
    )
    F = F.with_columns(
        dl=pl.col("atot") - pl.col("p") - pl.col("std"),
        day=pl.from_epoch((pl.col("atot") - pl.col("p")).cast(pl.Int64), time_unit="s").dt.date(),
        am=pl.from_epoch(pl.col("atot").cast(pl.Int64), time_unit="s").dt.strftime("%Y-%m"),
    )
    F = F.with_columns(ok=pl.col("day").dt.strftime("%Y-%m") == pl.col("am"))
    S = (
        F.filter(pl.col("ok"))
        .group_by("ADEP_mvt", "day")
        .agg(S=pl.col("dl").clip(lower_bound=0).sum(), n=pl.len(), nd=(pl.col("dl") > 0).sum())
    )
    R = pl.read_parquet(f"{ROOT}/data/external/xpr_pru_daily.parquet")
    F = F.join(S, on=["ADEP_mvt", "day"], how="left").join(R, on=["ADEP_mvt", "day"], how="left")
    Ps = pl.col("allpre_min") * 60.0
    F = F.with_columns(
        xP=Ps / pl.col("n_ifr"),
        xSh=pl.col("S") / pl.col("n"),
        xG=((Ps - pl.col("S")) / pl.max_horizontal(pl.col("nd"), pl.lit(5))).clip(-900, 900),
        xR=(Ps / pl.max_horizontal(pl.col("S"), pl.lit(60.0))).clip(0, 5),
        xdl=pl.col("dl").clip(-7200, 7200),
        xlate=pl.col("n_late") / pl.max_horizontal(pl.col("n_reg"), pl.lit(1.0)),
        xreg=pl.col("n_reg") / pl.col("n_dep1_sa"),
    )
    phi = pl.col("xdl").map_batches(
        lambda s: pl.Series(0.5 * (1 + np.vectorize(math.erf)(s.to_numpy() / (120 * math.sqrt(2))))),
        return_dtype=pl.Float64,
    )
    F = F.with_columns(xGd=pl.col("xG") * phi)
    F = F.with_columns([pl.when(pl.col("ok")).then(pl.col(c)).otherwise(None).alias(c) for c in COLS])
    return F.select(pl.col("mid").alias("MVT_ID"), *COLS, "ok", "ADEP_mvt")


def report(Z, tag):
    """missingness of every column, per airport, on this side's rows"""
    print(
        f"pru {tag}: rows {Z.height:,} | ok (month rule) {Z['ok'].mean():.4f} | null share "
        + " ".join(f"{c} {Z[c].null_count()/Z.height:.4f}" for c in COLS),
        flush=True,
    )
    g = (
        Z.group_by("ADEP_mvt")
        .agg(
            n=pl.len(),
            nulP=pl.col("xP").null_count() / pl.len(),
            nulG=pl.col("xG").null_count() / pl.len(),
            mP=pl.col("xP").mean(),
            mSh=pl.col("xSh").mean(),
            mG=pl.col("xG").mean(),
            sG=pl.col("xG").std(),
            mR=pl.col("xR").mean(),
        )
        .sort("ADEP_mvt")
    )
    for r in g.iter_rows(named=True):
        print(
            f"   {tag} {r['ADEP_mvt']} n {r['n']:,} nullP {r['nulP']:.4f} nullG {r['nulG']:.4f} | xP {r['mP']:.1f} xSh {r['mSh']:.1f} xG {r['mG']:+.1f} sd {r['sG']:.1f} xR {r['mR']:.3f}",
            flush=True,
        )
