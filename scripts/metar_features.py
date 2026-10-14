#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""METAR weather features for the PRC Data Challenge 2026 (taxi-out prediction).

Requires data/external/metar_10apt.parquet (built by scripts/download_metar.py + scripts/build_metar.py).

    from scripts.metar_features import add_metar_features
    df = add_metar_features(df)                       # ADEP_mvt / MVT_TIME_UTC_mvt, lags 0/30/60 min

For each row and each lag L in `lags_min`, the most recent METAR at or before (time_col - L min)
for that airport is attached (polars join_asof backward, by airport, within `max_age`) as columns
`wx{L}_<name>`:
    tmpc, dwpc, sknt, gust, drct, vsby_m, ceiling_ft, qnh_hpa,
    rain, snow, freezing, ts, fog, mist, drizzle, precip, lowvis, lvp, deice_proxy,
    age_min  (minutes between that METAR and time_col - L)
plus per airport UTC-day aggregates `wxd_*`: min_vsby_m, max_sknt, max_gust, any_snow, any_ts,
any_fog, any_lvp, min_tmpc, n_obs  (NB: computed over the whole calendar day, i.e. they look
slightly into the future of an early-morning movement -- fine for the challenge, but keep in mind).

Rows whose airport is not among the 10 METAR stations (or whose METAR is older than `max_age`)
get nulls.  The row order of the input frame is preserved.
"""

from __future__ import annotations

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import warnings
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
METAR_PATH = ROOT / "data" / "external" / "metar_10apt.parquet"

WX_COLS = [
    "tmpc",
    "dwpc",
    "sknt",
    "gust",
    "drct",
    "vsby_m",
    "ceiling_ft",
    "qnh_hpa",
    "rain",
    "snow",
    "freezing",
    "ts",
    "fog",
    "mist",
    "drizzle",
    "precip",
    "lowvis",
    "lvp",
    "deice_proxy",
]
FLAG_COLS = ["rain", "snow", "freezing", "ts", "fog", "mist", "drizzle", "precip", "lowvis", "lvp", "deice_proxy"]

_METAR_CACHE: dict[str, pl.DataFrame] = {}


def load_metar(path: str | Path = METAR_PATH) -> pl.DataFrame:
    """Load + cache the parsed METAR table (station, valid[UTC, us], WX_COLS)."""
    key = str(path)
    if key not in _METAR_CACHE:
        m = pl.read_parquet(path).select(["station", "valid"] + WX_COLS)
        m = m.with_columns(
            pl.col("valid").cast(pl.Datetime("us")).dt.convert_time_zone("UTC"),
            pl.col("gust").fill_null(0.0),  # no gust group => no gust
            *[pl.col(c).cast(pl.Int8) for c in FLAG_COLS],
        ).sort(["station", "valid"])
        _METAR_CACHE[key] = m
    return _METAR_CACHE[key]


def daily_metar(metar: pl.DataFrame) -> pl.DataFrame:
    return metar.group_by("station", pl.col("valid").dt.date().alias("_date")).agg(
        pl.col("vsby_m").min().alias("wxd_min_vsby_m"),
        pl.col("sknt").max().alias("wxd_max_sknt"),
        pl.col("gust").max().alias("wxd_max_gust"),
        pl.col("snow").max().alias("wxd_any_snow"),
        pl.col("ts").max().alias("wxd_any_ts"),
        pl.col("fog").max().alias("wxd_any_fog"),
        pl.col("lvp").max().alias("wxd_any_lvp"),
        pl.col("tmpc").min().alias("wxd_min_tmpc"),
        pl.len().cast(pl.Int32).alias("wxd_n_obs"),
    )


def add_metar_features(
    df: pl.DataFrame,
    airport_col: str = "ADEP_mvt",
    time_col: str = "MVT_TIME_UTC_mvt",
    lags_min: tuple[int, ...] = (0, 30, 60),
    max_age: str = "6h",
    metar_path: str | Path = METAR_PATH,
    daily: bool = True,
) -> pl.DataFrame:
    """Attach `wx{lag}_*` (asof-backward METAR at time_col - lag) and `wxd_*` (airport-day aggregates)."""
    metar = load_metar(metar_path)
    ts = pl.col(time_col)
    if df.schema[time_col] == pl.Datetime or df.schema[time_col].time_zone is None:
        ts = ts.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    else:
        ts = ts.cast(pl.Datetime("us")).dt.convert_time_zone("UTC")

    out = df.with_row_index("_ri").with_columns(ts.alias("_t"), pl.col(airport_col).cast(pl.Utf8).alias("_apt"))

    warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")  # left is sorted below
    for lag in lags_min:
        pre = f"wx{lag}_"
        right = metar.rename({c: pre + c for c in WX_COLS}).rename({"valid": pre + "valid"})
        out = (
            out.with_columns((pl.col("_t") - pl.duration(minutes=lag)).alias("_tl"))
            .sort("_tl")
            .with_columns(pl.col("_tl").set_sorted())  # globally sorted => sorted per group
            .join_asof(
                right,
                left_on="_tl",
                right_on=pre + "valid",
                by_left="_apt",
                by_right="station",
                strategy="backward",
                tolerance=max_age,
                coalesce=True,
            )
            .with_columns(
                ((pl.col("_tl") - pl.col(pre + "valid")).dt.total_seconds() / 60.0).round(1).alias(pre + "age_min")
            )
            .drop("_tl", pre + "valid")
        )

    if daily:
        d = daily_metar(metar)
        out = (
            out.with_columns(pl.col("_t").dt.date().alias("_date"))
            .join(d, left_on=["_apt", "_date"], right_on=["station", "_date"], how="left")
            .drop("_date")
        )

    return out.sort("_ri").drop("_ri", "_t", "_apt")


if __name__ == "__main__":
    import sys

    rk = pl.read_parquet(ROOT / "data" / "raw" / "ranking.parquet")
    dep = rk.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"ranking departures: {dep.height:,} rows, {dep['ADEP_mvt'].n_unique()} airports")
    feat = add_metar_features(dep)
    new = [c for c in feat.columns if c.startswith("wx")]
    print(f"added {len(new)} columns")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=40):
        nulls = feat.select([pl.col(c).is_null().mean().round(4).alias(c) for c in new]).transpose(
            include_header=True, header_name="column", column_names=["null_rate"]
        )
        print("\n== null rates on ranking departures ==")
        print(nulls)
        print("\n== null rate of wx0_tmpc per airport ==")
        print(
            feat.group_by("ADEP_mvt")
            .agg(
                pl.len().alias("n"),
                pl.col("wx0_tmpc").is_null().mean().round(4).alias("wx0_tmpc_null"),
                pl.col("wx0_age_min").median().alias("med_age_min"),
                pl.col("wx0_age_min").max().alias("max_age_min"),
            )
            .sort("n", descending=True)
        )
        print("\n== feature means (ranking departures) ==")
        print(
            feat.select(
                [
                    pl.col(c).mean().round(3).alias(c)
                    for c in ["wx0_" + f for f in FLAG_COLS] + ["wxd_any_snow", "wxd_any_ts", "wxd_any_lvp"]
                ]
            ).transpose(include_header=True, header_name="column", column_names=["mean"])
        )
        print("\n== sample rows ==")
        print(
            feat.select(
                [
                    "ADEP_mvt",
                    "MVT_TIME_UTC_mvt",
                    "TAXITIME_SEC_mvt",
                    "wx0_tmpc",
                    "wx0_sknt",
                    "wx0_drct",
                    "wx0_gust",
                    "wx0_vsby_m",
                    "wx0_ceiling_ft",
                    "wx0_rain",
                    "wx0_lowvis",
                    "wx0_deice_proxy",
                    "wx0_age_min",
                    "wx30_tmpc",
                    "wx60_sknt",
                    "wxd_min_vsby_m",
                    "wxd_max_sknt",
                ]
            ).sample(8, seed=1)
        )
        # the ranking set carries no target -> check the signal on a training month if available
        trf = sorted((ROOT / "data" / "raw").glob("training_*.parquet"))
        if trf:
            tr = add_metar_features(pl.read_parquet(trf[0]).filter(pl.col("PHASE_mvt") == "DEP"))
            print(f"\n== median/mean taxi-out (s) by flag on {trf[0].name} ==")
            for f in ["wx0_lvp", "wx0_snow", "wx0_ts", "wx0_lowvis", "wx0_fog", "wx0_deice_proxy"]:
                print(
                    tr.group_by(f)
                    .agg(
                        pl.len().alias("n"),
                        pl.col("TAXITIME_SEC_mvt").median().alias("median_taxi_s"),
                        pl.col("TAXITIME_SEC_mvt").mean().round(0).alias("mean_taxi_s"),
                    )
                    .sort(f)
                )
