#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Parse the raw IEM METAR CSVs (data/external/metar_raw/*.csv) into
data/external/metar_10apt.parquet and print a coverage report.

Output columns (one row per station & observation time, UTC):
  station, valid, tmpc, dwpc, relh, sknt, drct, gust, vsby (miles), vsby_m, alti (inHg),
  qnh_hpa (Q-group from raw METAR, else alti converted), skyc1-4, skyl1-4 (ft), ceiling_ft
  (lowest BKN/OVC/VV layer), wxcodes, metar (raw), is_special (off-schedule minute ~ SPECI),
  flags: rain, snow, freezing, ts, fog, mist, drizzle, hail, precip, lowvis, lvp, deice_proxy

Usage:  uv run scripts/build_metar.py [--raw DIR] [--out FILE]
"""

import argparse, glob
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "metar_raw"
OUT = ROOT / "data" / "external" / "metar_10apt.parquet"

NUM_COLS = [
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "peak_wind_gust",
    "peak_wind_drct",
]
STR_COLS = ["station", "valid", "skyc1", "skyc2", "skyc3", "skyc4", "wxcodes", "metar", "peak_wind_time"]


def read_raw(raw_dir: Path) -> pl.DataFrame:
    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not files:
        raise SystemExit(f"no raw CSVs in {raw_dir}; run scripts/download_metar.py first")
    frames = []
    for f in files:
        d = pl.read_csv(f, infer_schema_length=0, null_values=[""], quote_char='"')
        keep = [c for c in STR_COLS + NUM_COLS if c in d.columns]
        d = d.select(keep)
        for c in NUM_COLS:
            if c in d.columns:
                d = d.with_columns(pl.col(c).cast(pl.Float64, strict=False))
        d = d.with_columns(pl.lit(Path(f).name).alias("_src"))
        frames.append(d)
    return pl.concat(frames, how="diagonal_relaxed")


def parse(raw: pl.DataFrame) -> pl.DataFrame:
    df = raw.with_columns(
        pl.col("valid").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M", strict=True).dt.replace_time_zone("UTC"),
    )
    # --- de-duplicate (report_type 3 and 4 can both carry the same ob; keep longest raw METAR) ---
    df = (
        df.with_columns(pl.col("metar").str.len_chars().fill_null(0).alias("_len"))
        .sort(["station", "valid", "_len"], descending=[False, False, True])
        .unique(subset=["station", "valid"], keep="first", maintain_order=True)
        .drop("_len")
    )

    # --- units ---
    df = (
        df.with_columns(
            ((pl.col("tmpf") - 32) * 5 / 9).round(1).alias("tmpc"),
            ((pl.col("dwpf") - 32) * 5 / 9).round(1).alias("dwpc"),
            # IEM caps 9999 m at 6.21 mi; convert back to metres, snap to 50 m and 10 km cap
            pl.when(pl.col("vsby") >= 6.2)
            .then(10000.0)
            .otherwise(((pl.col("vsby") * 1609.344) / 50).round() * 50)
            .alias("vsby_m"),
            pl.col("metar").str.extract(r"\bQ(\d{4})\b", 1).cast(pl.Float64, strict=False).alias("_q"),
            (pl.col("alti") * 33.8639).round(1).alias("_alti_hpa"),
        )
        .with_columns(
            pl.coalesce(["_q", "_alti_hpa"]).alias("qnh_hpa"),
        )
        .drop("_q", "_alti_hpa")
    )

    # --- specials: IEM drops the METAR/SPECI keyword, so flag obs whose minute is not one of the
    #     station's two routine minutes (e.g. :20/:50) -> is_special (SPECI or off-schedule ob) ---
    routine = (
        df.group_by("station", pl.col("valid").dt.minute().alias("_min"))
        .len()
        .sort(["station", "len"], descending=[False, True])
        .group_by("station", maintain_order=True)
        .head(2)
        .group_by("station")
        .agg(pl.col("_min").alias("_routine_min"))
    )
    df = (
        df.join(routine, on="station", how="left")
        .with_columns(pl.col("valid").dt.minute().is_in(pl.col("_routine_min")).not_().alias("is_special"))
        .drop("_routine_min")
    )

    # --- ceiling: lowest BKN / OVC / VV layer (ft) ---
    layer_exprs = []
    for i in range(1, 5):
        layer_exprs.append(
            pl.when(pl.col(f"skyc{i}").is_in(["BKN", "OVC", "VV"])).then(pl.col(f"skyl{i}")).otherwise(None)
        )
    df = df.with_columns(pl.min_horizontal(layer_exprs).alias("ceiling_ft"))
    # vertical visibility from raw METAR (VV003) when IEM did not put it into the sky columns
    df = df.with_columns(
        pl.coalesce(
            [
                pl.col("ceiling_ft"),
                pl.col("metar").str.extract(r"\bVV(\d{3})\b", 1).cast(pl.Float64, strict=False) * 100,
            ]
        ).alias("ceiling_ft")
    )

    # --- weather phenomena flags from wxcodes (fallback: raw METAR body up to the RMK/TEMPO part) ---
    wx = pl.coalesce(
        [
            pl.col("wxcodes"),
            pl.col("metar")
            .str.replace(r"\s(TEMPO|BECMG|NOSIG|RMK).*$", "")
            .str.extract(
                r"((?:\s[-+]?(?:VC)?(?:MI|BC|PR|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)+)+)",
                1,
            ),
            pl.lit(""),
        ]
    )
    df = (
        df.with_columns(wx.alias("_wx"))
        .with_columns(
            pl.col("_wx").str.contains(r"RA").alias("rain"),
            pl.col("_wx").str.contains(r"SN|SG|GS").alias("snow"),
            pl.col("_wx").str.contains(r"FZ").alias("freezing"),
            pl.col("_wx").str.contains(r"TS").alias("ts"),
            pl.col("_wx").str.contains(r"FG").alias("fog"),
            pl.col("_wx").str.contains(r"BR").alias("mist"),
            pl.col("_wx").str.contains(r"DZ").alias("drizzle"),
            pl.col("_wx").str.contains(r"GR|PL").alias("hail"),
            pl.col("_wx").str.contains(r"SH").alias("showers"),
        )
        .with_columns(
            (
                pl.col("rain") | pl.col("snow") | pl.col("drizzle") | pl.col("hail") | pl.col("_wx").str.contains("UP")
            ).alias("precip"),
        )
        .with_columns(
            (pl.col("vsby_m") < 1500).alias("lowvis"),
            ((pl.col("vsby_m") < 600) | (pl.col("ceiling_ft") < 200)).fill_null(False).alias("lvp"),
            ((pl.col("tmpc") <= 3) & (pl.col("precip") | ((pl.col("tmpc") - pl.col("dwpc")) < 2) | pl.col("fog")))
            .fill_null(False)
            .alias("deice_proxy"),
        )
        .drop("_wx")
    )

    cols = [
        "station",
        "valid",
        "is_special",
        "tmpc",
        "dwpc",
        "relh",
        "sknt",
        "drct",
        "gust",
        "vsby",
        "vsby_m",
        "alti",
        "qnh_hpa",
        "mslp",
        "skyc1",
        "skyc2",
        "skyc3",
        "skyc4",
        "skyl1",
        "skyl2",
        "skyl3",
        "skyl4",
        "ceiling_ft",
        "wxcodes",
        "metar",
        "rain",
        "snow",
        "freezing",
        "ts",
        "fog",
        "mist",
        "drizzle",
        "hail",
        "showers",
        "precip",
        "lowvis",
        "lvp",
        "deice_proxy",
        "peak_wind_gust",
        "peak_wind_drct",
        "_src",
    ]
    return df.select([c for c in cols if c in df.columns]).sort(["station", "valid"])


def coverage(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    per_month = (
        df.group_by("station", pl.col("valid").dt.strftime("%Y-%m").alias("month"))
        .agg(
            pl.len().alias("n_obs"),
            pl.col("tmpc").is_null().mean().round(3).alias("tmpc_null"),
            pl.col("vsby").is_null().mean().round(3).alias("vsby_null"),
        )
        .sort("station", "month")
    )
    gaps = (
        df.sort("station", "valid")
        .with_columns(pl.col("valid").diff().over("station").alias("gap"))
        .filter(pl.col("gap") > pl.duration(hours=3))
        .select(
            "station",
            (pl.col("valid") - pl.col("gap")).alias("gap_start"),
            pl.col("valid").alias("gap_end"),
            (pl.col("gap").dt.total_minutes() / 60).alias("gap_h"),
        )
        .sort("station", "gap_start")
    )
    return per_month, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    raw = read_raw(Path(a.raw))
    print(f"raw rows: {raw.height:,} from {raw['_src'].n_unique()} files")
    df = parse(raw)
    print(
        f"parsed rows: {df.height:,}  stations={df['station'].n_unique()}  "
        f"{df['valid'].min()} .. {df['valid'].max()}"
    )
    df.write_parquet(a.out)
    print(f"wrote {a.out}")

    per_month, gaps = coverage(df)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print("\n== obs per station per month ==")
        print(per_month.pivot(on="station", index="month", values="n_obs").sort("month"))
        print("\n== null share of tmpc per station ==")
        print(
            df.group_by("station")
            .agg(
                pl.len().alias("n"),
                pl.col("tmpc").is_null().mean().round(4).alias("tmpc_null"),
                pl.col("sknt").is_null().mean().round(4).alias("sknt_null"),
                pl.col("vsby").is_null().mean().round(4).alias("vsby_null"),
                pl.col("qnh_hpa").is_null().mean().round(4).alias("qnh_null"),
                (pl.col("valid").diff().dt.total_minutes()).median().alias("median_dt_min"),
                pl.col("rain").mean().round(3).alias("rain"),
                pl.col("snow").mean().round(4).alias("snow"),
                pl.col("ts").mean().round(4).alias("ts"),
                pl.col("fog").mean().round(4).alias("fog"),
                pl.col("lowvis").mean().round(4).alias("lowvis"),
                pl.col("lvp").mean().round(4).alias("lvp"),
                pl.col("deice_proxy").mean().round(4).alias("deice"),
            )
            .sort("station")
        )
        print(f"\n== gaps > 3 h: {gaps.height} ==")
        print(
            gaps.group_by("station")
            .agg(
                pl.len().alias("n_gaps"),
                pl.col("gap_h").max().alias("max_gap_h"),
                pl.col("gap_h").sum().round(1).alias("total_gap_h"),
            )
            .sort("station")
        )
        print(gaps.sort("gap_h", descending=True).head(25))


if __name__ == "__main__":
    main()
