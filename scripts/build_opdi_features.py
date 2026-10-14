#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds OPDI (ADS-B ground-trajectory) features for every departure row, train 2025 plus ranking Jan/Jul 2026.
Reads data/external/opdi/flight_list_*.parquet and the matching event files; writes
data/external/opdi_features.parquet keyed by MVT_ID_mvt. Group prefix: opdi_.

Per-flight (linked by callsign + airport + day, time-window checked): seconds from first ADS-B contact, pushback
(exit-parking_position), apron exit, taxiway entry, runway entry to the take-off time; link flag. Airport context
from ALL OPDI flights at the airport: counts of pushbacks / taxiway entries / runway entries in the 15 and 30 min
before take-off, de-icing pad entries in the previous 60 min and same day, and the rolling median line-up wait
(take-off minus runway entry) of linked departures in the previous 30 min."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import glob, os, time, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/data/external/opdi"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
GROUND = ["exit-parking_position", "entry-taxiway", "entry-runway", "exit-apron", "entry-deicing_pad", "take-off"]


def _fl(f):
    df = (
        pl.scan_parquet(f)
        .select("id", "flt_id", "adep", "ades", "icao24", "first_seen", "last_seen")
        .filter(pl.col("adep").is_in(APTS) | pl.col("ades").is_in(APTS))
        .collect()
    )
    return df.with_columns(
        [
            pl.col(c).str.to_datetime(time_unit="us") if df.schema[c] == pl.Utf8 else pl.col(c).cast(pl.Datetime("us"))
            for c in ("first_seen", "last_seen")
        ]
    ).with_columns(
        pl.col("id").reinterpret(signed=False) if df.schema["id"] == pl.Int64 else pl.col("id").cast(pl.UInt64),
        pl.col("flt_id").str.strip_chars(),
    )


fl_all = pl.concat([_fl(f) for f in sorted(glob.glob(f"{D}/flight_list_*.parquet"))])
fl = fl_all.filter(pl.col("adep").is_in(APTS))
fl_arr = fl_all.filter(pl.col("ades").is_in(APTS))
ids = fl.select("id").unique()
log("OPDI departures at our airports", fl.height, "arrivals", fl_arr.height)
parts = []
for f in sorted(glob.glob(f"{D}/flight_events_*.parquet")):
    e = pl.scan_parquet(f).select("flight_id", "type", "event_time").filter(pl.col("type").is_in(GROUND)).collect()
    e = e.with_columns(
        (
            pl.col("flight_id").reinterpret(signed=False)
            if e.schema["flight_id"] == pl.Int64
            else pl.col("flight_id").cast(pl.UInt64)
        ),
        (
            pl.col("event_time").str.to_datetime(time_unit="us")
            if e.schema["event_time"] == pl.Utf8
            else pl.col("event_time").cast(pl.Datetime("us"))
        ),
    )
    parts.append(e.join(ids.rename({"id": "flight_id"}), on="flight_id", how="inner"))
ev = pl.concat(parts)
log("ground events", ev.height)
first = (
    ev.group_by("flight_id", "type")
    .agg(pl.col("event_time").min().alias("t"))
    .pivot(on="type", index="flight_id", values="t")
)
fl = fl.join(first, left_on="id", right_on="flight_id", how="left")
# ---- our departures
raw = pl.concat(
    [
        pl.scan_parquet(f)
        .select("MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt", "CALLSIGN_flt")
        .filter(pl.col("PHASE_mvt") == "DEP")
        .collect()
        for f in sorted(glob.glob(f"{ROOT}/data/raw/training_*.parquet")) + [f"{ROOT}/data/raw/ranking.parquet"]
    ],
    how="diagonal_relaxed",
)
raw = raw.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).alias("mvt")).with_columns(
    pl.col("mvt").dt.date().alias("day"),
    pl.coalesce(pl.col("CALLSIGN_flt"), pl.col("FLIGHT_mvt")).str.strip_chars().alias("cs"),
)
cand = fl.select(
    pl.col("adep").alias("ADEP_mvt"),
    pl.col("flt_id").alias("cs"),
    pl.col("first_seen").dt.date().alias("day"),
    "id",
    "icao24",
    "first_seen",
    "last_seen",
    *[c for c in GROUND if c in fl.columns],
)
j = raw.join(cand, on=["ADEP_mvt", "cs", "day"], how="left")
ok = (pl.col("mvt") >= pl.col("first_seen") - pl.duration(hours=1)) & (
    pl.col("mvt") <= pl.col("last_seen") + pl.duration(hours=1)
)
j = j.with_columns(
    [
        pl.when(ok).then(pl.col(c)).otherwise(None).alias(c)
        for c in ["id", "icao24", "first_seen"] + [c for c in GROUND if c in fl.columns]
    ]
)
j = j.sort("MVT_ID_mvt", pl.col("id").is_null()).unique(subset=["MVT_ID_mvt"], keep="first")
sec = lambda c: (pl.col("mvt") - pl.col(c)).dt.total_seconds()
feat = j.select(
    "MVT_ID_mvt",
    "ADEP_mvt",
    "mvt",
    pl.col("id").is_not_null().cast(pl.Int8).alias("opdi_linked"),
    sec("first_seen").alias("opdi_first_seen_before"),
    *[
        sec(c).alias(
            "opdi_"
            + c.replace("-", "_")
            .replace("exit_parking_position", "pushback")
            .replace("entry_", "")
            .replace("exit_", "")
            + "_before"
        )
        for c in ("exit-parking_position", "exit-apron", "entry-taxiway", "entry-runway")
        if c in j.columns
    ],
    (sec("take-off") if "take-off" in j.columns else pl.lit(None, pl.Int64)).alias("opdi_takeoff_delta"),
)
log("per-flight features", feat.shape)
# ---- airframe linkage (opdir_): the same icao24's most recent OPDI arrival at this airport -> OUR arrival row -> its substitution flag / offsets
log("airframe (icao24) inbound-leg linkage")
oa = (
    fl_arr.select(
        pl.col("ades").alias("ADEP_mvt"),
        "icao24",
        pl.col("flt_id").alias("arr_cs"),
        pl.col("last_seen").alias("t_arr_seen"),
    )
    .drop_nulls(["icao24", "t_arr_seen"])
    .sort("t_arr_seen")
)
jj = (
    j.select("MVT_ID_mvt", "ADEP_mvt", "mvt", "icao24")
    .filter(pl.col("icao24").is_not_null())
    .sort("mvt")
    .join_asof(
        oa, left_on="mvt", right_on="t_arr_seen", by=["ADEP_mvt", "icao24"], strategy="backward", tolerance="36h"
    )
)
ours_arr = pl.concat(
    [
        pl.scan_parquet(f)
        .select(
            "ADES_mvt",
            "FLIGHT_mvt",
            "CALLSIGN_flt",
            "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "ARVT_3_flt",
            "PHASE_mvt",
        )
        .filter(pl.col("PHASE_mvt") != "DEP")
        .collect()
        for f in sorted(glob.glob(f"{ROOT}/data/raw/training_*.parquet")) + [f"{ROOT}/data/raw/ranking.parquet"]
    ],
    how="diagonal_relaxed",
)
ours_arr = ours_arr.with_columns(
    pl.col("ADES_mvt").alias("ADEP_mvt"),
    pl.coalesce(pl.col("CALLSIGN_flt"), pl.col("FLIGHT_mvt")).str.strip_chars().alias("arr_cs"),
    pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).alias("t_land"),
    pl.col("BLOCK_TIME_UTC_mvt").dt.replace_time_zone(None).alias("t_inblock"),
    pl.col("SCHED_TIME_UTC_mvt").dt.replace_time_zone(None).alias("t_asched"),
    pl.col("ARVT_3_flt").dt.replace_time_zone(None).alias("t_arvt3"),
)
ours_arr = (
    ours_arr.select("ADEP_mvt", "arr_cs", "t_land", "t_inblock", "t_asched", "t_arvt3")
    .drop_nulls(["arr_cs", "t_land"])
    .sort("t_land")
)
jj = (
    jj.filter(pl.col("t_arr_seen").is_not_null())
    .sort("t_arr_seen")
    .join_asof(
        ours_arr, left_on="t_arr_seen", right_on="t_land", by=["ADEP_mvt", "arr_cs"], strategy="nearest", tolerance="1h"
    )
)
sec2 = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()
reg = jj.select(
    "MVT_ID_mvt",
    pl.col("t_arr_seen").is_not_null().cast(pl.Int8).alias("opdir_linked"),
    pl.col("t_land").is_not_null().cast(pl.Int8).alias("opdir_arr_found"),
    ((pl.col("t_inblock") - pl.col("t_asched")).dt.total_seconds().abs() <= 60).cast(pl.Int8).alias("opdir_arr_afb"),
    sec2("t_inblock", "t_arvt3").alias("opdir_arr_off"),
    sec2("t_inblock", "t_asched").alias("opdir_arr_delay"),
    sec2("mvt", "t_inblock").alias("opdir_turn"),
    sec2("mvt", "t_land").alias("opdir_since_land"),
    sec2("mvt", "t_arr_seen").alias("opdir_since_seen"),
)
feat = feat.join(reg, on="MVT_ID_mvt", how="left")
log(
    "  opdir linked",
    float(feat["opdir_linked"].fill_null(0).mean()),
    "arrival row found",
    float(feat["opdir_arr_found"].fill_null(0).mean()),
)
# ---- airport context from ALL OPDI flights (events keyed by airport via the flight list)
ev_a = ev.join(fl.select("id", "adep"), left_on="flight_id", right_on="id", how="inner").with_columns(
    pl.col("event_time").dt.epoch("s").alias("t")
)
feat = feat.with_columns(pl.col("mvt").dt.epoch("s").alias("t"))
out_cols = {}
mvt_t = feat["t"].to_numpy()
apt = feat["ADEP_mvt"].to_numpy()
for typ, tag in [
    ("exit-parking_position", "pushback"),
    ("entry-taxiway", "taxiway"),
    ("entry-runway", "rwyentry"),
    ("entry-deicing_pad", "deice"),
]:
    for w in ((900, 1800) if typ != "entry-deicing_pad" else (3600, 86400)):
        out_cols[f"opdi_n_{tag}_m{w//60}"] = np.full(len(feat), np.nan, dtype=np.float32)
    sub = ev_a.filter(pl.col("type") == typ)
    for a in APTS:
        et = np.sort(sub.filter(pl.col("adep") == a)["t"].to_numpy())
        ii = np.where(apt == a)[0]
        if len(et) == 0:
            continue
        for w in ((900, 1800) if typ != "entry-deicing_pad" else (3600, 86400)):
            lo = np.searchsorted(et, mvt_t[ii] - w, side="left")
            hi = np.searchsorted(et, mvt_t[ii], side="right")
            out_cols[f"opdi_n_{tag}_m{w//60}"][ii] = hi - lo
    log("  context", typ)
# rolling median line-up wait (mvt - entry-runway) of linked departures in the previous 30 min, per airport
lw = (
    feat.filter(
        pl.col("opdi_runway_before").is_not_null()
        & (pl.col("opdi_runway_before") > 0)
        & (pl.col("opdi_runway_before") < 3600)
    )
    .select("ADEP_mvt", "t", pl.col("opdi_runway_before").alias("lw"))
    .sort("ADEP_mvt", "t")
)
lw = (
    lw.with_columns(pl.from_epoch("t", time_unit="s").alias("ts"))
    .with_columns(
        pl.col("lw")
        .rolling_median_by("ts", window_size="30m", closed="both")
        .over("ADEP_mvt")
        .alias("opdi_lineup_m30_med")
    )
    .drop("ts", "lw")
    .rename({"t": "t_key"})
    .sort("t_key")
)
feat = (
    feat.with_columns([pl.Series(k, v) for k, v in out_cols.items()])
    .sort("t")
    .join_asof(lw, left_on="t", right_on="t_key", by="ADEP_mvt", strategy="backward", tolerance=3600)
    .drop("t_key", "t", "mvt", "ADEP_mvt")
)
feat.write_parquet(f"{ROOT}/data/external/opdi_features.parquet")
log("wrote data/external/opdi_features.parquet", feat.shape)
print(feat.select([c for c in feat.columns if c != "MVT_ID_mvt"]).describe())
