#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the engineered feature table for taxi-out prediction: departures of the training data (2025) plus the ranking
batch (Jan/Jul 2026), one row per flight and one column block per feature group.
Inputs: --train-glob (default data/raw/training_*.parquet, labelled movements), --rank (default data/raw/ranking.parquet,
the unlabelled batch to predict). Output: --out (default data/features_v2.parquet).

Feature groups (train2.py --feats accepts a comma list): base, tl (timeline/queue bank), ref (stand-runway reference stats,
leave-one-month-out), turn (stand turnaround linkage), te (cross-fitted target encodings), geo (OSM distances, if
data/external/stand_runway_distance.parquet exists), wx (METAR via scripts/metar_features.py, if data exists).
usage: uv run scripts/build_features.py [--out data/features_v2.parquet]
"""

import argparse, glob, time, sys
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path(__file__).resolve().parents[1]
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(ROOT / "data/features_v2.parquet"), help="output features parquet")
ap.add_argument("--train-glob", default=str(ROOT / "data/raw/training_*.parquet"), help="training movements (labelled)")
ap.add_argument(
    "--rank",
    default=str(ROOT / "data/raw/ranking.parquet"),
    help="the batch to predict (unlabelled departures); phase 2 passes a new file here",
)
a = ap.parse_args()
tr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(a.train_glob))]).with_columns(
    pl.lit(0, pl.Int8).alias("is_rank")
)
rk = pl.read_parquet(a.rank).with_columns(pl.lit(1, pl.Int8).alias("is_rank"))
allm = pl.concat([tr, rk]).with_columns(
    pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"),
    (pl.col("PHASE_mvt") == "DEP").alias("is_dep"),
    (pl.col("ADEP_mvt") + "_" + pl.col("RUNWAY_mvt")).alias("apt_rwy"),
    pl.col("MVT_TIME_UTC_mvt").dt.month().alias("m"),
    pl.col("FLIGHT_ID_mvt").is_null().cast(pl.Int8).alias("unmatched"),
    pl.col("WK_TBL_CAT_flt").is_in(["H", "J"]).cast(pl.Int8).alias("heavy"),
    pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").alias("t_sched"),
    pl.col("EOBT_1_flt").dt.epoch("s").alias("t_eobt"),
    pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").alias("t_block"),
)
log("loaded", allm.height)


# ---------- helpers: counts in time windows via searchsorted per group ----------
def window_counts(keys_df, key_cols, times_col, query_df, q_key_cols, q_time_col, windows, prefix, weights=None):
    """For each query row, count events (from keys_df) with the same key in [t+lo, t+hi] for each (lo,hi) in windows."""
    ev = keys_df.select(key_cols + [times_col] + ([weights] if weights else [])).sort(key_cols + [times_col])
    q = query_df.select(q_key_cols + [q_time_col, "MVT_ID_mvt"])
    out = {f"{prefix}_{name}": np.zeros(q.height, dtype=np.float32) for name, _, _ in windows}
    # group iteration
    ev_groups = {k: g for k, g in ev.group_by(key_cols)}
    qk = q.select(q_key_cols).rows()
    qt = q[q_time_col].to_numpy()
    # index queries by key
    import collections

    idx = collections.defaultdict(list)
    for i, k in enumerate(qk):
        idx[k].append(i)
    for k, ii in idx.items():
        g = ev_groups.get(k)
        if g is None:
            continue
        et = g[times_col].to_numpy()
        ii = np.array(ii)
        tq = qt[ii]
        if weights:
            w = g[weights].to_numpy().astype(np.float64)
            cs = np.concatenate([[0.0], np.cumsum(w)])
        for name, lo, hi in windows:
            a = np.searchsorted(et, tq + lo, side="left")
            b = np.searchsorted(et, tq + hi, side="right")
            out[f"{prefix}_{name}"][ii] = (cs[b] - cs[a]) if weights else (b - a)
    return pl.DataFrame({"MVT_ID_mvt": q["MVT_ID_mvt"]}).with_columns([pl.Series(n, v) for n, v in out.items()])


dep = allm.filter(pl.col("is_dep")).sort("t")
W = [
    ("m5", -300, 0),
    ("m10", -600, 0),
    ("m15", -900, 0),
    ("m30", -1800, 0),
    ("m60", -3600, 0),
    ("m120", -7200, 0),
    ("p5", 0, 300),
    ("p10", 0, 600),
    ("p15", 0, 900),
    ("p30", 0, 1800),
    ("p60", 0, 3600),
]
log("timeline: runway dep counts")
f_rd = window_counts(dep, ["apt_rwy"], "t", dep, ["apt_rwy"], "t", W, "rwy_dep")
log("timeline: airport dep/arr counts")
f_ad = window_counts(dep, ["ADEP_mvt"], "t", dep, ["ADEP_mvt"], "t", W, "apt_dep")
arr = allm.filter(~pl.col("is_dep"))
f_aa = window_counts(arr, ["ADES_mvt"], "t", dep, ["ADEP_mvt"], "t", W, "apt_arr")
f_ra = window_counts(
    arr.with_columns((pl.col("ADES_mvt") + "_" + pl.col("RUNWAY_mvt")).alias("apt_rwy")),
    ["apt_rwy"],
    "t",
    dep,
    ["apt_rwy"],
    "t",
    W[:6],
    "rwy_arr",
)
log("timeline: heavy predecessors, scheduled demand, EOBT demand")
f_hv = window_counts(
    dep.filter(pl.col("heavy") == 1),
    ["apt_rwy"],
    "t",
    dep,
    ["apt_rwy"],
    "t",
    [("m10", -600, 0), ("m30", -1800, 0)],
    "rwy_heavy",
)
sch = dep.filter(pl.col("t_sched").is_not_null())
f_sd = window_counts(
    sch,
    ["ADEP_mvt"],
    "t_sched",
    dep,
    ["ADEP_mvt"],
    "t",
    [("m30", -1800, 0), ("m60", -3600, 0), ("pm30", -1800, 1800), ("p30", 0, 1800)],
    "sched_dep",
)
eob = dep.filter(pl.col("t_eobt").is_not_null())
f_ed = window_counts(
    eob,
    ["ADEP_mvt"],
    "t_eobt",
    dep,
    ["ADEP_mvt"],
    "t",
    [("m30", -1800, 0), ("m60", -3600, 0), ("m15", -900, 0)],
    "eobt_dep",
)
# gaps and queue runs on the runway sequence
log("timeline: gaps / queue runs")
dep = dep.sort("apt_rwy", "t").with_columns(
    [
        *[(pl.col("t") - pl.col("t").shift(k)).over("apt_rwy").alias(f"gap_prev{k}_rwy") for k in (1, 2, 3, 5, 10)],
        *[(pl.col("t").shift(-k) - pl.col("t")).over("apt_rwy").alias(f"gap_next{k}_rwy") for k in (1, 2, 3, 5, 10)],
        pl.col("heavy").shift(1).over("apt_rwy").alias("prev_heavy"),
        pl.col("heavy").shift(2).over("apt_rwy").alias("prev2_heavy"),
        pl.col("WK_TBL_CAT_flt").shift(1).over("apt_rwy").alias("prev_wtc"),
    ]
)
for thr in (90, 120, 180):
    dep = dep.with_columns((pl.col("gap_prev1_rwy") <= thr).cast(pl.Int32).alias("tight"))
    dep = dep.with_columns(
        ((pl.col("tight") == 0) | pl.col("tight").is_null()).cast(pl.Int32).cum_sum().over("apt_rwy").alias("run_id")
    )
    dep = dep.with_columns(
        pl.col("tight").cum_sum().over(["apt_rwy", "run_id"]).alias(f"queue_run{thr}"),
        pl.len().over(["apt_rwy", "run_id"]).alias(f"run_len{thr}"),
    )
    dep = dep.with_columns((pl.col(f"queue_run{thr}") / pl.col(f"run_len{thr}")).alias(f"queue_pos{thr}"))
dep = (
    dep.drop("tight", "run_id")
    .sort("ADEP_mvt", "t")
    .with_columns(
        [
            (pl.col("t") - pl.col("t").shift(1)).over("ADEP_mvt").alias("gap_prev1_apt"),
            (pl.col("t") - pl.col("t").shift(10)).over("ADEP_mvt").alias("gap_prev10_apt"),
            (pl.col("t").shift(-10) - pl.col("t")).over("ADEP_mvt").alias("gap_next10_apt"),
            pl.col("apt_rwy").n_unique().over(["ADEP_mvt", (pl.col("t") // 3600)]).alias("n_rwy_in_hour"),
        ]
    )
)
# runway share in the hour
hr = dep.group_by(["apt_rwy", (pl.col("t") // 3600).alias("hr")]).agg(pl.len().alias("rwy_hr_n"))
dep = dep.with_columns((pl.col("t") // 3600).alias("hr")).join(hr, on=["apt_rwy", "hr"], how="left")
ah = dep.group_by(["ADEP_mvt", "hr"]).agg(pl.len().alias("apt_hr_n"))
dep = (
    dep.join(ah, on=["ADEP_mvt", "hr"], how="left")
    .with_columns((pl.col("rwy_hr_n") / pl.col("apt_hr_n")).alias("rwy_share_hr"))
    .drop("hr")
)
for f in (f_rd, f_ad, f_aa, f_ra, f_hv, f_sd, f_ed):
    dep = dep.join(f, on="MVT_ID_mvt", how="left")

# ---------- turnaround / stand linkage ----------
log("turnaround linkage")
arr_s = (
    arr.select(
        pl.col("ADES_mvt").alias("ADEP_mvt"),
        "STAND_mvt",
        pl.col("t_block").alias("t_arr_inblock"),
        pl.col("t").alias("t_arr_land"),
        (pl.col("t_block") - pl.col("t_sched")).alias("arr_block_delay"),
    )
    .drop_nulls(["t_arr_inblock"])
    .sort("t_arr_inblock")
)
dep = dep.sort("t").join_asof(
    arr_s, left_on="t", right_on="t_arr_inblock", by=["ADEP_mvt", "STAND_mvt"], strategy="backward"
)
dep = dep.with_columns(
    (pl.col("t") - pl.col("t_arr_inblock")).alias("stand_since_arr_inblock"),
    (pl.col("t") - pl.col("t_arr_land")).alias("stand_since_arr_land"),
).drop("t_arr_inblock", "t_arr_land")
prev_dep = dep.select("ADEP_mvt", "STAND_mvt", pl.col("t").alias("t_prev_dep")).sort("t_prev_dep")
dep = dep.sort("t").join_asof(
    prev_dep,
    left_on="t",
    right_on="t_prev_dep",
    by=["ADEP_mvt", "STAND_mvt"],
    strategy="backward",
    allow_exact_matches=False,
)
dep = dep.with_columns((pl.col("t") - pl.col("t_prev_dep")).alias("stand_since_prev_dep")).drop("t_prev_dep")

# ---------- base ----------
log("base features")


def secs(a, b):
    return (pl.col(a) - pl.col(b)).dt.total_seconds()


dep = (
    dep.with_columns(
        [
            pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
            pl.col("MVT_TIME_UTC_mvt").dt.ordinal_day().alias("doy"),
            pl.col("SCHED_TIME_UTC_mvt").dt.minute().alias("sched_min"),
            secs("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("mvt_aobt3"),
            secs("MVT_TIME_UTC_mvt", "LOBT_flt").alias("mvt_lobt"),
            secs("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("mvt_eobt1"),
            secs("MVT_TIME_UTC_mvt", "IOBT_flt").alias("mvt_iobt"),
            secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mvt_sched"),
            secs("EOBT_1_flt", "SCHED_TIME_UTC_mvt").alias("eobt_sched"),
            secs("AOBT_3_flt", "EOBT_1_flt").alias("aobt3_eobt1"),
            secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt3_sched"),
            secs("ARVT_3_flt", "AOBT_3_flt").alias("blocktime3"),
            secs("ARVT_1_flt", "EOBT_1_flt").alias("blocktime1"),
            (pl.col("ADES_FILED_flt") != pl.col("ADES_flt")).cast(pl.Int8).alias("diverted"),
            pl.col("FLIGHT_mvt").str.slice(0, 3).alias("cs_pre"),
            pl.col("FLIGHT_mvt").str.len_chars().alias("cs_len"),
            pl.col("FLIGHT_mvt").str.contains(r"^[A-Z]{3}[0-9]").cast(pl.Int8).alias("cs_airline_style"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            (pl.col("ADEP_mvt") + "_" + pl.col("unmatched").cast(pl.Utf8)).alias("apt_unm"),
            (pl.col("ADEP_mvt") + "_" + pl.col("STAND_mvt")).alias("apt_stand"),
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60)
            .cast(pl.Int8)
            .alias("fb"),
        ]
    )
    .with_columns((pl.col("fb") == 1) | (pl.col("y") > 7200))
    .with_columns(pl.col("fb").alias("fb_pru"))
    if False
    else dep.with_columns(
        [
            pl.col("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
            pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
            pl.col("MVT_TIME_UTC_mvt").dt.ordinal_day().alias("doy"),
            pl.col("SCHED_TIME_UTC_mvt").dt.minute().alias("sched_min"),
            secs("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("mvt_aobt3"),
            secs("MVT_TIME_UTC_mvt", "LOBT_flt").alias("mvt_lobt"),
            secs("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("mvt_eobt1"),
            secs("MVT_TIME_UTC_mvt", "IOBT_flt").alias("mvt_iobt"),
            secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("mvt_sched"),
            secs("EOBT_1_flt", "SCHED_TIME_UTC_mvt").alias("eobt_sched"),
            secs("AOBT_3_flt", "EOBT_1_flt").alias("aobt3_eobt1"),
            secs("AOBT_3_flt", "SCHED_TIME_UTC_mvt").alias("aobt3_sched"),
            secs("ARVT_3_flt", "AOBT_3_flt").alias("blocktime3"),
            secs("ARVT_1_flt", "EOBT_1_flt").alias("blocktime1"),
            (pl.col("ADES_FILED_flt") != pl.col("ADES_flt")).cast(pl.Int8).alias("diverted"),
            pl.col("FLIGHT_mvt").str.slice(0, 3).alias("cs_pre"),
            pl.col("FLIGHT_mvt").str.len_chars().alias("cs_len"),
            pl.col("FLIGHT_mvt").str.contains(r"^[A-Z]{3}[0-9]").cast(pl.Int8).alias("cs_airline_style"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"),
            (pl.col("ADEP_mvt") + "_" + pl.col("unmatched").cast(pl.Utf8)).alias("apt_unm"),
            (pl.col("ADEP_mvt") + "_" + pl.col("STAND_mvt")).alias("apt_stand"),
            ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60)
            .cast(pl.Int8)
            .alias("fb"),
        ]
    )
)
dep = dep.with_columns(((pl.col("fb") == 1) | (pl.col("y") > 7200)).cast(pl.Int8).alias("fb_pru"))

# ---------- reference stats & target encodings (leave-one-month-out for train rows; all-2025 for ranking rows) ----------
log("reference stats / target encodings")
trn = dep.filter(pl.col("is_rank") == 0)


def lomo_stats(keys, name, agg_exprs):
    tot = trn.group_by(keys).agg(agg_exprs["tot"])
    per = trn.group_by(keys + ["m"]).agg(agg_exprs["per"])
    # train rows: subtract own month; rank rows: use totals
    j = (
        dep.select(["MVT_ID_mvt", "is_rank", "m"] + keys)
        .join(tot, on=keys, how="left")
        .join(per, on=keys + ["m"], how="left")
    )
    return j


# P10 via quantile can't be subtracted; approximate LOMO by computing per-month-excluded stats directly for the 12 months (cost ok)
def lomo_quantile(keys, prefix):
    parts = []
    for mm in range(1, 13):
        sub = (
            trn.filter(pl.col("m") != mm)
            .group_by(keys)
            .agg(
                pl.col("y").quantile(0.1).alias(f"{prefix}_p10"),
                pl.col("y").median().alias(f"{prefix}_med"),
                pl.col("y").mean().alias(f"{prefix}_mean"),
                pl.len().alias(f"{prefix}_n"),
            )
        )
        parts.append(
            dep.filter((pl.col("is_rank") == 0) & (pl.col("m") == mm))
            .select(["MVT_ID_mvt"] + keys)
            .join(sub, on=keys, how="left")
            .drop(keys)
        )
    full = trn.group_by(keys).agg(
        pl.col("y").quantile(0.1).alias(f"{prefix}_p10"),
        pl.col("y").median().alias(f"{prefix}_med"),
        pl.col("y").mean().alias(f"{prefix}_mean"),
        pl.len().alias(f"{prefix}_n"),
    )
    parts.append(
        dep.filter(pl.col("is_rank") == 1).select(["MVT_ID_mvt"] + keys).join(full, on=keys, how="left").drop(keys)
    )
    return pl.concat(parts)


for keys, prefix in [
    (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], "ref_sr"),
    (["ADEP_mvt", "STAND_mvt"], "ref_s"),
    (["ADEP_mvt", "RUNWAY_mvt"], "ref_r"),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], "te_op"),
    (["ADEP_mvt", "FLIGHT_mvt"], "te_fno"),
    (["ADEP_mvt", "cs_pre"], "te_cs"),
    (["ADEP_mvt", "ADES_mvt"], "te_des"),
    (["ADEP_mvt", "STAND_mvt", "hour"], "ref_sh"),
]:
    dep = dep.join(lomo_quantile(keys, prefix), on="MVT_ID_mvt", how="left")
    log("  ", prefix)
dep = dep.with_columns(
    (pl.col("mvt_aobt3") - pl.col("ref_sr_p10")).alias("aobt3_minus_ref"),
    (pl.col("y") - pl.col("ref_sr_p10")).alias("y_minus_ref"),
)

# ---------- cross-airport linkage: the same flight's arrival record at the destination (if it is one of the 10 airports) ----------
log("destination arrival linkage")
arr_link = (
    arr.filter(pl.col("ADES_mvt").is_in(dep["ADEP_mvt"].unique().to_list()))
    .select(
        pl.col("FLIGHT_mvt"),
        pl.col("ADEP_mvt").alias("link_adep"),
        pl.col("ADES_mvt"),
        pl.col("t").alias("t_land"),
        pl.col("t_block").alias("t_inblock"),
        pl.col("FLIGHT_ID_mvt").is_null().cast(pl.Int8).alias("dst_unmatched"),
        pl.col("AOBT_3_flt").dt.epoch("s").alias("dst_aobt3"),
        pl.col("EOBT_1_flt").dt.epoch("s").alias("dst_eobt1"),
        pl.col("LOBT_flt").dt.epoch("s").alias("dst_lobt"),
        pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").alias("dst_sched"),
    )
    .drop_nulls(["FLIGHT_mvt"])
    .sort("t_land")
)
dep = dep.sort("t").join_asof(
    arr_link, left_on="t", right_on="t_land", by=["FLIGHT_mvt", "ADES_mvt"], strategy="forward", tolerance=8 * 3600
)
dep = dep.with_columns(
    pl.col("t_land").is_not_null().cast(pl.Int8).alias("dst_found"),
    (pl.col("link_adep") == pl.col("ADEP_mvt")).cast(pl.Int8).alias("dst_adep_match"),
    (pl.col("t_land") - pl.col("t")).alias("dst_flight_time"),
    (pl.col("t") - pl.col("dst_aobt3")).alias("dst_mvt_aobt3"),
    (pl.col("t") - pl.col("dst_eobt1")).alias("dst_mvt_eobt1"),
    (pl.col("t") - pl.col("dst_lobt")).alias("dst_mvt_lobt"),
    (pl.col("t_inblock") - pl.col("dst_sched")).alias("dst_arr_delay"),
).drop("link_adep", "t_land", "t_inblock", "dst_aobt3", "dst_eobt1", "dst_lobt", "dst_sched")

# ---------- airport-day / airport-hour disruption context (test-visible: all take-off/landing times known) ----------
log("day/hour context")
dep = dep.with_columns(
    (pl.col("t") // 86400).alias("day_idx"),
    (pl.col("t") // 3600).alias("hr_idx"),
    pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("wd"),
)
allmv = allm.with_columns(
    (pl.col("t") // 86400).alias("day_idx"),
    (pl.col("t") // 3600).alias("hr_idx"),
    pl.col("MVT_TIME_UTC_mvt").dt.weekday().alias("wd"),
    pl.when(pl.col("is_dep")).then(pl.col("ADEP_mvt")).otherwise(pl.col("ADES_mvt")).alias("apt"),
)
dayagg = allmv.group_by(["apt", "day_idx"]).agg(
    pl.col("is_dep").sum().alias("day_ndep"),
    (~pl.col("is_dep")).sum().alias("day_narr"),
    pl.col("t").sort().diff().median().alias("day_med_gap"),
    pl.col("wd").first().alias("wd"),
)
# typical volume for the same airport & weekday over the surrounding 4 weeks (rolling median over 8 same-weekday days)
dayagg = dayagg.sort(["apt", "wd", "day_idx"]).with_columns(
    pl.col("day_ndep")
    .rolling_median(window_size=9, center=True, min_samples=3)
    .over(["apt", "wd"])
    .alias("day_ndep_typ"),
    pl.col("day_narr")
    .rolling_median(window_size=9, center=True, min_samples=3)
    .over(["apt", "wd"])
    .alias("day_narr_typ"),
)
dayagg = dayagg.with_columns(
    (pl.col("day_ndep") / pl.col("day_ndep_typ")).alias("day_dep_ratio"),
    (pl.col("day_narr") / pl.col("day_narr_typ")).alias("day_arr_ratio"),
)
# departures' actual vs scheduled time spread per airport-day (delay level of the day) using sched of departures
dd = allmv.filter(pl.col("is_dep") & pl.col("t_sched").is_not_null()).with_columns(
    (pl.col("t") - pl.col("t_sched")).alias("dsch")
)
dd = dd.group_by(["apt", "day_idx"]).agg(
    pl.col("dsch").median().alias("day_med_dep_delay"),
    pl.col("dsch").quantile(0.9).alias("day_p90_dep_delay"),
    (pl.col("dsch") > 3600).mean().alias("day_share_dep_late1h"),
)
hh = allmv.group_by(["apt", "hr_idx"]).agg(
    pl.col("is_dep").sum().alias("hr_ndep"), (~pl.col("is_dep")).sum().alias("hr_narr")
)
hh = (
    hh.with_columns((pl.col("hr_idx") % 24).alias("hod"))
    .sort(["apt", "hod", "hr_idx"])
    .with_columns(
        pl.col("hr_ndep")
        .rolling_median(window_size=15, center=True, min_samples=3)
        .over(["apt", "hod"])
        .alias("hr_ndep_typ")
    )
    .with_columns((pl.col("hr_ndep") / (pl.col("hr_ndep_typ") + 1)).alias("hr_dep_ratio"))
)
dep = dep.join(
    dayagg.select("apt", "day_idx", "day_ndep", "day_narr", "day_med_gap", "day_dep_ratio", "day_arr_ratio").rename(
        {"apt": "ADEP_mvt"}
    ),
    on=["ADEP_mvt", "day_idx"],
    how="left",
)
dep = dep.join(dd.rename({"apt": "ADEP_mvt"}), on=["ADEP_mvt", "day_idx"], how="left")
dep = dep.join(
    hh.select("apt", "hr_idx", "hr_ndep", "hr_narr", "hr_dep_ratio").rename({"apt": "ADEP_mvt"}),
    on=["ADEP_mvt", "hr_idx"],
    how="left",
).drop("day_idx", "hr_idx", "wd")

# ---------- same-day arrival substitution evidence (arrivals' block times are known in train AND test) ----------
log("same-day arrival substitution shares")
arr_ev = allm.filter(~pl.col("is_dep")).with_columns(
    pl.col("ADES_mvt").alias("apt"),
    pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
    pl.col("FLIGHT_mvt").str.slice(0, 3).alias("op3"),
    ((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60)
    .cast(pl.Int8)
    .alias("afb"),
    (pl.col("t") - pl.col("t_sched") > 1800).cast(pl.Int8).alias("alate"),
)
arr_ev = arr_ev.with_columns((pl.col("afb") * pl.col("alate")).alias("aart"))
a_day = arr_ev.group_by("apt", "day").agg(
    pl.col("afb").mean().alias("arrfb_day"), pl.col("aart").sum().alias("arrart_day_n"), pl.len().alias("arr_day_n")
)
a_dop = arr_ev.group_by("apt", "day", "op3").agg(
    pl.col("afb").mean().alias("arrfb_dayop"),
    pl.col("aart").sum().alias("arrart_dayop_n"),
    pl.col("alate").sum().alias("arrlate_dayop_n"),
    pl.len().alias("arr_dayop_n"),
)
a_op = arr_ev.group_by("apt", "op3").agg(pl.col("afb").mean().alias("arrfb_op"))
dep = dep.with_columns(
    pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"), pl.col("FLIGHT_mvt").str.slice(0, 3).alias("op3_")
)
dep = dep.join(a_day.rename({"apt": "ADEP_mvt"}), on=["ADEP_mvt", "day"], how="left")
dep = dep.join(a_dop.rename({"apt": "ADEP_mvt", "op3": "op3_"}), on=["ADEP_mvt", "day", "op3_"], how="left")
dep = dep.join(a_op.rename({"apt": "ADEP_mvt", "op3": "op3_"}), on=["ADEP_mvt", "op3_"], how="left")
dep = dep.with_columns(
    (pl.col("arrart_dayop_n") / pl.col("arrlate_dayop_n").clip(1)).alias("arrart_dayop_share"),
    (pl.col("arrfb_dayop") - pl.col("arrfb_op")).alias("arrfb_dayop_excess"),
).drop("day", "op3_")

# ---------- neighbour realised-taxi proxies: NM off-block (AOBT_3) is known for matched departures in train AND test, so
#            MVT - AOBT_3 of the departures just before / after a flight measures the queue it sits in ----------
log("neighbour taxi proxies")
nb = dep.filter(pl.col("mvt_aobt3").is_not_null() & (pl.col("mvt_aobt3") > 0) & (pl.col("mvt_aobt3") < 7200)).select(
    "ADEP_mvt", "apt_rwy", "t", pl.col("mvt_aobt3").alias("tx")
)


def nbr_stats(keys, tag):
    for direction in ("m", "p"):
        k = (
            nb.select(keys + ["t", "tx"])
            .with_columns((pl.col("t") if direction == "m" else -pl.col("t")).alias("tt"))
            .sort(keys + ["tt"])
        )
        k = k.with_columns(pl.from_epoch("tt", time_unit="s").alias("ts"), pl.lit(1.0).alias("one"))
        ex = []
        for w in (15, 30, 60):
            ex += [
                pl.col("tx")
                .rolling_median_by("ts", window_size=f"{w}m", closed="both")
                .over(keys)
                .alias(f"nbr_{tag}_{direction}{w}_med"),
                pl.col("tx")
                .rolling_max_by("ts", window_size=f"{w}m", closed="both")
                .over(keys)
                .alias(f"nbr_{tag}_{direction}{w}_max"),
                pl.col("tx")
                .rolling_mean_by("ts", window_size=f"{w}m", closed="both")
                .over(keys)
                .alias(f"nbr_{tag}_{direction}{w}_mean"),
                pl.col("one")
                .rolling_sum_by("ts", window_size=f"{w}m", closed="both")
                .over(keys)
                .alias(f"nbr_{tag}_{direction}{w}_n"),
            ]
        k = k.with_columns(ex).drop("ts", "one", "tx").with_columns(pl.col("t").alias("t_key")).drop("tt")
        k = k.sort("t_key")
        globals()["dep"] = (
            globals()["dep"]
            .sort("t")
            .join_asof(
                k.drop("t") if "t" in k.columns else k,
                left_on="t",
                right_on="t_key",
                by=keys,
                strategy="backward" if direction == "m" else "forward",
                tolerance=3600,
            )
            .drop("t_key")
        )


nbr_stats(["ADEP_mvt"], "apt")
nbr_stats(["apt_rwy"], "rwy")
# arrivals' taxi-in (block - landing) is also known in test: surface congestion proxy
ta = (
    allm.filter(~pl.col("is_dep") & pl.col("t_block").is_not_null())
    .with_columns((pl.col("t_block") - pl.col("t")).alias("tin"))
    .filter((pl.col("tin") > 0) & (pl.col("tin") < 7200))
)
ta = (
    ta.select(pl.col("ADES_mvt").alias("ADEP_mvt"), "t", "tin")
    .sort("ADEP_mvt", "t")
    .with_columns(pl.from_epoch("t", time_unit="s").alias("ts"))
)
ta = (
    ta.with_columns(
        [
            pl.col("tin")
            .rolling_median_by("ts", window_size=f"{w}m", closed="both")
            .over("ADEP_mvt")
            .alias(f"nbr_arr_m{w}_med")
            for w in (30, 60)
        ]
        + [
            pl.col("tin")
            .rolling_max_by("ts", window_size="30m", closed="both")
            .over("ADEP_mvt")
            .alias("nbr_arr_m30_max")
        ]
    )
    .drop("ts", "tin")
    .rename({"t": "t_key"})
    .sort("t_key")
)
dep = (
    dep.sort("t")
    .join_asof(ta, left_on="t", right_on="t_key", by=["ADEP_mvt"], strategy="backward", tolerance=3600)
    .drop("t_key")
)
log("  nbr columns:", len([c for c in dep.columns if c.startswith("nbr_")]))

# ---------- nbrx: extended neighbour stats (short 5-min and long 120-min windows, upper quantile, long-taxi share) ----------
log("neighbour taxi proxies, extended")


def nbrx_stats(keys, tag):
    for direction in ("m", "p"):
        k = (
            nb.select(keys + ["t", "tx"])
            .with_columns((pl.col("t") if direction == "m" else -pl.col("t")).alias("tt"))
            .sort(keys + ["tt"])
        )
        k = k.with_columns(
            pl.from_epoch("tt", time_unit="s").alias("ts"),
            pl.lit(1.0).alias("one"),
            (pl.col("tx") > 1200).cast(pl.Float64).alias("long"),
        )
        ex = [
            pl.col("tx")
            .rolling_median_by("ts", window_size="5m", closed="both")
            .over(keys)
            .alias(f"nbrx_{tag}_{direction}5_med"),
            pl.col("one")
            .rolling_sum_by("ts", window_size="5m", closed="both")
            .over(keys)
            .alias(f"nbrx_{tag}_{direction}5_n"),
            pl.col("tx")
            .rolling_quantile_by("ts", window_size="30m", quantile=0.9, closed="both")
            .over(keys)
            .alias(f"nbrx_{tag}_{direction}30_q90"),
            pl.col("long")
            .rolling_mean_by("ts", window_size="30m", closed="both")
            .over(keys)
            .alias(f"nbrx_{tag}_{direction}30_longshare"),
            pl.col("tx")
            .rolling_median_by("ts", window_size="120m", closed="both")
            .over(keys)
            .alias(f"nbrx_{tag}_{direction}120_med"),
            pl.col("tx")
            .rolling_quantile_by("ts", window_size="120m", quantile=0.9, closed="both")
            .over(keys)
            .alias(f"nbrx_{tag}_{direction}120_q90"),
        ]
        k = (
            k.with_columns(ex)
            .drop("ts", "one", "long", "tx")
            .with_columns(pl.col("t").alias("t_key"))
            .drop("tt")
            .sort("t_key")
        )
        globals()["dep"] = (
            globals()["dep"]
            .sort("t")
            .join_asof(
                k.drop("t"),
                left_on="t",
                right_on="t_key",
                by=keys,
                strategy="backward" if direction == "m" else "forward",
                tolerance=7200,
            )
            .drop("t_key")
        )


nbrx_stats(["ADEP_mvt"], "apt")
nbrx_stats(["apt_rwy"], "rwy")
# trend: short-window median minus long-window median (queue building up or draining)
dep = dep.with_columns(
    (pl.col("nbrx_rwy_m5_med") - pl.col("nbrx_rwy_m120_med")).alias("nbrx_rwy_trend"),
    (pl.col("nbr_apt_m15_med") - pl.col("nbrx_apt_m120_med")).alias("nbrx_apt_trend"),
)
log("  nbrx columns:", len([c for c in dep.columns if c.startswith("nbrx_")]))

# ---------- shrinkage (hierarchical) target encodings, leave-one-month-out, clipped target ----------
log("shrinkage target encodings")
dep = dep.with_columns((pl.col("dow") * 24 + pl.col("hour")).alias("how"))
trn2 = dep.filter(pl.col("is_rank") == 0).with_columns(pl.col("y").clip(0, 3600).alias("yc"))


def shrink_te(keys, parent, prefix, m_smooth=30):
    tot = trn2.group_by(keys).agg(pl.col("yc").sum().alias("s_"), pl.len().alias("n_"))
    per = trn2.group_by(keys + ["m"]).agg(pl.col("yc").sum().alias("sm_"), pl.len().alias("nm_"))
    ptot = trn2.group_by(parent).agg(pl.col("yc").sum().alias("ps_"), pl.len().alias("pn_"))
    pper = trn2.group_by(parent + ["m"]).agg(pl.col("yc").sum().alias("psm_"), pl.len().alias("pnm_"))
    cols = list(dict.fromkeys(keys + parent))
    j = (
        dep.select(["MVT_ID_mvt", "is_rank", "m"] + cols)
        .join(tot, on=keys, how="left")
        .join(per, on=keys + ["m"], how="left")
        .join(ptot, on=parent, how="left")
        .join(pper, on=parent + ["m"], how="left")
    )
    j = j.with_columns([pl.col(c).fill_null(0) for c in ("s_", "n_", "sm_", "nm_", "ps_", "pn_", "psm_", "pnm_")])
    own = pl.col("is_rank") == 0
    j = j.with_columns(
        (pl.col("s_") - pl.when(own).then(pl.col("sm_")).otherwise(0)).alias("s_"),
        (pl.col("n_") - pl.when(own).then(pl.col("nm_")).otherwise(0)).alias("n_"),
        (pl.col("ps_") - pl.when(own).then(pl.col("psm_")).otherwise(0)).alias("ps_"),
        (pl.col("pn_") - pl.when(own).then(pl.col("pnm_")).otherwise(0)).alias("pn_"),
    )
    j = j.with_columns((pl.col("ps_") / pl.col("pn_").clip(1)).alias("prior_"))
    j = j.with_columns(
        ((pl.col("s_") + m_smooth * pl.col("prior_")) / (pl.col("n_") + m_smooth)).alias(f"{prefix}_te"),
        pl.col("n_").cast(pl.Float32).alias(f"{prefix}_n"),
    )
    return j.select("MVT_ID_mvt", f"{prefix}_te", f"{prefix}_n")


for keys, parent, prefix in [
    (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], "te2_sr"),
    (["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "hour"], ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], "te2_srh"),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_flt"], ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"], "te2_opt"),
    (["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], "te2_opr"),
    (["ADEP_mvt", "how"], ["ADEP_mvt"], "te2_how"),
    (["ADEP_mvt", "RUNWAY_mvt", "how"], ["ADEP_mvt", "RUNWAY_mvt"], "te2_rhow"),
    (["ADEP_mvt", "ADES_mvt"], ["ADEP_mvt"], "te2_des"),
    (["ADEP_mvt", "AIRCRAFT_TYPE_flt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], "te2_tr"),
    (["ADEP_mvt", "cs_pre", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], "te2_csr"),
    (["ADEP_mvt", "STAND_mvt", "ADES_mvt"], ["ADEP_mvt", "STAND_mvt"], "te2_sdes"),
]:
    dep = dep.join(shrink_te(keys, parent, prefix), on="MVT_ID_mvt", how="left")
    log("  ", prefix)
dep = dep.drop("how")

# ---------- arrx: transductive arrival-offset context (arrivals are fully labelled in train AND ranking months) ----------
log("arrival-offset context (arrx)")
arx = allm.filter(~pl.col("is_dep") & pl.col("t_block").is_not_null()).with_columns(
    pl.col("ADES_mvt").alias("apt"),
    pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"),
    pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
    pl.col("FLIGHT_mvt").str.slice(0, 3).alias("op3"),
    (pl.col("t_block") - pl.col("ARVT_3_flt").dt.epoch("s")).alias("off_arr"),
    (pl.col("t_block") - pl.col("t")).alias("taxi_in"),
)
arx = arx.filter(pl.col("taxi_in").is_between(0, 7200))
a_day = arx.group_by("apt", "day").agg(
    pl.col("off_arr").median().alias("arrx_off_day_med"),
    (pl.col("off_arr").quantile(0.75) - pl.col("off_arr").quantile(0.25)).alias("arrx_off_day_iqr"),
    pl.col("off_arr").is_null().mean().alias("arrx_off_day_nullshare"),
    pl.col("taxi_in").median().alias("arrx_tin_day_med"),
    pl.col("taxi_in").quantile(0.9).alias("arrx_tin_day_p90"),
    pl.len().alias("arrx_day_n"),
)
a_mon = arx.group_by("apt", "ym").agg(
    pl.col("off_arr").median().alias("off_mon_med"), pl.col("taxi_in").median().alias("tin_mon_med")
)
a_dop = arx.group_by("apt", "day", "op3").agg(
    pl.col("off_arr").median().alias("arrx_off_dayop_med"), pl.len().alias("arrx_dayop_n")
)
a_hr = (
    arx.with_columns((pl.col("t") // 3600).alias("hr"))
    .group_by("apt", "hr")
    .agg(pl.col("taxi_in").median().alias("arrx_tin_hr_med"), pl.len().alias("arrx_hr_n"))
)
dep = dep.with_columns(
    pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day_"),
    pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym_"),
    pl.col("FLIGHT_mvt").str.slice(0, 3).alias("op3_"),
    (pl.col("t") // 3600).alias("hr_"),
)
dep = dep.join(a_day.rename({"apt": "ADEP_mvt", "day": "day_"}), on=["ADEP_mvt", "day_"], how="left").join(
    a_mon.rename({"apt": "ADEP_mvt", "ym": "ym_"}), on=["ADEP_mvt", "ym_"], how="left"
)
dep = dep.join(
    a_dop.rename({"apt": "ADEP_mvt", "day": "day_", "op3": "op3_"}), on=["ADEP_mvt", "day_", "op3_"], how="left"
).join(a_hr.rename({"apt": "ADEP_mvt", "hr": "hr_"}), on=["ADEP_mvt", "hr_"], how="left")
dep = dep.with_columns(
    (pl.col("arrx_off_day_med") - pl.col("off_mon_med")).alias("arrx_off_day_dev"),
    (pl.col("arrx_tin_day_med") - pl.col("tin_mon_med")).alias("arrx_tin_day_dev"),
    (pl.col("arrx_off_dayop_med") - pl.col("off_mon_med")).alias("arrx_off_dayop_dev"),
).drop("day_", "ym_", "op3_", "hr_", "off_mon_med", "tin_mon_med")
log("  arrx columns:", len([c for c in dep.columns if c.startswith("arrx_")]))

# ---------- te3: hierarchical Gaussian target encoding (variance-weighted shrinkage, prior = parent level, posterior precision as feature) ----------
log("hierarchical Gaussian target encodings (te3)")
_yc = pl.col("y").clip(0, 3600)
trn3 = dep.filter(pl.col("is_rank") == 0).with_columns(_yc.alias("yc"), (_yc**2).alias("yc2"))


def _lomo_stats(keys):
    """per-row LOMO (n, mean, var) of clipped y for the group defined by keys: totals minus own month for train rows, totals for rank rows"""
    tot = trn3.group_by(keys).agg(pl.col("yc").sum().alias("s"), pl.col("yc2").sum().alias("s2"), pl.len().alias("n"))
    per = trn3.group_by(keys + ["m"]).agg(
        pl.col("yc").sum().alias("sm"), pl.col("yc2").sum().alias("s2m"), pl.len().alias("nm")
    )
    j = (
        dep.select(["MVT_ID_mvt", "is_rank", "m"] + keys)
        .join(tot, on=keys, how="left")
        .join(per, on=keys + ["m"], how="left")
        .fill_null(0)
    )
    own = pl.col("is_rank") == 0
    j = j.with_columns(
        (pl.col("s") - pl.when(own).then(pl.col("sm")).otherwise(0)).alias("s"),
        (pl.col("s2") - pl.when(own).then(pl.col("s2m")).otherwise(0)).alias("s2"),
        (pl.col("n") - pl.when(own).then(pl.col("nm")).otherwise(0)).alias("n"),
    )
    j = j.with_columns((pl.col("s") / pl.col("n").clip(1)).alias("mean"))
    j = j.with_columns(
        ((pl.col("s2") / pl.col("n").clip(1)) - pl.col("mean") ** 2).clip(1e4, None).alias("var")
    )  # variance floor 100 s^2
    return j.select("MVT_ID_mvt", "n", "mean", "var")


N0 = 20.0
glob_mu = float(trn3["yc"].mean())
glob_var = float(trn3["yc"].var())
chains = [
    ("sr", [["ADEP_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]]),
    ("rh", [["ADEP_mvt"], ["ADEP_mvt", "hour"], ["ADEP_mvt", "RUNWAY_mvt", "hour"]]),
    (
        "ot",
        [
            ["ADEP_mvt"],
            ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt"],
            ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_flt"],
        ],
    ),
    ("sd", [["ADEP_mvt"], ["ADEP_mvt", "ADES_mvt"], ["ADEP_mvt", "ADES_mvt", "STAND_mvt"]]),
]
for tag, levels in chains:
    prior_mu = pl.lit(glob_mu)
    prior_prec = pl.lit(1.0 / glob_var)
    acc = dep.select("MVT_ID_mvt")
    for li, keys in enumerate(levels):
        st = _lomo_stats(keys).rename({"n": f"n{li}", "mean": f"mean{li}", "var": f"var{li}"})
        acc = acc.join(st, on="MVT_ID_mvt", how="left")
        # posterior: prior weight N0 * prior precision, group weight n * (1/var)
        acc = acc.with_columns(
            (
                (N0 * prior_prec * prior_mu + pl.col(f"n{li}") / pl.col(f"var{li}") * pl.col(f"mean{li}"))
                / (N0 * prior_prec + pl.col(f"n{li}") / pl.col(f"var{li}"))
            ).alias(f"mu{li}"),
            (N0 * prior_prec + pl.col(f"n{li}") / pl.col(f"var{li}")).alias(f"prec{li}"),
        )
        prior_mu = pl.col(f"mu{li}")
        prior_prec = pl.col(f"prec{li}")
    out = acc.select(
        "MVT_ID_mvt",
        pl.col("mu1").alias(f"te3_{tag}1_mu"),
        pl.col("prec1").alias(f"te3_{tag}1_prec"),
        pl.col("mu2").alias(f"te3_{tag}2_mu"),
        pl.col("prec2").alias(f"te3_{tag}2_prec"),
        pl.col("n2").alias(f"te3_{tag}2_n"),
    )
    dep = dep.join(out, on="MVT_ID_mvt", how="left")
    log("  te3", tag)
log("  te3 columns:", len([c for c in dep.columns if c.startswith("te3_")]))

# ---------- atfm: EUROCONTROL PRU daily airport ATFM delay by cause + traffic (2019-2026-07, all 10 airports) ----------
_atfm = ROOT / "data/external/atfm_daily.parquet"
if _atfm.exists():
    _a = pl.read_parquet(_atfm)
    dep = (
        dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_day"))
        .join(_a.rename({"day": "_day"}), on=["ADEP_mvt", "_day"], how="left")
        .drop("_day")
    )
    log("atfm joined", len([c for c in dep.columns if c.startswith("atfm_")]), "cols")
else:
    log("no atfm file")
_ert = ROOT / "data/external/enroute_daily.parquet"
if _ert.exists():
    _e = pl.read_parquet(_ert)
    dep = (
        dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_day"))
        .join(_e.rename({"day": "_day"}), on=["ADEP_mvt", "_day"], how="left")
        .drop("_day")
    )
    log("enroute joined", len([c for c in dep.columns if c.startswith("ert_")]), "cols")
else:
    log("no enroute file")

# ---------- optional external joins ----------
geo = ROOT / "data/external/stand_runway_distance.parquet"
if geo.exists():
    g = pl.read_parquet(geo)
    dep = dep.join(g, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"], how="left")
    log("geo joined", g.columns)
else:
    log("no geometry file yet")
opdi = ROOT / "data/external/opdi_features.parquet"
if opdi.exists():
    o = pl.read_parquet(opdi).with_columns(pl.col("MVT_ID_mvt").cast(dep.schema["MVT_ID_mvt"]))
    dep = dep.join(o, on="MVT_ID_mvt", how="left")
    log("opdi joined", len(o.columns) - 1, "cols")
else:
    log("no opdi features file")
# Fail closed: scripts/metar_guard.py raises when the METAR join would leave the 69 wx columns empty.
sys.path.insert(0, str(ROOT / "scripts"))
from metar_guard import join_metar_or_fail

dep = join_metar_or_fail(
    dep, log
)  # raises on a missing file, a failed join or < 99 % coverage; PRC_ALLOW_NO_METAR=1 opts out
dep.write_parquet(a.out)
log("wrote", a.out, dep.shape)
