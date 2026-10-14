# SPDX-License-Identifier: GPL-3.0-only
"""Characterise the huge-error unmatched rows.
2025: for every unmatched departure, what is the label BLOCK relative to every observable timestamp (SCHED, MVT, the inbound leg at the
same stand, the same flight number's previous/next departure, its arrival, same-day duplicates), with day/tz offsets?
2026: the same label-free anchors for the 412 ranking rows of the frozen classes (+ all unmatched ranking rows).
Proxy predictor for 'huge error' on 2025 = queue_eb_ra3qa 12-month OOF (6 two-month folds). Read-only on data/; writes scratchpad/xhr/ only.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, sys, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
os.makedirs("scratchpad/xhr", exist_ok=True)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(300)
pl.Config.set_tbl_rows(200)
pl.Config.set_fmt_str_lengths(24)
print("polars", pl.__version__)
TZ = {
    "LIRF": "Europe/Rome",
    "EHAM": "Europe/Amsterdam",
    "EDDF": "Europe/Berlin",
    "EDDM": "Europe/Berlin",
    "LFPG": "Europe/Paris",
    "LSZH": "Europe/Zurich",
    "LEBL": "Europe/Madrid",
    "LEMD": "Europe/Madrid",
    "EGLL": "Europe/London",
    "LTFM": "Europe/Istanbul",
}
RC = [
    "MVT_ID_mvt",
    "FLIGHT_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "PHASE_mvt",
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "TAXITIME_SEC_mvt",
]
raw = pl.concat(
    [
        pl.read_parquet(f, columns=RC).with_columns(rank=pl.lit(0, pl.Int8))
        for f in sorted(glob.glob("data/raw/training_*.parquet"))
    ]
    + [pl.read_parquet("data/raw/ranking.parquet", columns=RC).with_columns(rank=pl.lit(1, pl.Int8))],
    how="diagonal_relaxed",
)
raw = raw.with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
# ranking DEP rows: BLOCK is a placeholder (label hidden) -> null it
raw = raw.with_columns(
    BLOCK_TIME_UTC_mvt=pl.when((pl.col("rank") == 1) & (pl.col("PHASE_mvt") == "DEP"))
    .then(None)
    .otherwise(pl.col("BLOCK_TIME_UTC_mvt")),
    TAXITIME_SEC_mvt=pl.when((pl.col("rank") == 1) & (pl.col("PHASE_mvt") == "DEP"))
    .then(None)
    .otherwise(pl.col("TAXITIME_SEC_mvt")),
)
print("raw rows", raw.height, raw.group_by("PHASE_mvt", "rank").len().sort("PHASE_mvt", "rank").rows())


def utc_off(col):  # UTC offset in seconds of the airport's local zone at time col (DST-aware)
    return pl.coalesce(
        [
            pl.when(pl.col("ADEP_mvt") == a).then(
                (
                    pl.col(col).dt.convert_time_zone(z).dt.replace_time_zone(None)
                    - pl.col(col).dt.replace_time_zone(None)
                ).dt.total_seconds()
            )
            for a, z in TZ.items()
        ]
    )


def loc_date(col):  # local calendar day number
    return (pl.col(col).dt.epoch("s") + utc_off(col)) // 86400


def loc_tod(col):  # local seconds of day
    return (pl.col(col).dt.epoch("s") + utc_off(col)) % 86400


F = pl.read_parquet(
    "data/features_v9.parquet",
    columns=[
        "MVT_ID_mvt",
        "unmatched",
        "mvt_sched",
        "is_rank",
        "m",
        "y",
        "fb",
        "stand_since_arr_inblock",
        "stand_since_arr_land",
        "stand_since_prev_dep",
        "arr_block_delay",
    ],
).with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
dep = raw.filter(pl.col("PHASE_mvt") == "DEP").join(F, on="MVT_ID_mvt", how="inner")
arr = raw.filter(pl.col("PHASE_mvt") == "ARR")
print("dep", dep.height, "arr", arr.height, "| dep unmatched", int((dep["unmatched"] == 1).sum()))

# ---- proxy OOF: queue_eb_ra3qa 6 folds (pred, p, taxi, art) ----
P = []
for f in sorted(glob.glob("data/oof/queue_eb_ra3qa_months*_months*.npy")):
    a = np.load(f)
    P.append(
        pl.DataFrame(
            {
                "MVT_ID_mvt": a[0].astype(np.int64),
                "px": a[1],
                "px_p": a[2],
                "px_taxi": a[3],
                "px_art": a[4],
                "fold": pl.Series([os.path.basename(f).split("_")[2]] * a.shape[1]),
            }
        )
    )
P = pl.concat(P).unique("MVT_ID_mvt", keep="first")
print("proxy OOF rows", P.height, P.group_by("fold").len().sort("fold").rows())
dep = dep.join(P, on="MVT_ID_mvt", how="left")

# ---- classes (label-free, as unm_resid_lib.cls_expr / frozen_weight_u1) ----
dep = dep.with_columns(
    cls=pl.when(pl.col("unmatched") != 1)
    .then(pl.lit("matched"))
    .when(pl.col("mvt_sched").fill_null(0) > 28800)
    .then(pl.lit("exposed"))
    .when(pl.col("ADEP_mvt") == "LIRF")
    .then(pl.lit("unm_LIRF"))
    .otherwise(pl.lit("unm_other"))
)
U = dep.filter(pl.col("cls") != "matched")
U = U.with_columns(
    sch_day=loc_date("SCHED_TIME_UTC_mvt"),
    mvt_day=loc_date("MVT_TIME_UTC_mvt"),
    sch_tod=loc_tod("SCHED_TIME_UTC_mvt"),
    mvt_tod=loc_tod("MVT_TIME_UTC_mvt"),
    blk_tod=loc_tod("BLOCK_TIME_UTC_mvt"),
    blk_day=loc_date("BLOCK_TIME_UTC_mvt"),
    off_s=utc_off("SCHED_TIME_UTC_mvt"),
    off_m=utc_off("MVT_TIME_UTC_mvt"),
)
U = U.with_columns(
    cross=(pl.col("mvt_day") - pl.col("sch_day")).cast(pl.Int32),
    cross_utc=(pl.col("MVT_TIME_UTC_mvt").dt.date() != pl.col("SCHED_TIME_UTC_mvt").dt.date()),
)


# ---- anchors ----
def asof(left, right, lkey, rkey, by, strategy, tol, prefix, cols):
    r = right.select(by + [pl.col(rkey).alias("_k")] + [pl.col(c).alias(prefix + c) for c in cols]).sort("_k")
    l = left.select("MVT_ID_mvt", *by, pl.col(lkey).alias("_k")).sort("_k")
    j = l.join_asof(r, on="_k", by=by, strategy=strategy, tolerance=tol).drop("_k", *by)
    return left.join(j, on="MVT_ID_mvt", how="left")


ACOLS = ["MVT_ID_mvt", "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]
# inbound: latest arrival at the same airport+stand that landed before MVT (48 h)
arr_s = arr.filter(pl.col("STAND_mvt").is_not_null()).with_columns(_t=pl.col("MVT_TIME_UTC_mvt"))
U = asof(
    U.with_columns(_t=pl.col("MVT_TIME_UTC_mvt")),
    arr_s,
    "_t",
    "_t",
    ["ADEP_mvt", "STAND_mvt"],
    "backward",
    "48h",
    "inb_",
    ACOLS,
)
# inbound by in-block time before SCHED (the aircraft that was at the stand at schedule time)
arr_b = arr.filter(pl.col("STAND_mvt").is_not_null() & pl.col("BLOCK_TIME_UTC_mvt").is_not_null()).with_columns(
    _t=pl.col("BLOCK_TIME_UTC_mvt")
)
U = asof(
    U.with_columns(_t=pl.col("SCHED_TIME_UTC_mvt")),
    arr_b,
    "_t",
    "_t",
    ["ADEP_mvt", "STAND_mvt"],
    "backward",
    "48h",
    "inbS_",
    ACOLS,
)
# previous departure at the same stand (before MVT - 5 min)
dep_s = dep.filter(pl.col("STAND_mvt").is_not_null()).with_columns(_t=pl.col("MVT_TIME_UTC_mvt"))
U = asof(
    U.with_columns(_t=pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=5)),
    dep_s,
    "_t",
    "_t",
    ["ADEP_mvt", "STAND_mvt"],
    "backward",
    "48h",
    "pds_",
    ACOLS + ["unmatched"],
)
# same flight number, same airport: previous departure (SCHED <= own SCHED - 1 h, within 47 h) and next (SCHED >= own + 1 h)
dep_f = dep.with_columns(_t=pl.col("SCHED_TIME_UTC_mvt"))
U = asof(
    U.with_columns(_t=pl.col("SCHED_TIME_UTC_mvt") - pl.duration(hours=1)),
    dep_f,
    "_t",
    "_t",
    ["ADEP_mvt", "FLIGHT_mvt"],
    "backward",
    "47h",
    "sp_",
    ACOLS + ["unmatched"],
)
U = asof(
    U.with_columns(_t=pl.col("SCHED_TIME_UTC_mvt") + pl.duration(hours=1)),
    dep_f,
    "_t",
    "_t",
    ["ADEP_mvt", "FLIGHT_mvt"],
    "forward",
    "47h",
    "sn_",
    ACOLS + ["unmatched"],
)
# same flight number arriving at the same airport before MVT (48 h)
arr_f = arr.with_columns(_t=pl.col("MVT_TIME_UTC_mvt"))
U = asof(
    U.with_columns(_t=pl.col("MVT_TIME_UTC_mvt")),
    arr_f,
    "_t",
    "_t",
    ["ADEP_mvt", "FLIGHT_mvt"],
    "backward",
    "48h",
    "fa_",
    ACOLS,
)
# same flight number departing from another of the 10 airports nearest before MVT (the previous leg)
dep_o = dep.select(pl.col("ADEP_mvt").alias("_o"), *ACOLS).with_columns(_t=pl.col("MVT_TIME_UTC_mvt"))
U = asof(
    U.with_columns(_t=pl.col("MVT_TIME_UTC_mvt") - pl.duration(minutes=30)),
    dep_o,
    "_t",
    "_t",
    ["FLIGHT_mvt"],
    "backward",
    "48h",
    "ol_",
    ACOLS + ["_o"],
)
# same-day duplicates of the flight number at the airport (local schedule day)
dd = (
    dep.with_columns(sch_day=loc_date("SCHED_TIME_UTC_mvt"))
    .group_by("ADEP_mvt", "FLIGHT_mvt", "sch_day")
    .agg(dup_n=pl.len(), dup_unm=pl.col("unmatched").sum())
)
U = U.join(dd, on=["ADEP_mvt", "FLIGHT_mvt", "sch_day"], how="left")
U = U.drop("_t")

# ---- label geometry (2025) ----
sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()
L = U.filter(pl.col("rank") == 0).with_columns(
    b_sch=sec("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
    b_mvt=sec("BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"),
    b_inb_blk=sec("BLOCK_TIME_UTC_mvt", "inb_BLOCK_TIME_UTC_mvt"),
    b_inb_mvt=sec("BLOCK_TIME_UTC_mvt", "inb_MVT_TIME_UTC_mvt"),
    b_inb_sch=sec("BLOCK_TIME_UTC_mvt", "inb_SCHED_TIME_UTC_mvt"),
    b_inbS_blk=sec("BLOCK_TIME_UTC_mvt", "inbS_BLOCK_TIME_UTC_mvt"),
    b_pds_blk=sec("BLOCK_TIME_UTC_mvt", "pds_BLOCK_TIME_UTC_mvt"),
    b_pds_mvt=sec("BLOCK_TIME_UTC_mvt", "pds_MVT_TIME_UTC_mvt"),
    b_sp_sch=sec("BLOCK_TIME_UTC_mvt", "sp_SCHED_TIME_UTC_mvt"),
    b_sp_blk=sec("BLOCK_TIME_UTC_mvt", "sp_BLOCK_TIME_UTC_mvt"),
    b_sp_mvt=sec("BLOCK_TIME_UTC_mvt", "sp_MVT_TIME_UTC_mvt"),
    b_sn_sch=sec("BLOCK_TIME_UTC_mvt", "sn_SCHED_TIME_UTC_mvt"),
    b_sn_blk=sec("BLOCK_TIME_UTC_mvt", "sn_BLOCK_TIME_UTC_mvt"),
    b_sn_mvt=sec("BLOCK_TIME_UTC_mvt", "sn_MVT_TIME_UTC_mvt"),
    b_fa_blk=sec("BLOCK_TIME_UTC_mvt", "fa_BLOCK_TIME_UTC_mvt"),
    b_fa_mvt=sec("BLOCK_TIME_UTC_mvt", "fa_MVT_TIME_UTC_mvt"),
    b_ol_blk=sec("BLOCK_TIME_UTC_mvt", "ol_BLOCK_TIME_UTC_mvt"),
    b_ol_mvt=sec("BLOCK_TIME_UTC_mvt", "ol_MVT_TIME_UTC_mvt"),
    err=pl.col("px") - pl.col("y"),
)
OFFS = {
    "0": 0,
    "+1h": 3600,
    "-1h": -3600,
    "+2h": 7200,
    "-2h": -7200,
    "+12h": 43200,
    "-12h": -43200,
    "+1d": 86400,
    "-1d": -86400,
    "-1d-1h": -90000,
    "-1d+1h": -82800,
    "-1d-2h": -93600,
    "-1d+2h": -79200,
    "+1d+1h": 90000,
    "+1d-1h": 82800,
    "-2d": -172800,
    "+2d": 172800,
}
ANCH = [
    "b_sch",
    "b_inb_blk",
    "b_inb_mvt",
    "b_inb_sch",
    "b_inbS_blk",
    "b_pds_blk",
    "b_pds_mvt",
    "b_sp_sch",
    "b_sp_blk",
    "b_sp_mvt",
    "b_sn_sch",
    "b_sn_blk",
    "b_sn_mvt",
    "b_fa_blk",
    "b_fa_mvt",
    "b_ol_blk",
    "b_ol_mvt",
]
TOL = 120


# explanation: ORD if 60 <= y <= 5400 (BLOCK is a real off-block before MVT) - checked LAST; otherwise nearest (anchor, offset) within TOL
def explain(df):
    A = {a: df[a].to_numpy().astype(float) for a in ANCH}
    y = df["y"].to_numpy().astype(float)
    ms = df["mvt_sched"].to_numpy().astype(float)
    n = len(y)
    best = np.full(n, "", object)
    bres = np.full(n, np.nan)
    order = [("b_sch", "0")] + [(a, o) for a in ANCH for o in OFFS if not (a == "b_sch" and o == "0")]
    for a, o in order:
        r = A[a] - OFFS[o]
        hit = (np.abs(r) <= TOL) & (best == "")
        best[hit] = f"{a[2:]}{'' if o == '0' else o}"
        bres[hit] = r[hit]
    # DS-like generic: y within 2 h of 86400 (BLOCK = real clock, previous date) if MVT-1d not otherwise explained
    ds = (best == "") & (np.abs(y - 86400) < 7200)
    best[ds] = "MVT-1d(taxi)"
    bres[ds] = y[ds] - 86400
    ord_ = (best == "") & (y >= 60) & (y <= 5400)
    best[ord_] = "ORD"
    best[best == ""] = "UNEXPLAINED"
    return df.with_columns(expl=pl.Series(best.astype(str)), expl_res=pl.Series(bres))


L = explain(L)
print("\n=== 2025 unmatched departures by class ===")
print(
    L.group_by("cls")
    .agg(
        n=pl.len(),
        lirf=(pl.col("ADEP_mvt") == "LIRF").sum(),
        y_med=pl.col("y").median(),
        ms_med=pl.col("mvt_sched").median(),
        proxy_mse=(pl.col("err") ** 2).mean(),
        n_oof=pl.col("px").is_not_null().sum(),
    )
    .sort("cls")
)
print("\n=== explanation of BLOCK, all 2025 unmatched rows, by class ===")
print(
    L.group_by("cls", "expl")
    .agg(
        n=pl.len(),
        sse=(pl.col("err") ** 2).sum() / 1e6,
        y_med=pl.col("y").median(),
        ms_med=pl.col("mvt_sched").median(),
    )
    .sort("cls", "n", descending=[False, True])
)
for thr in (3600, 7200, 14400):
    H = L.filter(pl.col("err").abs() > thr)
    print(
        f"\n=== rows with proxy |err| > {thr} s: n {H.height}, SSE share of all unmatched {float((H['err']**2).sum() / (L['err']**2).sum()):.3f} ==="
    )
    print(
        H.group_by("cls", "ADEP_mvt", "expl")
        .agg(
            n=pl.len(),
            sse_M=(pl.col("err") ** 2).sum() / 1e6,
            y_med=pl.col("y").median(),
            ms_med=pl.col("mvt_sched").median(),
            cross=pl.col("cross").mean(),
            fb=pl.col("fb").mean(),
        )
        .sort("sse_M", descending=True)
        .head(60)
    )
# the time-of-day / day decomposition of BLOCK vs SCHED and vs MVT for the huge rows
H = L.filter(pl.col("err").abs() > 7200).with_columns(
    dday_s=((pl.col("blk_day") - pl.col("sch_day"))),
    dtod_s=(pl.col("blk_tod") - pl.col("sch_tod")),
    dday_m=(pl.col("blk_day") - pl.col("mvt_day")),
    dtod_m=(pl.col("blk_tod") - pl.col("mvt_tod")),
)
print("\n=== huge rows (|err|>7200): BLOCK vs SCHED as (day diff, tod diff bucket 10 min) ===")
print(
    H.group_by("cls", "dday_s", (pl.col("dtod_s") / 600).round().cast(pl.Int32).alias("dtod_s_10m"))
    .agg(n=pl.len(), sse_M=(pl.col("err") ** 2).sum() / 1e6)
    .sort("n", descending=True)
    .head(40)
)
print("\n=== huge rows: BLOCK vs MVT (day diff, tod diff bucket 10 min) ===")
print(
    H.group_by("cls", "dday_m", (pl.col("dtod_m") / 600).round().cast(pl.Int32).alias("dtod_m_10m"))
    .agg(n=pl.len(), sse_M=(pl.col("err") ** 2).sum() / 1e6)
    .sort("n", descending=True)
    .head(40)
)
SHOW = [
    "cls",
    "ADEP_mvt",
    "FLIGHT_mvt",
    "STAND_mvt",
    "SCHED_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "MVT_TIME_UTC_mvt",
    "y",
    "mvt_sched",
    "px",
    "px_p",
    "err",
    "cross",
    "expl",
    "expl_res",
    "inb_FLIGHT_mvt",
    "b_inb_blk",
    "b_inb_mvt",
    "sp_FLIGHT_mvt",
    "b_sp_sch",
    "b_sp_blk",
    "sn_FLIGHT_mvt",
    "b_sn_blk",
    "b_fa_blk",
    "ol__o",
    "b_ol_blk",
    "dup_n",
    "fb",
]
print("\n=== the 120 largest |err| rows (proxy) ===")
print(L.sort(pl.col("err").abs(), descending=True).head(120).select(SHOW))
print("\n=== exposed 2025 by airport x cross x expl ===")
print(
    L.filter(pl.col("cls") == "exposed")
    .group_by("ADEP_mvt", "cross", "expl")
    .agg(n=pl.len(), y_med=pl.col("y").median(), ms_h=pl.col("mvt_sched").median() / 3600)
    .sort("ADEP_mvt", "cross", "n", descending=[False, False, True])
)
print("\n=== unm_LIRF 2025: expl x month ===")
print(
    L.filter(pl.col("cls") == "unm_LIRF")
    .group_by("expl", "m")
    .agg(n=pl.len(), sse_M=(pl.col("err") ** 2).sum() / 1e6)
    .sort("expl", "m")
)
print("\n=== unm_LIRF 2025 rows with |err| > 3600 ===")
print(L.filter((pl.col("cls") == "unm_LIRF") & (pl.col("err").abs() > 3600)).sort("err").select(SHOW))
L.write_parquet("scratchpad/xhr/unm2025.parquet")

# ---- 2026 ranking rows: label-free anchors for the frozen classes ----
R = U.filter(pl.col("rank") == 1)
v62 = (
    pl.read_parquet(os.environ.get("unmatched_rule_file", "stages/unmatched_rule.parquet"))
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    .rename({"TAXITIME_SEC_mvt": "v62"})
)
base = (
    pl.read_parquet(os.environ.get("postrule_line_file", "scratchpad/xpr/xprBX_xsx.parquet"))
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    .rename({"TAXITIME_SEC_mvt": "base"})
)
R = (
    R.join(v62, on="MVT_ID_mvt", how="left")
    .join(base, on="MVT_ID_mvt", how="left")
    .with_columns(
        inb_gap=sec("MVT_TIME_UTC_mvt", "inb_BLOCK_TIME_UTC_mvt"),
        sp_dsch=sec("SCHED_TIME_UTC_mvt", "sp_SCHED_TIME_UTC_mvt"),
        sp_y=pl.col("sp_TAXITIME_SEC_mvt"),
        sn_dsch=sec("sn_SCHED_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        sn_y=pl.col("sn_TAXITIME_SEC_mvt"),
    )
)
print("\n=== 2026 ranking unmatched rows by class ===")
print(
    R.group_by("cls")
    .agg(
        n=pl.len(),
        ms_med=pl.col("mvt_sched").median(),
        v62_med=pl.col("v62").median(),
        base_eq_v62=(pl.col("base") == pl.col("v62")).mean(),
        inb_found=pl.col("inb_MVT_ID_mvt").is_not_null().mean(),
        sp_found=pl.col("sp_MVT_ID_mvt").is_not_null().mean(),
        cross=pl.col("cross").mean(),
    )
    .sort("cls")
)
RS = [
    "cls",
    "ADEP_mvt",
    "FLIGHT_mvt",
    "STAND_mvt",
    "SCHED_TIME_UTC_mvt",
    "MVT_TIME_UTC_mvt",
    "mvt_sched",
    "cross",
    "v62",
    "base",
    "inb_FLIGHT_mvt",
    "inb_gap",
    "sp_FLIGHT_mvt",
    "sp_dsch",
    "sp_y",
    "sn_dsch",
    "sn_y",
    "dup_n",
    "px",
]
print("\n=== 2026 exposed rows (51) ===")
print(R.filter(pl.col("cls") == "exposed").sort("ADEP_mvt", "mvt_sched").select(RS))
print("\n=== 2026 unm_LIRF rows (361), top 60 by v62 ===")
print(R.filter(pl.col("cls") == "unm_LIRF").sort("v62", descending=True).head(60).select(RS))
R.write_parquet("scratchpad/xhr/unm2026.parquet")
print("done")
