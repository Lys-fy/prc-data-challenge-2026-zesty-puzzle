# SPDX-License-Identifier: GPL-3.0-only
"""Step 2 of the unmatched-row characterisation: the 2025 context table scratchpad/xhr/unm2025b.parquet.
Input scratchpad/xhr/unm2025.parquet (written by scripts/exposed_characterise.py before its 2026 part). Fills the proxy OOF of the rows the
six two-month folds do not cover with the janjul fold of the same member (queue_eb_ra3qa), adds the label regime (SUB / DS / ORD / OTHER) and
readable hour columns, prints the largest-error rows, the unexplained rows with all anchors, and SUB/ORD/DS shares by label-free cells at LIRF.
This table is the 2025 context of Kumo F4 (kumo_icl_lirf.py) and the input of exposed_mechanisms.py, exposed_ord_bounds.py.
usage: exposed_context.py [--ref <saved unm2025b.parquet>]   (cwd = repository root or PRC_ROOT; --ref compares the new table with a saved copy)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
pl.Config.set_tbl_cols(60)
pl.Config.set_tbl_width_chars(400)
pl.Config.set_tbl_rows(300)
pl.Config.set_fmt_str_lengths(20)
pl.Config.set_fmt_float("mixed")
OUT = os.environ.get("OUT", "scratchpad/xhr/unm2025b.parquet")
L = pl.read_parquet("scratchpad/xhr/unm2025.parquet")
a = np.load("data/oof/queue_eb_ra3qa_janjul_months17.npy")
J = pl.DataFrame({"MVT_ID_mvt": a[0].astype(np.int64), "pj": a[1], "pj_p": a[2], "pj_taxi": a[3], "pj_art": a[4]})
L = L.join(J, on="MVT_ID_mvt", how="left").with_columns(
    px=pl.coalesce("px", "pj"),
    px_p=pl.coalesce("px_p", "pj_p"),
    px_taxi=pl.coalesce("px_taxi", "pj_taxi"),
    px_art=pl.coalesce("px_art", "pj_art"),
)
L = L.with_columns(
    err=pl.col("px") - pl.col("y"),
    sch_h=(pl.col("sch_tod") / 3600).round(1),
    mvt_h=(pl.col("mvt_tod") / 3600).round(1),
    blk_h=(pl.col("blk_tod") / 3600).round(1),
    al=pl.col("FLIGHT_mvt").str.slice(0, 3),
    ms_h=(pl.col("mvt_sched") / 3600).round(1),
    y_h=(pl.col("y") / 3600).round(2),
    regime=pl.when((pl.col("y") - pl.col("mvt_sched")).abs() < 600)
    .then(pl.lit("SUB"))
    .when((pl.col("y") - 86400).abs() < 7200)
    .then(pl.lit("DS"))
    .when((pl.col("y") >= 60) & (pl.col("y") <= 5400))
    .then(pl.lit("ORD"))
    .otherwise(pl.lit("OTHER")),
    inb_land_after_sched=(pl.col("inb_MVT_TIME_UTC_mvt") > pl.col("SCHED_TIME_UTC_mvt")),
    inb_gap_h=((pl.col("MVT_TIME_UTC_mvt") - pl.col("inb_BLOCK_TIME_UTC_mvt")).dt.total_seconds() / 3600).round(1),
    sp_ms_h=(((pl.col("sp_MVT_TIME_UTC_mvt") - pl.col("sp_SCHED_TIME_UTC_mvt")).dt.total_seconds()) / 3600).round(1),
    sp_y=pl.col("sp_TAXITIME_SEC_mvt"),
    sn_y=pl.col("sn_TAXITIME_SEC_mvt"),
    sn_ms_h=(((pl.col("sn_MVT_TIME_UTC_mvt") - pl.col("sn_SCHED_TIME_UTC_mvt")).dt.total_seconds()) / 3600).round(1),
)
print(
    "proxy coverage by class:",
    L.group_by("cls").agg(n=pl.len(), oof=pl.col("px").is_not_null().sum()).sort("cls").rows(),
)
print("\n=== SSE (proxy) by class x regime, all 2025 unmatched ===")
print(
    L.group_by("cls", "regime")
    .agg(
        n=pl.len(),
        sse_M=(pl.col("err") ** 2).sum() / 1e6,
        mse=(pl.col("err") ** 2).mean(),
        ms_h_med=pl.col("ms_h").median(),
        p_med=pl.col("px_p").median(),
    )
    .sort("cls", "sse_M", descending=[False, True])
)
SHOW = [
    "cls",
    "ADEP_mvt",
    "FLIGHT_mvt",
    "STAND_mvt",
    "m",
    "SCHED_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "MVT_TIME_UTC_mvt",
    "y",
    "mvt_sched",
    "px",
    "px_p",
    "err",
    "cross",
    "regime",
    "expl",
    "inb_FLIGHT_mvt",
    "inb_gap_h",
    "inb_land_after_sched",
    "sp_FLIGHT_mvt",
    "sp_ms_h",
    "sp_y",
    "sn_ms_h",
    "sn_y",
    "dup_n",
    "lk" if "lk" in L.columns else "fb",
]
print("\n=== 80 largest |err| rows ===")
print(L.filter(pl.col("err").is_not_null()).sort(pl.col("err").abs(), descending=True).head(80).select(SHOW))
B = [c for c in L.columns if c.startswith("b_")]
print("\n=== OTHER-regime rows (not SUB/DS/ORD) with y > 3600 or y < 0: all anchors (BLOCK - anchor, s) ===")
O = L.filter((pl.col("regime") == "OTHER") & ((pl.col("y") > 3600) | (pl.col("y") < 0))).sort(
    "cls", "y", descending=[False, True]
)
print(
    O.select(
        [
            "cls",
            "ADEP_mvt",
            "FLIGHT_mvt",
            "STAND_mvt",
            "SCHED_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt",
            "MVT_TIME_UTC_mvt",
            "y",
            "mvt_sched",
            "err",
            "blk_h",
            "sch_h",
            "mvt_h",
        ]
        + B[:9]
    ).head(120)
)
print(
    O.select(
        [
            "cls",
            "FLIGHT_mvt",
            "y",
            "inb_FLIGHT_mvt",
            "pds_FLIGHT_mvt",
            "sp_FLIGHT_mvt",
            "sn_FLIGHT_mvt",
            "fa_FLIGHT_mvt",
            "ol__o",
            "ol_FLIGHT_mvt",
            "dup_n",
        ]
        + B[9:]
    ).head(120)
)
print(
    "\nOTHER rows: y - mvt_sched (s), and (y - mvt_sched) mod 3600:",
    O.select(d=(pl.col("y") - pl.col("mvt_sched"))).with_columns(mod=pl.col("d") % 3600).head(60).rows(),
)
# ---- LIRF: SUB / ORD / DS shares by label-free cells ----
LI = L.filter(pl.col("ADEP_mvt") == "LIRF").with_columns(
    ms_band=pl.when(pl.col("mvt_sched") < 3600)
    .then(pl.lit("a<1h"))
    .when(pl.col("mvt_sched") < 7200)
    .then(pl.lit("b1-2h"))
    .when(pl.col("mvt_sched") < 14400)
    .then(pl.lit("c2-4h"))
    .when(pl.col("mvt_sched") < 28800)
    .then(pl.lit("d4-8h"))
    .when(pl.col("mvt_sched") < 43200)
    .then(pl.lit("e8-12h"))
    .when(pl.col("mvt_sched") < 57600)
    .then(pl.lit("f12-16h"))
    .otherwise(pl.lit("g>16h"))
)


def shares(df, keys, minn=1):
    return (
        df.group_by(keys)
        .agg(
            n=pl.len(),
            SUB=(pl.col("regime") == "SUB").mean(),
            ORD=(pl.col("regime") == "ORD").mean(),
            DS=(pl.col("regime") == "DS").mean(),
            OTH=(pl.col("regime") == "OTHER").mean(),
            p_med=pl.col("px_p").median(),
            sse_M=(pl.col("err") ** 2).sum() / 1e6,
        )
        .filter(pl.col("n") >= minn)
        .sort(keys)
    )


L7 = LI.filter(pl.col("mvt_sched") >= 7200)
print("\n=== LIRF unmatched 2025: regime shares by MS band ===")
print(shares(LI, ["ms_band"]))
print("\n=== by MS band x cross ===")
print(shares(LI, ["ms_band", "cross"]))
print("\n=== by MS band x month ===")
print(shares(L7, ["ms_band", "m"]))
print("\n=== by MS band x inbound landed after SCHED (same stand, <48h) ===")
print(shares(L7, ["ms_band", "inb_land_after_sched"]))
print("\n=== by MS band x inbound gap (h, MVT - inbound in-block) bucket ===")
print(
    shares(
        L7.with_columns(
            gapb=pl.when(pl.col("inb_gap_h").is_null())
            .then(pl.lit("none"))
            .when(pl.col("inb_gap_h") < 1)
            .then(pl.lit("<1h"))
            .when(pl.col("inb_gap_h") < 2)
            .then(pl.lit("1-2h"))
            .when(pl.col("inb_gap_h") < 4)
            .then(pl.lit("2-4h"))
            .when(pl.col("inb_gap_h") < 12)
            .then(pl.lit("4-12h"))
            .otherwise(pl.lit(">12h"))
        ),
        ["ms_band", "gapb"],
    )
)
print("\n=== by MS band x airline (n>=5) ===")
print(shares(L7, ["ms_band", "al"], 5))
print("\n=== by MS band x sched local hour band ===")
print(shares(L7.with_columns(hb=(pl.col("sch_tod") // 10800).cast(pl.Int32)), ["ms_band", "hb"]))
print("\n=== by MS band x duplicate flight number same day ===")
print(shares(L7, ["ms_band", "dup_n"]))
print("\n=== by MS band x previous-day sibling regime (sp_y ~ sp_ms => SUB yesterday) ===")
LI = LI.with_columns(
    sp_reg=pl.when(pl.col("sp_y").is_null())
    .then(pl.lit("none"))
    .when((pl.col("sp_y") - pl.col("sp_ms_h") * 3600).abs() < 600)
    .then(pl.lit("SUB"))
    .when((pl.col("sp_y") >= 60) & (pl.col("sp_y") <= 5400))
    .then(pl.lit("ORD"))
    .otherwise(pl.lit("OTH")),
    sp_unm=pl.col("sp_unmatched"),
)
L7 = LI.filter(pl.col("mvt_sched") >= 7200)
print(shares(L7, ["ms_band", "sp_reg"]))
print("\n=== by MS band x previous-day sibling unmatched? ===")
print(shares(L7, ["ms_band", "sp_unm"]))
print("\n=== by MS band x stand prefix (first char) ===")
print(shares(L7.with_columns(sp1=pl.col("STAND_mvt").str.slice(0, 1)), ["ms_band", "sp1"], 3))
print("\n=== by MS band x runway ===")
print(shares(L7, ["ms_band", "RUNWAY_mvt"], 3))
print("\n=== by MS band x aircraft type (n>=4) ===")
print(shares(L7, ["ms_band", "AIRCRAFT_TYPE_mvt"], 4))
print("\n=== SUB rows: exactness of BLOCK == SCHED (b_sch distribution) ===")
print(
    L.filter(pl.col("regime") == "SUB")
    .group_by("cls")
    .agg(
        n=pl.len(),
        exact0=(pl.col("b_sch") == 0).mean(),
        within5=(pl.col("b_sch").abs() <= 5).mean(),
        within60=(pl.col("b_sch").abs() <= 60).mean(),
        med=pl.col("b_sch").median(),
    )
    .sort("cls")
)
print("\n=== the exposed rows: LIRF by cross x regime with sched hour, mvt hour ===")
print(
    L.filter((pl.col("cls") == "exposed") & (pl.col("ADEP_mvt") == "LIRF"))
    .sort("cross", "regime", "mvt_sched")
    .select(
        [
            "FLIGHT_mvt",
            "m",
            "STAND_mvt",
            "RUNWAY_mvt",
            "sch_h",
            "mvt_h",
            "blk_h",
            "ms_h",
            "y_h",
            "cross",
            "regime",
            "px",
            "px_p",
            "inb_FLIGHT_mvt",
            "inb_gap_h",
            "sp_ms_h",
            "sp_y",
            "sn_ms_h",
            "sn_y",
            "dup_n",
        ]
    )
)
L.write_parquet(OUT)
print("wrote", OUT, L.shape)
# ---- optional: compare with a saved copy (rows, column set, max |diff| per numeric column, exact match of the other columns) ----
if "--ref" in sys.argv:
    R = pl.read_parquet(sys.argv[sys.argv.index("--ref") + 1])
    print(
        f"\n=== compare with the saved table: rows new {L.height} / saved {R.height}; columns only new {sorted(set(L.columns) - set(R.columns))}, only saved {sorted(set(R.columns) - set(L.columns))}"
    )
    k = ["MVT_ID_mvt"]
    R = R.with_columns(pl.col("MVT_ID_mvt").cast(L.schema["MVT_ID_mvt"]))
    J = L.join(R, on=k, how="inner", suffix="__ref")
    worst = 0.0
    bad = []
    print("rows only new / only saved:", L.join(R, on=k, how="anti").height, R.join(L, on=k, how="anti").height)
    for c in sorted(set(L.columns) & set(R.columns) - set(k)):
        a_, b_ = J[c], J[c + "__ref"]
        if a_.dtype.is_numeric() and b_.dtype.is_numeric():
            x, z = a_.cast(pl.Float64).to_numpy(), b_.cast(pl.Float64).to_numpy()
            nan_ok = np.array_equal(np.isnan(x), np.isnan(z))
            d = float(np.nanmax(np.abs(x - z))) if np.isfinite(x - z).any() else 0.0
            worst = max(worst, d)
            if d > 0 or not nan_ok:
                bad.append((c, d, nan_ok))
        elif not a_.cast(pl.Utf8).eq_missing(b_.cast(pl.Utf8)).all():
            bad.append((c, "differs", None))
    print(f"max |diff| over numeric columns {worst:.6g}; columns not identical: {bad if bad else 'none'}")
