# SPDX-License-Identifier: GPL-3.0-only
"""Callsign link of unmatched rows, step 1: characterise the unmatched departures (2025 labelled + 2026 ranking) and write the row frame used by every lever.
Frame scratchpad/xur2/frame.parquet: one row per unmatched departure with label-free fields, the label (2025), the unm_resid/C1 reference
(fold OOF; janjul on the real C1 = V47L x10 + CATV17x x20 predictor) and, for ranking rows, the value of stages/unmatched_resid.parquet (column v76).
Reference (the 'current unmatched predictor'): unm_other non-LTFM = C1 + unm_resid resb correction (S_xLTFM_HUB_resb_c14400_k0_s0.5_g900),
LTFM / unm_LIRF / exposed = C1 proxy (the frozen/px_safe rows are out of scope and only reported).
Also writes the IATA->ICAO callsign map from MATCHED rows (label-free: FLIGHT_mvt vs CALLSIGN_flt, both years)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, re, numpy as np, polars as pl

sys.path.insert(0, "scratchpad/xur")
from unm_resid_lib import *

os.makedirs("scratchpad/xur2", exist_ok=True)
C = [
    "MVT_ID_mvt",
    "is_rank",
    "unmatched",
    "ADEP_mvt",
    "ADES_mvt",
    "FLIGHT_mvt",
    "FLIGHT_RULE_mvt",
    "AIRCRAFT_TYPE_mvt",
    "STAND_mvt",
    "RUNWAY_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "y",
    "mvt_sched",
    "cs_pre",
    "cs_airline_style",
    "CALLSIGN_flt",
    "stand_since_arr_inblock",
    "stand_since_prev_dep",
]
A = pl.read_parquet("data/features_v9.parquet", columns=C)
# ---- callsign map from matched rows (label-free)
M = A.filter((pl.col("unmatched") == 0) & pl.col("CALLSIGN_flt").is_not_null()).select(
    apt="ADEP_mvt",
    fl=pl.col("FLIGHT_mvt").str.strip_chars(),
    cs=pl.col("CALLSIGN_flt").str.strip_chars(),
    t=pl.col("MVT_TIME_UTC_mvt"),
)
MAP = (
    M.group_by("apt", "fl", "cs").agg(n=pl.len(), t0=pl.col("t").min(), t1=pl.col("t").max()).sort("n", descending=True)
)
MAP.write_parquet("scratchpad/xur2/csmap.parquet")
U = A.filter(pl.col("unmatched") == 1)
for f, cs in [
    ("adsb", ["adsb_linked"]),
    ("adsbU", ["adsbU_linked", "adsbU_best", "adsbU_lb", "adsbU_dt_tko"]),
    ("adsb2", ["adsb2_leave", "adsb2_at"]),
]:
    U = U.join(
        pl.read_parquet(f"data/external/{f}_features.parquet", columns=["MVT_ID_mvt"] + cs), on="MVT_ID_mvt", how="left"
    )
inv = {m: f for f, ms in FOLDS.items() for m in ms}
U = U.with_columns(
    cls_expr(),
    lk=pl.col("adsb_linked").fill_null(0).cast(pl.Int8),
    lkU=pl.col("adsbU_linked").fill_null(0).cast(pl.Int8),
    mon=pl.col("MVT_TIME_UTC_mvt").dt.month(),
    fl=pl.col("FLIGHT_mvt").str.strip_chars(),
).sort("MVT_ID_mvt")
U = U.with_columns(
    fold=pl.when(pl.col("is_rank") == 1)
    .then(pl.lit("rank"))
    .otherwise(pl.col("mon").replace_strict(inv, default="none")),
    style=pl.when(pl.col("fl").str.contains(r"^[A-Z]{3}[0-9][0-9A-Z]*$"))
    .then(pl.lit("ICAO"))
    .when(pl.col("fl").str.contains(r"^([A-Z]{2}|[A-Z][0-9]|[0-9][A-Z])[0-9]{1,4}[A-Z]?$"))
    .then(pl.lit("IATA"))
    .when(pl.col("fl").str.contains(r"^[A-Z]{1,2}[A-Z0-9]{3,5}$") & ~pl.col("fl").str.contains(r"[0-9]{3}"))
    .then(pl.lit("REG"))
    .otherwise(pl.lit("OTHER")),
)
# known map hit: same airport and flight number seen on a matched row
hit = MAP.group_by("apt", "fl").agg(ncs=pl.col("cs").n_unique(), nmap=pl.col("n").sum())
U = U.join(hit, left_on=["ADEP_mvt", "fl"], right_on=["apt", "fl"], how="left").with_columns(
    inmap=pl.col("nmap").fill_null(0) > 0
)
# ---- reference predictions
z = np.load("scratchpad/xur/S_xLTFM_HUB_resb_c14400_k0_s0.5_g900.npz", allow_pickle=True)
ids = U["MVT_ID_mvt"].to_numpy().astype(float)
assert np.array_equal(z["ids"].astype(float), ids)
fo = U["fold"].to_numpy()
base = z["base"]
final = z["final"]
v53j = np.maximum(
    0.5 * np.mean([member(U, f"V47L_s{i}") for i in range(10)], axis=0)
    + 0.5 * np.mean([member(U, f"CATV17x_s{i}") for i in range(20)], axis=0),
    60.0,
)
jj = fo == "janjul"
ref = np.where(jj & np.isfinite(v53j), np.maximum(v53j + (final - base), 60.0), final)
c1 = np.where(jj & np.isfinite(v53j), v53j, base)
V76 = pl.read_parquet("stages/unmatched_resid.parquet").rename({"TAXITIME_SEC_mvt": "v76"})
U = U.with_columns(ref=pl.Series(ref), c1=pl.Series(c1)).join(V76, on="MVT_ID_mvt", how="left")
isr = U["is_rank"].to_numpy() == 1
print(
    "ranking: ref (unm_resid S final) == v76 on unm_other non-LTFM?",
    np.mean(np.abs(np.rint(U["ref"].to_numpy()[isr]) - U["v76"].to_numpy()[isr]) < 1.5),
)
U.write_parquet("scratchpad/xur2/frame.parquet")
# ---------------- characterisation
U = U.with_columns(
    grp=pl.when(pl.col("cls") == "unm_other")
    .then(pl.when(pl.col("ADEP_mvt") == "LTFM").then(pl.lit("unm_LTFM")).otherwise(pl.lit("unm_oth")))
    .otherwise(pl.col("cls")),
    yr=pl.when(pl.col("is_rank") == 1).then(pl.lit("2026")).otherwise(pl.lit("2025")),
)
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
print("\n== counts by group x year")
print(U.group_by("grp", "yr").len().pivot(on="yr", index="grp", values="len"))
S = U.filter(pl.col("grp").is_in(["unm_oth", "unm_LTFM"]))
print("\n== airport x year (unm_other incl. LTFM)")
print(S.group_by("ADEP_mvt", "yr").len().pivot(on="yr", index="ADEP_mvt", values="len").sort("ADEP_mvt"))
print("\n== flight-number style x airport (share %, both years)")
print(
    S.group_by("ADEP_mvt", "style")
    .len()
    .with_columns(pct=(100 * pl.col("len") / pl.col("len").sum().over("ADEP_mvt")).round(1))
    .pivot(on="style", index="ADEP_mvt", values="pct")
    .sort("ADEP_mvt")
)
print("\n== flight rule / map / linked by airport x year")
print(
    S.group_by("ADEP_mvt", "yr")
    .agg(
        n=pl.len(),
        vfr=(pl.col("FLIGHT_RULE_mvt") == "V").mean().round(3),
        inmap=pl.col("inmap").mean().round(3),
        lk=pl.col("lk").mean().round(3),
        lkU=pl.col("lkU").mean().round(3),
        lk_any=((pl.col("lk") == 1) | (pl.col("lkU") == 1)).mean().round(3),
        ms_med=pl.col("mvt_sched").median(),
        y_med=pl.col("y").median(),
        y_p90=pl.col("y").quantile(0.9),
    )
    .sort("ADEP_mvt", "yr")
)
print("\n== top airline prefixes per airport (both years, unm_other)")
for a in APTS:
    t = (
        S.filter(pl.col("ADEP_mvt") == a)
        .group_by("cs_pre")
        .agg(n=pl.len(), n26=(pl.col("yr") == "2026").sum())
        .sort("n", descending=True)
        .head(8)
    )
    print(a, ", ".join(f"{p}:{n}/{m}" for p, n, m in t.rows()))
print("\n== aircraft type classes (unm_other): top types")
print(S.group_by("AIRCRAFT_TYPE_mvt").len().sort("len", descending=True).head(20).rows())
# label distribution: unmatched vs matched (2025)
L = pl.read_parquet("data/features_v9.parquet", columns=["ADEP_mvt", "unmatched", "is_rank", "y", "mvt_sched"]).filter(
    pl.col("is_rank") == 0
)
print("\n== label distribution 2025: matched vs unm_other (quantiles of y, s) and share near-SCHED |y-ms|<300")
q = (
    L.group_by("ADEP_mvt", "unmatched")
    .agg(
        n=pl.len(),
        q10=pl.col("y").quantile(0.1),
        q50=pl.col("y").median(),
        q90=pl.col("y").quantile(0.9),
        q99=pl.col("y").quantile(0.99),
        mean=pl.col("y").mean().round(0),
        sd=pl.col("y").std().round(0),
        nearS=((pl.col("y") - pl.col("mvt_sched")).abs() < 300).mean().round(3),
    )
    .sort("ADEP_mvt", "unmatched")
)
print(q)
# reference error on 2025 folds
T = S.filter((pl.col("yr") == "2025") & pl.col("y").is_not_null() & pl.col("ref").is_not_null()).with_columns(
    e2=(pl.col("ref") - pl.col("y")) ** 2, e2c1=(pl.col("c1") - pl.col("y")) ** 2
)
print("\n== 2025 reference MSE (ref = unm_resid+C1) by airport, linked-any vs not; share of SSE in top 1 % rows")
print(
    T.group_by("ADEP_mvt")
    .agg(
        n=pl.len(),
        mse=pl.col("e2").mean().round(0),
        mse_c1=pl.col("e2c1").mean().round(0),
        med_abs=pl.col("e2").sqrt().median().round(0),
        mse_lk=pl.col("e2").filter((pl.col("lk") == 1) | (pl.col("lkU") == 1)).mean().round(0),
        mse_dark=pl.col("e2").filter((pl.col("lk") == 0) & (pl.col("lkU") == 0)).mean().round(0),
        top1=(pl.col("e2").sort(descending=True).head(pl.len() // 100 + 1).sum() / pl.col("e2").sum()).round(3),
    )
    .sort("ADEP_mvt")
)
print("\n== by style (2025)")
print(
    T.group_by("style")
    .agg(
        n=pl.len(),
        mse=pl.col("e2").mean().round(0),
        y50=pl.col("y").median(),
        vfr=(pl.col("FLIGHT_RULE_mvt") == "V").mean().round(3),
    )
    .sort("n", descending=True)
)
print("\n== by flight rule (2025)")
print(
    T.group_by("FLIGHT_RULE_mvt").agg(
        n=pl.len(), mse=pl.col("e2").mean().round(0), y50=pl.col("y").median(), ms50=pl.col("mvt_sched").median()
    )
)
# why unmatched: is the same flight number matched on other days? how often is the number unmatched overall?
fl_all = A.group_by("ADEP_mvt", pl.col("FLIGHT_mvt").str.strip_chars().alias("fl")).agg(
    n_all=pl.len(), n_unm=pl.col("unmatched").sum()
)
S2 = S.join(fl_all, left_on=["ADEP_mvt", "fl"], right_on=["ADEP_mvt", "fl"], how="left").with_columns(
    unm_share=pl.col("n_unm") / pl.col("n_all")
)
print(
    "\n== why unmatched: share of the same (airport, FLIGHT_mvt) rows that are unmatched, by airport x year (median, and share of rows whose number is ALWAYS unmatched)"
)
print(
    S2.group_by("ADEP_mvt", "yr")
    .agg(
        med_unm_share=pl.col("unm_share").median().round(2),
        always=(pl.col("unm_share") == 1).mean().round(2),
        single=(pl.col("n_all") == 1).mean().round(2),
    )
    .sort("ADEP_mvt", "yr")
)
# same-day duplicate flight numbers (duplicates can steal the NM match)
dup = (
    A.with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date(), fl=pl.col("FLIGHT_mvt").str.strip_chars())
    .group_by("ADEP_mvt", "fl", "day")
    .agg(nday=pl.len(), nmatch=(pl.col("unmatched") == 0).sum())
)
S3 = S.with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date()).join(dup, on=["ADEP_mvt", "fl", "day"], how="left")
print("\n== same-day duplicates: share of unmatched rows whose (airport, number, day) has another row that IS matched")
print(
    S3.group_by("ADEP_mvt", "yr")
    .agg(dup_matched=(pl.col("nmatch") > 0).mean().round(3), dup_any=(pl.col("nday") > 1).mean().round(3))
    .sort("ADEP_mvt", "yr")
)
print("\n== unmatched rows per day distribution (2026 vs 2025), by airport: mean / max per day")
print(
    S.with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date())
    .group_by("ADEP_mvt", "yr", "day")
    .len()
    .group_by("ADEP_mvt", "yr")
    .agg(mean=pl.col("len").mean().round(1), mx=pl.col("len").max())
    .sort("ADEP_mvt", "yr")
)
