# SPDX-License-Identifier: GPL-3.0-only
"""Schedule-copy rule at LIRF, relink step: relink unmatched departures of a ranking file to orphan NM arrival records and price
the LIRF rows the rule covers. Inputs: data/raw/training_*.parquet (root $PRC_ROOT, else the current directory) and the ranking file.
Output: <out dir>/final_dryrun.parquet (MVT_ID_mvt, mon, ordc, subc, ms, mvt_aobt3_rec, mvt_lobt_rec, off_med, margin_s, new, new_int).
usage: rule_schedule_copy_lirf_relink.py [<ranking parquet> [<out dir> [<Jan/Jul cross-check parquet>]]]
       defaults: data/final/final_ranking.parquet, scratchpad/JOINRULE_GATE, none.

Relink (match()): an unmatched departure (no FLIGHT_ID_mvt) is paired with an arrival record of a flight that has no departure row
(an orphan) from the same airport, AOBT_3 between 4 h and 60 s before the movement time, landing 10 min to 8 h after it, destination
consistent; the pair score is callsign digits + aircraft family (4), callsign digits (3), family + schedule within 3 h (2), schedule (1);
a departure keeps its unique best pair and an arrival record used by more than one departure is dropped. The final ranking file gives
Feb/Jun 2026 their own recovered NM records.
Rule (schedule_copy()): on the relinked LIRF rows with MVT - SCHED <= 28,800 s (longer gaps are the exposed class), ORD rows
(|AOBT_3 - LOBT| > 3,600 s on the recovered record) get (MVT - AOBT_3) + the 2025 airport median of TAXI - (MVT - AOBT_3); SUB rows
(not ORD and |SCHED - LOBT| > 3,600 s) get MVT - SCHED; both are floored at 60 s; other rows are not priced.
The optional Jan/Jul cross-check compares with the same rule run on the public ranking file (a parquet with MVT_ID_mvt, ordc, subc, new).
Read-only on its inputs."""

import os, sys, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
cols = [
    "MVT_ID_mvt",
    "FLIGHT_ID_mvt",
    "FLIGHT_mvt",
    "PHASE_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "MVT_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "AIRCRAFT_TYPE_mvt",
    "LOBT_flt",
    "CALLSIGN_flt",
    "ADEP_flt",
    "ADES_flt",
    "AOBT_3_flt",
    "EOBT_1_flt",
    "AIRCRAFT_TYPE_flt",
    "AIRCRAFT_OPERATOR_flt",
    "FLIGHT_RULE_mvt",
]
ep = lambda c: pl.col(c).dt.epoch("s").cast(pl.Float64)
DCOLS = [
    "MVT_ID_mvt",
    "ADEP_mvt",
    "ADES_mvt",
    "FLIGHT_mvt",
    "digits",
    "fam",
    "AIRCRAFT_TYPE_mvt",
    "t",
    "ts",
    "yr",
    "mon",
    "is_rank",
    "ms",
    "FLIGHT_RULE_mvt",
]


def pool(Arr):
    return Arr.select(
        [
            "FLIGHT_ID_mvt",
            "ADEP_flt",
            pl.col("ADES_mvt").alias("arr_apt"),
            "ADES_flt",
            "CALLSIGN_flt",
            "cs_digits",
            "fam_flt",
            pl.col("fam").alias("arr_fam"),
            "ta",
            "te",
            "tl",
            pl.col("t").alias("t_land"),
            pl.col("ts").alias("ts_arr"),
            pl.col("tb").alias("tb_arr"),
            "AIRCRAFT_OPERATOR_flt",
        ]
    ).filter(pl.col("ta").is_not_null())


def pairs(D, PO):
    out = []
    for a in APTS:
        d = D.filter(pl.col("ADEP_mvt") == a).select(DCOLS)
        p = PO.filter(pl.col("ADEP_flt") == a).sort("ta")
        if d.height == 0 or p.height == 0:
            continue
        ta = p["ta"].to_numpy()
        t = d["t"].to_numpy()
        lo = np.searchsorted(ta, t - 4 * 3600, side="left")
        hi = np.searchsorted(ta, t - 60, side="right")
        n = np.maximum(hi - lo, 0)
        di = np.repeat(np.arange(d.height), n)
        pi = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l]) if n.sum() else np.zeros(0, int)
        out.append(pl.concat([d[di], p[pi]], how="horizontal"))
    return pl.concat(out, how="vertical_relaxed")


def match(D, PO):
    J = pairs(D, PO)
    J = J.filter(
        (pl.col("t_land") >= pl.col("t") + 600)
        & (pl.col("t_land") <= pl.col("t") + 8 * 3600)
        & (
            pl.col("ADES_mvt").is_null()
            | (pl.col("ADES_mvt") == pl.col("arr_apt"))
            | (pl.col("ADES_mvt") == pl.col("ADES_flt"))
        )
    )
    J = J.with_columns(
        cs_ok=(pl.col("digits").str.len_chars() > 0) & (pl.col("digits") == pl.col("cs_digits")),
        typ_ok=(pl.col("fam") == pl.col("fam_flt")) | (pl.col("fam") == pl.col("arr_fam")),
        sch_ok=(pl.col("te") - pl.col("ts")).abs() <= 3 * 3600,
    )
    J = J.with_columns(
        score=pl.when(pl.col("cs_ok") & pl.col("typ_ok"))
        .then(4)
        .when(pl.col("cs_ok"))
        .then(3)
        .when(pl.col("typ_ok") & pl.col("sch_ok"))
        .then(2)
        .when(pl.col("sch_ok"))
        .then(1)
        .otherwise(0)
    ).filter(pl.col("score") >= 1)
    J = J.with_columns(smax=pl.col("score").max().over("MVT_ID_mvt"))
    J = J.with_columns(nbest=(pl.col("score") == pl.col("smax")).cast(pl.Int32).sum().over("MVT_ID_mvt")).filter(
        (pl.col("score") == pl.col("smax")) & (pl.col("nbest") == 1)
    )
    J = J.with_columns(narr=pl.len().over("FLIGHT_ID_mvt")).filter(pl.col("narr") == 1)
    return J


def schedule_copy(aobt3_rec, lobt_rec, ms, off_med):
    """Rule cells and prices. Arrays in s: aobt3_rec = MVT - AOBT_3 and lobt_rec = MVT - LOBT of the recovered record,
    ms = MVT - SCHED. ordc = |aobt3_rec - lobt_rec| > 3,600 -> aobt3_rec + off_med; subc = not ordc and |ms - lobt_rec| > 3,600 -> ms;
    prices floored at 60 s, NaN on the other rows."""
    ordc = np.abs(aobt3_rec - lobt_rec) > 3600
    subc = (~ordc) & (np.abs(ms - lobt_rec) > 3600)
    new = np.where(ordc, np.maximum(aobt3_rec + off_med, 60.0), np.where(subc, np.maximum(ms, 60.0), np.nan))
    return ordc, subc, new


def main(argv):
    if argv[:1] in (["-h"], ["--help"]):
        sys.exit(print(__doc__))
    os.chdir(os.environ.get("PRC_ROOT", os.getcwd()))
    RANK = argv[0] if len(argv) > 0 else "data/final/final_ranking.parquet"
    O = argv[1] if len(argv) > 1 else "scratchpad/JOINRULE_GATE"
    RELINK_CHECK_ARG = argv[2] if len(argv) > 2 else None
    os.makedirs(O, exist_ok=True)
    fs = sorted(glob.glob("data/raw/training_*.parquet")) + [RANK]
    M = pl.concat(
        [
            pl.read_parquet(f, columns=cols + (["TAXITIME_SEC_mvt"] if "training" in f else [])).with_columns(
                pl.lit(int(f == RANK)).cast(pl.Int8).alias("is_rank")
            )
            for f in fs
        ],
        how="diagonal_relaxed",
    )
    M = M.with_columns(
        t=ep("MVT_TIME_UTC_mvt"),
        ts=ep("SCHED_TIME_UTC_mvt"),
        tb=ep("BLOCK_TIME_UTC_mvt"),
        ta=ep("AOBT_3_flt"),
        te=ep("EOBT_1_flt"),
        tl=ep("LOBT_flt"),
        yr=pl.col("MVT_TIME_UTC_mvt").dt.year(),
        mon=pl.col("MVT_TIME_UTC_mvt").dt.month(),
        digits=pl.col("FLIGHT_mvt").str.replace_all(r"[^0-9]", "").str.strip_chars_start("0"),
        cs_digits=pl.col("CALLSIGN_flt").str.replace_all(r"[^0-9]", "").str.strip_chars_start("0"),
        fam=pl.col("AIRCRAFT_TYPE_mvt").str.slice(0, 3),
        fam_flt=pl.col("AIRCRAFT_TYPE_flt").str.slice(0, 3),
    )
    DEP = M.filter(pl.col("PHASE_mvt") == "DEP").with_columns(ms=pl.col("t") - pl.col("ts"))
    ARR = M.filter(pl.col("PHASE_mvt") == "ARR")
    Dm = DEP.filter(pl.col("FLIGHT_ID_mvt").is_not_null())
    dep_ids = Dm.select(pl.col("FLIGHT_ID_mvt").alias("fid")).unique()
    print(
        f"rows {M.height:,}; final rows {M.filter(pl.col('is_rank') == 1).height:,}; DEP {DEP.height:,} matched {Dm.height:,}"
    )
    A = ARR.filter(pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("ADEP_flt").is_in(APTS)).join(
        dep_ids, left_on="FLIGHT_ID_mvt", right_on="fid", how="anti"
    )
    ORPH = pool(A)
    U = DEP.filter(pl.col("FLIGHT_ID_mvt").is_null())
    R = match(U, ORPH).with_columns(mvt_aobt3_rec=pl.col("t") - pl.col("ta"), mvt_lobt_rec=pl.col("t") - pl.col("tl"))
    Mm = Dm.filter(pl.col("yr") == 2025).with_columns(
        d=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64) - (pl.col("t") - pl.col("ta"))
    )
    off = Mm.group_by("ADEP_mvt").agg(off_med=pl.col("d").median())
    R = R.join(off, on="ADEP_mvt", how="left")
    print(
        "2025 orphan rows:",
        R.filter(pl.col("yr") == 2025).height,
        "; 2026 by month:",
        dict(sorted(R.filter(pl.col("yr") == 2026).group_by("mon").len().iter_rows())),
    )
    # UF rows of the final file and the rule
    F = R.filter((pl.col("is_rank") == 1) & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("ms") <= 28800))
    nuf = (
        DEP.filter(
            (pl.col("is_rank") == 1)
            & (pl.col("ADEP_mvt") == "LIRF")
            & pl.col("FLIGHT_ID_mvt").is_null()
            & (pl.col("ms") <= 28800)
        )
        .group_by("mon")
        .len()
    )
    print("final UF rows per month (all, with or without a record):", dict(sorted(nuf.iter_rows())))
    A_, L_, ms_, of_ = [F[c].to_numpy().astype(float) for c in ["mvt_aobt3_rec", "mvt_lobt_rec", "ms", "off_med"]]
    ordc, subc, new = schedule_copy(A_, L_, ms_, of_)
    F = F.with_columns(
        ordc=pl.Series(ordc),
        subc=pl.Series(subc),
        new=pl.Series(new),
        new_int=pl.Series(np.rint(new)).cast(pl.Int32, strict=False),
    )
    mar_a = np.abs(np.abs(A_ - L_) - 3600)
    mar_m = np.where(~ordc, np.abs(np.abs(ms_ - L_) - 3600), np.inf)
    F = F.with_columns(margin_s=pl.Series(np.minimum(mar_a, mar_m)))
    print(
        F.group_by("mon")
        .agg(
            orphan=pl.len(),
            ORD=pl.col("ordc").sum(),
            SUB=pl.col("subc").sum(),
            changed=(pl.col("ordc") | pl.col("subc")).sum(),
            min_margin=pl.col("margin_s").min(),
            new_mean=pl.col("new").mean(),
        )
        .sort("mon")
    )
    # optional cross-check of Jan/Jul against the ranking.parquet run (ids identical; times re-jittered +-12 s)
    if RELINK_CHECK_ARG:
        j = pl.read_parquet(RELINK_CHECK_ARG).select(
            pl.col("MVT_ID_mvt").cast(pl.Float64),
            pl.col("ordc").alias("o0"),
            pl.col("subc").alias("s0"),
            pl.col("new").alias("new0"),
        )
        c = (
            F.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
            .filter(pl.col("mon").is_in([1, 7]))
            .join(j, on="MVT_ID_mvt", how="full", coalesce=True)
        )
        print(
            f"Jan/Jul cross-check: final-run rows {F.filter(pl.col('mon').is_in([1, 7])).height}, ranking-run rows {j.height}, both {c.filter(pl.col('o0').is_not_null() & pl.col('ordc').is_not_null()).height};"
            f" same cell {c.filter((pl.col('o0') == pl.col('ordc')) & (pl.col('s0') == pl.col('subc'))).height}; |new - new0| max {c.select((pl.col('new') - pl.col('new0')).abs().max()).item()}"
        )
    F.filter(pl.col("ordc") | pl.col("subc")).select(
        "MVT_ID_mvt",
        "mon",
        "ordc",
        "subc",
        "ms",
        "mvt_aobt3_rec",
        "mvt_lobt_rec",
        "off_med",
        "margin_s",
        "new",
        "new_int",
    ).sort("mon", "MVT_ID_mvt").write_parquet(f"{O}/final_dryrun.parquet")
    print("wrote", f"{O}/final_dryrun.parquet")


if __name__ == "__main__":
    main(sys.argv[1:])
