# SPDX-License-Identifier: GPL-3.0-only
"""Schedule-copy rule for unmatched LIRF departures (docs/METHOD.md §6).

Faithful re-implementation of the orphan link in scripts/rule_schedule_copy_lirf_relink.py (function `match`) and of the
rule it feeds (applied in the full pipeline by scripts/rule_schedule_copy_lirf_apply.py). Label-free except `off_med`."""
import numpy as np
import polars as pl

from .config import FLOOR_S, JOIN_EDGE_S, SCHED_WINDOW_S

AIRPORT = "LIRF"
digits = lambda c: pl.col(c).str.replace_all(r"[^0-9]", "").str.strip_chars_start("0")
family = lambda c: pl.col(c).str.slice(0, 3)


def orphan_arrivals(rows: pl.DataFrame) -> pl.DataFrame:
    """Arrival records from AIRPORT whose NM flight has no departure row: the departure's NM record, recovered."""
    dep_ids = rows.filter(pl.col("is_dep") & pl.col("matched"))["FLIGHT_ID_mvt"].unique()
    arr = rows.filter(~pl.col("is_dep") & pl.col("matched") & (pl.col("ADEP_flt") == AIRPORT)
                      & ~pl.col("FLIGHT_ID_mvt").is_in(dep_ids.implode()) & pl.col("ta").is_not_null())
    return arr.select("FLIGHT_ID_mvt", "ADES_flt", "ta", "te", "tl", arr_apt="ADES_mvt", t_land="t", cs_digits=digits("CALLSIGN_flt"),
                      fam_flt=family("AIRCRAFT_TYPE_flt"), arr_fam=family("AIRCRAFT_TYPE_mvt"))


def link(rows: pl.DataFrame) -> pl.DataFrame:
    """Unmatched AIRPORT departures linked to a unique best orphan record: MVT_ID_mvt, A = MVT - AOBT_3,rec, L = MVT - LOBT_rec."""
    dep = rows.filter(pl.col("is_dep") & ~pl.col("matched") & (pl.col("ADEP_mvt") == AIRPORT)).select(
        "MVT_ID_mvt", "ADES_mvt", "t", "ts", digits=digits("FLIGHT_mvt"), fam=family("AIRCRAFT_TYPE_mvt"))
    j = dep.join_where(orphan_arrivals(rows), pl.col("ta") >= pl.col("t") - 4 * 3600, pl.col("ta") <= pl.col("t") - 60)
    j = j.filter(pl.col("t_land").is_between(pl.col("t") + 600, pl.col("t") + 8 * 3600) & (pl.col("ADES_mvt").is_null()
                 | (pl.col("ADES_mvt") == pl.col("arr_apt")) | (pl.col("ADES_mvt") == pl.col("ADES_flt"))))
    cs_ok = (pl.col("digits").str.len_chars() > 0) & (pl.col("digits") == pl.col("cs_digits"))
    typ_ok = (pl.col("fam") == pl.col("fam_flt")) | (pl.col("fam") == pl.col("arr_fam"))
    sch_ok = (pl.col("te") - pl.col("ts")).abs() <= 3 * 3600
    score = pl.when(cs_ok & typ_ok).then(4).when(cs_ok).then(3).when(typ_ok & sch_ok).then(2).when(sch_ok).then(1).otherwise(0)
    j = j.with_columns(score=score).filter(pl.col("score") >= 1)
    best = (pl.col("score") == pl.col("score").max().over("MVT_ID_mvt"))
    j = j.filter(best & (best.cast(pl.Int32).sum().over("MVT_ID_mvt") == 1))
    j = j.filter(pl.len().over("FLIGHT_ID_mvt") == 1)
    return j.select("MVT_ID_mvt", A=pl.col("t") - pl.col("ta"), L=pl.col("t") - pl.col("tl"))


def schedule_copy(a: np.ndarray, l: np.ndarray, ms: np.ndarray, off_med: float) -> np.ndarray:
    """Rule value per row, NaN where the rule leaves the prediction unchanged.
    ORD |A - L| > edge: the stamp is real -> max(A + off_med, 60); SUB |A - L| <= edge and |MS - L| > edge: BLOCK is the
    substituted scheduled time -> max(MS, 60). Rows with MS > 8 h (exposed) are never changed."""
    ordc = np.abs(a - l) > JOIN_EDGE_S
    subc = ~ordc & (np.abs(ms - l) > JOIN_EDGE_S)
    out = np.where(ordc, np.maximum(a + off_med, FLOOR_S), np.where(subc, np.maximum(ms, FLOOR_S), np.nan))
    return np.where(ms <= SCHED_WINDOW_S, out, np.nan)


def apply(dep: pl.DataFrame, pred: np.ndarray, links: pl.DataFrame, off_med: float) -> tuple[np.ndarray, int]:
    """Overwrite `pred` (aligned with `dep`) on linked rows where the rule fires; returns the new vector and the rows changed."""
    j = dep.select("MVT_ID_mvt", "mvt_sched").join(links, on="MVT_ID_mvt", how="left", maintain_order="left")
    a, l, ms = (j[c].fill_null(np.nan).to_numpy() for c in ("A", "L", "mvt_sched"))
    with np.errstate(invalid="ignore"):
        new = schedule_copy(a, l, ms, off_med)
    fire = ~np.isnan(new)
    return np.where(fire, new, pred), int(fire.sum())
