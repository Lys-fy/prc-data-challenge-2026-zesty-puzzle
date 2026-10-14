# SPDX-License-Identifier: GPL-3.0-only
"""The anchor (MVT - AOBT_3) and a small label-free feature set computed from the competition rows only."""
import numpy as np
import polars as pl

from .config import CONGESTION_WINDOW_S

CAT_COLS = ["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "airline", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]
NUM_COLS = ["hour", "weekday", "matched", "mvt_aobt3", "mvt_lobt", "mvt_eobt1", "mvt_iobt", "mvt_sched", "aobt3_lobt",
            "dep_before", "dep_after", "arr_around", "taxiing_at_mvt", "taxiing_at_aobt3"]


def add_row_features(dep: pl.DataFrame) -> pl.DataFrame:
    """Calendar, operator and time-delta features of each departure (seconds; null when the NM record is missing)."""
    return dep.with_columns(
        hour=pl.col("MVT_TIME_UTC_mvt").dt.hour(), weekday=pl.col("MVT_TIME_UTC_mvt").dt.weekday(),
        airline=pl.coalesce("AIRCRAFT_OPERATOR_flt", pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]+)")),
        mvt_aobt3=pl.col("t") - pl.col("ta"), mvt_lobt=pl.col("t") - pl.col("tl"), mvt_eobt1=pl.col("t") - pl.col("te"),
        mvt_iobt=pl.col("t") - pl.col("ti"), mvt_sched=pl.col("t") - pl.col("ts"), aobt3_lobt=pl.col("ta") - pl.col("tl"),
        matched=pl.col("matched").cast(pl.Int8))


def count_le(sorted_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Number of sorted times <= each query time."""
    return np.searchsorted(sorted_times, query, side="right")


def airport_counts(dep: pl.DataFrame, arr: pl.DataFrame, w: float = CONGESTION_WINDOW_S) -> pl.DataFrame:
    """Congestion counts of one airport: departures and arrivals near take-off, and aircraft taxiing out (AOBT_3 passed,
    not yet airborne) at take-off and at the departure's own AOBT_3."""
    t_dep, t_arr = np.sort(dep["t"].to_numpy()), np.sort(arr["t"].to_numpy())
    nm = dep.filter(pl.col("ta").is_not_null())
    t_off, t_out = np.sort(nm["ta"].to_numpy()), np.sort(nm["t"].to_numpy())
    q, q_off = dep["t"].to_numpy(), dep["ta"].fill_null(np.nan).to_numpy()
    taxiing = lambda x: (count_le(t_off, x) - count_le(t_out, x)).astype(float)
    return dep.select("MVT_ID_mvt").with_columns(
        dep_before=count_le(t_dep, q) - count_le(t_dep, q - w), dep_after=count_le(t_dep, q + w) - count_le(t_dep, q),
        arr_around=count_le(t_arr, q + w) - count_le(t_arr, q - w), taxiing_at_mvt=taxiing(q),
        taxiing_at_aobt3=np.where(np.isnan(q_off), np.nan, taxiing(np.nan_to_num(q_off))))


def add_congestion(rows: pl.DataFrame, dep: pl.DataFrame) -> pl.DataFrame:
    """Join the per-airport congestion counts (computed over all rows of the same file set) to the departures."""
    arr = rows.filter(~pl.col("is_dep"))
    parts = [airport_counts(dep.filter(pl.col("ADEP_mvt") == a), arr.filter(pl.col("ADES_mvt") == a))
             for a in dep["ADEP_mvt"].unique().sort()]
    return dep.join(pl.concat(parts), on="MVT_ID_mvt", how="left", maintain_order="left")


def build(rows: pl.DataFrame) -> pl.DataFrame:
    """Departures of `rows` with every feature of CAT_COLS + NUM_COLS."""
    dep = add_row_features(rows.filter(pl.col("is_dep")))
    return add_congestion(rows, dep)


def anchor(dep: pl.DataFrame, fallback: dict[str, float]) -> np.ndarray:
    """MVT - AOBT_3 (METHOD §1); rows without AOBT_3 use the airport's median taxi-out of unmatched training rows."""
    fb = dep["ADEP_mvt"].replace_strict(fallback, default=float(np.median(list(fallback.values()))), return_dtype=pl.Float64)
    return dep["mvt_aobt3"].fill_null(fb).to_numpy()
