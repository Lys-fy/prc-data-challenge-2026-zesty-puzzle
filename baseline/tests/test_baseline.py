# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic-row checks of the anchor and the schedule-copy rule (no competition data needed)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from baseline import data, features, rule

T0 = datetime(2025, 7, 1, 12, tzinfo=timezone.utc)
ts = lambda s: T0 + timedelta(seconds=s)


def row(mvt_id: float, phase: str, fid: float | None, **kw) -> dict:
    """One competition-like row; times in seconds from T0."""
    base = {"MVT_ID_mvt": mvt_id, "FLIGHT_ID_mvt": fid, "PHASE_mvt": phase, "FLIGHT_mvt": "AZ123", "ADEP_mvt": "LIRF",
            "ADES_mvt": "LFPG", "AIRCRAFT_TYPE_mvt": "A320", "AIRCRAFT_TYPE_flt": "A320", "CALLSIGN_flt": "AZA123",
            "ADEP_flt": "LIRF", "ADES_flt": "LFPG", "AIRCRAFT_OPERATOR_flt": "AZA", "TAXITIME_SEC_mvt": 600}
    for c in data.TIME_COLS.values():
        base[c] = ts(kw.pop(c)) if c in kw else None
    return {**base, **kw}


def frame(rows: list[dict]) -> pl.DataFrame:
    """Rows as the loader would return them."""
    df = pl.DataFrame(rows, schema_overrides={c: pl.Datetime("us", "UTC") for c in data.TIME_COLS.values()}, strict=False)
    epochs = [pl.col(c).dt.epoch("s").cast(pl.Float64).alias(k) for k, c in data.TIME_COLS.items()]
    return df.with_columns(*epochs, month=pl.lit(7), is_dep=pl.col("PHASE_mvt") == "DEP", matched=pl.col("FLIGHT_ID_mvt").is_not_null())


def test_anchor_is_mvt_minus_aobt3_with_fallback() -> None:
    rows = frame([row(1.0, "DEP", 10.0, MVT_TIME_UTC_mvt=900, AOBT_3_flt=0, LOBT_flt=0),
                  row(2.0, "DEP", None, MVT_TIME_UTC_mvt=900, SCHED_TIME_UTC_mvt=0)])
    dep = features.build(rows)
    assert np.allclose(features.anchor(dep, {"LIRF": 777.0}), [900.0, 777.0])


def test_schedule_copy_cells() -> None:
    a, l, ms = np.array([5000.0, 900.0, 900.0, 900.0]), np.array([800.0, 800.0, 800.0, 800.0]), np.array([900.0, 5000.0, 900.0, 30000.0])
    out = rule.schedule_copy(a, l, ms, off_med=50.0)
    assert out[0] == 5050.0          # ORD: |A - L| > 3,600 -> A + off_med
    assert out[1] == 5000.0          # SUB: |A - L| <= 3,600, |MS - L| > 3,600 -> MS
    assert np.isnan(out[2])          # neither cell
    assert np.isnan(out[3])          # exposed (MS > 8 h): unchanged


def test_link_and_apply_on_an_orphan_arrival() -> None:
    dep = row(1.0, "DEP", None, MVT_TIME_UTC_mvt=7200, SCHED_TIME_UTC_mvt=7000, CALLSIGN_flt=None, ADEP_flt=None)
    arr = row(2.0, "ARR", 99.0, ADEP_mvt="LIRF", ADES_mvt="LFPG", MVT_TIME_UTC_mvt=14400, AOBT_3_flt=6000, LOBT_flt=0, EOBT_1_flt=0)
    rows = frame([dep, arr])
    links = rule.link(rows)
    assert links.to_dicts() == [{"MVT_ID_mvt": 1.0, "A": 1200.0, "L": 7200.0}]
    pred, n = rule.apply(features.build(rows), np.array([300.0]), links, off_med=40.0)
    assert n == 1 and pred[0] == 1240.0  # ORD cell: |1200 - 7200| > 3,600
