# SPDX-License-Identifier: GPL-3.0-only
"""Schedule-copy rule at LIRF on synthetic rows: relink (pairs/match), rule cells and prices, apply step."""

import numpy as np
import polars as pl
import pytest

import rule_schedule_copy_lirf_apply as apply_mod
import rule_schedule_copy_lirf_relink as relink

T = 1_750_000_000.0  # movement time of the synthetic departure (epoch s)


def dep(mvt_id=1.0, t=T, digits="123", fam="A32", ades="LFPG", adep="LIRF", ts=None):
    ts = t - 900 if ts is None else ts
    return {
        "MVT_ID_mvt": mvt_id,
        "ADEP_mvt": adep,
        "ADES_mvt": ades,
        "FLIGHT_mvt": "AZ" + digits,
        "digits": digits,
        "fam": fam,
        "AIRCRAFT_TYPE_mvt": fam + "0",
        "t": t,
        "ts": ts,
        "yr": 2026,
        "mon": 2,
        "is_rank": 1,
        "ms": t - ts,
        "FLIGHT_RULE_mvt": "I",
    }


def orphan(fid=10.0, ta=T - 1800, t_land=T + 3600, cs="123", fam="A32", ades="LFPG", adep="LIRF", te=None, tl=None):
    return {
        "FLIGHT_ID_mvt": fid,
        "ADEP_flt": adep,
        "arr_apt": ades,
        "ADES_flt": ades,
        "CALLSIGN_flt": "AZA" + cs,
        "cs_digits": cs,
        "fam_flt": fam,
        "arr_fam": fam,
        "ta": ta,
        "te": ta if te is None else te,
        "tl": ta if tl is None else tl,
        "t_land": t_land,
        "ts_arr": t_land,
        "tb_arr": t_land + 300,
        "AIRCRAFT_OPERATOR_flt": "AZA",
    }


def run(deps, orphans):
    D = pl.DataFrame(deps).select(relink.DCOLS)
    return relink.match(D, pl.DataFrame(orphans))


def test_match_pairs_callsign_and_family():
    J = run([dep()], [orphan()])
    assert J.height == 1 and J["FLIGHT_ID_mvt"][0] == 10.0 and J["score"][0] == 4


@pytest.mark.parametrize(
    "o",
    [
        orphan(ta=T - 30),  # AOBT_3 less than 60 s before the movement
        orphan(ta=T - 4 * 3600 - 1),  # AOBT_3 more than 4 h before
        orphan(t_land=T + 599),  # landing less than 10 min after the movement
        orphan(t_land=T + 8 * 3600 + 1),  # landing more than 8 h after
        orphan(ades="EGLL"),  # destination differs
        orphan(cs="999", fam="B77", te=T + 4 * 3600),  # no evidence: score 0
    ],
)
def test_match_rejects(o):
    assert run([dep()], [o]).height == 0


def test_match_keeps_the_unique_best_pair():
    J = run([dep()], [orphan(fid=10.0), orphan(fid=11.0, cs="999", ta=T - 1700)])  # 4 beats 2 (family + schedule)
    assert J["FLIGHT_ID_mvt"].to_list() == [10.0]


def test_match_stays_within_the_departure_airport():
    J = run(
        [dep()], [orphan(fid=10.0, adep="LFPG"), orphan(fid=11.0, cs="999", ta=T - 1700)]
    )  # exact pair at another airport
    assert J["FLIGHT_ID_mvt"].to_list() == [11.0]


def test_match_drops_ties_and_shared_arrivals():
    assert run([dep()], [orphan(fid=10.0), orphan(fid=11.0, ta=T - 1700)]).height == 0  # two best pairs
    assert run([dep(1.0), dep(2.0, t=T + 60)], [orphan()]).height == 0  # one arrival record, two departures


def test_schedule_copy_cells_and_prices():
    a3 = np.array([600.0, 600.0, 600.0, 10.0, 600.0])  # MVT - AOBT_3
    lo = np.array([600.0 + 3601, 600.0 + 3600, 600.0, 3700.0, 600.0])  # MVT - LOBT
    ms = np.array([0.0, 0.0, 600.0 + 3601, 0.0, 900.0])  # MVT - SCHED
    off = np.full(5, -20.0)
    ordc, subc, new = relink.schedule_copy(a3, lo, ms, off)
    assert ordc.tolist() == [True, False, False, True, False]  # |a3 - lo| > 3600 strictly
    assert subc.tolist() == [False, True, True, False, False]  # not ordc and |ms - lo| > 3600
    assert new[0] == 580.0 and new[3] == 60.0  # AOBT_3 anchor + airport offset, floored at 60 s
    assert new[1] == 60.0 and new[2] == 600.0 + 3601  # MVT - SCHED, floored at 60 s
    assert np.isnan(new[4])


def test_apply_changes_only_rule_rows():
    B = pl.DataFrame(
        {"MVT_ID_mvt": [1.0, 2.0, 3.0], "TAXITIME_SEC_mvt": [700, 800, 900]},
        schema={"MVT_ID_mvt": pl.Float64, "TAXITIME_SEC_mvt": pl.Int32},
    )
    D = pl.DataFrame({"MVT_ID_mvt": [2.0], "mon": [2], "new_int": [1234]})
    O, _ = apply_mod.apply_rule(B, D)
    assert O.schema == B.schema and O["TAXITIME_SEC_mvt"].to_list() == [700, 1234, 900]
    with pytest.raises(AssertionError):
        apply_mod.apply_rule(B, D.with_columns(new_int=pl.lit(59)))  # below the floor
    with pytest.raises(AssertionError):
        apply_mod.apply_rule(B, D.with_columns(MVT_ID_mvt=pl.lit(9.0)))  # id not in the base
