# SPDX-License-Identifier: GPL-3.0-only
"""AOBT_3 anchor of the residual target: mvt_aobt3 = MVT - AOBT_3 (s) in build_features.py; the train2 trainer and the Kumo
members fit y - anchor with the anchor 0 where AOBT_3 is missing, and add the anchor back to the prediction."""

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from conftest import extract


def test_mvt_aobt3_is_seconds_between_movement_and_aobt3():
    ns = {"pl": pl}
    exec(extract("scripts/build_features.py", {"secs"}), ns)
    t0 = datetime(2025, 7, 1, 12, tzinfo=timezone.utc)
    d = pl.DataFrame({"MVT_TIME_UTC_mvt": [t0, t0], "AOBT_3_flt": [t0 - timedelta(seconds=754), None]})
    assert d.select(ns["secs"]("MVT_TIME_UTC_mvt", "AOBT_3_flt"))[:, 0].to_list() == [754, None]


def anchor(path, names, d, target="resid_aobt3"):
    y = np.array([900.0, 700.0, 1200.0])
    ns = {"np": np, "d": d, "y": y, "a": argparse.Namespace(target=target), "ref": np.zeros(3), "d_lobt": np.zeros(3)}
    exec(extract(path, names), ns)
    return ns


def test_trainer_and_kumo_members_share_the_anchor():
    d = pl.DataFrame({"mvt_aobt3": [754.0, None, 1300.0], "ref_sr_p10": [500.0, 500.0, 500.0]})
    tr = anchor("scripts/train2.py", {"aobt", "off"}, d)
    km = anchor("scripts/kumo_km_member.py", {"off"}, d)
    assert tr["off"].tolist() == [754.0, 0.0, 1300.0]  # NaN anchor -> 0, so the model fits the raw target there
    assert np.array_equal(tr["off"], km["off"])
    r = tr["y"] - tr["off"]  # residual target
    assert np.array_equal(tr["off"] + r, tr["y"])  # the prediction adds the anchor back
