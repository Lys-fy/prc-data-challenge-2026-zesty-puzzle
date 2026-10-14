# SPDX-License-Identifier: GPL-3.0-only
"""Submission validator on synthetic rows: schema, id set, floor, LOBT law, P2_NRANK."""

import numpy as np
import polars as pl
import pytest

import validate_submission as vs

F = pl.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "is_rank": [1, 1, 1], "mvt_lobt": [1000.0, None, 500.0]})


def sub(ids=(1.0, 2.0, 3.0), y=(900, 60, 4106)):
    return pl.DataFrame(
        {"MVT_ID_mvt": list(ids), "TAXITIME_SEC_mvt": list(y)},
        schema={"MVT_ID_mvt": pl.Float64, "TAXITIME_SEC_mvt": pl.Int32},
    )


def test_valid_file_passes():
    n, y = vs.check(sub(), F)
    assert n == 3 and sorted(y.tolist()) == [60, 900, 4106]  # |4106 - 500| = 3606 is inside the law


@pytest.mark.parametrize(
    "d",
    [
        sub(y=(900, 59, 600)),  # below the 60 s floor
        sub(y=(900, 60, 4107)),  # LOBT law: |4107 - 500| > 3606.5
        sub(ids=(1.0, 2.0, 4.0)),  # id outside the ranking set
        sub(ids=(1.0, 2.0, 2.0)),  # duplicate id
        sub(ids=(1.0, 2.0), y=(900, 60)),  # missing row
        sub().with_columns(pl.col("TAXITIME_SEC_mvt").cast(pl.Int64)),  # schema
        sub().with_columns(TAXITIME_SEC_mvt=pl.Series([900, None, 600], dtype=pl.Int32)),  # null
    ],
)
def test_invalid_files_fail(d):
    with pytest.raises(AssertionError):
        vs.check(d, F)


def test_p2_nrank_must_match_the_features_table():
    vs.check(sub(), F, nrank="3")
    with pytest.raises(AssertionError):
        vs.check(sub(), F, nrank="4")


def test_rows_without_lobt_are_not_bounded():
    assert vs.check(sub(y=(900, 90000, 600)), F)[1].max() == 90000 and np.isnan(F["mvt_lobt"].to_numpy()[1])
