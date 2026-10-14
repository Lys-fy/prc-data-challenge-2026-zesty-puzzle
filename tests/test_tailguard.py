# SPDX-License-Identifier: GPL-3.0-only
"""KCATLg tail guard: KCAT8 rows where KCAT8 predicts more than T = 2,400 s, KCATL16 elsewhere."""

import numpy as np
import pytest

import kumo_kcat_logtarget_tailguard as tg


def member(ids, taxi, k):
    return np.vstack([ids, taxi, np.full(len(ids), k), np.full(len(ids), 10 * k)]).astype(float)


def test_threshold_is_2400():
    assert tg.T == 2400.0


def test_guard_takes_whole_column_from_kcat8_above_threshold():
    ids = np.array([1.0, 2.0, 3.0, 4.0])
    a = member(ids, [600.0, 2400.0, 2400.5, 9000.0], 1)  # KCAT8
    b = member(ids, [610.0, 2300.0, 2000.0, 5000.0], 2)  # KCATL16
    out, use8 = tg.guard(a, b)
    assert use8.tolist() == [False, False, True, True]  # strictly above T
    assert out[1].tolist() == [610.0, 2300.0, 2400.5, 9000.0]
    assert out[2].tolist() == [2, 2, 1, 1] and out[3].tolist() == [20, 20, 10, 10]  # spread and g_sd follow the source
    assert np.array_equal(out[0], ids)


def test_guard_rejects_misaligned_members():
    a = member(np.array([1.0, 2.0]), [1.0, 2.0], 1)
    with pytest.raises(AssertionError):
        tg.guard(a, member(np.array([2.0, 1.0]), [1.0, 2.0], 2))
