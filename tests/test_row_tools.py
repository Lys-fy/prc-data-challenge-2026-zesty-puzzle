# SPDX-License-Identifier: GPL-3.0-only
"""Row-level tools of the final path, run as subprocesses on small synthetic parquet files: pool_mix.py (weighted chain pool),
staged_delta_apply.py (member swap as a delta, steps S3 / S5 of the staged assembly) and isolate_delta.py (chain-variant delta on
the base line with unmatched and exposed rows protected)."""

import hashlib
import os
import shutil
import stat
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = "TAXITIME_SEC_mvt"


def sub(ids, vals):
    return pl.DataFrame({"MVT_ID_mvt": np.asarray(ids, float), C: np.asarray(vals, np.int32)})


def run(args, cwd, env=None):
    e = dict(os.environ, POLARS_MAX_THREADS="1", **(env or {}))
    return subprocess.run([sys.executable] + args, cwd=cwd, env=e, capture_output=True, text=True)


# ---------------------------------------------------------------- pool_mix.py
def test_pool_mix(tmp_path):
    ids = [1, 2, 3, 4, 5]
    os.makedirs(tmp_path / "data")
    pl.DataFrame(
        {"MVT_ID_mvt": np.asarray(ids, float), "is_rank": [1] * 5, "unmatched": [0, 0, 0, 1, 0]}
    ).write_parquet(tmp_path / "data/features_v9.parquet")
    sub(ids, [500, 500, 500, 7000, 500]).write_parquet(tmp_path / "ref.parquet")
    sub(ids, [600, 700, 800, 7000, 900]).write_parquet(tmp_path / "a.parquet")
    sub(ids, [600, 900, 1000, 7000, 901]).write_parquet(tmp_path / "b.parquet")
    r = run([f"{ROOT}/scripts/pool_mix.py", "ref.parquet", "out.parquet", "a.parquet", "b.parquet"], tmp_path, {"W": "3,1"})
    assert r.returncode == 0, r.stderr
    o = pl.read_parquet(tmp_path / "out.parquet")
    assert o.schema == {"MVT_ID_mvt": pl.Float64, C: pl.Int32}
    # row 1: both chains agree -> chain 1; rows 2, 3, 5: rint(0.75 a + 0.25 b); row 4: unmatched -> REF
    assert o[C].to_list() == [600, 750, 850, 7000, 900]


def test_pool_mix_rejects_unmatched_rows_that_differ_from_ref(tmp_path):
    ids = [1, 2]
    os.makedirs(tmp_path / "data")
    pl.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "is_rank": [1, 1], "unmatched": [0, 1]}).write_parquet(
        tmp_path / "data/features_v9.parquet"
    )
    sub(ids, [500, 7000]).write_parquet(tmp_path / "ref.parquet")
    sub(ids, [600, 7001]).write_parquet(tmp_path / "a.parquet")
    r = run([f"{ROOT}/scripts/pool_mix.py", "ref.parquet", "out.parquet", "a.parquet", "a.parquet"], tmp_path)
    assert r.returncode != 0 and "unmatched rows differ" in r.stderr


# ---------------------------------------------------------------- staged_delta_apply.py
@pytest.fixture
def staged(tmp_path):
    ids = [1, 2, 3, 4]
    sub(ids, [900, 900, 900, 900]).write_parquet(tmp_path / "base.parquet")
    sub(ids, [800, 800, 800, 800]).write_parquet(tmp_path / "from.parquet")
    # deltas: 0, +5,000 (beyond the LOBT bound), -2,000 (below the 60 s floor), +10.4 (rounded)
    pl.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], C: [800.0, 5800.0, -1200.0, 810.4]}).write_parquet(tmp_path / "to.parquet")
    pl.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "is_rank": [1] * 4, "mvt_lobt": [1000.0] * 4}).write_parquet(
        tmp_path / "feat.parquet"
    )
    return tmp_path


def test_staged_delta_apply_changed_only(staged):
    a = ["base.parquet", "from.parquet", "to.parquet", "feat.parquet", "out.parquet", "--changed-only"]
    r = run([f"{ROOT}/scripts/staged_delta_apply.py"] + a, staged)
    assert r.returncode == 0, r.stderr
    o = pl.read_parquet(staged / "out.parquet")
    assert o.schema == {"MVT_ID_mvt": pl.Float64, C: pl.Int32}
    assert o[C].to_list() == [900, 1000 + 3606, 60, 910]  # unchanged, LOBT-clipped, floored, rounded
    r = run([f"{ROOT}/scripts/staged_delta_apply.py"] + a, staged)
    assert r.returncode != 0 and "refusing" in r.stderr  # never overwrites its output


def test_staged_delta_apply_protect(staged):
    sub([2], [0]).write_parquet(staged / "sched.parquet")  # schedule-copy rows: row 2 carries a delta
    pl.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "cls": ["matched"] * 4}).write_parquet(staged / "U.parquet")
    a = ["base.parquet", "from.parquet", "to.parquet", "feat.parquet", "out.parquet", "--protect", "U.parquet", "sched.parquet"]
    r = run([f"{ROOT}/scripts/staged_delta_apply.py"] + a, staged)
    assert r.returncode != 0 and "delta touches rule-priced" in r.stderr
    assert not (staged / "out.parquet").exists()


# ---------------------------------------------------------------- isolate_delta.py
N = 344841  # the script checks the public ranking size


def fake_root(tmp_path):
    for s in ("isolate_delta.py", "combine_nofloor.py", "final_combine.py", "validate_submission.py"):
        os.makedirs(tmp_path / "scripts", exist_ok=True)
        shutil.copy(f"{ROOT}/scripts/{s}", tmp_path / "scripts" / s)
    os.makedirs(tmp_path / ".venv/bin")
    py = tmp_path / ".venv/bin/python"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    ids = np.arange(1, N + 1, dtype=float)
    os.makedirs(tmp_path / "data")
    pl.DataFrame(
        {
            "MVT_ID_mvt": ids,
            "is_rank": np.ones(N, np.int8),
            "unmatched": (ids % 10 == 0).astype(np.int8),
            "mvt_lobt": np.full(N, 1000.0),
            "ADEP_mvt": np.where(ids % 2 == 0, "LFPG", "EGLL"),
        }
    ).with_columns(MVT_TIME_UTC_mvt=pl.datetime(2026, 1, 15)).write_parquet(tmp_path / "data/features_v9.parquet")
    os.makedirs(tmp_path / "scratchpad/xhr")
    pl.DataFrame({"MVT_ID_mvt": [3.0, 7.0]}).write_parquet(tmp_path / "scratchpad/xhr/exposed_rows.parquet")
    sub(ids, np.full(N, 900)).write_parquet(tmp_path / "v99.parquet")
    sub(ids, np.full(N, 800)).write_parquet(tmp_path / "ref.parquet")
    return ids


def isolate(tmp_path, delta):
    ids = fake_root(tmp_path)
    sub(ids, 800 + delta).write_parquet(tmp_path / "c.parquet")

    m = hashlib.md5(open(tmp_path / "v99.parquet", "rb").read()).hexdigest()
    a = [str(tmp_path / f) for f in ("out.parquet", "c.parquet", "ref.parquet", "v99.parquet")]
    return run([str(tmp_path / "scripts/isolate_delta.py")] + a, tmp_path, {"BASEMD5": m[:8], "PRC_ROOT": str(tmp_path)})


def test_isolate_delta(tmp_path):
    d = np.zeros(N, np.int64)
    d[0], d[1], d[3] = 50, 5000, -2000  # ids 1, 2, 4: matched and not exposed
    r = isolate(tmp_path, d)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    o = pl.read_parquet(tmp_path / "out.parquet")
    assert o.schema == {"MVT_ID_mvt": pl.Float64, C: pl.Int32} and o.height == N
    v = o[C].to_numpy()
    assert v[:4].tolist() == [950, 1000 + 3606, 900, 60]  # delta, LOBT-clipped, untouched, floored
    assert (v[4:] == 900).all()
    assert "VALIDATOR OK" in r.stdout


def test_isolate_delta_protects_unmatched_and_exposed_rows(tmp_path):
    d = np.zeros(N, np.int64)
    d[2] = 50  # id 3: exposed
    r = isolate(tmp_path, d)
    assert r.returncode != 0 and "PROTECTED" in r.stdout + r.stderr
    assert not (tmp_path / "out.parquet").exists()
