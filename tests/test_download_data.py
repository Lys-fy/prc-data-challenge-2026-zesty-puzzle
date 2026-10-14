# SPDX-License-Identifier: GPL-3.0-only
"""download_data.py: final_* objects go to data/final/, known files are checked by md5, a mismatch is an error."""

import hashlib

import pytest

import download_data as dd


def test_destination_of_bucket_objects():
    assert dd.dest("final_ranking.parquet") == dd.FINAL / "final_ranking.parquet"
    assert dd.dest("2026/final_submitting.parquet") == dd.FINAL / "final_submitting.parquet"
    assert dd.dest("training_2025.parquet") == dd.OUT / "training_2025.parquet"


def test_md5_skip_and_mismatch(tmp_path, monkeypatch):
    f = tmp_path / "final_ranking.parquet"
    f.write_bytes(b"fake")
    monkeypatch.setitem(dd.KNOWN_MD5, "final_ranking.parquet", "0" * 32)
    assert not dd.up_to_date(f, 4)  # same size, wrong md5: download again
    with pytest.raises(SystemExit, match="md5 mismatch"):
        dd.verify(f)
    monkeypatch.setitem(dd.KNOWN_MD5, "final_ranking.parquet", hashlib.md5(b"fake").hexdigest())
    assert dd.up_to_date(f, 999)  # md5 match decides, not size
    dd.verify(f)
    g = tmp_path / "training_x.parquet"
    g.write_bytes(b"abc")
    assert dd.up_to_date(g, 3) and not dd.up_to_date(g, 4)
    assert not dd.up_to_date(tmp_path / "missing.parquet", 0)
