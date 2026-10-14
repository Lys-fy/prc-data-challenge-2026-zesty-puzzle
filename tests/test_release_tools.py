# SPDX-License-Identifier: GPL-3.0-only
"""Repository checks: 2025 constants against the code, and the pre-publication scan of the tree."""

import os
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT


def test_constants_match_the_code():
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools/check_constants.py"), ROOT], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_release_scan_is_clean():
    r = subprocess.run(["bash", os.path.join(ROOT, "tools/check_release.sh"), ROOT], capture_output=True, text=True)
    assert r.returncode == 0 and "RESULT: clean" in r.stdout, r.stdout[-2000:]
