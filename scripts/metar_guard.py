# SPDX-License-Identifier: GPL-3.0-only
"""Fail-closed METAR join for build_features.py.
A missing or broken METAR file would give a feature table without the 69 wx columns and a different model; join_metar_or_fail() raises
in that case, and also checks
coverage: in the production features_v9 wx0_tmpc is non-null on 99.996 % (2025) / 99.998 % (Jan/Jul 2026) of the rows, so a share
below MIN_COVERAGE on either year means the METAR file does not cover the period.
Opt-out for experiments only: PRC_ALLOW_NO_METAR=1 warns and continues without the wx columns."""

import os
import polars as pl

MIN_COVERAGE = float(os.environ.get("PRC_METAR_MIN_COVERAGE", "0.99"))
N_WX_COLS = 69  # wx0_/wx30_/wx60_ x 20 + wxd_* + wx_age columns in features_v9


class MetarError(RuntimeError):
    pass


def join_metar_or_fail(dep: pl.DataFrame, log=print, metar_path=None, rank_col: str = "is_rank") -> pl.DataFrame:
    from metar_features import add_metar_features, METAR_PATH

    path = metar_path or METAR_PATH
    if os.environ.get("PRC_ALLOW_NO_METAR") == "1":  # explicit, logged opt-out (old behaviour)
        try:
            out = add_metar_features(dep, metar_path=path)
            log("metar joined")
            return out
        except Exception as e:
            log("WARNING no metar features (PRC_ALLOW_NO_METAR=1):", repr(e)[:120])
            return dep
    if not os.path.exists(path):
        raise MetarError(
            f"METAR file missing: {path} (run scripts/download_metar.py + scripts/build_metar.py, or set PRC_ALLOW_NO_METAR=1)"
        )
    out = add_metar_features(dep, metar_path=path)  # any error propagates
    wx = [c for c in out.columns if c.startswith(("wx0_", "wx30_", "wx60_", "wxd_", "wx_age"))]
    if len(wx) != N_WX_COLS:
        raise MetarError(f"METAR join produced {len(wx)} wx columns, expected {N_WX_COLS}")
    groups = [pl.lit(True)] if rank_col not in out.columns else [pl.col(rank_col) == 0, pl.col(rank_col) == 1]
    for g in groups:
        sub = out.filter(g)
        if sub.height == 0:
            continue
        cov = sub.select(pl.col("wx0_tmpc").is_not_null().mean()).item()
        if cov < MIN_COVERAGE:
            raise MetarError(
                f"METAR coverage {cov:.4f} < {MIN_COVERAGE} on {sub.height:,} rows ({g}): the METAR file does not cover the period"
            )
    log("metar joined", len(wx), "cols")
    return out
