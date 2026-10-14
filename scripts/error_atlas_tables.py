#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Row-level error table of the matched 2025 fold line (v94 fold equivalent), the matched-row input of scripts/tw1_rule_build.py.
No fits, no 2026 labels.
Line  = wideleaf_le3ba + (pseudolabel_pl3_le3ba - pseudolabel_b0_le3ba): the wide-leaf line plus the pseudo-label delta over its host twin, each the
        (B) + (A) output of pru_daily_calib.py on its three chains (scripts/error_atlas_line.sh), cf-clipped.
Rows  = labelled, NM-matched, non-ranking fold rows of the 6 folds (sealed_score.ev 'ok').
w     = sealed_score convention: month-aware cell weights on all scored fold rows, rescaled by sum w_all / sum w_uns (board-s2 scale
        per fold); uns = row on an unsealed airport-day (data/sealed_days.parquet); drop = False.
usage: error_atlas_tables.py [--line <mix spec>] [--out scratchpad/xatlas]      writes <out>/rows_matched.parquet"""

import os, sys, argparse, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B, S = "scratchpad/xbase", "scratchpad/xsyn"
ap = argparse.ArgumentParser()
ap.add_argument(
    "--line", default=f"mix:1*npy:{B}/wideleaf_le3ba+1*npy:{S}/pseudolabel_pl3_le3ba+-1*npy:{S}/pseudolabel_b0_le3ba"
)
ap.add_argument("--out", default="scratchpad/xatlas")
a = ap.parse_args()
# the scorer's loaders and weights (sealed_score.py up to its pair loop)
src = open(f"{ROOT}/scripts/sealed_score.py").read().split('print("## pairs:')[0]
sys.argv = [f"{ROOT}/scripts/sealed_score.py"]
H = {"__file__": f"{ROOT}/scripts/sealed_score.py", "__name__": "error_atlas"}
exec(src, H)
load, ev, L = H["load"], H["ev"], H["L"]
os.makedirs(a.out, exist_ok=True)
FO6 = ["months39", "months28", "months410", "janjul", "months511", "months612"]
EXTRA_F9 = ["MVT_ID_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "SCHED_TIME_UTC_mvt"]
F9 = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=EXTRA_F9).with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Float64)
)
parts = []
for t in FO6:
    r = load(a.line, t)
    assert r is not None, f"line missing for fold {t}: {a.line}"
    ids, v = r
    Fr = L.frame(ids)
    lo = Fr["mvt_lobt"].to_numpy().astype(float)
    p = L.cf(v, lo)
    E = ev(t, ids)
    ok = E["ok"]
    U = ~E["sealed"]
    wm = E["wm"]
    sc = wm.sum() / wm[U].sum()
    D = Fr.filter(pl.Series(ok)).select(
        "MVT_ID_mvt",
        "y",
        "ADEP_mvt",
        "mvt_lobt",
        "mvt_aobt3",
        "mvt_eobt1",
        "mvt_sched",
        "hour",
        "MVT_TIME_UTC_mvt",
        "LOBT_flt",
        "lk",
        "adsb2_leave",
        "adsb3_mv_gs",
        "adsb_move",
    )
    D = D.with_columns(
        pl.col("ADEP_mvt").cast(pl.Utf8),
        fold=pl.lit(t),
        pred=pl.Series(p[ok]),
        w=pl.Series(wm * sc),
        uns=pl.Series(U),
        drop=pl.lit(False),
    )
    parts.append(D)
    print(t, D.height, f"UNS {U.sum():,}", flush=True)
M = pl.concat(parts).join(F9, on="MVT_ID_mvt", how="left")
M.write_parquet(f"{a.out}/rows_matched.parquet")
print("matched rows", M.height, "->", f"{a.out}/rows_matched.parquet", flush=True)
