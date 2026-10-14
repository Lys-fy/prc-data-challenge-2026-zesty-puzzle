#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Runs a clockresid_lib-based clock-stage script with the Kumo member KM visible to the stage as one extra design column
dd_kmx_taxi = base - KM taxi (added through clockresid_lib.TIMES, so every design() call of chain_clock / chain_leave (fold side) and
chain_clock_apply / leave_apply (production side) gets it).
KM table = data/oof/<chain_fold_line_km>_<fold>.npy (6 folds, [id, taxi]) + data/preds/<chain_fold_line_kmp>_s0.parquet [MVT_ID_mvt, taxi] (ranking rows).
It calls scripts/clockstage_run.py, which adds the adsbL clocks and calls pushstart_run.py (adsb6) and adsb5_stage_run.py (adsb5).
usage: chain_fold_line_km=kumo_member_screen_km chain_fold_line_kmp=kumo_member_screen_kmp kumo_clock_run.py <script.py> [args...]
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, runpy, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib

KM, KMP = os.environ["chain_fold_line_km"], os.environ["chain_fold_line_kmp"]
ids, tx = [], []
for t in ["months39", "months28", "months410", "janjul", "months612", "months511"]:
    z = np.load(f"{ROOT}/data/oof/{KM}_{t}.npy")
    ids.append(z[0])
    tx.append(z[1])
P = pl.read_parquet(f"{ROOT}/data/preds/{KMP}_s0.parquet")
ids.append(P["MVT_ID_mvt"].cast(pl.Float64).to_numpy())
tx.append(P["taxi"].cast(pl.Float64).to_numpy())
T = pl.DataFrame(
    {"MVT_ID_mvt": np.concatenate(ids).astype(float), "kmx_taxi": np.concatenate(tx).astype(float)}
).unique("MVT_ID_mvt", keep="first")
print(f"CLOCK_CHAIN_FIX kmx: KM {KM} + {KMP}: {T.height:,} ids, taxi mean {T['kmx_taxi'].mean():.1f}", flush=True)
_frame0 = clockresid_lib.frame


def _frame(ids_, _f=_frame0):
    F = _f(ids_)
    F = F.join(T, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert F.height == len(ids_)
    print(
        f"CLOCK_CHAIN_FIX kmx: frame {F.height:,} rows, kmx cover {F['kmx_taxi'].is_not_null().mean():.4f}", flush=True
    )
    return F


clockresid_lib.frame = _frame
clockresid_lib.TIMES = clockresid_lib.TIMES + ["kmx_taxi"]
sys.argv = [f"{ROOT}/scripts/clockstage_run.py"] + sys.argv[1:]
runpy.run_path(f"{ROOT}/scripts/clockstage_run.py", run_name="__main__")
