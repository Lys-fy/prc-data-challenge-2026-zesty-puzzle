#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Production post-rules (standfloor stand floor and offblock_pull late-off-block pull, the fixed arms of phase2/params/postrules_fixed.json)
applied to a fold vector, so that the 2025 fold line (scripts/krs_fold_line.sh) includes the post-rule step of the production line.
usage: chain_post.py <in fold prefix> <out fold prefix>      (6 folds, [MVT_ID, pred]); prints rows changed per fold."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
IN, OUT = sys.argv[1], sys.argv[2]
# --- fixed production arms (= phase2/params/postrules_fixed.json) ---
standfloor = {
    "EDDF": (600, 300, 1.0),
    "EGLL": (900, 300, 1.0),
    "LFPG": (900, 300, 1.0),
    "LIRF": (600, 0, 0.5),
    "LTFM": (900, 300, 1.0),
}
offblock_pull = {
    "EGLL": ("px", 7200, 0.5),
    "LFPG": ("px", 7200, 0.5),
    "LIRF": ("px", 7200, 1.0),
    "LSZH": ("px", 7200, 0.5),
    "EDDM": ("px", 7200, 0.25),
    "LEBL": ("px", 7200, 0.25),
    "LEMD": ("px", 7200, 0.5),
    "LTFM": ("px", 7200, 0.25),
    "EDDF": ("sh", 5400, -47.0),
    "EHAM": ("sh", 7200, 35.0),
}


def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo) - 3606, np.nan_to_num(lo) + 3606))
    return np.maximum(v, 60.0)


def anchor(ssai):
    return np.where(np.isfinite(ssai) & (ssai < 7200), ssai, np.nan)


def floor_arm(p, a_, lo, g):
    W, b, k = g
    sel = np.isfinite(a_) & (a_ - p > 0) & (a_ - p < W)
    return cf(np.where(sel, p + k * (a_ + b - p), p), lo)


def overlay(arm, pp, al, lo, s=0.0):
    f, t, k = arm
    sel = np.nan_to_num(al, nan=-1e9) > t
    if f == "px":
        return cf(np.where(sel & np.isfinite(lo), pp + k * (np.maximum(lo, 0) - pp), pp), lo)
    return cf(np.where(sel, pp + s, pp), lo)


ep = lambda c: pl.col(c).dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)
# --- fold rows instead of ranking rows ---
R = (
    pl.read_parquet(
        "data/features_v9.parquet",
        columns=[
            "MVT_ID_mvt",
            "is_rank",
            "unmatched",
            "ADEP_mvt",
            "mvt_lobt",
            "stand_since_arr_inblock",
            "AOBT_3_flt",
            "LOBT_flt",
        ],
    )
    .filter(pl.col("is_rank") == 0)
    .with_columns(
        mid=pl.col("MVT_ID_mvt").cast(pl.Float64),
        ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8),
        al=ep("AOBT_3_flt") - ep("LOBT_flt"),
    )
    .drop("AOBT_3_flt", "LOBT_flt", "MVT_ID_mvt")
)
for t in ["months39", "months28", "months410", "janjul", "months612", "months511"]:
    g = sorted(glob.glob(f"{IN}_{t}.npy"))
    if not g:
        print(t, "missing")
        continue
    z = np.load(g[0])
    J = pl.DataFrame({"mid": z[0]}).join(R, on="mid", how="left", maintain_order="left")
    assert J.height == z.shape[1]
    N = lambda c: J[c].to_numpy().astype(float)
    p = z[1].astype(float)
    lo = N("mvt_lobt")
    ap = J["ADEP_mvt"].fill_null("?").to_numpy().astype(str)
    um = J["unmatched"].fill_null(1).to_numpy() == 1
    a26 = anchor(N("stand_since_arr_inblock"))
    al = N("al")
    p1 = p.copy()
    for A, gg in standfloor.items():
        m = (~um) & (ap == A)
        p1[m] = floor_arm(p[m], a26[m], lo[m], gg)
    n1 = int((p1 != p).sum())
    p2 = p1.copy()
    for A, arm in offblock_pull.items():
        m = (~um) & (ap == A)
        s = arm[2] if arm[0] == "sh" else 0.0
        p2[m] = overlay(arm, p1[m], al[m], lo[m], s)
    n2 = int((p2 != p1).sum())  # no rint: fold vectors stay float (rint is a +-0.5 s no-op for the ladder)
    np.save(f"{OUT}_{t}.npy", np.vstack([z[0], p2]))
    print(f"{t}: rows {len(p):,} floor changed {n1:,} overlay changed {n2:,} unmatched {um.sum():,}", flush=True)
