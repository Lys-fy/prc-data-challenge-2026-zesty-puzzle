#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Validate member npz files made ON a box (fixed agg): rows map into the payload frame, taxi - g == payload offset (mvt_aobt3, NaN->0),
recomputed clean RMSE of the ensemble and of every instance == meta rmse_clean / rmse_inst. usage: check_member.py <member_dir> <PAYLOAD> NAME [NAME..]
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, glob, os, json, numpy as np, polars as pl

D, PAY = sys.argv[1:3]
bad = 0
n = 0
for NAME in sys.argv[3:]:
    for f in sorted(glob.glob(f"{D}/{NAME}_*.npz")):
        fold, apt = os.path.basename(f)[len(NAME) + 1 : -4].rsplit("_", 1)
        d = pl.read_parquet(
            f"{PAY}/member/km_{apt}.parquet",
            columns=["_row", "meta__mvt_aobt3", "meta__y", "meta__fb_pru", "meta__MVT_ID_mvt"],
        )
        R = d["_row"].to_numpy()
        off = np.nan_to_num(d["meta__mvt_aobt3"].to_numpy().astype(float))
        y = d["meta__y"].to_numpy().astype(float)
        cl = d["meta__fb_pru"].to_numpy() != 1
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        row = z["row"]
        s = np.argsort(R, kind="stable")
        loc = s[np.searchsorted(R, row, sorter=s)]
        ok_rows = np.array_equal(R[loc], row) and np.array_equal(
            d["meta__MVT_ID_mvt"].to_numpy().astype(float)[loc], z["id"]
        )
        doff = float(np.abs(z["taxi"] - z["g"] - off[loc]).max())
        ok = cl[loc] & np.isfinite(y[loc])
        yy = y[loc]
        rm = float(np.sqrt(((z["taxi"] - yy)[ok] ** 2).mean()))
        ri = [float(np.sqrt(((off[loc] + G - yy)[ok] ** 2).mean())) for G in z["g_inst"]]
        dr = abs(rm - m["rmse_clean"])
        dri = max(abs(a - b) for a, b in zip(ri, m["rmse_inst"])) if "rmse_inst" in m else float("nan")
        good = ok_rows and doff < 1e-3 and dr < 0.05 and (dri < 0.05 or np.isnan(dri))
        bad += not good
        n += 1
        print(
            f"{NAME} {fold} {apt}: rows {'OK' if ok_rows else 'BAD'} | max|taxi-g-off| {doff:.2e} | rmse {rm:.2f} vs meta {m['rmse_clean']:.2f} | inst max diff {dri:.4f} | {'OK' if good else 'FAIL'}"
        )
print(f"CHECK_MEMBER {n} files, {bad} fail")
