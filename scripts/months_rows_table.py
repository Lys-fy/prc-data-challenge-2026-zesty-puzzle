#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Per-row fold table of the v107 fold line: matched scored 2025 rows of all 6 month-pair LOFO folds, one row per (fold, movement).
Lines read (6 folds, [id, pred]): <LINE>_K2_KRS (v107 fold line), <LINE>_POST (before KRS), <LINE>_LE3B (before post rules), from scripts/krs_fold_line.sh.
Columns: id fold month day hour apt lnk cls src slot y ao3 lobt ws (month-blind board mix weight) wm (month-aware, clock_lib.cell_weights_mw)
         K sp (K250 and its spread, data/oof/KSBWPc100_<fold>.npy) v107 post le3b sealed (data/sealed_days.parquet).
Read by scripts/krs_nonlinear.py fit.
usage: months_rows_table.py [LINE prefix, default scratchpad/xchain2/v106K] [out, default scratchpad/xmonths/rows.parquet]
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K

FO = ["janjul", "months28", "months39", "months410", "months511", "months612"]
LN = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/xchain2/v106K"
OUT = sys.argv[2] if len(sys.argv) > 2 else "scratchpad/xmonths/rows.parquet"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
LINES = {"v107": f"{LN}_K2_KRS", "post": f"{LN}_POST", "le3b": f"{LN}_LE3B"}
Rk = K.rank_frame()
rk = K.keys(Rk)
rslot = K.slot(Rk)
SD = pl.read_parquet("data/sealed_days.parquet")


def lv(prefix, t):
    z = np.load(f"{prefix}_{t}.npy")
    o = np.argsort(z[0])
    return z[:, o]


parts = []
for t in FO:
    z = lv(LINES["v107"], t)
    ids = z[0]
    Fr = L.frame(ids)
    y = Fr["y"].to_numpy().astype(float)
    ok = np.isfinite(y) & (Fr["unmatched"].to_numpy() == 0) & (Fr["is_rank"].to_numpy() == 0)
    fk = K.keys(Fr)
    sub = tuple(x[ok] for x in fk)
    fs = K.slot(Fr)[ok]
    ws = K.cell_weights(sub, rk)[0]
    wm = K.cell_weights_mw(sub, rk, fs, rslot)[0]
    P = {"v107": z[1][ok]}
    for k in ("post", "le3b"):
        zz = lv(LINES[k], t)
        assert np.array_equal(zz[0], ids)
        P[k] = zz[1][ok]
    Kz = np.load(f"data/oof/KSBWPc100_{t}.npy")
    m = pl.DataFrame({"i": Kz[0], "K": Kz[1], "sp": Kz[2]})
    KK = pl.DataFrame({"i": ids[ok]}).join(m, on="i", how="left", maintain_order="left")
    F = Fr.filter(pl.Series(ok))
    D = pl.DataFrame(
        {
            "id": ids[ok],
            "fold": t,
            "month": F["MVT_TIME_UTC_mvt"].dt.month().to_numpy(),
            "day": F["MVT_TIME_UTC_mvt"].dt.date(),
            "hour": F["MVT_TIME_UTC_mvt"].dt.hour().to_numpy(),
            "apt": sub[0],
            "lnk": sub[1],
            "cls": sub[2],
            "src": sub[3],
            "slot": fs,
            "y": y[ok],
            "ao3": F["mvt_aobt3"].cast(pl.Float64).to_numpy(),
            "lobt": F["mvt_lobt"].cast(pl.Float64).to_numpy(),
            "ws": ws,
            "wm": wm,
            "K": KK["K"].to_numpy(),
            "sp": KK["sp"].to_numpy(),
            **P,
        }
    )
    D = D.join(SD.select(pl.col("ADEP_mvt").alias("apt"), "day", "sealed"), on=["apt", "day"], how="left")
    parts.append(D)
    print(f"[{t}] rows {D.height:,} sum ws {ws.sum():,.0f} wm {wm.sum():,.0f}", flush=True)
A = pl.concat(parts)
A.write_parquet(OUT)
print(f"wrote {OUT} rows {A.height:,}", flush=True)
