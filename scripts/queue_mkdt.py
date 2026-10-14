#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""XQ taxi-only variant 'dT' for production: per seed i, p and art from runway_queue_v47lq_s<i> (classifier WITHOUT q, i.e. v57's
member), taxi branch from queue_decomp_v47lqq_s<i> (regressor WITH the nested regime q). Exact re-mix (the recipe has no
--mix-additive, p_power 1): pred = p*art + (1-p)*taxi. Writes NEW files data/preds/queue_decomp_v47ldt_s<i>.parquet (same columns).
Fold equivalent: data/oof/queue_decomp_dt_<fold>.npy (scripts/queue_mkdt_oof.py)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PA, PB, PO = (
    os.environ.get("PA", "runway_queue_v47lq"),
    os.environ.get("PB", "queue_decomp_v47lqq"),
    os.environ.get("PO", "queue_decomp_v47ldt"),
)  # p/art from PA, taxi from PB, output PO
for i in range(n):
    A = pl.read_parquet(f"{ROOT}/data/preds/{PA}_s{i}.parquet").sort("MVT_ID_mvt")
    B = pl.read_parquet(f"{ROOT}/data/preds/{PB}_s{i}.parquet").sort("MVT_ID_mvt")
    assert A["MVT_ID_mvt"].equals(B["MVT_ID_mvt"]) and A.height == NRANK
    p, art, ta, tb = (A["p_raw"].to_numpy(), A["art"].to_numpy(), A["taxi"].to_numpy(), B["taxi"].to_numpy())
    assert np.allclose(p * art + (1 - p) * ta, A["TAXITIME_SEC_mvt"].to_numpy(), atol=1e-3), "XE does not reconstruct"
    assert np.allclose(art, B["art"].to_numpy()), "art differs"
    out = A.with_columns(TAXITIME_SEC_mvt=pl.Series(p * art + (1 - p) * tb), taxi=pl.Series(tb))
    fn = f"{ROOT}/data/preds/{PO}_s{i}.parquet"
    assert not os.path.exists(fn)
    out.write_parquet(fn)
    print(
        i,
        "mean |dTaxi|",
        float(np.mean(np.abs(tb - ta))),
        "mean pred",
        float(out["TAXITIME_SEC_mvt"].mean()),
        "vs XE",
        float(A["TAXITIME_SEC_mvt"].mean()),
    )
