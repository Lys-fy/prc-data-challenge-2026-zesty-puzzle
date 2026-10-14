#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the Kumo F1 / F3 arms onto a base line: writes the scope rows of ARM_F1 (UL) and ARM_F3 (UU) onto a copy
of the base line file. Inputs: positional OUT, SCOPE[,SCOPE] (UL and/or UU), env kumo_icl_lirf_base (required base
file), kumo_icl_lirf_base_md5, P2_MODE, P2_NRANK (expected row count, default the standard ranking size). Output:
OUT (same schema as the base file: MVT_ID_mvt + TAXITIME_SEC_mvt).

Only the scope's 2026 unm_other rows take the ARM's values (ARM_F1 for UL, ARM_F3 for UU; the 2026 values come from the LOMO 'rank' fit on all 12
months, already in the npz). Rows whose base value is NOT the chain the arm was scored against (e.g. the 58 UNM_LINK3 A2L rows, later isolates) are left
at the base value and reported. Everything else is asserted byte-identical; moves <= 7,200 s.
usage: kumo_icl_build.py OUT SCOPE[,SCOPE]     SCOPE in UL (ARM_F1) | UU (ARM_F3)
env: kumo_icl_lirf_base = base file (required; phase2/run_pipeline.sh passes the workspace line), kumo_icl_lirf_base_md5 = md5 prefix it must have
(default empty = no check), P2_MODE=new skips the md5 check. P2_NRANK = expected row count (default the standard ranking size, as phase2).
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, hashlib, numpy as np, polars as pl

ROOT = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + "")
os.chdir(ROOT)
OUT, SCOPES = sys.argv[1], sys.argv[2].split(",")
BP = os.environ.get("kumo_icl_lirf_base", "")
BP_MD5 = os.environ.get("kumo_icl_lirf_base_md5", "")
assert BP, "set kumo_icl_lirf_base to the base line file"
if BP_MD5 and os.environ.get("P2_MODE") != "new":
    assert hashlib.md5(open(BP, "rb").read()).hexdigest().startswith(BP_MD5), "base md5"
B = pl.read_parquet(BP)
base = B["TAXITIME_SEC_mvt"].to_numpy()
assert B.height == int(os.environ.get("P2_NRANK", "344841"))
pos = {v: i for i, v in enumerate(B["MVT_ID_mvt"].to_numpy().astype(float))}
out = base.copy()
scope_all = np.zeros(B.height, bool)
F9 = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "is_rank"]).filter(
    pl.col("is_rank") == 1
)
ap = dict(zip(F9["MVT_ID_mvt"].to_numpy().astype(float), F9["ADEP_mvt"].to_list()))
for sc in SCOPES:
    arm = {"UL": "F1", "UU": "F3"}[sc]
    z = np.load(f"scratchpad/xtfm2/ARM_{arm}.npz", allow_pickle=True)
    r = z["isr"] & (z["ul"] if sc == "UL" else z["uu"])
    ids = z["ids"][r].astype(float)
    ix = np.array([pos[v] for v in ids])
    chain = np.maximum(np.rint(z["chain"][r]), 60).astype(base.dtype)
    new = np.maximum(np.rint(z["cand"][r]), 60).astype(base.dtype)
    same = chain == base[ix]
    foreign = ~same
    fa = np.array([ap[v] for v in ids[foreign]]) if foreign.any() else np.array([])
    print(
        f"{sc} ({arm}): ranking rows {r.sum()}, base == scored chain on {same.sum()} rows; {foreign.sum()} rows differ from the chain (kept at base)"
        + (f" by airport {dict(zip(*np.unique(fa, return_counts=True)))}" if foreign.any() else "")
    )
    assert (np.abs(new[same].astype(float) - base[ix[same]]) <= 7200).all()
    out[ix[same]] = new[same]
    scope_all[ix[same]] = True
    d = new[same].astype(float) - base[ix[same]]
    c = d != 0
    ra = np.array([ap[v] for v in ids[same][c]])
    print(
        f"   changed {c.sum()}, mean {d[c].mean() if c.any() else 0:+.1f}, RMS {np.sqrt((d[c] ** 2).mean()) if c.any() else 0:.1f}, max |d| {np.abs(d).max():.0f}; by airport {dict(zip(*np.unique(ra, return_counts=True)))}"
    )
assert (out[~scope_all] == base[~scope_all]).all(), "rows outside the scope changed"
pl.DataFrame({"MVT_ID_mvt": B["MVT_ID_mvt"], "TAXITIME_SEC_mvt": out}).write_parquet(OUT)
print(
    f"wrote {OUT} md5 {hashlib.md5(open(OUT, 'rb').read()).hexdigest()} | rows changed vs base {int((out != base).sum())}"
)
