#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the generalised KRS production line (not an upload): A = BASE + beta_cell * (K250_prod - BASE) on matched,
unprotected, non-TW1 ranking rows, cell = group*3 + spread tercile. Inputs: positional BASE (line without any KRS
step), TW1_IDS.parquet, COEF.json (coefficients written by krs_fit2.py, modes KRS / KRSL / KRSAR). Output: OUT
(written via isolate_delta.py OUT A BASE BASE, with protections, LOBT law, floor 60, validator; env BASEMD5 = md5
prefix of BASE, VN = validator name).

BASE must be the line WITHOUT any KRS step (e.g. v103 + F1K + CPL + KSTK isolates); the coefficients must come from
that line's fold vectors (krs_fold_line.sh).
usage: krs_build.py BASE TW1_IDS.parquet COEF.json OUT"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, subprocess, numpy as np, polars as pl
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
c = "TAXITIME_SEC_mvt"
import clockresid_lib as L

BASE, TW, COEF, OUT = sys.argv[1:5]
cf = json.load(open(COEF))
within = cf["within_airport"]
g = cf["group"]
v = pl.read_parquet(BASE).with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
K = pl.read_parquet("data/preds/KSBWPc100P_s0.parquet").select(mid=pl.col("MVT_ID_mvt").cast(pl.Float64), K="taxi")
z = np.load("data/oof/KSBWPc100_prod.npy")
S = pl.DataFrame({"mid": z[0].astype(float), "sp": z[2].astype(float)})
J = v.join(K, on="mid", how="left", maintain_order="left").join(S, on="mid", how="left", maintain_order="left")
assert J.height == NRANK and J["K"].null_count() == 0 and J["sp"].null_count() == 0
Fr = L.frame(J["mid"].to_numpy())
um = Fr["unmatched"].fill_null(1).to_numpy() == 1
ap = Fr["ADEP_mvt"].cast(pl.Utf8).fill_null("?").to_numpy()
ex = (
    pl.read_parquet("scratchpad/xhr/exposed_rows.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
    .cast(pl.Float64)
    .to_numpy()
)
tw = pl.read_parquet(TW)["MVT_ID_mvt"].cast(pl.Float64).to_numpy()
keep = um | np.isin(J["mid"].to_numpy(), ex) | np.isin(J["mid"].to_numpy(), tw)
sp = J["sp"].to_numpy()
t = np.zeros(len(sp), int)
if within:
    for A, e in cf["edges"].items():
        m = ap == A
        t[m] = np.digitize(sp[m], np.array(e))
else:
    t = np.digitize(sp, np.array(cf["edges"]["None"]))
grp = {"z0": np.zeros(len(sp), int), "lk": Fr["lk"].to_numpy().astype(int)}[g]
cell = grp * 3 + t
beta = np.array([cf["beta"].get(str(k), 0.0) for k in cell])
p = J[c].to_numpy().astype(float)
new = np.where(keep, p, p + beta * (J["K"].to_numpy() - p))
if os.environ.get(
    "TW1ADD"
):  # "V103:V102F2KM": base has no TW1 -> add the TW1 shift s = v103 - v102_F2KM on the TW1 rows (TW1 = rint(base + s))
    a_, b_ = os.environ["TW1ADD"].split(":")
    T1 = (
        pl.read_parquet(a_)
        .join(pl.read_parquet(b_), on="MVT_ID_mvt", suffix="_b")
        .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64), s=(pl.col(c) - pl.col(c + "_b")).cast(pl.Float64))
    )
    sh = (
        J.select("mid")
        .join(T1.select("mid", "s"), on="mid", how="left", maintain_order="left")["s"]
        .fill_null(0.0)
        .to_numpy()
    )
    istw = np.isin(J["mid"].to_numpy(), tw)
    assert (sh[~istw] == 0).all()
    new = np.where(istw, p + sh, new)
    print(f"TW1ADD: shift on {int((sh != 0).sum())} TW1 rows, values {np.unique(sh[istw]).tolist()[:5]}", flush=True)
print(
    f"KRS build ({cf.get('mode')}): cells {np.bincount(cell).tolist()} | kept rows {int(keep.sum())} (TW1 {int(np.isin(J['mid'].to_numpy(), tw).sum())}) | delta mean {np.mean(new - p):+.3f} RMS {np.sqrt(np.mean((new - p) ** 2)):.2f}",
    flush=True,
)
A = OUT.replace(".parquet", "_A.parquet")
v.select("MVT_ID_mvt").with_columns(pl.Series(c, np.rint(np.maximum(new, 60))).cast(v[c].dtype)).write_parquet(A)
r = subprocess.run([".venv/bin/python", "scripts/isolate_delta.py", OUT, A, BASE, BASE], env=dict(os.environ))
sys.exit(r.returncode)
