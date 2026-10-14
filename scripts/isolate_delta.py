#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Adds (C88 - REF88), the matched-row change of a chain variant against its same-recipe twin, to the base line V99,
with unmatched and exposed rows protected. Inputs: positional OUT.parquet C88.parquet REF88.parquet V99.parquet, env
BASEMD5 (expected md5 prefix of V99, default 5e87ff38). Writes OUT.parquet only; no upload, nothing written to submissions/.
usage: isolate_delta.py OUT.parquet C88.parquet REF88.parquet V99.parquet

 1. protect = unmatched ranking ids ∪ the px_safe exposed ids (scratchpad/xhr/exposed_rows.parquet), i.e. the frozen rows.
 2. scripts/combine_nofloor.py add --base V99 -c C88@REF88 --allow-stale --protect <protect>   (no validate: the LOBT law is enforced in step 3)
 3. LOBT bound: changed matched rows clipped to [mvt_lobt − 3606, mvt_lobt + 3606] and floored at 60 (rint, Int32).
 4. checks: row set == the 344,841 ranking ids; 0 unmatched/exposed rows changed; out == V99 elsewhere; validate_submission.py; md5.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, subprocess, hashlib, shutil, tempfile, numpy as np, polars as pl
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P = ".venv/bin/python"
c = "TAXITIME_SEC_mvt"
OUT, C88, REF = sys.argv[1:4]
V99 = sys.argv[4]  # the base line the delta is applied to
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
assert md5(V99).startswith(os.environ.get("BASEMD5", "5e87ff38")), md5(V99)
F = pl.read_parquet(
    "data/features_v9.parquet",
    columns=["MVT_ID_mvt", "is_rank", "unmatched", "mvt_lobt", "ADEP_mvt", "MVT_TIME_UTC_mvt"],
).filter(pl.col("is_rank") == 1)
ex = pl.read_parquet("scratchpad/xhr/exposed_rows.parquet", columns=["MVT_ID_mvt"]).with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Float64)
)
prot = pl.concat([F.filter(pl.col("unmatched") == 1).select(pl.col("MVT_ID_mvt").cast(pl.Float64)), ex]).unique()
pf = OUT.replace(".parquet", "_protect.parquet")
prot.write_parquet(pf)
tmp = OUT.replace(".parquet", "_fc.parquet")
for f in (OUT, tmp):
    if os.path.exists(f):
        os.remove(f)
r = subprocess.run(
    [
        P,
        "scripts/combine_nofloor.py",
        "add",
        "--base",
        V99,
        "-c",
        f"{C88}@{REF}",
        "--allow-stale",
        "--protect",
        pf,
        "--out",
        tmp,
        "--label",
        f"CHAIN_HOOK: base {md5(V99)[:8]} + ({os.path.basename(C88)} - {os.path.basename(REF)} {md5(REF)[:8]}); unmatched + exposed protected",
    ],
    capture_output=True,
    text=True,
)
print(
    "\n".join(
        l for l in (r.stdout + r.stderr).splitlines() if "polars" not in l and "implode" not in l and "is_in" not in l
    )[-1500:]
)
assert r.returncode == 0
o = pl.read_parquet(tmp)
b = pl.read_parquet(V99)
J = F.join(b.rename({c: "b"}), on="MVT_ID_mvt", how="left").join(o.rename({c: "o"}), on="MVT_ID_mvt", how="left")
assert J["o"].null_count() == 0 and o.height == NRANK and o["MVT_ID_mvt"].n_unique() == NRANK, "row set"
lo = J["mvt_lobt"].cast(pl.Float64).to_numpy()
ov = J["o"].to_numpy().astype(float)
bv = J["b"].to_numpy().astype(float)
ch = ov != bv
has = ~np.isnan(lo)
cl = ov.copy()
m = ch & has
cl[m] = np.clip(ov[m], lo[m] - 3606, lo[m] + 3606)
cl = np.maximum(np.where(ch, cl, bv), 60)
print(f"rows < 60 before floor: {int((ov < 60).sum())} (min {ov.min():.0f})")
nclip = int((cl != ov).sum())
print(f"LOBT-law clip: {nclip} changed rows clipped (max move {np.abs(cl - ov).max():.0f} s)")
res = (
    o.join(J.select("MVT_ID_mvt").with_columns(new=pl.Series(np.rint(cl))), on="MVT_ID_mvt", how="left")
    .with_columns(pl.col("new").cast(pl.Int32).alias(c))
    .select("MVT_ID_mvt", c)
)
assert res.schema == o.schema and res["MVT_ID_mvt"].equals(o["MVT_ID_mvt"])
res.write_parquet(OUT)
os.remove(tmp)
# ---- checks ----
R = (
    F.join(res.rename({c: "o"}), on="MVT_ID_mvt")
    .join(b.rename({c: "b"}), on="MVT_ID_mvt")
    .join(pl.read_parquet(C88).rename({c: "k"}), on="MVT_ID_mvt")
    .join(pl.read_parquet(REF).rename({c: "r"}), on="MVT_ID_mvt")
)
assert R.height == NRANK
u = R["unmatched"].to_numpy() == 1
e = R["MVT_ID_mvt"].is_in(ex["MVT_ID_mvt"].implode()).to_numpy()
d = (R["o"] - R["b"]).to_numpy().astype(float)
dk = (R["k"] - R["r"]).to_numpy().astype(float)
print(
    f"ROWSET OK 344,841 | rows changed {(d != 0).sum():,} (matched {int(((d != 0) & ~u).sum()):,}; unmatched {int(((d != 0) & u).sum())}; exposed {int(((d != 0) & e).sum())}) | "
    f"delta mean {d.mean():+.3f} sd {d.std():.2f} RMS {np.sqrt((d ** 2).mean()):.2f} max|d| {np.abs(d).max():.0f} | on changed rows: mean {d[d != 0].mean():+.2f} sd {d[d != 0].std():.2f}"
)
assert int(((d != 0) & (u | e)).sum()) == 0
rest = ~(u | e)
print(
    f"|out − (v99 + (C88 − REF))| on unprotected rows: max {np.abs(d[rest] - dk[rest]).max():.0f}, rows ≠ {int((d[rest] != dk[rest]).sum())} (rint/60 floor/LOBT clip)"
)
L = (
    R.with_columns(d=pl.Series(d), mo=pl.col("MVT_TIME_UTC_mvt").dt.month())
    .filter(pl.Series(~u))
    .group_by(["ADEP_mvt", "mo"])
    .agg(pl.col("d").mean().alias("s"))
)
print("airport x month level shift (matched): max |shift|", round(float(L["s"].abs().max()), 3))
print(
    "per airport mean/sd of d:",
    {
        a: (round(float(g["d"].mean()), 2), round(float(g["d"].std()), 1))
        for (a,), g in R.with_columns(d=pl.Series(d)).group_by(["ADEP_mvt"])
    },
)
td = tempfile.mkdtemp()
vp = f"{td}/isolate_check.parquet"
shutil.copy(OUT, vp)
v = subprocess.run([P, "scripts/validate_submission.py", vp, V99], capture_output=True, text=True)
print(v.stdout.strip(), v.stderr.strip()[-400:])
assert v.returncode == 0, "VALIDATOR FAILED"
print("VALIDATOR OK | md5", md5(OUT), OUT)
