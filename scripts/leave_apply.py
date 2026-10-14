#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Applies an optional second correction stage on top of a clock_stage Wh6 parquet file: (A) with --lirf-excl, resets LIRF
editable rows to the --pre values; (B) with --arm L|M|X, fits a LightGBM stage on linked matched rows of that arm and
applies it to that arm's ranking-editable rows. Every other row is copied byte-identically from --target.
Inputs: --target, --pre, --npz (clock_rw.npz), optional --gnpz, --arm, --lirf-excl, --seeds, --nt. Output: --out parquet.

leave_stage production: on top of an clock_stage Wh6 file (--target, = --pre + clock_resid_stage/clock_stage Wh6 stage), optionally
 (A) --lirf-excl: LIRF editable rows reset to the pre-stage target (--pre) values, and
 (B) --arm L|M|X: a second stage (clockresid_lib.PRM, 400 rounds, clkonly with base = Wh6, +src for X, --seeds, unweighted) fitted on the
     four folds' linked matched rows of that arm with base = the Wh6 LOFO OOF (from clock_stage's npz), applied on ranking editable rows
     (matched, LOBT, linked) of that arm (not LIRF if --lirf-excl) with base = the target's value; clip +-300, rint.
Every other row is copied byte-identically from --target (unmatched asserted).
usage: leave_apply.py --target T --pre P --npz clock_rw.npz --out O [--arm L] [--lirf-excl] [--seeds 3] [--nt 8]"""

import os, sys, argparse, hashlib, numpy as np, polars as pl, lightgbm as lgb
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--target", required=True, help="clock_stage Wh6 parquet to apply the leave_stage correction on top of")
ap.add_argument("--pre", required=True, help="pre-stage parquet, used as the LIRF reset value")
ap.add_argument("--npz", required=True, help="clock_rw.npz with the clock_stage LOFO OOF arrays")
ap.add_argument("--out", required=True, help="output parquet path; must not already exist")
ap.add_argument(
    "--arm", default="none", choices=["none", "L", "M", "X"], help="which arm to fit/apply the second stage for"
)
ap.add_argument("--lirf-excl", action="store_true", help="reset LIRF editable rows to the --pre values")
ap.add_argument("--seeds", type=int, default=3, help="number of LightGBM seeds to average")
ap.add_argument("--nt", type=int, default=8, help="LightGBM num_threads")
ap.add_argument("--gnpz", default=None, help="huber_clock: stage-1 OOF from this npz's G_Hh (Huber) instead of G_Wh")
a = ap.parse_args()
pl.Config.set_tbl_rows(-1)
V = pl.read_parquet(a.target)
vc = [c for c in V.columns if c != "MVT_ID_mvt"][0]
PRE = pl.read_parquet(a.pre)
R = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank"]).filter(
    pl.col("is_rank") == 1
)
Rk = L.frame(R["MVT_ID_mvt"].to_numpy())
for nm, D in (("P1", V), ("P0", PRE)):
    Rk = Rk.join(
        D.select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(vc).cast(pl.Float64).alias(nm)),
        on="MVT_ID_mvt",
        how="left",
        maintain_order="left",
    )
assert Rk.height == NRANK and Rk["P1"].null_count() == 0 and Rk["P0"].null_count() == 0
P1 = Rk["P1"].to_numpy()
P0 = Rk["P0"].to_numpy()
rlo = Rk["mvt_lobt"].to_numpy().astype(float)
rapt = Rk["ADEP_mvt"].cast(pl.Utf8).to_numpy().astype(str)
edit = (Rk["unmatched"].to_numpy() == 0) & Rk["LOBT_flt"].is_not_null().to_numpy() & (Rk["lk"].to_numpy() == 1)
rs = K.src(Rk)
print(
    f"editable {edit.sum():,}; target vs pre: changed {(P1 != P0).sum():,} (all editable: {((P1 != P0) <= edit).all()})",
    flush=True,
)
new = P1.copy()
if a.arm != "none":
    Z = np.load(a.npz)
    ids, ok, lk, b, y, lo = (Z[k] for k in ("ids", "ok", "lk", "b", "y", "lo"))
    LK = ok & lk
    Wh6 = np.where(
        LK,
        L.cf(b + np.clip(np.nan_to_num((np.load(a.gnpz)["G_Hh"] if a.gnpz else Z["G_Wh"]).mean(0)), -300, 300), lo),
        b,
    )
    Fr = L.frame(ids)
    sn = K.src(Fr)
    clk = L.clock_cols(Fr)
    am = {"L": sn == 0, "M": sn == 1, "X": np.ones(len(sn), bool)}[a.arm]
    tr = LK & am
    ram = {"L": rs == 0, "M": rs == 1, "X": np.ones(len(rs), bool)}[a.arm]
    ap_ = edit & ram & ~((rapt == "LIRF") & a.lirf_excl)
    Xf, cols = L.design(Fr.filter(pl.Series(tr)), Wh6[tr], "clkonly", clk)
    Xr, cr = L.design(Rk.filter(pl.Series(ap_)), P1[ap_], "clkonly", clk)
    assert cols == cr
    if a.arm == "X":
        Xf = np.hstack([Xf, sn[tr, None].astype(np.float32)])
        Xr = np.hstack([Xr, rs[ap_, None].astype(np.float32)])
    rtr = np.clip(y[tr] - Wh6[tr], -1800, 1800)
    g = np.zeros(ap_.sum())
    print(f"stage 2 arm {a.arm}: fit rows {tr.sum():,}, apply rows {ap_.sum():,}", flush=True)
    for s in range(a.seeds):
        m = lgb.train(
            dict(L.PRM, seed=s, num_threads=a.nt), lgb.Dataset(Xf, rtr, categorical_feature=[cols.index("apt")]), 400
        )
        g += m.predict(Xr, num_threads=a.nt) / a.seeds
    new[ap_] = L.cf(P1[ap_] + np.clip(g, -300, 300), rlo[ap_])
if a.lirf_excl:
    lm = edit & (rapt == "LIRF")
    new[lm] = P0[lm]
    print(f"LIRF editable rows reset to pre: {lm.sum():,} (of which differed {(P1[lm] != P0[lm]).sum():,})")
if V.schema[vc].is_integer():
    new = np.rint(new)
dv = new - P1
assert (dv[~edit] == 0).all()
print(
    f"rows changed vs target {(dv != 0).sum():,}; mean {dv[edit].mean():+.3f} sd {dv[edit].std():.2f} max|d| {np.abs(dv).max():.0f}; RMS all rows {np.sqrt((dv**2).mean()):.2f}"
)
Rk = Rk.with_columns(dv=pl.Series(dv), mon=pl.col("MVT_TIME_UTC_mvt").dt.month())
LM = Rk.filter(pl.col("unmatched") == 0).group_by(["ADEP_mvt", "mon"]).agg(pl.col("dv").mean().round(2).alias("m"))
print(LM.pivot(on="mon", index="ADEP_mvt", values="m", sort_columns=True).sort("ADEP_mvt"))
print(f"max |airport x month shift| {LM['m'].abs().max():.2f} s")
out = V.join(
    pl.DataFrame({"MVT_ID_mvt": Rk["MVT_ID_mvt"], "new": new}),
    left_on=pl.col("MVT_ID_mvt").cast(pl.Float64),
    right_on="MVT_ID_mvt",
    how="left",
    suffix="_r",
)
out = out.with_columns(pl.col("new").cast(V.schema[vc]).alias(vc)).select(V.columns)
assert out.height == V.height and out["MVT_ID_mvt"].equals(V["MVT_ID_mvt"])
um = set(Rk.filter(pl.col("unmatched") == 1)["MVT_ID_mvt"].to_list())
chk = out.join(V, on="MVT_ID_mvt", suffix="_o").filter(pl.col("MVT_ID_mvt").cast(pl.Float64).is_in(list(um)))
assert chk.height == len(um) and (chk[vc] == chk[vc + "_o"]).all(), "unmatched rows changed"
print("unmatched rows byte-identical:", len(um))
assert not os.path.exists(a.out), f"{a.out} exists"
out.write_parquet(a.out)
print("wrote", a.out, out.shape, out.schema[vc], "md5", hashlib.md5(open(a.out, "rb").read()).hexdigest())
