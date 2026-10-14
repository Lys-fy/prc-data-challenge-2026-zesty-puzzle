#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Clock-residual stage, production side (docstring below).
"""Fits a LightGBM clock-residual correction on ADS-B-linked matched fold rows and applies it to a submission-format
file's matched, LOBT-present, linked ranking rows; every other row is copied unchanged.
Inputs: --target parquet, --fit-oof OOF fold-vector prefix(es), data/features_v9.parquet.
Output: --out parquet (same schema as --target), unless --dry is given.

Clock-residual stage, production side: applies the clock stage to a submission-format file.
Fit: LightGBM (clockresid_lib settings, --seeds averaged) on clip(y - base, +-1800) over the ADS-B-linked matched rows of all fold months,
     base = the stacked OOF vector <fit-oof>_<fold>.npy ([MVT_ID, pred]) of the same pipeline as the target.
     --wmode none|sqrt|full reweights the training rows to the airport x class x anchor-source mix of the ranking rows (clock_lib).
Apply: ranking rows that are matched (features_v9 unmatched == 0 and LOBT present) and linked (adsb_linked == 1):
       new = cf(target + shrink * clip(g, +-clip)), g computed with base = the target's value; rounded (np.rint) for an
       integer target. Every other row (no ADS-B link, unmatched, protected) is copied unchanged from the target.
--objective huber --alpha 600 switches the stage loss (default l2).
--fit-oof A,B,C trains on the linked rows under A (target chain) plus the same rows under B, C (a single prefix: A only).
env clock_tune_cleaves / clock_tune_crounds (default 400): stage capacity.
usage: stage12_clock_apply.py --target <submission.parquet> --fit-oof <prefix> --out <out.parquet> [--fset clkonly|full]
                     [--shrink S] [--clip C] [--seeds 2] [--wmode none|sqrt|full] [--v53 stages/unmatched_rule.parquet] [--nt 4] [--dry]
"""

import os, sys, glob, argparse, hashlib, numpy as np, polars as pl
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K, lightgbm as lgb

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--target", required=True, help="submission-format parquet to apply the clock stage on top of")
ap.add_argument("--fit-oof", required=True, help="comma list of OOF fold-vector prefixes (first = target chain)")
ap.add_argument("--out", required=True, help="output parquet path")
ap.add_argument("--fset", default="clkonly", help="clockresid_lib feature set")
ap.add_argument("--shrink", type=float, required=True, help="shrinkage applied to the predicted correction")
ap.add_argument("--clip", type=float, required=True, help="symmetric clip on the predicted correction, in seconds")
ap.add_argument("--seeds", type=int, default=2, help="number of LightGBM seeds to average")
ap.add_argument("--v53", default=None, help="optional unmatched_rule.parquet for a target == v53 comparison report")
ap.add_argument("--nt", type=int, default=4, help="LightGBM num_threads")
ap.add_argument("--dry", action="store_true", help="run and report but do not write --out")
ap.add_argument("--wmode", default="sqrt", choices=["none", "sqrt", "full"], help="row reweighting mode")
ap.add_argument("--objective", default="l2", choices=["l2", "huber"], help="stage loss")
ap.add_argument(
    "--alpha", type=float, default=600.0, help="huber alpha (only with --objective huber)"
)  # huber_clock; default l2 = unchanged
a = ap.parse_args()
XP = {} if a.objective == "l2" else dict(objective="huber", alpha=a.alpha)
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(20)


def find(prefix, fold):
    for pat in [f"{prefix}_{fold}.npy", f"{prefix}_{fold}_*.npy"] + [
        f"{d}/{os.path.basename(prefix)}_{fold}*.npy"
        for d in (f"{ROOT}/data/oof", f"{ROOT}/scratchpad/xcr", f"{ROOT}/scratchpad/xz")
    ]:
        g = sorted(glob.glob(pat))
        if g:
            return g[0]
    raise FileNotFoundError(f"{prefix} {fold}")


PREF = a.fit_oof.split(
    ","
)  # clock_oof_prefix: extra comma-separated prefixes = augmenting chain variants (same linked rows, their base); one prefix = unchanged
ids, pred = [], []
for t in L.FOLDS:
    p = find(PREF[0], t)
    z = np.load(p)
    ids.append(z[0])
    pred.append(z[1])
    print("fit-oof", t, p, z.shape[1], flush=True)
ids = np.concatenate(ids)
pred = np.concatenate(pred)
Fr = L.frame(ids)
y = Fr["y"].to_numpy().astype(float)
lo = Fr["mvt_lobt"].to_numpy().astype(float)
ok = np.isfinite(y) & (Fr["unmatched"].to_numpy() == 0) & (Fr["is_rank"].to_numpy() == 0) & np.isfinite(pred)
lk = Fr["lk"].to_numpy() == 1
b = L.cf(pred, lo)
clk = L.clock_cols(Fr)
Xf, cols = L.design(Fr, b, a.fset, clk)
tr = ok & lk
print(f"fit rows {tr.sum():,} (linked matched fold rows of {ok.sum():,}); features {len(cols)}", flush=True)
# ranking side
V = pl.read_parquet(a.target)
vc = [c for c in V.columns if c != "MVT_ID_mvt"][0]
R = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank"]).filter(
    pl.col("is_rank") == 1
)
Rk = L.frame(R["MVT_ID_mvt"].to_numpy())
Rk = Rk.join(
    V.select(pl.col("MVT_ID_mvt").cast(pl.Float64), pl.col(vc).cast(pl.Float64).alias("P0")),
    on="MVT_ID_mvt",
    how="left",
    maintain_order="left",
)
assert Rk.height == NRANK and Rk["P0"].null_count() == 0
P0 = Rk["P0"].to_numpy()
rlo = Rk["mvt_lobt"].to_numpy().astype(float)
edit = (Rk["unmatched"].to_numpy() == 0) & Rk["LOBT_flt"].is_not_null().to_numpy() & (Rk["lk"].to_numpy() == 1)
Xr, _ = L.design(Rk, P0, a.fset, clk)
wt = None
if a.wmode != "none":
    fk = K.keys(Fr)
    rk = K.keys(Rk)
    wt, _, unc = K.cell_weights(tuple(x[tr] for x in fk), tuple(x[edit] for x in rk), matched_only=False)
    wt = wt / wt.mean()
    wt = np.clip(wt, 0.2, 5.0)
    wt /= wt.mean()
    if a.wmode == "sqrt":
        wt = np.sqrt(wt) / np.sqrt(wt).mean()
    print(
        f"weights ({a.wmode}): sd {wt.std():.3f} min {wt.min():.2f} max {wt.max():.2f}, ranking rows uncovered {unc}; by src "
        + " ".join(f"{n} {wt[fk[3][tr] == n].mean():.2f}" for n in K.SRC),
        flush=True,
    )
g = np.zeros(edit.sum())
imp = np.zeros(len(cols))
rtr = np.clip(y[tr] - b[tr], -1800, 1800)
Xtr = Xf[tr]
if (
    len(PREF) > 1
):  # clock_oof_prefix: append the same linked rows under each augmenting chain's fold vector; weights copied; min_data_in_leaf x n_copies
    pos = {v: i for i, v in enumerate(ids)}
    Xs, rs, ws = [Xtr], [rtr], [wt if wt is not None else np.ones(tr.sum())]
    wfull = np.full(len(ids), np.nan)
    wfull[tr] = ws[0]
    for pf in PREF[1:]:
        v = np.full(len(ids), np.nan)
        for t in L.FOLDS:
            z = np.load(find(pf, t))
            ix = np.array([pos.get(i, -1) for i in z[0]])
            v[ix[ix >= 0]] = z[1][ix >= 0]
        v = L.cf(v, lo)
        mm = tr & np.isfinite(v)
        X_, _ = L.design(Fr, v, a.fset, clk)
        Xs.append(X_[mm])
        rs.append(np.clip(y[mm] - v[mm], -1800, 1800))
        ws.append(wfull[mm])
        print(
            f"aug {pf}: {mm.sum():,} rows, rmse vs target base {np.sqrt(np.mean((v - b)[mm] ** 2)):.1f} s", flush=True
        )
    Xtr, rtr, wt = np.concatenate(Xs), np.concatenate(rs), np.concatenate(ws)
    XP = dict(XP, min_data_in_leaf=500 * len(PREF))
XP = dict(XP, **{k: tp(os.environ[e]) for k, e, tp in (("num_leaves", "clock_tune_cleaves", int),) if e in os.environ})
XR_ = int(os.environ.get("clock_tune_crounds", "400"))
print("oof_build capacity", XP, XR_, flush=True)  # capacity of the clock stage
for sd in range(a.seeds):
    mdl = lgb.train(
        dict(L.PRM, seed=sd, num_threads=a.nt, **XP),
        lgb.Dataset(Xtr, rtr, weight=wt, categorical_feature=[cols.index("apt")]),
        XR_,
    )
    g += mdl.predict(Xr[edit], num_threads=a.nt) / a.seeds
    imp += mdl.feature_importance("gain")
print("top gain: " + ", ".join(f"{cols[i]} {100*imp[i]/imp.sum():.0f}%" for i in np.argsort(-imp)[:10]))
new = P0.copy()
new[edit] = L.cf(P0[edit] + a.shrink * np.clip(g, -a.clip, a.clip), rlo[edit])
if V.schema[vc].is_integer():
    new = np.rint(new)
dv = new - P0
m = Rk["unmatched"].to_numpy() == 0
print(
    f"ranking rows {len(P0):,} | matched {m.sum():,} | editable (matched, LOBT, linked) {edit.sum():,} | rows changed {(dv != 0).sum():,}"
)
print(
    f"correction on editable rows: mean {dv[edit].mean():+.3f} sd {dv[edit].std():.2f} p1 {np.percentile(dv[edit],1):+.1f} p99 {np.percentile(dv[edit],99):+.1f} max|d| {np.abs(dv).max():.0f}; RMS over all rows {np.sqrt((dv**2).mean()):.2f}"
)
assert (dv[~edit] == 0).all()
Rk = Rk.with_columns(dv=pl.Series(dv), ed=pl.Series(edit), mon=pl.col("MVT_TIME_UTC_mvt").dt.month())
print("level shift per airport (mean over matched rows | over edited rows):")
print(
    Rk.filter(pl.col("unmatched") == 0)
    .group_by("ADEP_mvt")
    .agg(
        pl.col("dv").mean().round(3).alias("matched"),
        pl.col("dv").filter(pl.col("ed")).mean().round(3).alias("linked"),
        pl.col("ed").mean().round(3).alias("lk_share"),
    )
    .sort("ADEP_mvt")
)
LM = Rk.filter(pl.col("unmatched") == 0).group_by(["ADEP_mvt", "mon"]).agg(pl.col("dv").mean().round(2).alias("m"))
print("level shift per airport x month (mean over matched rows):")
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
if a.v53:
    W = V.join(pl.read_parquet(a.v53), on="MVT_ID_mvt", suffix="_53").join(out, on="MVT_ID_mvt", suffix="_n")
    fz = W.filter(pl.col(vc) == pl.col(vc + "_53"))
    print(
        "rows with target == v53:",
        fz.height,
        "| of which changed:",
        fz.filter(pl.col(vc) != pl.col(vc + "_n")).height,
        "| unmatched changed: 0 (asserted,",
        len(um),
        "rows)",
    )
if a.dry:
    sys.exit(0)
assert not os.path.exists(a.out), f"{a.out} exists"
out.write_parquet(a.out)
print("wrote", a.out, out.shape, "dtype", out.schema[vc], "md5", hashlib.md5(open(a.out, "rb").read()).hexdigest())
