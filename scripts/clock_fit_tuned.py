#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Clock stage of a chain, fold side: the chain's clock stage on a saved fold vector, leave-one-fold-out over clockresid_lib.FOLDS: the clock-residual stage Wh6 (L2 loss by default) (cell
clkonly, shrink 1.0, clip +-300 s, sqrt source weights, seeds 0..NS-1) on the ADS-B-linked matched rows, then the leave stage A0 (stage
correction set to zero at LIRF). --objective huber --alpha A switches the stage loss; --jit / --ncopy add jittered training copies.
Stage capacity env knobs (defaults = clockresid_lib settings): clock_tune_cleaves, clock_tune_cminleaf (500), clock_tune_crounds (400),
clock_tune_clr, clock_tune_cl2, clock_tune_cff.
Writes <out>_Wh6_<fold>.npy and <out>_A0_<fold>.npy ([MVT_ID, pred]).
usage: clock_fit_tuned.py <base prefix> <out prefix> [--ns 6] [--nt 6]"""

import os, sys, glob, argparse, time, numpy as np, lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("out")
ap.add_argument("--ns", type=int, default=6)
ap.add_argument("--nt", type=int, default=6)
ap.add_argument("--objective", default="l2")
ap.add_argument("--alpha", type=float, default=600.0)
ap.add_argument("--npz", default=None)
ap.add_argument("--jit", type=float, default=0.0)
ap.add_argument("--ncopy", type=int, default=2)
a = ap.parse_args()
T0 = time.time()


def log(*x):
    print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)


ids, pred, fo = [], [], []
for t in L.FOLDS:
    p = sorted(glob.glob(f"{a.base}_{t}.npy") or glob.glob(f"{a.base}_{t}_*.npy"))[0]
    z = np.load(p)
    ids.append(z[0])
    pred.append(z[1])
    fo.append(np.full(z.shape[1], t))
ids = np.concatenate(ids)
pred = np.concatenate(pred)
fo = np.concatenate(fo)
Fr = L.frame(ids)
y = Fr["y"].to_numpy().astype(float)
lo = Fr["mvt_lobt"].to_numpy().astype(float)
ok = np.isfinite(y) & (Fr["unmatched"].to_numpy() == 0) & (Fr["is_rank"].to_numpy() == 0) & np.isfinite(pred)
lk = Fr["lk"].to_numpy() == 1
b = L.cf(pred, lo)
clk = L.clock_cols(Fr)
Xf, cols = L.design(Fr, b, "clkonly", clk)
r = np.clip(y - b, -1800, 1800)
fk = K.keys(Fr)
Rk = K.rank_frame()
rk = K.keys(Rk)
redit = np.char.startswith(rk[2].astype(str), "M_") & (rk[1] == "lnk") & Rk["LOBT_flt"].is_not_null().to_numpy()
rk_ed = tuple(x[redit] for x in rk)
log(f"folds {L.FOLDS} base {a.base}: rows {len(y):,} ok {ok.sum():,} linked {(ok & lk).sum():,}")
G = np.full((a.ns, len(y)), np.nan)
XJ = (
    []
)  # jitter (clock_tune_jitter T11 / control): training copies with base b' = cf(b + N(0, jit)), target y - b', same weights, min_leaf x (1+ncopy)
if a.jit > 0:
    rng = np.random.default_rng(7)
    for _ in range(a.ncopy):
        bj = L.cf(b + rng.normal(0, a.jit, len(b)), lo)
        Xj, _ = L.design(Fr, bj, "clkonly", clk)
        XJ.append((Xj, np.clip(y - bj, -1800, 1800)))
    log(f"jitter sd {a.jit} x {a.ncopy} copies")
# clock_tune: env capacity knobs (defaults = stage12_clock / clockresid_lib.PRM, 400 rounds)
CLOCK_TUNE_PARAMS = {
    k: tp(os.environ[e])
    for k, e, tp in (
        ("num_leaves", "clock_tune_cleaves", int),
        ("learning_rate", "clock_tune_clr", float),
        ("lambda_l2", "clock_tune_cl2", float),
        ("feature_fraction", "clock_tune_cff", float),
    )
    if e in os.environ
}
CLOCK_TUNE_ROUNDS = int(os.environ.get("clock_tune_crounds", "400"))
ML = int(os.environ.get("clock_tune_cminleaf", "500"))
CLOCK_TUNE_PARAMS["min_data_in_leaf"] = ML * (1 + len(XJ))
clock_tune_jparams = {}
print("clock_tune stage params", CLOCK_TUNE_PARAMS, "rounds", CLOCK_TUNE_ROUNDS, flush=True)
for t in L.FOLDS:
    tr = ok & lk & (fo != t)
    te = ok & lk & (fo == t)
    w, cov, unc = K.cell_weights(tuple(x[tr] for x in fk), rk_ed, matched_only=False)
    w = w / w.mean()
    w = np.clip(w, 0.2, 5.0)
    w /= w.mean()
    wh = np.sqrt(w) / np.sqrt(w).mean()
    for s in range(a.ns):
        Xtr = np.concatenate([Xf[tr]] + [x[tr] for x, _ in XJ])
        rtr = np.concatenate([r[tr]] + [q[tr] for _, q in XJ])
        wtr = np.tile(wh, 1 + len(XJ))
        m = lgb.train(
            dict(
                L.PRM,
                seed=s,
                **CLOCK_TUNE_PARAMS,
                num_threads=a.nt,
                **({"objective": "huber", "alpha": a.alpha} if a.objective == "huber" else {}),
                **clock_tune_jparams,
            ),
            lgb.Dataset(Xtr, rtr, weight=wtr, categorical_feature=[cols.index("apt")]),
            CLOCK_TUNE_ROUNDS,
        )
        G[s, te] = m.predict(Xf[te], num_threads=a.nt)
    log(f"{t}: train {tr.sum():,} test {te.sum():,} done")
Wh6 = np.where(ok & lk, L.cf(b + np.clip(np.nan_to_num(G.mean(0)), -300, 300), lo), b)
apt = Fr["ADEP_mvt"].cast(str).to_numpy().astype(str)
A0 = np.where(apt == "LIRF", b, Wh6)
for t in L.FOLDS:
    m = fo == t
    np.save(f"{a.out}_{'Hu6' if a.objective == 'huber' else 'Wh6'}_{t}.npy", np.vstack([ids[m], Wh6[m]]))
    np.save(f"{a.out}_{'HA0' if a.objective == 'huber' else 'A0'}_{t}.npy", np.vstack([ids[m], A0[m]]))
    mm = m & ok
    print(
        f"  {t}: Wh6 vs base {np.mean((Wh6[mm]-y[mm])**2-(b[mm]-y[mm])**2):+.1f}  A0 vs base {np.mean((A0[mm]-y[mm])**2-(b[mm]-y[mm])**2):+.1f}",
        flush=True,
    )
if (
    a.npz
):  # leave_apply/leave_eval-compatible npz (stage-1 per-seed LOFO predictions under G_U/G_Wh/G_Hh = this run's G)
    np.savez_compressed(a.npz, ids=ids, fo=fo, ok=ok, lk=lk, b=b, y=y, lo=lo, G_U=G, G_Wh=G, G_Hh=G)
log("done")
