#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Pool weights of the stacking chains: one global simplex fit (w >= 0, sum 1, MSE on matched 2025 rows, never per airport) on the
6-fold vectors of the chains, all 6 folds pooled. The same fit leave-one-fold-out is printed as a stability check.
Chains = saved fold vectors <prefix>_<fold>.npy ([id, pred]) after the Hu6 clock stage (HA0 form), in the order dTK, Kq, NNA3;
extra chains may be given (they get weight 0 when they add nothing). Writes the weights to <out.json>.
usage: pool_weights_fit.py <out.json> <chain prefix> [<chain prefix> ...]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, json, numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L

out, P = sys.argv[1], sys.argv[2:]
F6 = ["months39", "months28", "months410", "janjul", "months612", "months511"]


def ld(pref, t):
    z = np.load(sorted(glob.glob(f"{pref}_{t}.npy") or glob.glob(f"{pref}_{t}_*.npy"))[0])
    o = np.argsort(z[0])
    return z[0][o], z[1][o]


ids, fo, X = [], [], []
for t in F6:
    i0, _ = ld(P[0], t)
    cols = []
    for p in P:
        i1, v1 = ld(p, t)
        k = np.searchsorted(i1, i0)
        assert np.array_equal(i1[np.clip(k, 0, len(i1) - 1)], i0), (p, t)
        cols.append(v1[k])
    ids.append(i0)
    fo.append(np.full(len(i0), t))
    X.append(np.stack(cols, 1))
ids = np.concatenate(ids)
fo = np.concatenate(fo)
X = np.concatenate(X)
Fr = L.frame(ids)
y = Fr["y"].to_numpy().astype(float)
ok = np.isfinite(y) & (Fr["unmatched"].to_numpy() == 0) & (Fr["is_rank"].to_numpy() == 0) & np.isfinite(X).all(1)
k = X.shape[1]
E = np.full(k, 1 / k)
names = [os.path.basename(p) for p in P]


def simplex(msk):
    Xm = X[msk]
    G = Xm.T @ Xm / msk.sum()
    g = Xm.T @ y[msk] / msk.sum()
    r = minimize(
        lambda w: w @ G @ w - 2 * g @ w,
        E.copy(),
        method="SLSQP",
        bounds=[(0, 1)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return r.x


print("chains", names)
for t in F6:
    print(f"leave-one-fold-out {t:10s} " + " ".join(f"{x:.3f}" for x in simplex(ok & (fo != t))))
w = simplex(ok)
print("all 6 folds (production) " + ",".join(f"{x:.4f}" for x in w))
json.dump(
    {"chains": names, "weights": [round(float(x), 6) for x in w], "rows": int(ok.sum())}, open(out, "w"), indent=1
)
