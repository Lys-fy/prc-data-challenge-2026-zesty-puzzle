#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""KRS fit with beta per (group x spread tercile). MODES: KRSL (group = adsb link), RND2 (group = random 2-way, placebo), KRSAR (terciles within airport, global beta), KRS (spread terciles, below). Env MODES=KRSL,RND2,KRSAR.
Arm KRS: end-of-chain re-step toward K250 with beta per K250-spread tercile, on a fold line (6 folds, [id, pred]).
Fit: UNS rows (sealed never read), month-aware weights wm (as scripts/sealed_score.py), fit folds = other 3 gate folds (fresh folds: the 4 gate folds).
Applies to every matched row of the fold vector that has K250 (sealed rows get the step too; their labels are not read). cf (LOBT clip + floor 60).
Writes <out>_<MODE>_<fold>.npy for MODE in KRS (spread terciles, decision), RND (random terciles, placebo guard), CST (one beta, info), GSD (g_sd terciles, info).
Also prints beta_k per fold and the 2026 ranking-row tercile shares under the 2025 edges (info), and the production betas/edges (fit on the 4 gate folds).
usage: krs_fit2.py <base fold prefix> <out prefix>"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
BASE, OUT = sys.argv[1], sys.argv[2]
src = open(f"{ROOT}/scripts/sealed_score.py").read().split('print("## pairs:')[0]
sys.argv = [f"{ROOT}/scripts/sealed_score.py"]
H = {"__file__": f"{ROOT}/scripts/sealed_score.py", "__name__": "xchain2krs"}
exec(src, H)
load, ev = H["load"], H["ev"]
import clockresid_lib as L

GATE = ["months39", "months28", "months410", "janjul"]
FRESH = ["months511", "months612"]
ALL = GATE + FRESH
rng = np.random.default_rng(11)
D = {}
for t in ALL:
    ids, p = load("npy:" + BASE, t)
    z = np.load(f"data/oof/KSBWPc100_{t}.npy")
    o = np.argsort(z[0])
    z = z[:, o]
    pos = np.clip(np.searchsorted(z[0], ids), 0, z.shape[1] - 1)
    has = z[0][pos] == ids
    Fr = L.frame(ids)
    lkv = Fr["lk"].to_numpy().astype(int)
    apv = Fr["ADEP_mvt"].cast(pl.Utf8).fill_null("?").to_numpy()
    um = Fr["unmatched"].fill_null(1).to_numpy() == 1
    lo = Fr["mvt_lobt"].to_numpy().astype(float)
    E = ev(t, ids)
    ok = E["ok"]
    U = ~E["sealed"]
    sel = np.flatnonzero(ok)[U]
    app = has & ~um
    D[t] = dict(
        ids=ids,
        p=p,
        K=np.where(has, z[1][pos], np.nan),
        sp=np.where(has, z[2][pos], np.nan),
        sd=np.where(has, z[3][pos], np.nan),
        rnd=rng.random(len(ids)),
        lk=lkv,
        r2=(rng.random(len(ids)) < 0.5).astype(int),
        ap=apv,
        z0=np.zeros(len(ids), int),
        lo=lo,
        app=app,
        sel=sel,
        y=E["y"][sel],
        w=E["wm"][U],
    )
    assert app[sel].all(), t
    print(f"[load] {t}: rows {len(ids):,} apply {app.sum():,} UNS fit rows {len(sel):,}", flush=True)


def fitset(t):
    return [f for f in GATE if f != t] if t in GATE else GATE


APTS = sorted(set(np.concatenate([D[t]["ap"] for t in ALL])))


def edges(fs, within):
    if not within:
        v = np.concatenate([D[f]["sp"][D[f]["sel"]] for f in fs])
        return {None: np.quantile(v, [1 / 3, 2 / 3])}
    out = {}
    for A in APTS:
        v = np.concatenate([D[f]["sp"][D[f]["sel"]][D[f]["ap"][D[f]["sel"]] == A] for f in fs])
        out[A] = np.quantile(v, [1 / 3, 2 / 3]) if len(v) > 30 else np.array([np.inf, np.inf])
    return out


def tert(d, idx, E, within):
    if not within:
        return np.digitize(d["sp"][idx], E[None])
    t = np.zeros(len(idx), int)
    ap = d["ap"][idx]
    sp = d["sp"][idx]
    for A, e in E.items():
        m = ap == A
        t[m] = np.digitize(sp[m], e)
    return t


def fit(fs, gkey, within):
    E = edges(fs, within)
    b = {}
    xs, ys, ws, cs = [], [], [], []
    for f in fs:
        d = D[f]
        s = d["sel"]
        cs.append(d[gkey][s] * 3 + tert(d, s, E, within))
        xs.append(d["K"][s] - d["p"][s])
        ys.append(d["y"] - d["p"][s])
        ws.append(d["w"])
    x, yy, w, c = map(np.concatenate, (xs, ys, ws, cs))
    for k in np.unique(c):
        j = c == k
        b[int(k)] = float((w[j] * x[j] * yy[j]).sum() / (w[j] * x[j] * x[j]).sum())
    return E, b


MODES = {"KRS": ("z0", False), "KRSL": ("lk", False), "RND2": ("r2", False), "KRSAR": ("z0", True)}
SEL = os.environ.get("MODES", "KRS").split(",")
for mode in SEL:
    gkey, within = MODES[mode]
    for t in ALL:
        E, b = fit(fitset(t), gkey, within)
        d = D[t]
        a = np.flatnonzero(d["app"])
        c = d[gkey][a] * 3 + tert(d, a, E, within)
        bb = np.array([b.get(int(k), 0.0) for k in c])
        new = d["p"].copy()
        new[a] = L.cf(d["p"][a] + bb * (d["K"][a] - d["p"][a]), d["lo"][a])
        np.save(f"{OUT}_{mode}_{t}.npy", np.vstack([d["ids"], new]))
        print(
            f"[beta] {mode} {t}: beta(cell=group*3+tercile) "
            + " ".join(f"{k}:{v:.3f}" for k, v in sorted(b.items()))
            + f" | delta rms {np.sqrt(((new - d['p']) ** 2).mean()):.2f} s",
            flush=True,
        )
    E, b = fit(GATE, gkey, within)
    json.dump(
        {
            "mode": mode,
            "group": gkey,
            "within_airport": within,
            "edges": {str(k): v.tolist() for k, v in E.items()},
            "beta": {str(k): v for k, v in b.items()},
            "base": BASE,
        },
        open(f"{OUT}_{mode}_prod_coef.json", "w"),
    )
    print(f"[prod] {mode}: beta " + " ".join(f"{k}:{v:.4f}" for k, v in sorted(b.items())), flush=True)
