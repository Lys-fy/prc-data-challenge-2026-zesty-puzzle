#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""leave_stage ): on top of clock_stage's Wh6 clock stage (clock_rw npz, base unm_resid3_blkbtd), evaluate
(A) zeroing the stage at LIRF (A0 apply-only; A1 LIRF also out of stage-1 training, info) and
(B) second stages on linked rows: L (leave-only), M (mv_gs-only), X (all linked + source feature); strict nesting (inner LOFO
    stage-1 Wh on the three training folds) and plain LOFO (training base = outer Wh6 OOF, info).
Metrics: raw, w/o top-3, comp, src per ranking row. usage: leave_fit_tuned.py <clock npz> --tag T [--ns 3] [--nt 8]
"""

import os, sys, argparse, time, numpy as np, lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--tag", required=True)
ap.add_argument("--ns", type=int, default=3)
ap.add_argument("--nt", type=int, default=8)
ap.add_argument(
    "--hnpz", default=None
)  # huber_clock: stage 1 = Huber-600 Wh6 (G_Hh of this npz), inner stage-1 Huber too
a = ap.parse_args()
HP = dict(objective="huber", alpha=600.0) if a.hnpz else {}
# clock_tune: stage-1 (inner, strict nesting) capacity from env clock_tune_cleaves/clock_tune_cminleaf/clock_tune_crounds, identical to clock_fit_tuned; stage 2 (L) unchanged
HP = dict(
    HP,
    **{
        k: tp(os.environ[e])
        for k, e, tp in (("num_leaves", "clock_tune_cleaves", int), ("min_data_in_leaf", "clock_tune_cminleaf", int))
        if e in os.environ
    },
)
HPR = int(os.environ.get("clock_tune_crounds", "400"))
print("clock_tune inner stage-1", HP, HPR)
T0 = time.time()
os.makedirs("scratchpad/xlv", exist_ok=True)


def log(*x):
    print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)


Z = np.load(a.npz)
ids, fo, ok, lk, b, y, lo = (Z[k] for k in ("ids", "fo", "ok", "lk", "b", "y", "lo"))
GU, GW = Z["G_U"], (np.load(a.hnpz)["G_Hh"] if a.hnpz else Z["G_Wh"])
Fr = L.frame(ids)
assert np.allclose(Fr["y"].to_numpy().astype(float)[ok], y[ok])
apt = Fr["ADEP_mvt"].cast(str).to_numpy().astype(str)
lir = apt == "LIRF"
clk = L.clock_cols(Fr)
X0, cols = L.design(Fr, b, "clkonly", clk)
ib = cols.index("base")
idd = [i for i, c in enumerate(cols) if c.startswith("dd_")]
sn = K.src(Fr)
fk = K.keys(Fr)
Rk = K.rank_frame()
rk = K.keys(Rk)
redit = np.char.startswith(rk[2].astype(str), "M_") & (rk[1] == "lnk") & Rk["LOBT_flt"].is_not_null().to_numpy()
rk_ed = tuple(x[redit] for x in rk)
LK = ok & lk
log(
    f"rows {len(y):,} ok {ok.sum():,} linked {LK.sum():,}; src on linked: "
    + " ".join(f"{n} {(sn[LK]==k).mean():.3f}" for k, n in enumerate(K.SRC))
)


def Xb(base, rows, extra=False):
    X = X0[rows].copy()
    d = (base[rows] - b[rows]).astype(np.float32)
    X[:, ib] = base[rows]
    X[:, idd] += d[:, None]
    return np.hstack([X, sn[rows, None].astype(np.float32)]) if extra else X


def whw(tr):
    w, _, _ = K.cell_weights(tuple(x[tr] for x in fk), rk_ed, matched_only=False)
    w = w / w.mean()
    w = np.clip(w, 0.2, 5.0)
    w /= w.mean()
    return np.sqrt(w) / np.sqrt(w).mean()


def fit(tr, base, w=None, extra=False, seeds=None, xp={}):
    r = np.clip(y[tr] - base[tr], -1800, 1800)
    ds = lgb.Dataset(Xb(base, tr, extra), r, weight=w, categorical_feature=[cols.index("apt")], free_raw_data=False)
    return [
        lgb.train(dict(L.PRM, seed=s, num_threads=a.nt, **xp), ds, HPR if xp is HP else 400)
        for s in (seeds if seeds is not None else range(a.ns))
    ]


def pred(ms, base, te, extra=False):
    X = Xb(base, te, extra)
    return np.mean([m.predict(X, num_threads=a.nt) for m in ms], 0)


def corr1(g, rows, base=None):
    base = b if base is None else base
    out = base.copy()
    out[rows] = L.cf(base[rows] + np.clip(g, -300, 300), lo[rows])
    return out


Wh6 = np.where(LK, L.cf(b + np.clip(np.nan_to_num(GW.mean(0)), -300, 300), lo), b)
B6 = np.where(LK, L.cf(b + np.clip(np.nan_to_num(GU.mean(0)), -300, 300), lo), b)
Wh3 = np.where(LK, L.cf(b + np.clip(np.nan_to_num(GW[:3].mean(0)), -300, 300), lo), b)
# ---- evaluation weights (as clock_rw)
evw = {}
for t in L.FOLDS:
    m = (fo == t) & ok
    sub = tuple(x[m] for x in fk)
    comp = (sub[0], sub[1], sub[2], np.where(sub[1] == "lnk", "x", "-"))
    rcomp = (rk[0], rk[1], rk[2], np.where(rk[1] == "lnk", "x", "-"))
    evw[t] = (m, K.cell_weights(comp, rcomp)[0], K.cell_weights(sub, rk)[0])


def score(new, ref):
    out = []
    for t in L.FOLDS:
        m, wc, ws = evw[t]
        d = (new[m] - y[m]) ** 2 - (ref[m] - y[m]) ** 2
        k = np.ones(len(d), bool)
        k[np.argsort(-np.abs(d))[:3]] = False
        out.append((d.mean(), d[k].mean(), (wc * d).sum() / K.RN, (ws * d).sum() / K.RN))
    return np.array(out)


def fmt(R):
    return (
        " | ".join(f"{t} {x[0]:+7.1f} [{x[1]:+7.1f}] c{x[2]:+7.1f} s{x[3]:+7.1f}" for t, x in zip(L.FOLDS, R))
        + f" | mean raw {R[:,0].mean():+6.1f} c {R[:,2].mean():+6.1f} s {R[:,3].mean():+6.1f}"
    )


def byapt(new, ref):
    s = []
    for t in L.FOLDS:
        m, wc, ws = evw[t]
        d = ws * ((new[m] - y[m]) ** 2 - (ref[m] - y[m]) ** 2)
        s.append({ap_: d[apt[m] == ap_].sum() / K.RN for ap_ in L.APTS})
    return "  ".join(f"{ap_} " + "/".join(f"{x[ap_]:+.0f}" for x in s) for ap_ in L.APTS)


def gate(R):
    neg_r = R[:, 0] < 0
    neg_s = R[:, 3] < 0
    same = all(np.sign(R[i, 1]) == np.sign(R[i, 0]) for i in range(4) if neg_r[i])
    return dict(
        raw=int(neg_r.sum()),
        src=int(neg_s.sum()),
        wo3=same,
        jj_raw=R[3, 0] <= 0,
        jj_src=R[3, 3] <= 0,
        mean_s=R[:, 3].mean(),
    )


res = {}


def rep(name, new, ref, refname):
    R = score(new, ref)
    res[name] = R
    print(f"  {name:10s} vs {refname:5s} {fmt(R)}  gate {gate(R)}", flush=True)
    return R


print("\n## reference: stage vs base")
rep("B6", B6, b, "base")
rep("Wh6", Wh6, b, "base")
rep("Wh3", Wh3, b, "base")
print("\n## (A) LIRF exclusion (increments)")
A0 = np.where(lir, b, Wh6)
rep("A0", A0, Wh6, "Wh6")
print("    by airport (src):", byapt(A0, Wh6))
rep("A0_B6", np.where(lir, b, B6), B6, "B6")
rep("A0_Wh3", np.where(lir, b, Wh3), Wh3, "Wh3")
if not os.environ.get("leave_stage_noa1"):
    g = np.full(len(y), np.nan)
    for t in L.FOLDS:
        tr = LK & (fo != t) & ~lir
        te = LK & (fo == t)
        g[te] = pred(fit(tr, b, whw(tr), xp=HP), b, te)
    A1 = np.where(LK & ~lir, L.cf(b + np.clip(np.nan_to_num(g), -300, 300), lo), b)
    rep("A1", A1, np.where(lir, b, Wh3), "A0_3")
log("A done")
# ---- (B) stage 2
ARMS = {"L": LK & (sn == 0), "M": LK & (sn == 1), "X": LK}
ARMS = {
    k: v for k, v in ARMS.items() if k in os.environ.get("leave_stage_arms", "LMX")
}  # leave_arms_view: leave_stage_arms=L limits arms
OUT = {f"{k}_{n}": Wh6.copy() for k in ARMS for n in ("strict", "plain")}
for t in L.FOLDS:
    oth = [s for s in L.FOLDS if s != t]
    te = LK & (fo == t)
    inner = b.copy()
    for s in oth:
        tri = LK & np.isin(fo, [u for u in oth if u != s])
        tei = LK & (fo == s)
        inner[tei] = L.cf(b[tei] + np.clip(pred(fit(tri, b, whw(tri), xp=HP), b, tei), -300, 300), lo[tei])
    log(f"{t}: inner stage-1 done")
    for k, am in ARMS.items():
        for n, trb in (("strict", inner), ("plain", Wh6)):
            tr = am & (fo != t)
            tt = am & (fo == t)
            ex = k == "X"
            gg = pred(fit(tr, trb, extra=ex), Wh6, tt, extra=ex)
            OUT[f"{k}_{n}"][tt] = L.cf(Wh6[tt] + np.clip(gg, -300, 300), lo[tt])
        log(f"  {t} arm {k} done")
print("\n## (B) second stage increments vs Wh6")
for k, v in OUT.items():
    rep(k, v, Wh6, "Wh6")
for k in ARMS:
    print(f"    {k}_strict by airport (src):", byapt(OUT[f"{k}_strict"], Wh6))
print("\n## combined A0 + stage 2 (stage 2 not at LIRF), strict, vs Wh6")
for k in ARMS:
    rep(f"A0+{k}", np.where(lir, b, OUT[f"{k}_strict"]), Wh6, "Wh6")
    rep(f"A0+{k}vA0", np.where(lir, b, OUT[f"{k}_strict"]), A0, "A0")
    print("    by airport (src):", byapt(np.where(lir, b, OUT[f"{k}_strict"]), A0))
for k, v in list(OUT.items()) + [("A0", A0), ("Wh6", Wh6)]:
    for t in L.FOLDS:
        np.save(f"scratchpad/xlv/leave_stage_{a.tag}_{k}_{t}.npy", np.vstack([ids[fo == t], v[fo == t]]))
log("done")
