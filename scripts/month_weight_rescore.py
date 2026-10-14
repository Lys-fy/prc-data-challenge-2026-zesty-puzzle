#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Re-score stored fold vectors with the src weights AND the month-aware (MW) weights, plus a LIRF Jul-dark column.
Pair spec: LABEL=NEW~REF, each side one of
  npy:PREFIX                 oof_build-style per-fold [ID, pred] arrays (PREFIX_<fold>.npy or PREFIX_<fold>_*.npy); folds without a file are skipped
  chain:RW.npz:HU.npz        lgb_airport_eval final form cf(b + clip(mean G_Hh, +-300)) on linked ok rows, b elsewhere and at LIRF (Hu6 + A0)
  pre:RW.npz                 the chain's pre-stage b
  mix:w1*SPEC1+w2*SPEC2      (npy only) weighted mean
usage: month_weight_rescore.py [--level SPEC] PAIR [PAIR ...]
Columns per fold: raw [w/o top-3] | s = src old | m = src MW | m3 = src MW w/o top-3 |w d| | LJd = LIRF Jul-dark, LWd = LIRF Jan-dark, EJd = LEMD Jul-dark
(board s2 under MW weights; LJo = LIRF Jul-dark under old weights). Rules: OLD (unweighted) and MW."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K

FO = ["months39", "months28", "months410", "janjul"]
args = sys.argv[1:]
level = None
MWD = False
if args and args[0] == "--mwd":
    MWD = True
    args = args[1:]
if args and args[0] == "--level":
    level = args[1]
    args = args[2:]
Rk = K.rank_frame()
rk = K.keys(Rk)
rslot = K.slot(Rk)
rD = K.dbin(Rk["MVT_ID_mvt"].to_numpy()) if MWD else None
NPZ = {}


def npz(p):
    if p not in NPZ:
        NPZ[p] = dict(np.load(p))
    return NPZ[p]


def load(spec, t):
    kind, rest = spec.split(":", 1)
    if kind == "npy":
        g = sorted(glob.glob(f"{rest}_{t}.npy") or glob.glob(f"{rest}_{t}_*.npy"))
        if not g:
            return None
        z = np.load(g[0])
        o = np.argsort(z[0])
        return z[0][o], z[1][o]
    if kind == "mix":
        parts = []
        for term in rest.split("+"):
            w, s = term.split("*", 1)
            r = load(s, t)
            if r is None:
                return None
            parts.append((float(w), r))
        ids = parts[0][1][0]
        for _, (i, _) in parts[1:]:
            assert np.array_equal(i, ids)
        return ids, sum(w * v for w, (_, v) in parts)
    if kind in ("chain", "pre"):
        rw = rest.split(":")[0]
        Z = npz(rw)
        m = Z["fo"].astype(str) == t
        if kind == "pre":
            v = Z["b"]
        else:
            H = npz(rest.split(":")[1])["G_Hh"]
            ok = Z["ok"].astype(bool)
            lk = Z["lk"].astype(bool)
            lir = LIR.get(rw)
            if lir is None:
                lir = L.frame(Z["ids"])["ADEP_mvt"].cast(str).to_numpy().astype(str) == "LIRF"
                LIR[rw] = lir
            v = np.where(ok & lk & ~lir, L.cf(Z["b"] + np.clip(np.nan_to_num(H.mean(0)), -300, 300), Z["lo"]), Z["b"])
        ids = Z["ids"][m]
        o = np.argsort(ids)
        return ids[o], v[m][o]
    raise ValueError(spec)


LIR = {}
EV = {}


def ev(t, ids):
    key = (t, len(ids), float(ids[::997].sum()))
    if key not in EV:
        Fr = L.frame(ids)
        y = Fr["y"].to_numpy().astype(float)
        ok = np.isfinite(y) & (Fr["unmatched"].to_numpy() == 0) & (Fr["is_rank"].to_numpy() == 0)
        fk = K.keys(Fr)
        sub = tuple(x[ok] for x in fk)
        fs = K.slot(Fr)[ok]
        ws = K.cell_weights(sub, rk)[0]
        lv = {}
        wm, cov, unc = K.cell_weights_mw(sub, rk, fs, rslot, levels_out=lv)
        apt = sub[0]
        dark = sub[1] != "lnk"
        cells = {
            "LJd": (apt == "LIRF") & dark & (fs == "S"),
            "LWd": (apt == "LIRF") & dark & (fs == "W"),
            "EJd": (apt == "LEMD") & dark & (fs == "S"),
        }
        lo = Fr["mvt_lobt"].to_numpy().astype(float)
        EV[key] = dict(y=y, ok=ok, ws=ws, wm=wm, cells=cells, lv=lv, cov=cov, unc=unc, lo=lo, apt=apt)
        if MWD:
            fD = K.dbin(np.asarray(ids)[ok])
            inf = {}
            wd_, _ = K.cell_weights_mwd(sub, rk, fs, rslot, fD, rD, info=inf)
            hi = fD >= 3
            EV[key].update(wd=wd_, hi=hi, hi20=fD >= 5)
            print(
                f"  [MWD {t}] sum w MW {wm.sum():,.1f} MWD {wd_.sum():,.1f} (diff {wd_.sum() - wm.sum():+.2f}) | groups kept at MW {inf['groups_kept_MW']} | D>=0.05 fold rows {hi.sum():,} "
                f"sum w old {ws[hi].sum():,.0f} MW {wm[hi].sum():,.0f} MWD {wd_[hi].sum():,.0f} | D>=0.20 MW {wm[fD >= 5].sum():,.0f} MWD {wd_[fD >= 5].sum():,.0f}",
                flush=True,
            )
        cs = cells["LJd"]
        print(
            f"  [weights {t}] rows {ok.sum():,} | MW coverage by level "
            + ", ".join(f"{k.split()[0]} {v:,}" for k, v in lv.items())
            + f" | uncovered {unc}"
            f" | LIRF Jul-dark fold rows {cs.sum()} sum w old {ws[cs].sum():,.0f} MW {wm[cs].sum():,.0f} (max w MW {wm[cs].max() if cs.any() else 0:.1f})"
            f" | LIRF Jan-dark sum w old {ws[cells['LWd']].sum():,.0f} MW {wm[cells['LWd']].sum():,.0f}",
            flush=True,
        )
    return EV[key]


def rules(R, folds):
    j = folds.index("janjul") if "janjul" in folds else None
    n = len(folds)
    need = 3 if n == 4 else n
    old = (R[:, 2] < 0).sum() >= need and (j is None or R[j, 2] <= 0) and (R[:, 1] < 0).sum() >= need
    mw = (
        (R[:, 3] < 0).sum() >= need
        and (j is None or R[j, 3] <= 0)
        and (R[:, 1] < 0).sum() >= need
        and (R[:, 4] < 0).sum() >= need
    )
    return old, mw


if level:
    print(f"## level check of {level}: board MSE (sum w e^2 / RN) under old vs MW weights; LIRF dark cell contribution")
    for t in FO:
        r = load(level, t)
        if r is None:
            continue
        ids, v = r
        E = ev(t, ids)
        ok = E["ok"]
        e = (v[ok] - E["y"][ok]) ** 2
        ld = (E["apt"] == "LIRF") & (E["cells"]["LJd"] | E["cells"]["LWd"])
        print(
            f"  {t:9s} MSE old {(E['ws'] * e).sum() / K.RN:8.0f} MW {(E['wm'] * e).sum() / K.RN:8.0f} | LIRF dark cell old {(E['ws'] * e)[ld].sum() / K.RN:6.0f} MW {(E['wm'] * e)[ld].sum() / K.RN:6.0f}"
            f" | LIRF Jul-dark old {(E['ws'] * e)[E['cells']['LJd']].sum() / K.RN:6.0f} MW {(E['wm'] * e)[E['cells']['LJd']].sum() / K.RN:6.0f} (MSE/row {e[E['cells']['LJd']].mean() if E['cells']['LJd'].any() else np.nan:,.0f})"
            f" | LIRF Jan-dark MSE/row {e[E['cells']['LWd']].mean() if E['cells']['LWd'].any() else np.nan:,.0f}",
            flush=True,
        )
        if MWD:
            h = E["hi"]
            tm = (E["wm"] * e).sum()
            td = (E["wd"] * e).sum()
            print(
                f"  {t:9s} MWD MSE {td / K.RN:8.0f} | D>=0.05 rows share of board s2: MW {100 * (E['wm'] * e)[h].sum() / tm:.1f} % -> MWD {100 * (E['wd'] * e)[h].sum() / td:.1f} % (old {100 * (E['ws'] * e)[h].sum() / (E['ws'] * e).sum():.1f} %)",
                flush=True,
            )
print(
    "## pairs (raw [w/o top-3] s=src old m=src MW m3=src MW w/o top-3 |w d| | LJd/LWd/EJd = LIRF Jul-dark / LIRF Jan-dark / LEMD Jul-dark board s2 MW; LJo = LIRF Jul-dark old)"
)
for p in args:
    lab, rest = p.split("=", 1)
    ns, rs = rest.split("~")
    R = []
    folds = []
    RD = []
    for t in FO:
        a, b = load(ns, t), load(rs, t)
        if a is None or b is None:
            continue
        (i1, v1), (i0, v0) = a, b
        common = np.intersect1d(i1, i0)
        v1 = v1[np.searchsorted(i1, common)]
        v0 = v0[np.searchsorted(i0, common)]
        E = ev(t, common)
        ok = E["ok"]
        y = E["y"][ok]
        lo = E["lo"][ok]
        assert np.isfinite(v1[ok]).all() and np.isfinite(v0[ok]).all(), (lab, t)
        d = (v1[ok] - y) ** 2 - (v0[ok] - y) ** 2  # no cf
        k = np.ones(len(d), bool)
        k[np.argsort(-np.abs(d))[:3]] = False
        wd = E["wm"] * d
        k2 = np.ones(len(d), bool)
        k2[np.argsort(-np.abs(wd))[:3]] = False
        c = E["cells"]
        if MWD:
            wdd = E["wd"] * d
            k3 = np.ones(len(d), bool)
            k3[np.argsort(-np.abs(wdd))[:3]] = False
            RD.append(
                (wdd.sum() / K.RN, wdd[k3].sum() / K.RN, wdd[E["hi"]].sum() / K.RN, (E["wm"] * d)[E["hi"]].sum() / K.RN)
            )
        R.append(
            (
                d.mean(),
                d[k].mean(),
                (E["ws"] * d).sum() / K.RN,
                wd.sum() / K.RN,
                wd[k2].sum() / K.RN,
                wd[c["LJd"]].sum() / K.RN,
                wd[c["LWd"]].sum() / K.RN,
                wd[c["EJd"]].sum() / K.RN,
                (E["ws"] * d)[c["LJd"]].sum() / K.RN,
            )
        )
        folds.append(t)
    if not R:
        print(f"[{lab}] no folds")
        continue
    R = np.array(R)
    o, m = rules(R, folds)
    print(
        f"[{lab}] "
        + " | ".join(
            f"{t} {x[0]:+7.1f} [{x[1]:+7.1f}] s{x[2]:+7.1f} m{x[3]:+7.1f} m3{x[4]:+7.1f} LJd{x[5]:+6.1f} LWd{x[6]:+6.1f} EJd{x[7]:+5.1f} LJo{x[8]:+5.1f}"
            for t, x in zip(folds, R)
        ),
        flush=True,
    )
    print(
        f"    mean raw {R[:,0].mean():+.1f} s {R[:,2].mean():+.1f} m {R[:,3].mean():+.1f} LJd {R[:,5].mean():+.1f} | RULE old {'PASS' if o else 'FAIL'} | RULE MW {'PASS' if m else 'FAIL'}"
        f"{'  <-- FLIP' if o != m else ''}",
        flush=True,
    )
    if MWD:
        D_ = np.array(RD)
        j = folds.index("janjul") if "janjul" in folds else None
        need = 3 if len(folds) == 4 else len(folds)
        md = (
            (D_[:, 0] < 0).sum() >= need
            and (j is None or D_[j, 0] <= 0)
            and (R[:, 1] < 0).sum() >= need
            and (D_[:, 1] < 0).sum() >= need
        )
        print(
            f"    MWD: "
            + " | ".join(
                f"{t} mD{x[0]:+7.1f} mD3{x[1]:+7.1f} Dhi{x[2]:+6.1f} (MW {x[3]:+6.1f})" for t, x in zip(folds, D_)
            )
            + f" | mean mD {D_[:,0].mean():+.1f} | RULE MWD {'PASS' if md else 'FAIL'}{'  <-- FLIP vs MW' if md != m else ''}",
            flush=True,
        )
