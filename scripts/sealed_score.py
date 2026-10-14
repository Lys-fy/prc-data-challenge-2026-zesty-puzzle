#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Sealed-split scorer: scores fold vectors separately on the sealed and the unsealed airport-days of data/sealed_days.parquet.
Same pair specs as month_weight_rescore.py (LABEL=NEW~REF; npy:PREFIX | mix:w*SPEC+... | chain:RW.npz:HU.npz | pre:RW.npz).
Per fold it prints three row sets:
  ALL  = every scored row -- must reproduce month_weight_rescore's raw [w/o top-3] / m (src_MW) / m3 exactly (self-check column ok/!!)
  UNS  = rows on the unsealed 80 % of airport-days (the search part experiments are gated on)
  SEAL = rows on the sealed 20 % of airport-days (data/sealed_days.parquet, md5 in)
Weights: the month-aware weights are computed ONCE on all fold rows (clock_lib.cell_weights_mw, exactly as month_weight_rescore), then summed over the
subset and rescaled by sum(w_all)/sum(w_subset), so UNS and SEAL are on the board-s2 scale of the full fold (sign is unaffected by the rescale).
Top-3 exclusions (raw w/o top-3, m3) are taken inside each subset.
Rules printed: MW(all) = standard rule; MW(uns) = the same rule on the unsealed rows; SEAL = sealed src_MW 4-fold mean < 0
(the proposed keep rule) with the per-fold count as info.
--boot N (default 0): N random stratified 20 % airport-day subsets (label-free, seed 7) -> P(sub-mean src_MW < 0), the noise reference for SEAL.
usage: sealed_score.py [--boot N] PAIR [PAIR ...]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, hashlib, numpy as np, polars as pl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, clock_lib as K
FO = ["months39", "months28", "months410", "janjul"]; SEALED = "data/sealed_days.parquet"
args = sys.argv[1:]; BOOT = 0
if args and args[0] == "--boot": BOOT = int(args[1]); args = args[2:]
SD = pl.read_parquet(SEALED); print(f"# sealed days {SEALED} md5 {hashlib.md5(open(SEALED, 'rb').read()).hexdigest()} "
                                   f"({int(SD['sealed'].sum())} of {SD.height} airport-days)", flush=True)
Rk = K.rank_frame(); rk = K.keys(Rk); rslot = K.slot(Rk)
NPZ = {}; LIR = {}; EV = {}
def npz(p):
    if p not in NPZ: NPZ[p] = dict(np.load(p))
    return NPZ[p]
def load(spec, t):   # identical to month_weight_rescore.load
    kind, rest = spec.split(":", 1)
    if kind == "npy":
        g = sorted(glob.glob(f"{rest}_{t}.npy") or glob.glob(f"{rest}_{t}_*.npy"))
        if not g: return None
        z = np.load(g[0]); o = np.argsort(z[0]); return z[0][o], z[1][o]
    if kind == "mix":
        parts = []
        for term in rest.split("+"):
            w, s = term.split("*", 1); r = load(s, t)
            if r is None: return None
            parts.append((float(w), r))
        ids = parts[0][1][0]
        for _, (i, _) in parts[1:]: assert np.array_equal(i, ids)
        return ids, sum(w * v for w, (_, v) in parts)
    if kind in ("chain", "pre"):
        rw = rest.split(":")[0]; Z = npz(rw); m = Z["fo"].astype(str) == t
        if kind == "pre": v = Z["b"]
        else:
            H = npz(rest.split(":")[1])["G_Hh"]; ok = Z["ok"].astype(bool); lk = Z["lk"].astype(bool)
            lir = LIR.get(rw)
            if lir is None:
                lir = L.frame(Z["ids"])["ADEP_mvt"].cast(str).to_numpy().astype(str) == "LIRF"; LIR[rw] = lir
            v = np.where(ok & lk & ~lir, L.cf(Z["b"] + np.clip(np.nan_to_num(H.mean(0)), -300, 300), Z["lo"]), Z["b"])
        ids = Z["ids"][m]; o = np.argsort(ids); return ids[o], v[m][o]
    raise ValueError(spec)
def ev(t, ids):
    key = (t, len(ids), float(ids[::997].sum()))
    if key not in EV:
        Fr = L.frame(ids); y = Fr["y"].to_numpy().astype(float); ok = np.isfinite(y) & (Fr["unmatched"].to_numpy() == 0) & (Fr["is_rank"].to_numpy() == 0)
        fk = K.keys(Fr); sub = tuple(x[ok] for x in fk); fs = K.slot(Fr)[ok]
        wm = K.cell_weights_mw(sub, rk, fs, rslot)[0]
        Dk = Fr.select(pl.col("ADEP_mvt").cast(pl.Utf8), day=pl.col("MVT_TIME_UTC_mvt").dt.date()).with_row_index("i")
        Dk = Dk.join(SD.select("ADEP_mvt", "day", "sealed"), on=["ADEP_mvt", "day"], how="left").sort("i")
        miss = int(Dk["sealed"].is_null().filter(pl.Series(ok)).sum()); assert miss == 0, f"{t}: {miss} scored rows without a sealed-day entry"
        sealed = Dk["sealed"].to_numpy()[ok].astype(bool)
        dk = (Dk["ADEP_mvt"] + "|" + Dk["day"].cast(pl.Utf8) + "|").to_numpy()[ok]
        strat = np.array([s[:4] + s[9:12] for s in dk])   # apt|MM- (airport x month)
        EV[key] = dict(y=y, ok=ok, wm=wm, sealed=sealed, day=np.unique(dk, return_inverse=True)[1], strat=strat)
        print(f"  [{t}] scored rows {ok.sum():,} sealed {sealed.sum():,} ({100 * sealed.mean():.1f} %), sum w MW all {wm.sum():,.0f} sealed {wm[sealed].sum():,.0f}", flush=True)
    return EV[key]
def cols(d, wd, S):
    """raw, raw w/o top-3, src_MW, src_MW w/o top-3 on subset S (unscaled; caller rescales m, m3)."""
    d = d[S]; wd = wd[S]
    k = np.ones(len(d), bool); k[np.argsort(-np.abs(d))[:3]] = False
    k2 = np.ones(len(d), bool); k2[np.argsort(-np.abs(wd))[:3]] = False
    return d.mean(), d[k].mean(), wd.sum() / K.RN, wd[k2].sum() / K.RN
def rule(R, folds):   # month_weight_rescore MW rule on columns (raw, raw3, m, m3)
    j = folds.index("janjul") if "janjul" in folds else None; need = 3 if len(folds) == 4 else len(folds)
    return (R[:, 2] < 0).sum() >= need and (j is None or R[j, 2] <= 0) and (R[:, 1] < 0).sum() >= need and (R[:, 3] < 0).sum() >= need
print("## pairs: per fold ALL raw [w/o top-3] m m3 | UNS m [m3] | SEAL m [m3] (m = src_MW board s2; UNS/SEAL rescaled to the full-fold weight sum)")
SUMMARY = []
for p in args:
    lab, rest = p.split("=", 1); ns, rs = rest.split("~"); RA, RU, RS, folds, BT = [], [], [], [], []
    for t in FO:
        a, b = load(ns, t), load(rs, t)
        if a is None or b is None: continue
        (i1, v1), (i0, v0) = a, b; common = np.intersect1d(i1, i0); v1 = v1[np.searchsorted(i1, common)]; v0 = v0[np.searchsorted(i0, common)]
        E = ev(t, common); ok = E["ok"]; y = E["y"][ok]
        assert np.isfinite(v1[ok]).all() and np.isfinite(v0[ok]).all(), (lab, t)
        d = (v1[ok] - y) ** 2 - (v0[ok] - y) ** 2; wm = E["wm"]; wd = wm * d; S = E["sealed"]
        out = []
        for M in (np.ones(len(d), bool), ~S, S):
            r = np.array(cols(d, wd, M)); sc = wm.sum() / wm[M].sum(); r[2:] *= sc; out.append(r)
        RA.append(out[0]); RU.append(out[1]); RS.append(out[2]); folds.append(t)
        if BOOT:
            g = E["day"]; nd = g.max() + 1; dsum = np.bincount(g, wd, nd); wsum = np.bincount(g, wm, nd)
            _, first = np.unique(g, return_index=True); dstrat = E["strat"][first]; rng = np.random.default_rng(7); vals = np.empty(BOOT)
            su, sinv = np.unique(dstrat, return_inverse=True)
            for bI in range(BOOT):
                pick = np.zeros(nd, bool)
                for s in range(len(su)):
                    idx = np.where(sinv == s)[0]; pick[rng.permutation(idx)[:int(round(0.2 * len(idx)))]] = True
                vals[bI] = dsum[pick].sum() / K.RN * wm.sum() / wsum[pick].sum()
            BT.append(vals)
    if not RA: print(f"[{lab}] no folds"); continue
    RA, RU, RS = map(np.array, (RA, RU, RS))
    print(f"[{lab}] " + " | ".join(f"{t} ALL {a[0]:+7.1f} [{a[1]:+7.1f}] m{a[2]:+7.1f} m3{a[3]:+7.1f} | UNS m{u[2]:+7.1f} [{u[3]:+7.1f}] | SEAL m{s[2]:+7.1f} [{s[3]:+7.1f}]"
                         for t, a, u, s in zip(folds, RA, RU, RS)), flush=True)
    ra, ru = rule(RA, folds), rule(RU, folds); smean = RS[:, 2].mean(); sneg = int((RS[:, 2] < 0).sum())
    line = (f"    mean m ALL {RA[:, 2].mean():+.1f} UNS {RU[:, 2].mean():+.1f} SEAL {smean:+.1f} | RULE MW(all) {'PASS' if ra else 'FAIL'} | RULE MW(uns) {'PASS' if ru else 'FAIL'}"
            f" | SEAL mean {'<0 PASS' if smean < 0 else '>=0 FAIL'} ({sneg}/{len(folds)} folds < 0)")
    if BOOT:
        bm = np.mean(BT, axis=0); line += f" | boot{BOOT} P(sub-mean<0) {100 * (bm < 0).mean():.0f} % (sub-mean sd {bm.std():.1f})"
    print(line, flush=True)
    SUMMARY.append((lab, RA[:, 2].mean(), ra, RU[:, 2].mean(), ru, smean, sneg, len(folds)))
print("## summary: label | m ALL | MW(all) | m UNS | MW(uns) | m SEAL | sealed folds < 0")
for lab, a, ra, u, ru, s, n, nf in SUMMARY:
    print(f"  {lab:14s} {a:+8.1f} {'PASS' if ra else 'FAIL'} | {u:+8.1f} {'PASS' if ru else 'FAIL'} | {s:+8.1f} {n}/{nf}", flush=True)
