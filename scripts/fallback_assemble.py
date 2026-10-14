#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Assembles any member set with the L2 stacker (v56 recipe), refit on the matching OOF; all inputs are environment
variables (member predictions under data/preds, OOF prefixes OL/OC/ON/QOOF, output name TAG). With TAG set, writes
stages/fallback_<TAG>_matched.parquet; without TAG, checks only and writes nothing.

v56 = cf(0.5*mean(PLG_s0..NLGB-1) + 0.5*mean(PCT_s0..NCAT-1) + clip(g, +-150)) on matched rows, g from a
members-only LightGBM meta (v53 inputs + q + q*gap; seed 0, 300 rounds) fitted on the three pairs' OOF (OL/OC/ON,
QOOF); unmatched rows copied byte-identically from BASE (v56). Rows listed in FALLBACK take BASE's values.
env (defaults reproduce v56):
  PLG=V47L NLGB=6  PCT=CATV17x NCAT=14  PNN=NNRA2 (all files)   production member preds in data/preds
  OL=RA3 OC=RAC2 ON=NNA2 QOOF=fallback_regpl                          OOF prefixes for the meta
  QRANK=fallback_regpl                                                 q on ranking rows (data/oof/<QRANK>_rank.npy)
  VAR=L2 (L2 | L2q | base)   FALLBACK=<parquet with MVT_ID_mvt>   BASE=stages/unmatched_rule.parquet
  TAG=<name> -> writes stages/fallback_<TAG>_matched.parquet (no TAG: check only, nothing written)
  FOLDCHECK=1 -> also score the candidate OOF members + meta nested against the default (RA3/RAC2) set, per fold
  NT threads.
  METAFIT=pairs|all (default pairs): production meta fitted on the three pairs' OOF, or on all four folds (+janjul) (XM)
  CLIP_LO / CLIP_HI (default 150 / 150): meta output clipped to [-CLIP_LO, +CLIP_HI]; FOLDCHECK uses the same flags (nested
  as in fallback_stack.py: METAFIT=all scores each fold with a meta fitted on the other three folds)
  EXTRA=lnk (default empty): append adsb_linked to the meta inputs (adsb_link_flag, ); ranking-row value from
  data/external/adsb_features.parquet. FOLDCHECK then scores candidate (COLS + lnk) against the reference WITHOUT lnk."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)
NT = int(os.environ.get("NT", "4"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
E = lambda k, d: os.environ.get(k, d)
PLG, NLGB, PCT, NCAT, PNN = E("PLG", "V47L"), int(E("NLGB", "6")), E("PCT", "CATV17x"), int(E("NCAT", "14")), E("PNN", "NNRA2")
VAR, TAG, FALLBACK, BASE, QRANK = E("VAR", "L2"), E("TAG", ""), E("FALLBACK", ""), E("BASE", "stages/unmatched_rule.parquet"), E("QRANK", "fallback_regpl")
SRC = open("scripts/fallback_stack.py").read().split("variants = sys.argv")[0]
def stack_env():
    g = {"__file__": os.path.abspath("scripts/fallback_stack.py"), "__name__": "xf"}; exec(compile(SRC, "xf_stack", "exec"), g); return g
g = stack_env(); D, X, prm, PAIRS, ALL, cf, BASEC, EXTRA, npy, al, FT, corr = (g[k] for k in ("D", "X", "prm", "PAIRS", "ALL", "cf", "BASEC", "EXTRA", "npy", "al", "FT", "corr"))
METAFIT, CLIP_LO, CLIP_HI = g["METAFIT"], g["CLIP_LO"], g["CLIP_HI"]; FIT = PAIRS if METAFIT == "pairs" else ALL
extra_inputs = g["extra_inputs"]; COLS0 = BASEC + (EXTRA[VAR] if VAR != "base" else []); COLS = COLS0 + sum((EXTRA[x] for x in extra_inputs), [])
print(f"PNEW {E('PNEW','-')} FBMODE {E('FBMODE','base')} | members: {PLG} x{NLGB} + {PCT} x{NCAT} + {PNN} | meta OOF {E('OL','RA3')}/{E('OC','RAC2')}/{E('ON','NNA2')} q {E('QOOF','fallback_regpl')} | VAR {VAR} | fallback {FALLBACK or '-'} | METAFIT {METAFIT} ({'+'.join(FIT)}) clip [-{CLIP_LO:g},+{CLIP_HI:g}] | EXTRA {','.join(extra_inputs) or '-'}")
meta = lgb.train(prm(0), lgb.Dataset(pd.concat([X(D[t], COLS) for t in FIT]), np.concatenate([D[t]["y"] - D[t]["v41"] for t in FIT])), 300)
if os.environ.get("FOLDCHECK"):
    cand = corr(COLS, 0); FC = []
    for k in ("OL", "OC", "ON", "QOOF", "PNEW"): os.environ.pop(k, None)
    for k in ("OL", "OC", "ON", "QOOF"):
        if os.environ.get("REF" + k): os.environ[k] = os.environ["REF" + k]
    g0 = stack_env(); ref = g0["corr"](COLS0, 0); D0 = g0["D"]
    print(f"FOLDCHECK (seed 0, nested, candidate OOF set + meta vs {os.environ.get('OL','RA3')}/{os.environ.get('OC','RAC2')}/NNA2 + meta, same rows by ID):")
    for t in ALL:
        a = pd.DataFrame({"id": D[t]["ids"] if "ids" in D[t] else np.arange(D[t]["n"])})
        # align by y/lobt row order: both loads come from the same features join, so align on MVT_ID via the saved id arrays
        idc = g["al"];
        ic, ir = D[t]["mid"], D0[t]["mid"]; common, xc, xr = np.intersect1d(ic, ir, return_indices=True)
        y = D[t]["y"][xc]; dd = (cand[t][xc] - y) ** 2 - (ref[t][xr] - y) ** 2; k = np.argsort(-np.abs(dd))[:3]
        FC.append(dd.mean()); print(f"  {t}: n {len(common):,} dMSE {dd.mean():+.1f} (w/o top3 {(dd.sum()-dd[k].sum())/len(dd):+.1f}) fb {dd[D[t]['fb'][xc]].sum()/len(dd):+.1f}")
# ---- production ----
def avg(pref, n, cols):
    acc = None
    for i in range(n):
        d = pl.read_parquet(f"data/preds/{pref}_s{i}.parquet").sort("MVT_ID_mvt"); a = np.vstack([d[c].to_numpy().astype(float) for c in cols])
        acc = a if acc is None else acc + a; ids = d["MVT_ID_mvt"].to_numpy()
    return ids, acc / n
F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched", "mvt_lobt", "ADEP_mvt"]).filter(pl.col("is_rank") == 1).sort("MVT_ID_mvt")
S = pl.read_parquet(BASE).sort("MVT_ID_mvt"); ids = F["MVT_ID_mvt"].to_numpy(); assert np.array_equal(S["MVT_ID_mvt"].to_numpy(), ids); vb = S["TAXITIME_SEC_mvt"].to_numpy()
V53 = pl.read_parquet("stages/unmatched_rule.parquet").sort("MVT_ID_mvt")["TAXITIME_SEC_mvt"].to_numpy()
nn = sorted(f for f in os.listdir("data/preds") if f.startswith(f"{PNN}_s")); iN, N = avg(PNN, len(nn), ["TAXITIME_SEC_mvt"]) if nn else (ids, None)
LS = [avg(x, NLGB, ["TAXITIME_SEC_mvt", "p_raw", "art", "taxi"]) for x in PLG.split(",")]; iL = LS[0][0]
assert all(np.array_equal(z[0], iL) for z in LS); L = np.mean([z[1] for z in LS], axis=0)
iC, C = avg(PCT, NCAT, ["TAXITIME_SEC_mvt", "p_raw", "art", "taxi"])
assert np.array_equal(iL, ids) and np.array_equal(iC, ids) and np.array_equal(iN, ids)
lo = F["mvt_lobt"].to_numpy().astype(float); m = F["unmatched"].to_numpy() == 0; apt = F["ADEP_mvt"].to_numpy().astype(str)
FBM = E("FBMODE", "base"); fbm = np.zeros(len(ids), bool)
if FALLBACK:
    fid = pl.read_parquet(FALLBACK)["MVT_ID_mvt"].to_numpy().astype(float); fbm = np.isin(ids.astype(float), fid)
    if FBM == "first": L[:, fbm] = LS[0][1][:, fbm]   # fallback rows: first member set only
PNEW = E("PNEW", "")
if PNEW:   # L1: production p = logit-mean of <stem>_rank files, applied to the seed-averaged branches of both halves
    zs = [al(npy(pp, "rank"), ids.astype(float))[0] for pp in PNEW.split("+")]; zs = [np.log(np.clip(z, 1e-7, 1 - 1e-7) / (1 - np.clip(z, 1e-7, 1 - 1e-7))) for z in zs]
    pn = 1 / (1 + np.exp(-np.mean(zs, axis=0)))
    L[0] = L[0] + (pn - L[1]) * (L[2] - L[3]); L[1] = pn; C[0] = C[0] + (pn - C[1]) * (C[2] - C[3]); C[1] = pn
Lv, p, art, taxi = L; Cv = C[0]; Nv = p * art + (1 - p) * N[0]; v = cf(0.5 * Lv + 0.5 * Cv, lo)
d = {"L": Lv, "C": Cv, "N": Nv, "taxiL": taxi, "artL": art, "pL": p, "v41": v, "taxiC": 0 * v, "pC": 0 * v, "artC": 0 * v}
if VAR != "base":
    q = al(npy(QRANK, "rank"), ids.astype(float))[0]
    gt = FT.group_by("ADEP_mvt").agg(hi=pl.col("r").filter(pl.col("r") > 120).median(), lo=pl.col("r").filter(pl.col("r") < -120).median())
    gap = {a: h - l for a, h, l in gt.select("ADEP_mvt", "hi", "lo").iter_rows()}; d["q"] = q; d["qgap"] = q * np.array([gap[a] for a in apt])
if "lnk" in extra_inputs:
    A = pl.read_parquet("data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"])
    lk = pl.DataFrame({"MVT_ID_mvt": ids}).join(A, on="MVT_ID_mvt", how="left", maintain_order="left")["adsb_linked"]
    if os.environ.get("MODE") == "new":   # a new period: a matched row without an adsb_features row is unlinked (lnk = 0)
        print("matched ranking rows without an adsb_features row (unlinked):", lk.filter(pl.Series(m)).null_count())
    else: assert lk.filter(pl.Series(m)).null_count() == 0, "matched ranking rows without adsb_linked"
    d["lnk"] = lk.fill_null(0).to_numpy().astype(float); print(f"ranking matched rows linked {d['lnk'][m].mean():.4f}")
gc = np.clip(meta.predict(X(d, COLS)), -CLIP_LO, CLIP_HI)
new = vb.astype(float).copy(); new[m] = np.floor(cf(v + gc, lo)[m])
nfb = int(fbm.sum())
if FALLBACK and FBM == "base": new[fbm] = vb[fbm]
assert (new[~m] == vb[~m]).all() and (vb[~m] == V53[~m]).all(), "unmatched rows must equal v53/v56"
dv = new - vb
print(f"vs BASE: matched rows changed {int((dv[m] != 0).sum()):,} | mean |diff| {np.abs(dv[m]).mean():.3f} s | max |diff| {np.abs(dv).max():.1f} | fallback rows {nfb:,}")
print("per airport mean(new - base) matched: " + " ".join(f"{a} {dv[m & (apt == a)].mean():+.2f}" for a in sorted(set(apt))))
if (dv == 0).all(): print("REPRODUCES BASE EXACTLY")
if not TAG: sys.exit(0)
if os.environ.get("FOLDCHECK") and sum(x < 0 for x in FC) < 3: print(f"FOLDCHECK wins {sum(x < 0 for x in FC)}/4 < 3: NOT writing {TAG}"); sys.exit(0)
out = pl.DataFrame({"MVT_ID_mvt": S["MVT_ID_mvt"], "TAXITIME_SEC_mvt": new}).with_columns(pl.col("TAXITIME_SEC_mvt").cast(S.schema["TAXITIME_SEC_mvt"]))
o2 = out["TAXITIME_SEC_mvt"].to_numpy(); assert out.height == NRANK and (o2[~m] == vb[~m]).all() and np.isfinite(o2.astype(float)).all()
fn = f"stages/fallback_{TAG}_matched.parquet"; assert not os.path.exists(fn), f"{fn} exists"
out.write_parquet(fn); print("wrote", fn)
