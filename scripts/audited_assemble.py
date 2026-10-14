#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Production side of the chain meta stacker: refits the meta of the stacker source on the 2025 fold OOF and applies it to the matched
ranking rows. Prediction = cf(0.5*mean(PLG_s0..NLGB-1) + 0.5*mean(PCT_s0..NCAT-1) + clip(g, -CLIP_LO, +CLIP_HI)) on matched rows, g from the
LightGBM meta (seed 0, 300 rounds; inputs of variant VAR) fitted on the fold OOF of OL / OC / ON and QOOF; cf() applies the LOBT bound
|pred - (MVT - LOBT)| <= 3,606 s and the 60 s floor. Unmatched rows are copied from BASE; rows listed in FALLBACK take BASE's values (FBMODE=base, default).
env (stacker settings METAFIT, CLIP_LO / CLIP_HI, OL / OC / ON, QOOF, EXTRA as in the stacker source):
  STACKSRC=<stacker script> (default scripts/stage12_stack.py)
  PLG=V47L NLGB=6  PCT=CATV17x NCAT=14  PNN=NNRA2                 production member predictions in data/preds
  QRANK=fallback_regpl                                            q on ranking rows (data/oof/<QRANK>_rank.npy)
  VAR=L2 (L2 | L2q | base)   FALLBACK=<parquet with MVT_ID_mvt>   BASE=stages/unmatched_rule.parquet
  FBMODE=base | first | member: rows of FALLBACK keep BASE's values (base), use the first member set of PLG only (first), or the member
         PLGFB x NLGBFB (default runway_queue_v47lq x 6) through the meta (member)
  PXLK=lgb_adsb_linked_a NXLK=3, PXDK=lgb_noclock_nc NXDK=2, PXAPT=lgb_airport_ltfm:LTFM:3   ranking-side members of EXTRA xlk / xdk / xapt
  EXTRA=lnk: adsb_linked of the ranking rows from data/external/adsb_features.parquet; FOLDCHECK=1 then scores (COLS + lnk) against COLS
  FOLDCHECK=1 also scores the candidate OOF members + meta nested against the default (RA3 / RAC2) set, per fold
  ROUND=floor | rint: rounding of the matched-row output (default floor)
  TAG=<name>; OUT=<parquet> (default scratchpad/xu/XU_<TAG>_pre.parquet; must not exist and must not lie under submissions/)
  NT threads (default 4)."""

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
SRC = open(os.environ.get("STACKSRC", "scripts/stage12_stack.py")).read().split("variants = sys.argv")[0]   # STACKSRC: the stack source exec'd for the meta model
def stack_env():
    g = {"__file__": os.path.abspath("scripts/fallback_stack.py"), "__name__": "xf"}; exec(compile(SRC, "xf_stack", "exec"), g); return g
g = stack_env(); D, X, prm, PAIRS, ALL, cf, BASEC, EXTRA, npy, al, FT, corr = (g[k] for k in ("D", "X", "prm", "PAIRS", "ALL", "cf", "BASEC", "EXTRA", "npy", "al", "FT", "corr"))
METAFIT, CLIP_LO, CLIP_HI = g["METAFIT"], g["CLIP_LO"], g["CLIP_HI"]; FIT = PAIRS if METAFIT == "pairs" else ALL
extra_inputs = g["extra_inputs"]; COLS0 = BASEC + (EXTRA[VAR] if VAR != "base" else []); COLS = COLS0 + sum((EXTRA[x] for x in extra_inputs), [])
print(f"PNEW {E('PNEW','-')} FBMODE {E('FBMODE','base')} | members: {PLG} x{NLGB} + {PCT} x{NCAT} + {PNN} | meta OOF {E('OL','RA3')}/{E('OC','RAC2')}/{E('ON','NNA2')} q {E('QOOF','fallback_regpl')} | VAR {VAR} | fallback {FALLBACK or '-'} | METAFIT {METAFIT} ({'+'.join(FIT)}) clip [-{CLIP_LO:g},+{CLIP_HI:g}] | EXTRA {','.join(extra_inputs) or '-'}")
_ds = lgb.Dataset(pd.concat([X(D[t], COLS) for t in FIT]), np.concatenate([D[t]["y"] - D[t]["v41"] for t in FIT]))
METAS = [lgb.train(prm(s_), _ds, g.get("MROUNDS", 300)) for s_ in (g.get("MSEEDS") or [0])]   # audited_stack: seed-averaged meta (default one seed-0 meta, 300 rounds = unchanged)
class _M:
    def predict(self, x): return np.mean([m_.predict(x) for m_ in METAS], axis=0)
meta = _M()
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
    if FBM == "member":   # audited_stack #6d: fallback rows take the xba-free LGB member (env PLGFB x NLGBFB, all four columns) through the current meta
        iF, LF = avg(E("PLGFB", "runway_queue_v47lq"), int(E("NLGBFB", "6")), ["TAXITIME_SEC_mvt", "p_raw", "art", "taxi"]); assert np.array_equal(iF, ids); L[:, fbm] = LF[:, fbm]
        print(f"FBMODE member: {int(fbm.sum()):,} fallback rows from {E('PLGFB', 'runway_queue_v47lq')} x{E('NLGBFB', '6')}")
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
for c in g.get("RAWC", []):   # audited_stack #K(b): raw features_v9 columns as meta inputs on ranking rows (aligned by MVT_ID)
    d["raw_" + c] = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", c]).filter(pl.col("is_rank") == 1).sort("MVT_ID_mvt")[c].cast(pl.Float64).to_numpy()
if g.get("RAWC"): print("raw meta inputs:", g["RAWC"])
if "lnk" in extra_inputs:
    A = pl.read_parquet("data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"])
    lk = pl.DataFrame({"MVT_ID_mvt": ids}).join(A, on="MVT_ID_mvt", how="left", maintain_order="left")["adsb_linked"]
    if os.environ.get("MODE") == "new":   # a new period: a matched row without an adsb_features row is unlinked (lnk = 0)
        print("matched ranking rows without an adsb_features row (unlinked):", lk.filter(pl.Series(m)).null_count())
    else: assert lk.filter(pl.Series(m)).null_count() == 0, "matched ranking rows without adsb_linked"
    d["lnk"] = lk.fill_null(0).to_numpy().astype(float); print(f"ranking matched rows linked {d['lnk'][m].mean():.4f}")
if "xlk" in extra_inputs:   # XU: K on linked matched ranking rows from the production lgb_adsb_linked member (seed mean of taxi), NaN elsewhere
    NXLK = int(E("NXLK", "3")); iK, KX = avg(E("PXLK", "lgb_adsb_linked_a"), NXLK, ["taxi"]); assert np.array_equal(iK, ids)
    A2 = pl.read_parquet("data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"])
    lk2 = pl.DataFrame({"MVT_ID_mvt": ids}).join(A2, on="MVT_ID_mvt", how="left", maintain_order="left")["adsb_linked"].fill_null(0).to_numpy() == 1
    tx = np.where(lk2 & m, KX[0], np.nan); d["K"] = p * art + (1 - p) * tx; d["KmL"] = d["K"] - Lv; print(f"lgb_adsb_linked: K on {np.isfinite(d['K']).sum():,} ranking rows ({NXLK} seeds)")
if "xdk" in extra_inputs:   # dark_form: KD on DARK matched ranking rows (adsb_linked NULL->0 == 0) from the production clock-free member (seed mean of taxi), NaN elsewhere
    NXDK = int(E("NXDK", "2")); iD, DX = avg(E("PXDK", "lgb_noclock_nc"), NXDK, ["taxi"]); assert np.array_equal(iD, ids)
    A3 = pl.read_parquet("data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"])
    dk3 = pl.DataFrame({"MVT_ID_mvt": ids}).join(A3, on="MVT_ID_mvt", how="left", maintain_order="left")["adsb_linked"].fill_null(0).to_numpy() == 0
    tx = np.where(dk3 & m, DX[0], np.nan); d["KD"] = p * art + (1 - p) * tx; d["KDmL"] = d["KD"] - Lv; print(f"dark_form: KD on {np.isfinite(d['KD']).sum():,} ranking rows ({NXDK} seeds)")
if "xapt" in extra_inputs:   # APT_MEMBER: KA on matched ranking rows of airport A from the production airport-only member (seed mean of taxi), NaN elsewhere; env PXAPT="pref:APT:n[,..]"
    tx = np.full(len(ids), np.nan); apt_ = F["ADEP_mvt"].cast(pl.Utf8).to_numpy().astype(str)
    for spec in E("PXAPT", "lgb_airport_ltfm:LTFM:3").split(","):
        pf, aa, nn = spec.split(":"); iA, AX = avg(pf, int(nn), ["taxi"]); o_ = np.argsort(iA); k_ = np.clip(np.searchsorted(iA[o_], ids), 0, len(iA) - 1); hit = (iA[o_][k_] == ids) & (apt_ == aa) & m
        tx[hit] = AX[0][o_][k_[hit]]; print(f"APT_MEMBER: {pf} x{nn} on {int(hit.sum()):,} {aa} ranking rows")
    d["KA"] = p * art + (1 - p) * tx; d["KAmL"] = d["KA"] - Lv
gc = np.clip(meta.predict(X(d, COLS)), -CLIP_LO, CLIP_HI)
new = vb.astype(float).copy(); RND = np.rint if E("ROUND", "floor") == "rint" else np.floor; new[m] = RND(cf(v + gc, lo)[m])   # XU: ROUND=rint
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
fn = E("OUT", f"scratchpad/xu/XU_{TAG}_pre.parquet"); assert not fn.startswith("submissions") and not os.path.exists(fn), f"{fn} exists or in submissions"; os.makedirs(os.path.dirname(fn), exist_ok=True)
out.write_parquet(fn); print("wrote", fn)
