#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fold screen of chain-stacker arms on the chain_stack.py stacker: an arm may replace the L member and add several extra members, combined
with ";", e.g. 'ol:pseudolabel_pl3;xtd:capacity_combine_pa2' (L member replaced and an extra member added); the CAPACITY_COLS columns are
appended to the raw_* columns of X(). Env capacity_combine_tag (default capacity_tag4); outputs scratchpad/xpla/<TAG>_<label>_{Kq,NNA3,mix}_<fold>.npy.
  spec  xtd:M1+M2[@1]+...   member j adds T_j = pL*artL + (1-pL)*taxi_Mj and T_j - L (all rows); taxi row 3 (tree OOF [id,pred,p,taxi,art]) or @1 (NN OOF [id,taxi,...]).
  The first member keeps the column names T, TmL (so 'xtd:wideleaf_retune_a2' gives the wide-leaf chain stack); later ones T2/TmL2, T3/TmL3.
env capacity_xtd_tag (default capacity_c4) and wideleaf_folds; outputs scratchpad/xcap/<TAG>_<label>_{Kq,NNA3,mix}_<fold>.npy.
--- original docstring ---
WIDELEAF: T2x L2 harness (chain_stack.py source exec'd) with four arm kinds.
usage: screen_l2_extra.py ARM [ARM ...]     ARM = [label=]spec, spec one of
  base            no change (OL queue_eb_ra3qa, KD lgb_noclock_nc)
  xtd:M           extra input T = pL*artL + (1-pL)*taxi_M and T - L on all rows (target_diversity form; M = data/oof prefix, row 3 = taxi)
  xtdd:M          the same on DARK rows only (adsb_linked NULL/0), NaN on linked rows (the dark_form form)
  ol:A[,B]        the L member replaced by the mean of the listed OOF sets (chain_stack OL list)
  xdk:M           the KD member (DARK_OOF) replaced by M
env wideleaf_folds (default months28,janjul): folds, meta cross-fitted over them (METAFIT=all), meta seeds 0,1,2; wideleaf_tag (default wideleaf_tag_default).
Writes scratchpad/xbase/<TAG>_<label>_{Kq,NNA3,mix}_<fold>.npy as [ID, pred]."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); OUT = "scratchpad/xpla"; os.makedirs(OUT, exist_ok=True)
TAG = os.environ.get("capacity_combine_tag", "capacity_tag4")
RAW = "te2_srh_te,mvt_eobt1,ref_sr_mean,mvt_aobt3,mvt_iobt,aobt3_minus_ref,te2_sr_te,mvt_sched,aobt3_eobt1,mvt_lobt"
COMMON = dict(oof_build_folds="6", QOOF="fallback_regpl", CLOCK_OOF="lgb_adsb_linked_a", DARK_OOF="lgb_noclock_nc", APT_MEMBER="lgb_airport_ltfm:LTFM,lgb_airport_lfpg:LFPG", audited_stack_raw=RAW,
              METAFIT="all", audited_stack_mseeds="0,1,2", OL="queue_eb_ra3qa", target_diversity_dark="0")
EXTRA0 = "lnk,xlk,xdk,xapt,spr,raw,xpr"
CFG = {"Kq": dict(OC="runway_queue_catq", ON="NNA2"), "NNA3": dict(OC="meta_oof_h_ctl", ON="meta_oof_realmlp_nna3")}
FOLDS = os.environ.get("wideleaf_folds", "months28,janjul").split(",")
src = open("scripts/chain_stack.py").read().split("variants = sys.argv")[0]
def rep(s, a, b):
    assert s.count(a) == 1, a; return s.replace(a, b)
src = rep(src, "D = {t: load(t) for t in ALL}", "ALL = list(wideleaf_retune_f); PAIRS = list(wideleaf_retune_f)\nD = {t: load(t) for t in ALL}")
src = rep(src, '"xl", "xpr"}, extra_inputs', '"xl", "xpr", "xtd"}, extra_inputs')
src = rep(src, '    for c in RAWC: d["raw_" + c] = V[c].cast(pl.Float64).to_numpy()   # audited_stack\n',
          '    for c in RAWC: d["raw_" + c] = V[c].cast(pl.Float64).to_numpy()   # audited_stack\n'
          '    if "xtd" in extra_inputs:   # CAPACITY_XTD: several extra members M_j (taxi row 3, or row r with M@r), all rows\n'
          '        for j, mm in enumerate(os.environ["EXTRA1_OOF"].split("+")):\n'
          '            mp, r = (mm.split("@")[0], int(mm.split("@")[1])) if "@" in mm else (mm, 3)\n'
          '            a = npy(mp, t); assert a is not None, (mp, t); o = np.argsort(a[0]); mid = V["MVT_ID_mvt"].to_numpy().astype(float)\n'
          '            k = np.clip(np.searchsorted(a[0][o], mid), 0, a.shape[1] - 1); hit = a[0][o][k] == mid\n'
          '            tx = np.full(len(mid), np.nan); tx[hit] = a[r][o][k[hit]]; d["txtd" + ("" if j == 0 else str(j + 1))] = tx\n')
src = rep(src, '    if "txap" in d: d["KA"]', '    for sfx in ("", "2", "3", "4"):\n        if "txtd" + sfx in d: d["T" + sfx] = d["pL"] * d["artL"] + (1 - d["pL"]) * d["txtd" + sfx]; d["TmL" + sfx] = d["T" + sfx] - d["L"]\n    if "txap" in d: d["KA"]')
src = rep(src, '"xpr": ["xP", "xSh", "xG", "xR", "xdl", "xGd", "xlate", "xreg"]}', '"xpr": ["xP", "xSh", "xG", "xR", "xdl", "xGd", "xlate", "xreg"], "xtd": CAPACITY_COLS}')
src = rep(src, '    for c in RAWC: x["raw_" + c] = d["raw_" + c]\n', '    for c in RAWC: x["raw_" + c] = d["raw_" + c]\n    for c in CAPACITY_COLS:\n        if c in d: x[c] = d[c]   # CAPACITY_XTD (screen_l2_extra: raw line kept; screen_capacity/screen_pseudo_capacity dropped it)\n')
code = compile(src, "p13_stack_xbase", "exec")
def parse(arm):
    label, spec = arm.split("=", 1) if "=" in arm else (arm.replace(":", "_").replace(",", "+"), arm)
    env = dict(COMMON); extra = EXTRA0
    for sp in spec.split(";"):   # CAPACITY_COMBINE: several kinds per arm
        if sp == "base": pass
        elif sp.startswith("xtd:"): extra += ",xtd"; env["EXTRA1_OOF"] = sp[4:]
        elif sp.startswith("xtdd:"): extra += ",xtd"; env["EXTRA1_OOF"] = sp[5:]; env["target_diversity_dark"] = "1"
        elif sp.startswith("ol:"): env["OL"] = sp[3:]
        elif sp.startswith("xdk:"): env["DARK_OOF"] = sp[4:]
        else: raise SystemExit(f"arm spec? {sp}")
    env["EXTRA"] = extra; return label, env
for arm in sys.argv[1:]:
    label, env = parse(arm); res = {}
    os.environ.pop("EXTRA1_OOF", None)
    for cfg, ce in CFG.items():
        os.environ.update(env); os.environ.update(ce)
        nm = len(env.get("EXTRA1_OOF", "").split("+")) if "EXTRA1_OOF" in env else 0
        xtdc = sum(([f"T{'' if j == 0 else j + 1}", f"TmL{'' if j == 0 else j + 1}"] for j in range(nm)), [])
        g = {"__file__": os.path.abspath("scripts/chain_stack.py"), "__name__": "xf", "wideleaf_retune_f": FOLDS, "CAPACITY_COLS": xtdc}; exec(code, g)
        cols = g["BASEC"] + g["EXTRA"]["L2"] + sum((g["EXTRA"][x] for x in g["extra_inputs"]), [])
        out = g["corr"](cols, 0); D = g["D"]; res[cfg] = {}
        for t, v in out.items():
            res[cfg][t] = (D[t]["mid"], v, D[t]["y"]); np.save(f"{OUT}/{TAG}_{label}_{cfg}_{t}.npy", np.vstack([D[t]["mid"], v]))
            msg = f"{label} [{arm}] {cfg} {t} n {len(v):,} cols {len(cols)} mse {np.mean((v - D[t]['y']) ** 2):.1f} | member L mse {np.mean((D[t]['L'] - D[t]['y']) ** 2):.1f}"
            for sfx in ("2", "3"):
                if "T" + sfx in D[t]:
                    ok = np.isfinite(D[t]["T" + sfx]); msg += f" | M{sfx} cover {ok.mean():.4f} T{sfx} mse {np.mean((D[t]['T' + sfx][ok] - D[t]['y'][ok]) ** 2):.1f} corr(T{sfx}-y,L-y) {np.corrcoef(D[t]['T' + sfx][ok] - D[t]['y'][ok], D[t]['L'][ok] - D[t]['y'][ok])[0, 1]:.4f} corr(T{sfx}-y,T-y) {np.corrcoef(D[t]['T' + sfx][ok] - D[t]['y'][ok], D[t]['T'][ok] - D[t]['y'][ok])[0, 1]:.4f}"
            if "T" in D[t]:
                ok = np.isfinite(D[t]["T"]); msg += f" | M cover {ok.mean():.4f} T mse {np.mean((D[t]['T'][ok] - D[t]['y'][ok]) ** 2):.1f} L mse same rows {np.mean((D[t]['L'][ok] - D[t]['y'][ok]) ** 2):.1f} corr(T-y,L-y) {np.corrcoef(D[t]['T'][ok] - D[t]['y'][ok], D[t]['L'][ok] - D[t]['y'][ok])[0, 1]:.4f}"
            if "KD" in D[t]:
                ok = np.isfinite(D[t]["KD"]); msg += f" | KD cover {ok.mean():.4f} KD mse {np.mean((D[t]['KD'][ok] - D[t]['y'][ok]) ** 2):.1f} L mse dark {np.mean((D[t]['L'][ok] - D[t]['y'][ok]) ** 2):.1f}"
            print(msg, flush=True)
    for t in res["Kq"]:
        (i1, v1, y1), (i2, v2, _) = res["Kq"][t], res["NNA3"][t]; c = np.intersect1d(i1, i2)
        a1 = np.argsort(i1); a2 = np.argsort(i2); m1 = v1[a1][np.searchsorted(i1[a1], c)]; m2 = v2[a2][np.searchsorted(i2[a2], c)]
        np.save(f"{OUT}/{TAG}_{label}_mix_{t}.npy", np.vstack([c, 0.5 * m1 + 0.5 * m2]))
        print(f"{label} mix {t} rows {len(c):,} (Kq {len(i1):,} NNA3 {len(i2):,})", flush=True)
