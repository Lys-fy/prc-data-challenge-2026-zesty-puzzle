#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Chain meta stacker on the 2025 month-pair folds: scripts/chain_stack.py with one extra member input (xtd), used for the wide-leaf and pseudo-label chains.
LightGBM meta model over the member outputs L, C, N (env OL / OC / ON, defaults RA3 / RAC2 / NNA2; OL may list several prefixes,
which are averaged), fitted to y - v41 with v41 = 0.5 L + 0.5 C (31 leaves, 300 rounds, lr 0.03, min_leaf 500, l2 50); the correction is
clipped to [-CLIP_LO, +CLIP_HI] (default 150 / 150 s). Folds: the month pairs months39, months28, months410 plus janjul; env
oof_build_folds=6 adds months612 and months511 (default 4). METAFIT=pairs (default): each fold is scored by a meta fitted on the pairs other
than itself, janjul by a meta fitted on all pairs; METAFIT=all: by a meta fitted on all other folds.
Variant columns:
  L3 : taxiC, pC, artC, taxiL - taxiC (rows 2-4 of the C member)
  L2 : q (fallback probability, env QOOF, default fallback_regpl), q x gap (per-airport mode gap of the fold's training months)
  env PNEW=<stem>[+<stem>]: pL and pC are replaced by the given classifier's p, and L and C are re-mixed with it
Optional extra meta inputs, env EXTRA (comma list, default empty):
  lnk  : adsb_linked (data/external/adsb_features.parquet, NULL -> 0) as column 'lnk'
  xlk  : K = pL*artL + (1-pL)*taxi of the clock-anchored member (env CLOCK_OOF, default lgb_adsb_linked_a) and K - L; NaN where it has no row
  xdk  : KD, KD - L: the same for the clock-free member (env DARK_OOF, default lgb_noclock_nc) on rows without an ADS-B link
  xapt : KA, KA - L: the same for airport-dedicated members (env APT_MEMBER="M1:A1[,M2:A2]") on the rows of their airport
  spr  : mean, std and range of L, C, N
  raw  : the features_v9 columns listed in env audited_stack_raw (comma list)
  xl   : the 8 same-day state columns PR5c..PAX15c from <P13_XLPREP>/prep_<fold>.parquet (env P13_XLPREP, default scratchpad/p13)
  xpr  : the 8 PRU daily inputs xP, xSh, xG, xR, xdl, xGd, xlate, xreg (scripts/pru_lib.py; needs oof_build_folds=6)
Meta knobs (defaults = the setting above): audited_stack_ridge=w blends a Ridge meta with weight w (default 0); audited_stack_rounds
(300), audited_stack_minleaf (500); audited_stack_mseeds=<comma list> averages metas over these seeds; audited_stack_special=<fold> keeps
that fold out of every meta under METAFIT=pairs.
  xtd  : T = pL*artL + (1-pL)*taxi_M, T - L for the extra member M = env EXTRA1_OOF (row 3 = taxi); NaN where M has no row
usage: wideleaf_stack.py <variant> [<variant> ...]  variant = '+'-joined subset of L2,L3,lnk (or 'base'); env SEEDS (default 5), NT (default 4)
Every variant is compared with the base stacker on the same rows and seeds, and with the fold prediction without a stacker.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
NT = int(os.environ.get("NT", "4")); os.environ.setdefault("POLARS_MAX_THREADS", str(NT)); os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = ["months39", "months28", "months410"]; ALL = PAIRS + ["janjul"]; SEEDS = int(os.environ.get("SEEDS", "5")); CLIP = 150.0
CLIP_LO, CLIP_HI = float(os.environ.get("CLIP_LO", "150")), float(os.environ.get("CLIP_HI", "150")); METAFIT = os.environ.get("METAFIT", "pairs"); assert METAFIT in ("pairs", "all"), METAFIT
HOLD = {"months39": [3, 9], "months28": [2, 8], "months410": [4, 10], "janjul": [1, 7], "months612": [6, 12], "months511": [5, 11]}
if os.environ.get("oof_build_folds", "4") == "6": PAIRS = PAIRS + ["months612", "months511"]; ALL = PAIRS + ["janjul"]   # oof_build: 6-fold pairs protocol
if os.environ.get("audited_stack_special"): PAIRS = [f for f in ALL if f != os.environ["audited_stack_special"]]; ALL = PAIRS + [os.environ["audited_stack_special"]]   # audited_stack #1b: the fold never in a meta under pairs
def npy(p, t):
    c = glob.glob(f"{ROOT}/data/oof/{p}_{t}.npy") or glob.glob(f"{ROOT}/data/oof/{p}_{t}_*.npy") or (glob.glob(f"{ROOT}/data/oof/{p}_months17.npy") if t == "janjul" else [])
    return np.load(sorted(c)[0]) if c else None
def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606)); return np.maximum(v, 60.0)
def al(x, ids):
    o = np.argsort(x[0]); k = np.searchsorted(x[0][o], ids); assert np.array_equal(x[0][o][k], ids); return x[1:, o][:, k]
RAWC = [c for c in os.environ.get("audited_stack_raw", "").split(",") if c]; RIDGE = float(os.environ.get("audited_stack_ridge", "0"))
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "y", "fb", "unmatched", "mvt_lobt", "mvt_aobt3", "ADEP_mvt", "MVT_TIME_UTC_mvt", "m", "is_rank"] + [c for c in RAWC if c not in ("mvt_lobt", "mvt_aobt3")])
FT = F.filter((pl.col("is_rank") == 0) & (pl.col("unmatched") == 0) & (pl.col("fb") == 0)).with_columns(r=pl.col("y") - pl.col("mvt_aobt3"))
def gap_table(t):   # per-airport mode gap from the fold's TRAINING months (clean matched rows)
    g = FT.filter(~pl.col("m").is_in(HOLD[t])).group_by("ADEP_mvt").agg(hi=pl.col("r").filter(pl.col("r") > 120).median(), lo=pl.col("r").filter(pl.col("r") < -120).median())
    return {a: h - l for a, h, l in g.select("ADEP_mvt", "hi", "lo").iter_rows()}
extra_inputs = [x for x in os.environ.get("EXTRA", "").split(",") if x]; assert set(extra_inputs) <= {"lnk", "xlk", "xdk", "xapt", "spr", "raw", "xl", "xpr", "xtd"}, extra_inputs
daystate_meta = ["PR5c", "PR15c", "PRT5c", "PRT15c", "PRX5c", "PRX15c", "PM15c", "PAX15c"]   # P13: XL state (month-demeaned)   # optional extra meta inputs (adsb_link_flag)
if any("lnk" in a for a in sys.argv[1:]) and "lnk" not in extra_inputs: extra_inputs.append("lnk")
PNEW = os.environ.get("PNEW", "")   # optional classifier stem (fallback_classifier...) replacing pL AND pC in the members (L1 -> stacker)
def load(t):
    OL_, OC_, ON_ = os.environ.get("OL", "RA3"), os.environ.get("OC", "RAC2"), os.environ.get("ON", "NNA2")   # OOF member prefixes (defaults = the v53/v56 meta's)
    OLs = OL_.split(","); L = npy(OLs[0], t); ids = L[0]
    if len(OLs) > 1:   # average several OOF member sets (all rows: pred, p, taxi, art), aligned by ID
        L = np.vstack([ids, np.mean([L[1:5]] + [al(npy(x, t), ids)[:4] for x in OLs[1:]], axis=0)])
    C = al(npy(OC_, t), ids); N = al(npy(ON_, t), ids)
    V = pl.DataFrame({"MVT_ID_mvt": ids, "L": L[1], "pL": L[2], "taxiL": L[3], "artL": L[4], "C": C[0], "pC": C[1], "taxiC": C[2], "artC": C[3], "Nt": N[0]})
    q = npy(os.environ.get("QOOF", "fallback_regpl"), t)
    if q is not None: V = V.with_columns(q=al(q, ids)[0])
    if PNEW:
        zs = [al(npy(pp, t), ids)[0] for pp in PNEW.split("+")]; zs = [np.log(np.clip(z, 1e-7, 1 - 1e-7) / (1 - np.clip(z, 1e-7, 1 - 1e-7))) for z in zs]
        V = V.with_columns(pn=1 / (1 + np.exp(-np.mean(zs, axis=0))))
    V = V.join(F, on="MVT_ID_mvt", how="inner").filter(pl.col("unmatched") == 0)
    if "xl" in extra_inputs: V = V.join(pl.read_parquet(f"{ROOT}/{os.environ.get('P13_XLPREP', 'scratchpad/p13')}/prep_{t}.parquet", columns=["MVT_ID_mvt"] + daystate_meta).with_columns(pl.col("MVT_ID_mvt").cast(V.schema["MVT_ID_mvt"])), on="MVT_ID_mvt", how="left", maintain_order="left")   # P13
    if "lnk" in extra_inputs: V = V.join(pl.read_parquet(f"{ROOT}/data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"]), on="MVT_ID_mvt", how="left", maintain_order="left").with_columns(lnk=pl.col("adsb_linked").fill_null(0))
    d = {c: V[c].to_numpy().astype(float) for c in V.columns if c in ("y", "L", "pL", "taxiL", "artL", "C", "pC", "taxiC", "artC", "Nt", "mvt_lobt", "q", "pn", "lnk")}
    if "xlk" in extra_inputs:   # XU: K = pL*artL + (1-pL)*taxi_XLK on rows the linked-only member covers, NaN elsewhere
        a = npy(os.environ.get("CLOCK_OOF", "lgb_adsb_linked_a"), t); o = np.argsort(a[0]); mid = V["MVT_ID_mvt"].to_numpy().astype(float)
        k = np.clip(np.searchsorted(a[0][o], mid), 0, a.shape[1] - 1); hit = a[0][o][k] == mid; tx = np.full(len(mid), np.nan); tx[hit] = a[3][o][k[hit]]; d["txlk"] = tx
    if "xdk" in extra_inputs:   # dark_form: KD = pL*artL + (1-pL)*taxi_M on DARK rows (adsb_linked NULL->0 == 0), NaN on linked rows; M = env DARK_OOF
        a = npy(os.environ.get("DARK_OOF", "lgb_noclock_nc"), t); o = np.argsort(a[0]); mid = V["MVT_ID_mvt"].to_numpy().astype(float)
        k = np.clip(np.searchsorted(a[0][o], mid), 0, a.shape[1] - 1); hit = a[0][o][k] == mid
        dk = V.select("MVT_ID_mvt").join(pl.read_parquet(f"{ROOT}/data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"]), on="MVT_ID_mvt", how="left", maintain_order="left")["adsb_linked"].fill_null(0).to_numpy() == 0
        tx = np.full(len(mid), np.nan); hit = hit & dk; tx[hit] = a[3][o][k[hit]]; d["txdk"] = tx
    if "xapt" in extra_inputs:   # APT_MEMBER: KA = pL*artL + (1-pL)*taxi_M on rows of airport A only, NaN elsewhere; env APT_MEMBER="M1:A1[,M2:A2]" (one member per airport)
        mid = V["MVT_ID_mvt"].to_numpy().astype(float); ap = V["ADEP_mvt"].cast(pl.Utf8).to_numpy().astype(str); tx = np.full(len(mid), np.nan)
        for spec in os.environ["APT_MEMBER"].split(","):
            mm, aa = spec.split(":"); a = npy(mm, t); o = np.argsort(a[0])
            k = np.clip(np.searchsorted(a[0][o], mid), 0, a.shape[1] - 1); hit = (a[0][o][k] == mid) & (ap == aa); tx[hit] = a[3][o][k[hit]]
        d["txap"] = tx
    for c in RAWC: d["raw_" + c] = V[c].cast(pl.Float64).to_numpy()   # audited_stack
    if "xtd" in extra_inputs:   # WIDELEAF ( step 2; target_diversity form): extra member M = env EXTRA1_OOF (row 3 = taxi) on all matched rows, NaN where M has no row
        a = npy(os.environ["EXTRA1_OOF"], t); o = np.argsort(a[0]); mid = V["MVT_ID_mvt"].to_numpy().astype(float)
        k = np.clip(np.searchsorted(a[0][o], mid), 0, a.shape[1] - 1); hit = a[0][o][k] == mid
        tx = np.full(len(mid), np.nan); tx[hit] = a[3][o][k[hit]]; d["txtd"] = tx
    if "xl" in extra_inputs:
        for c in daystate_meta: d[c] = V[c].cast(pl.Float64).fill_null(np.nan).to_numpy()   # P13
    if PNEW:   # re-mix both members with the new p
        d["L"] = d["pn"] * d["artL"] + (1 - d["pn"]) * d["taxiL"]; d["C"] = d["pn"] * d["artC"] + (1 - d["pn"]) * d["taxiC"]; d["pL"] = d["pn"]; d["pC"] = d["pn"]
    d["inc"] = cf(0.5 * al(L, V["MVT_ID_mvt"].to_numpy())[0] + 0.5 * al(npy(OC_, t), V["MVT_ID_mvt"].to_numpy())[0], d["mvt_lobt"])   # the no-stacker fold incumbent (stored members)
    d["N"] = d["pL"] * d["artL"] + (1 - d["pL"]) * d["Nt"]; d["v41"] = cf(0.5 * d["L"] + 0.5 * d["C"], d["mvt_lobt"])
    gp = gap_table(t); d["gap"] = np.array([gp.get(a, np.nan) for a in V["ADEP_mvt"].to_numpy()], dtype=float)
    if "q" in d: d["qgap"] = d["q"] * d["gap"]
    if "txlk" in d: d["K"] = d["pL"] * d["artL"] + (1 - d["pL"]) * d["txlk"]; d["KmL"] = d["K"] - d["L"]
    if "txtd" in d: d["T"] = d["pL"] * d["artL"] + (1 - d["pL"]) * d["txtd"]; d["TmL"] = d["T"] - d["L"]
    if "txap" in d: d["KA"] = d["pL"] * d["artL"] + (1 - d["pL"]) * d["txap"]; d["KAmL"] = d["KA"] - d["L"]
    if "txdk" in d: d["KD"] = d["pL"] * d["artL"] + (1 - d["pL"]) * d["txdk"]; d["KDmL"] = d["KD"] - d["L"]
    d["mid"] = V["MVT_ID_mvt"].to_numpy().astype(float); d["apt"] = V["ADEP_mvt"].to_numpy().astype(str); d["fb"] = V["fb"].to_numpy() == 1; d["mo"] = V["m"].to_numpy(); d["day"] = V["MVT_TIME_UTC_mvt"].dt.date().cast(pl.Utf8).to_numpy()
    ok = np.isfinite(d["y"]) & np.isfinite(d["v41"]); d = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in d.items()}; d["n"] = int(ok.sum())
    return d
D = {t: load(t) for t in ALL}
if "xpr" in extra_inputs:   # pru: per-row PRU daily inputs; day sums use every 2025 departure with its own OOF v41 (all 6 folds)
    import sys as _s; _s.path.insert(0, f"{ROOT}/scripts"); import pru_lib
    assert os.environ.get("oof_build_folds") == "6", "xpr needs the 12-month OOF (oof_build_folds=6)"
    _Z = pru_lib.feats(0, np.concatenate([D[t]["mid"] for t in ALL]), np.concatenate([D[t]["v41"] for t in ALL])); pru_lib.report(_Z, "fold2025")
    _Z = _Z.sort("MVT_ID"); _zi = _Z["MVT_ID"].to_numpy()
    for t in ALL:
        _k = np.searchsorted(_zi, D[t]["mid"]); assert np.array_equal(_zi[_k], D[t]["mid"])
        for c in pru_lib.COLS: D[t][c] = _Z[c].to_numpy().astype(float)[_k]
BASEC = ["L", "C", "N", "taxiL", "artL", "pL", "v41", "LmC", "NmL"]
BASEC = ["L", "C", "N", "taxiL", "artL", "pL", "v41", "LmC", "NmL"]
EXTRA = {"L3": ["taxiC", "pC", "artC", "TmT"], "L2": ["q", "qgap"], "L2q": ["q"], "lnk": ["lnk"], "xlk": ["K", "KmL"], "xdk": ["KD", "KDmL"], "xapt": ["KA", "KAmL"], "spr": ["mLCN", "sLCN", "rLCN"], "raw": ["raw_" + c for c in RAWC], "xl": daystate_meta, "xpr": ["xP", "xSh", "xG", "xR", "xdl", "xGd", "xlate", "xreg"], "xtd": ["T", "TmL"]}
def X(d, cols):
    x = pd.DataFrame({k: d[k] for k in ("L", "C", "N", "taxiL", "artL", "pL", "v41", "taxiC", "pC", "artC") + (("q", "qgap") if "q" in d else ()) + (("lnk",) if "lnk" in d else ()) + (("K", "KmL") if "K" in d else ()) + (("KD", "KDmL") if "KD" in d else ()) + (("KA", "KAmL") if "KA" in d else ())})
    x["LmC"] = d["L"] - d["C"]; x["NmL"] = d["N"] - d["L"]; x["TmT"] = d["taxiL"] - d["taxiC"]
    if "spr" in extra_inputs: M3 = np.vstack([d["L"], d["C"], d["N"]]); x["mLCN"] = M3.mean(0); x["sLCN"] = M3.std(0); x["rLCN"] = M3.max(0) - M3.min(0)   # audited_stack
    for c in RAWC: x["raw_" + c] = d["raw_" + c]
    if "T" in d: x["T"] = d["T"]; x["TmL"] = d["TmL"]   # WIDELEAF xtd
    if "xl" in extra_inputs:
        for c in daystate_meta: x[c] = d[c]   # P13
    if "xP" in d:
        for c in ("xP", "xSh", "xG", "xR", "xdl", "xGd", "xlate", "xreg"): x[c] = d[c]   # P13 (= pru_stack)
    return x[cols]
MROUNDS = int(os.environ.get("audited_stack_rounds", "300")); MLEAF = int(os.environ.get("audited_stack_minleaf", "500")); MSEEDS = [int(x) for x in os.environ.get("audited_stack_mseeds", "").split(",") if x]   # audited_stack: meta capacity / seed-average (defaults = unchanged)
def prm(seed): return dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=MLEAF, lambda_l2=50.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed, num_threads=NT)
RIDGE_OUT = {}
def ridge_design(x, st=None):
    a = x.to_numpy(dtype=float); nan = np.isnan(a); nc = np.where(nan.any(0))[0] if st is None else st[2]
    if st is None: mu = np.nanmean(a, 0); sd = np.nanstd(a, 0) + 1e-9
    else: mu, sd = st[0], st[1]
    z = np.where(nan, 0.0, (a - mu) / sd); return np.hstack([z, nan[:, nc].astype(float)]), (mu, sd, nc)
def ridge_fit(x, r):
    from sklearn.linear_model import Ridge
    z, st = ridge_design(x); return (Ridge(alpha=1.0).fit(z, r), st)
def ridge_pred(m, x):
    z, _ = ridge_design(x, m[1]); return m[0].predict(z)
def corr(cols, seed):
    out = {}
    for t in ALL:
        tags = [x for x in (PAIRS if METAFIT == "pairs" else ALL) if x != t]
        ds = lgb.Dataset(pd.concat([X(D[x], cols) for x in tags]), np.concatenate([D[x]["y"] - D[x]["v41"] for x in tags]))
        g = np.mean([lgb.train(prm(s_), ds, MROUNDS).predict(X(D[t], cols)) for s_ in (MSEEDS or [seed])], axis=0)
        if RIDGE > 0:   # audited_stack #K(a): Ridge on standardised inputs (NaN -> 0 + missing indicators), blended with weight RIDGE
            rm = ridge_fit(pd.concat([X(D[x], cols) for x in tags]), np.concatenate([D[x]["y"] - D[x]["v41"] for x in tags])); gr = ridge_pred(rm, X(D[t], cols))
            RIDGE_OUT[t] = (g, gr); g = (1 - RIDGE) * g + RIDGE * gr
        out[t] = cf(D[t]["v41"] + np.clip(g, -CLIP_LO, CLIP_HI), D[t]["mvt_lobt"])
    return out
variants = sys.argv[1:] or ["base"]
print(f"rows: " + " ".join(f"{t} {D[t]['n']:,}" for t in ALL) + (f" | PNEW={PNEW}" if PNEW else ""))
base = [corr(BASEC, s) for s in range(SEEDS)]
def dstat(pred_new, pred_ref, t):
    d = D[t]; dd = (pred_new - d["y"]) ** 2 - (pred_ref - d["y"]) ** 2; k = np.argsort(-np.abs(dd))[:3]
    return dd.mean(), (dd.sum() - dd[k].sum()) / len(dd), dd
print("\n[base stacker vs no-stacker incumbent] per fold (mean over seeds), pooled over the three pairs")
for s_ in range(SEEDS):
    r = [dstat(base[s_][t], D[t]["inc"], t)[0] for t in ALL]; pooled = sum(r[i] * D[t]["n"] for i, t in enumerate(PAIRS)) / sum(D[t]["n"] for t in PAIRS)
    print(f"  seed {s_}: " + " ".join(f"{t} {v:+7.1f}" for t, v in zip(ALL, r)) + f" | pairs pooled {pooled:+7.1f}")
for v in variants:
    if v == "base": continue
    cols = BASEC + sum((EXTRA[x] for x in v.split("+")), [])
    print(f"\n=== variant {v}: extra {cols[len(BASEC):]}  (vs base stacker, same seed; w/o top3 in brackets)")
    pool_s = []; allf = {t: [] for t in ALL}; allf3 = {t: [] for t in ALL}; fbc = {t: [] for t in ALL}; lirf = {t: [] for t in ALL}
    for s_ in range(SEEDS):
        c = corr(cols, s_); row = []
        for t in ALL:
            m_, m3, dd = dstat(c[t], base[s_][t], t); allf[t].append(m_); allf3[t].append(m3)
            fbc[t].append(dd[D[t]["fb"]].sum() / D[t]["n"]); lirf[t].append(dd[D[t]["apt"] == "LIRF"].sum() / D[t]["n"])
            row.append(f"{t} {m_:+7.1f} [{m3:+7.1f}]")
        pooled = sum(allf[t][-1] * D[t]["n"] for t in PAIRS) / sum(D[t]["n"] for t in PAIRS)
        pool_s.append(pooled); print(f"  seed {s_}: " + " ".join(row) + f" | pairs pooled {pooled:+7.1f}")
    NP = sum(D[t]["n"] for t in PAIRS)
    all4 = sum(np.mean(allf[t]) * D[t]["n"] for t in ALL) / sum(D[t]["n"] for t in ALL)
    print(f"  MEAN   : " + " ".join(f"{t} {np.mean(allf[t]):+7.1f} [{np.mean(allf3[t]):+7.1f}] (fb {np.mean(fbc[t]):+.1f}, LIRF {np.mean(lirf[t]):+.1f})" for t in ALL))
    print(f"  pairs pooled mean {np.mean(pool_s):+.1f} sd {np.std(pool_s):.1f} | 4-fold pooled {all4:+.1f} | folds negative {sum(np.mean(allf[t]) < 0 for t in ALL)}/4")
if os.environ.get("APT"):   # per-airport / per-cell breakdown of the last variant, seed 0, per fold row (sum over the 4 folds / total rows)
    c = corr(cols, 0); NT_ = sum(D[t]["n"] for t in ALL)
    apts = sorted(set(np.concatenate([D[t]["apt"] for t in ALL])))
    for a in apts:
        vals = [dstat(c[t], base[0][t], t)[2][D[t]["apt"] == a].sum() / D[t]["n"] for t in ALL]
        print(f"  APT {a}: " + " ".join(f"{t} {v:+7.1f}" for t, v in zip(ALL, vals)))
    for nm, fn in (("q<0.2", lambda d: d["q"] < 0.2), ("0.2-0.8", lambda d: (d["q"] >= 0.2) & (d["q"] <= 0.8)), ("q>0.8", lambda d: d["q"] > 0.8)):
        vals = [dstat(c[t], base[0][t], t)[2][fn(D[t])].sum() / D[t]["n"] for t in ALL]
        print(f"  Q {nm}: " + " ".join(f"{t} {v:+7.1f}" for t, v in zip(ALL, vals)))
