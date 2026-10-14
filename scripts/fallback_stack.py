#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""XF / L2+L3: members-only stacker (v53 protocol: RA3/RAC2/NNA2 single seeds, W 0.5/0.5/0, LightGBM 31 leaves /
300 rounds / lr 0.03 / min_leaf 500 / l2 50, target y - v41, clip |g| <= 150) refit NESTED -- each pair scored by a
meta fitted on the other two pairs, janjul by a meta fitted on all three pairs -- with extra input columns:
  L3 : taxiC, pC, artC, taxiL - taxiC (RAC2 rows 2-4)
  L2 : q (fallback_regpl, regime-sign classifier on all matched rows), q x gap (per-airport training-month mode gap)
  P:<stem> : replace pL by a new classifier's p (and re-mix L/C/N with it) -- L1 into the stacker
usage: fallback_stack.py <variant> [<variant> ...]  variant = '+'-joined subset of L2,L3 (or 'base'); env SEEDS (default 5), NT
Every variant is compared with the base stacker on the same rows and seeds, and with the no-stacker fold incumbent.
Optional protocol flags (XM,; defaults reproduce the protocol above exactly):
  METAFIT=pairs|all  pairs (default): meta fitted on the pairs other than the scored fold (janjul <- 3 pairs);
                     all: leave-one-fold-out over all 4 folds (each fold <- the other 3, janjul included for the pairs)
  CLIP_LO / CLIP_HI  meta output clipped to [-CLIP_LO, +CLIP_HI] (default 150 / 150)
Optional extra meta input (adsb_link_flag, ; default empty = unchanged):
  EXTRA=lnk          loads adsb_linked (data/external/adsb_features.parquet, NULL -> 0; fold months have no NULL days) as
                     column 'lnk'; fallback_stack variants may then use '+lnk' (e.g. L2+lnk); fallback_assemble appends it to COLS
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np, polars as pl, pandas as pd, lightgbm as lgb
NT = int(os.environ.get("NT", "4")); os.environ.setdefault("POLARS_MAX_THREADS", str(NT)); os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = ["months39", "months28", "months410"]; ALL = PAIRS + ["janjul"]; SEEDS = int(os.environ.get("SEEDS", "5")); CLIP = 150.0
CLIP_LO, CLIP_HI = float(os.environ.get("CLIP_LO", "150")), float(os.environ.get("CLIP_HI", "150")); METAFIT = os.environ.get("METAFIT", "pairs"); assert METAFIT in ("pairs", "all"), METAFIT
HOLD = {"months39": [3, 9], "months28": [2, 8], "months410": [4, 10], "janjul": [1, 7]}
def npy(p, t):
    c = glob.glob(f"{ROOT}/data/oof/{p}_{t}.npy") or glob.glob(f"{ROOT}/data/oof/{p}_{t}_*.npy") or (glob.glob(f"{ROOT}/data/oof/{p}_months17.npy") if t == "janjul" else [])
    return np.load(sorted(c)[0]) if c else None
def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606)); return np.maximum(v, 60.0)
def al(x, ids):
    o = np.argsort(x[0]); k = np.searchsorted(x[0][o], ids); assert np.array_equal(x[0][o][k], ids); return x[1:, o][:, k]
F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "y", "fb", "unmatched", "mvt_lobt", "mvt_aobt3", "ADEP_mvt", "MVT_TIME_UTC_mvt", "m", "is_rank"])
FT = F.filter((pl.col("is_rank") == 0) & (pl.col("unmatched") == 0) & (pl.col("fb") == 0)).with_columns(r=pl.col("y") - pl.col("mvt_aobt3"))
def gap_table(t):   # per-airport mode gap from the fold's TRAINING months (clean matched rows)
    g = FT.filter(~pl.col("m").is_in(HOLD[t])).group_by("ADEP_mvt").agg(hi=pl.col("r").filter(pl.col("r") > 120).median(), lo=pl.col("r").filter(pl.col("r") < -120).median())
    return {a: h - l for a, h, l in g.select("ADEP_mvt", "hi", "lo").iter_rows()}
extra_inputs = [x for x in os.environ.get("EXTRA", "").split(",") if x]; assert set(extra_inputs) <= {"lnk"}, extra_inputs   # optional extra meta inputs (adsb_link_flag)
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
    if "lnk" in extra_inputs: V = V.join(pl.read_parquet(f"{ROOT}/data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"]), on="MVT_ID_mvt", how="left", maintain_order="left").with_columns(lnk=pl.col("adsb_linked").fill_null(0))
    d = {c: V[c].to_numpy().astype(float) for c in V.columns if c in ("y", "L", "pL", "taxiL", "artL", "C", "pC", "taxiC", "artC", "Nt", "mvt_lobt", "q", "pn", "lnk")}
    if PNEW:   # re-mix both members with the new p
        d["L"] = d["pn"] * d["artL"] + (1 - d["pn"]) * d["taxiL"]; d["C"] = d["pn"] * d["artC"] + (1 - d["pn"]) * d["taxiC"]; d["pL"] = d["pn"]; d["pC"] = d["pn"]
    d["inc"] = cf(0.5 * al(L, V["MVT_ID_mvt"].to_numpy())[0] + 0.5 * al(npy(OC_, t), V["MVT_ID_mvt"].to_numpy())[0], d["mvt_lobt"])   # the no-stacker fold incumbent (stored members)
    d["N"] = d["pL"] * d["artL"] + (1 - d["pL"]) * d["Nt"]; d["v41"] = cf(0.5 * d["L"] + 0.5 * d["C"], d["mvt_lobt"])
    gp = gap_table(t); d["gap"] = np.array([gp.get(a, np.nan) for a in V["ADEP_mvt"].to_numpy()], dtype=float)
    if "q" in d: d["qgap"] = d["q"] * d["gap"]
    d["mid"] = V["MVT_ID_mvt"].to_numpy().astype(float); d["apt"] = V["ADEP_mvt"].to_numpy().astype(str); d["fb"] = V["fb"].to_numpy() == 1; d["mo"] = V["m"].to_numpy(); d["day"] = V["MVT_TIME_UTC_mvt"].dt.date().cast(pl.Utf8).to_numpy()
    ok = np.isfinite(d["y"]) & np.isfinite(d["v41"]); d = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in d.items()}; d["n"] = int(ok.sum())
    return d
D = {t: load(t) for t in ALL}
BASEC = ["L", "C", "N", "taxiL", "artL", "pL", "v41", "LmC", "NmL"]
EXTRA = {"L3": ["taxiC", "pC", "artC", "TmT"], "L2": ["q", "qgap"], "L2q": ["q"], "lnk": ["lnk"]}
def X(d, cols):
    x = pd.DataFrame({k: d[k] for k in ("L", "C", "N", "taxiL", "artL", "pL", "v41", "taxiC", "pC", "artC") + (("q", "qgap") if "q" in d else ()) + (("lnk",) if "lnk" in d else ())})
    x["LmC"] = d["L"] - d["C"]; x["NmL"] = d["N"] - d["L"]; x["TmT"] = d["taxiL"] - d["taxiC"]; return x[cols]
def prm(seed): return dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, lambda_l2=50.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed, num_threads=NT)
def corr(cols, seed):
    out = {}
    for t in ALL:
        tags = [x for x in (PAIRS if METAFIT == "pairs" else ALL) if x != t]
        b = lgb.train(prm(seed), lgb.Dataset(pd.concat([X(D[x], cols) for x in tags]), np.concatenate([D[x]["y"] - D[x]["v41"] for x in tags])), 300)
        out[t] = cf(D[t]["v41"] + np.clip(b.predict(X(D[t], cols)), -CLIP_LO, CLIP_HI), D[t]["mvt_lobt"])
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
