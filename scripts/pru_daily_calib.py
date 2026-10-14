#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fits the soft daily PRU calibration (B), plus an optional monthly taxi-out add-on (A), as a final stage on a
fixed-weight pool OOF, nested by month, and optionally applies the 2026 production shift.
Inputs: --pool (fold-vector prefixes) and --w (matching weights), data/features_v9.parquet, env PRU_DIR (PRU csv
files); optionally --prod (a production parquet to calibrate).
Outputs: --out fold vectors <out>P/B/BA_<fold>.npy (6 folds, [ID, pred]); with --prod, --prod-out parquet.

The (B) soft daily calibration as a final stage on a fixed-weight pool OOF, nested by month,
plus the optional (A) monthly taxi-out add-on, plus the 2026 production shift.
usage: pru_daily_calib.py --pool P1,P2,P3 --w w1,w2,w3 --out <fold prefix> [--ref oof_build_poolc] [--prod BASE.parquet --prod-out OUT.parquet]
Fold outputs: <out>B_<fold>.npy (B) and <out>BA_<fold>.npy (B then A), 6 folds, [ID, pred]."""
import os, sys, glob, argparse, numpy as np, polars as pl
from scipy.stats import norm
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, f"{ROOT}/scripts")
import pru_lib
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pool", required=True, help="comma list of pool OOF fold-vector prefixes")
ap.add_argument("--w", required=True, help="comma list of pool weights, matching --pool, summing to 1")
ap.add_argument("--out", required=True, help="output fold-vector prefix")
ap.add_argument("--ref", default="", help="regression-check these fold vectors against the pool (optional)")
ap.add_argument("--prod", default="", help="2026 production BASE.parquet to also calibrate (optional)")
ap.add_argument("--prod-out", default="", help="output parquet for the calibrated production file")
ap.add_argument("--prod-arm", default="B", help="which calibration arm to write in production: B or BA")
ap.add_argument("--prod-um", default="median", choices=["median", "prod"], help="unmatched-row proxy used in day sums")
ap.add_argument("--um25", default="", help="unm_resid OOF npz to substitute for the 2025 unmatched-row proxy")
ap.add_argument("--um-src", default="", help="source parquet for --prod-um=prod (default: --prod)")
a = ap.parse_args()
FOLDS = ["months39", "months28", "months410", "janjul", "months612", "months511"]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]; GOODA = ["EGLL", "EHAM", "LEBL", "LEMD", "LIRF"]
PRU = os.environ.get("PRU_DIR") or f"{ROOT}/data/external/pru_20260925"   # PRU_DIR = a fetch_pru.py output dir
cols = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "is_rank", "unmatched", "y", "mvt_lobt"]
FF = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=cols).with_columns(
    ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8), mid=pl.col("MVT_ID_mvt").cast(pl.Float64),
    atot=pl.col("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64),
    std=pl.col("SCHED_TIME_UTC_mvt").dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64),
    sr=pl.col("STAND_mvt").is_not_null() & pl.col("RUNWAY_mvt").is_not_null()).drop("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt")
MED = pru_lib.medians()
def cf(v, lo): v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606)); return np.maximum(v, 60.0)
# ---------------- pool OOF (6 folds), fixed weights, cf
W = [float(x) for x in a.w.split(",")]; PR = a.pool.split(","); assert len(W) == len(PR) and abs(sum(W) - 1) < 1e-3
lo_map = FF.filter(pl.col("is_rank") == 0).select("mid", "mvt_lobt")
POOL = {}
for t in FOLDS:
    V = []
    for p in PR:
        z = np.load(sorted(glob.glob(f"{p}_{t}.npy"))[0]); o = np.argsort(z[0]); V.append((z[0][o], z[1][o]))
    ids = V[0][0]
    for i, _ in V[1:]: assert np.array_equal(i, ids)
    lo = pl.DataFrame({"mid": ids}).join(lo_map, on="mid", how="left", maintain_order="left")["mvt_lobt"].to_numpy().astype(float)
    POOL[t] = (ids, cf(sum(w * v for w, (_, v) in zip(W, V)), lo))
if a.ref:
    for t in FOLDS[:4]:
        z = np.load(f"{a.ref}_{t}.npy"); o = np.argsort(z[0]); assert np.array_equal(z[0][o], POOL[t][0])
        print(f"REGR pool vs {a.ref} {t}: max|d| {np.abs(z[1][o] - POOL[t][1]).max():.2e}", flush=True)
PO = pl.concat([pl.DataFrame({"mid": POOL[t][0], "p": POOL[t][1], "fold": t}) for t in FOLDS]); assert PO["mid"].n_unique() == PO.height
# ---------------- day tables
def rdB(y): return pl.read_csv(f"{PRU}/all_pre_departure_delays_{y}.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APTS)).select(
    ADEP_mvt="APT_ICAO", day=pl.col("FLT_DATE").str.slice(0, 10).str.to_date("%Y-%m-%d"), P=pl.col("DLY_ALL_PRE_2").cast(pl.Float64, strict=False))
BT = pl.concat([rdB(2025), rdB(2026)])
def rdA(y): return pl.read_csv(f"{PRU}/taxi_out_additional_time_{y}.csv", infer_schema_length=0).filter(pl.col("APT_ICAO").is_in(APTS)).select(
    ADEP_mvt="APT_ICAO", month=pl.col("MONTH_NUM").cast(pl.Int32),
    pru=(pl.col("TOTAL_REF_TIME_MIN").cast(pl.Float64) + pl.col("TOTAL_ADD_TIME_MIN").cast(pl.Float64)) / pl.col("TOTAL_REF_NB_FL").cast(pl.Float64) * 60)
def side(rank, pred):
    """all departures of the side with pz, x, day, month-rule flag; pred = frame(mid, p) for rows with a base prediction"""
    F = FF.filter(pl.col("is_rank") == rank).join(pred, on="mid", how="left")
    F = F.with_columns(scored=pl.col("p").is_not_null() & (pl.col("unmatched") == 0),
                       pz=pl.when(pl.col("p").is_not_null() & (pl.col("unmatched") == 0)).then(pl.col("p")).otherwise(pl.col("ADEP_mvt").replace_strict(MED, return_dtype=pl.Float64)))
    F = F.with_columns(x=pl.col("atot") - pl.col("std") - pl.col("pz"), day=pl.from_epoch((pl.col("atot") - pl.col("pz")).cast(pl.Int64), time_unit="s").dt.date(),
                       am=pl.from_epoch(pl.col("atot").cast(pl.Int64), time_unit="s").dt.strftime("%Y-%m"))
    F = F.with_columns(ok=pl.col("day").dt.strftime("%Y-%m") == pl.col("am"), month=pl.col("day").dt.month().cast(pl.Int32))
    return F.join(BT, on=["ADEP_mvt", "day"], how="left").sort("ADEP_mvt", "day")
def repz(G, U):
    """§2f: replace pz on the rows in U (mid, pu) and recompute x/day/month rule/aggregate"""
    G = G.join(U, on="mid", how="left", maintain_order="left").with_columns(pz=pl.when(pl.col("pu").is_not_null() & pl.col("pu").is_finite()).then(pl.col("pu")).otherwise(pl.col("pz"))).drop("pu")
    G = G.with_columns(x=pl.col("atot") - pl.col("std") - pl.col("pz"), day=pl.from_epoch((pl.col("atot") - pl.col("pz")).cast(pl.Int64), time_unit="s").dt.date())
    return G.with_columns(ok=pl.col("day").dt.strftime("%Y-%m") == pl.col("am"), month=pl.col("day").dt.month().cast(pl.Int32)).drop("P").join(BT, on=["ADEP_mvt", "day"], how="left").sort("ADEP_mvt", "day")
def solve(F, sig):
    """per airport-day delta (soft, w = scored predicted-delayed), rows with ok only; returns delta per row (0 where not solvable/not ok)"""
    G = F.with_row_index("ri").filter(pl.col("ok"))
    if G.height == 0: return np.zeros(F.height), 0, 0   # a month without 2025 fold rows (dry run): zero deltas
    ad = G["ADEP_mvt"].to_numpy(); dd = G["day"].to_numpy(); x = G["x"].to_numpy(); w = (G["scored"].to_numpy() & (x > 0)).astype(float); T = G["P"].to_numpy()
    brk = np.r_[0, np.flatnonzero((ad[1:] != ad[:-1]) | (dd[1:] != dd[:-1])) + 1, len(G)]; out = np.zeros(len(G)); nd = 0; nb = 0
    for g in range(len(brk) - 1):
        s, e = brk[g], brk[g + 1]
        if not np.isfinite(T[s]): continue
        xx, ww, sg, tg = x[s:e], w[s:e], sig[ad[s]], T[s]
        def f(dl):
            mu = xx - dl * ww; z = mu / sg; return (mu * norm.cdf(z) + sg * norm.pdf(z)).sum() / 60 - tg
        L, H = -900.0, 900.0; fL, fH = f(L), f(H)
        if fL < 0 or fH > 0: v = L if fL < 0 else H; nb += 1
        else:
            for _ in range(40):
                M = 0.5 * (L + H)
                if f(M) > 0: L = M
                else: H = M
            v = 0.5 * (L + H)
        out[s:e] = np.clip(v, -600, 600); nd += 1
    r = np.zeros(F.height); r[G["ri"].to_numpy()] = out; return r, nd, nb
def mad(r): return 1.4826 * np.median(np.abs(r - np.median(r)))
def kfit(r, u):
    den = (u ** 2).sum(); return float(np.clip(-(r * u).sum() / den, 0, 1.5)) if den > 0 else 0.0
# ================= 2025 nested
F = side(0, PO.select("mid", "p"))
if a.um25:
    _z = np.load(a.um25, allow_pickle=True); _m = _z["fold"] != "rank"
    F = repz(F, pl.DataFrame({"mid": _z["ids"][_m].astype(float), "pu": _z["final"][_m].astype(float)}).join(FF.filter((pl.col("is_rank") == 0) & (pl.col("unmatched") == 1)).select("mid"), on="mid", how="semi"))
    print(f"um25: unmatched 2025 rows with the unm_resid OOF proxy from {a.um25}", flush=True)
S = F.filter(pl.col("scored")); print(f"2025: departures {F.height:,}, scored {S.height:,}, month-rule ok {F['ok'].mean():.4f}, days with P {F.filter(pl.col('P').is_not_null())['day'].n_unique()}", flush=True)
apt = F["ADEP_mvt"].to_numpy(); mon = F["month"].to_numpy(); x = F["x"].to_numpy(); sc = F["scored"].to_numpy(); y = F["y"].to_numpy().astype(float); p = F["pz"].to_numpy()
r = p - y; delta = np.zeros(F.height); q = np.zeros(F.height); SIG = {}
for m in range(1, 13):
    tr = sc & (mon != m); sig = {A: mad(r[tr & (apt == A)]) for A in APTS}; SIG[m] = sig
    Fm = F.with_row_index("gi").filter(pl.col("month") == m); d_, nd, nb = solve(Fm, sig); gi = Fm["gi"].to_numpy(); delta[gi] = d_
    q[gi] = norm.cdf(x[gi] / np.array([sig[A] for A in apt[gi]]))
    print(f"  month {m:2d}: sigma " + " ".join(f"{A[-2:]}{sig[A]:.0f}" for A in APTS) + f" | days solved {nd} bound-hit {nb}", flush=True)
u = delta * q; pdl = x > 0; new = p.copy(); KS = {}
for m in range(1, 13):
    te = sc & (mon == m); tr = sc & (mon != m)
    for A in APTS:
        for g in (True, False):
            sel = tr & (apt == A) & (pdl == g); k = kfit(r[sel], u[sel]); KS[(m, A, g)] = k
            idx = te & (apt == A) & (pdl == g); new[idx] = p[idx] + k * u[idx]
lo = F["mvt_lobt"].to_numpy().astype(float); newB = np.where(sc, cf(new, lo), p)
def info(nm, a_, b_):
    d = ((a_ - y) ** 2 - (b_ - y) ** 2)[sc]; mm = mon[sc]; aa = apt[sc]
    print(f"INFO {nm}: 12-month pooled dMSE {d.mean():+.1f} | Jan+Jul {d[np.isin(mm, [1, 7])].mean():+.1f} | w/o Feb {d[mm != 2].mean():+.1f} | months<0 {sum(d[mm == k].mean() < 0 for k in range(1, 13))}/12", flush=True)
    print("     by month: " + " ".join(f"{d[mm == k].mean():+.0f}" for k in range(1, 13)), flush=True)
    print("     by airport (contribution): " + " ".join(f"{A} {d[aa == A].sum() / len(d):+.1f}" for A in APTS), flush=True)
info("B - pool", newB, p)
print("  k (nested, mean over months) per apt [delayed/not]: " + " ".join(f"{A} {np.mean([KS[(m, A, True)] for m in range(1, 13)]):.2f}/{np.mean([KS[(m, A, False)] for m in range(1, 13)]):.2f}" for A in APTS), flush=True)
# ---------------- (A) add-on after (B): 5 airports, gap from adjacent calendar months, one pooled k from other months
A25 = rdA(2025); popl = sc & F["sr"].to_numpy() & (y > 0) & (y <= 7200)
Fa = pl.DataFrame({"ADEP_mvt": apt, "month": mon, "y": y, "pB": newB, "pl": popl, "pp": sc & F["sr"].to_numpy() & (newB <= 7200)})
LM = Fa.filter(pl.col("pl")).group_by("ADEP_mvt", "month").agg(lm=pl.col("y").mean()); PM = Fa.filter(pl.col("pp")).group_by("ADEP_mvt", "month").agg(pm=pl.col("pB").mean())
AM = A25.join(LM, on=["ADEP_mvt", "month"], how="left").join(PM, on=["ADEP_mvt", "month"], how="left").with_columns(gap=pl.col("lm") - pl.col("pru"))
gmap = {(r_["ADEP_mvt"], r_["month"]): r_ for r_ in AM.iter_rows(named=True)}
def dA(A, m, prm, pm):
    adj = [((m - 2) % 12) + 1, (m % 12) + 1]; g = [gmap[(A, k)]["gap"] for k in adj if (A, k) in gmap and gmap[(A, k)]["gap"] is not None]
    return prm + np.mean(g) - pm if g and prm is not None and pm is not None else 0.0
dAr = np.zeros(F.height)
for A in GOODA:
    for m in range(1, 13):
        if (A, m) in gmap: idx = (apt == A) & (mon == m); dAr[idx] = dA(A, m, gmap[(A, m)]["pru"], gmap[(A, m)]["pm"])
rB = newB - y; newBA = newB.copy(); kA = []
for m in range(1, 13):
    tr = sc & (mon != m) & np.isin(apt, GOODA); k = kfit(rB[tr], dAr[tr]); kA.append(k); idx = sc & (mon == m) & np.isin(apt, GOODA); newBA[idx] = newB[idx] + k * dAr[idx]
newBA = np.where(sc, cf(newBA, lo), p)
info("BA - B", newBA, newB); print(f"  kA per held-out month: {np.round(kA, 2).tolist()}", flush=True)
# ---------------- write fold vectors (same rows/order as the pool vectors)
Z = pl.DataFrame({"mid": F["mid"], "B": newB, "BA": newBA})
for t in FOLDS:
    ids = POOL[t][0]; J = pl.DataFrame({"mid": ids}).join(Z, on="mid", how="left", maintain_order="left"); assert J["B"].null_count() == 0
    np.save(f"{a.out}P_{t}.npy", np.vstack([ids, POOL[t][1]])); np.save(f"{a.out}B_{t}.npy", np.vstack([ids, J["B"].to_numpy()])); np.save(f"{a.out}BA_{t}.npy", np.vstack([ids, J["BA"].to_numpy()]))
print("wrote", a.out + "{P,B,BA}_<fold>.npy", flush=True)
# ================= 2026 production
if a.prod:
    sigA = {A: mad(r[sc & (apt == A)]) for A in APTS}
    K = {}
    # delta on 2025 with all-month sigma (production-form u for the k fit)
    qa = norm.cdf(x / np.array([sigA[A] for A in apt])); dfull = np.zeros(F.height)
    for m in range(1, 13):
        Fm = F.with_row_index("gi").filter(pl.col("month") == m); d_, _, _ = solve(Fm, sigA); dfull[Fm["gi"].to_numpy()] = d_
    ua = dfull * qa
    for A in APTS:
        for g in (True, False):
            sel = sc & (apt == A) & (pdl == g); K[(A, g)] = kfit(r[sel], ua[sel])
    print("PROD sigma: " + " ".join(f"{A} {sigA[A]:.0f}" for A in APTS) + "\nPROD k [delayed/not]: " + " ".join(f"{A} {K[(A, True)]:.2f}/{K[(A, False)]:.2f}" for A in APTS), flush=True)
    base = pl.read_parquet(a.prod); col = "TAXITIME_SEC_mvt"
    Rk = FF.filter(pl.col("is_rank") == 1).select("mid", "unmatched")
    bj = base.with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64)).join(Rk, on="mid", how="left", maintain_order="left")
    assert bj["unmatched"].null_count() == 0
    pr = bj.filter(pl.col("unmatched") == 0).select("mid", p=pl.col(col).cast(pl.Float64))
    G = side(1, pr)
    if a.prod_um == "prod":   # POST-HOC variant: unmatched ranking rows enter the day sums with the production file's value, not the median proxy
        _src = pl.read_parquet(a.um_src).with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64)) if a.um_src else bj
        um_ = _src.join(Rk.filter(pl.col("unmatched") == 1).select("mid"), on="mid", how="semi").select("mid", pu=pl.col(col).cast(pl.Float64)); print(f"prod-um source: {a.um_src or a.prod}, rows {um_.height}", flush=True)
        G = G.join(um_, on="mid", how="left", maintain_order="left").with_columns(pz=pl.when(pl.col("pu").is_not_null()).then(pl.col("pu")).otherwise(pl.col("pz"))).drop("pu")
        G = G.with_columns(x=pl.col("atot") - pl.col("std") - pl.col("pz"), day=pl.from_epoch((pl.col("atot") - pl.col("pz")).cast(pl.Int64), time_unit="s").dt.date())
        G = G.with_columns(ok=pl.col("day").dt.strftime("%Y-%m") == pl.col("am"), month=pl.col("day").dt.month().cast(pl.Int32)).drop("P").join(BT, on=["ADEP_mvt", "day"], how="left").sort("ADEP_mvt", "day")
    ga = G["ADEP_mvt"].to_numpy(); gx = G["x"].to_numpy(); gsc = G["scored"].to_numpy()
    gd, nd, nb = solve(G, sigA); gq = norm.cdf(gx / np.array([sigA[A] for A in ga]))
    kk = np.array([K[(A_, bool(v))] for A_, v in zip(ga, gx > 0)]); sh = np.where(gsc & G["ok"].to_numpy(), kk * gd * gq, 0.0)
    gp = G["pz"].to_numpy(); glo = G["mvt_lobt"].to_numpy().astype(float); gnew = np.where(gsc, cf(gp + sh, glo), gp)
    print(f"PROD 2026: departures {G.height:,}, scored {gsc.sum():,}, month-rule ok {G['ok'].mean():.4f}, days solved {nd}, bound-hit {nb}; shift mean {sh[gsc].mean():+.2f} rms {np.sqrt((sh[gsc]**2).mean()):.2f}", flush=True)
    for A in APTS:
        s_ = gsc & (ga == A); dd_ = gd[s_]; print(f"   {A}: shift mean {sh[s_].mean():+.2f} rms {np.sqrt((sh[s_]**2).mean()):.1f} | daily delta mean {dd_.mean():+.1f}", flush=True)
    if a.prod_arm == "BA":
        A26 = rdA(2026); Gm = G["month"].to_numpy(); gsr = G["sr"].to_numpy(); dA26 = np.zeros(G.height)
        # k for A in production: pooled over all 12 months on the nested-B 2025 vectors (same u as the fold form)
        kAp = kfit(rB[sc & np.isin(apt, GOODA)], dAr[sc & np.isin(apt, GOODA)])
        for A in GOODA:
            for m in (1, 7):
                s_ = (ga == A) & (Gm == m) & gsc & gsr & (gnew <= 7200); rr = A26.filter((pl.col("ADEP_mvt") == A) & (pl.col("month") == m))
                if s_.sum() and rr.height: dA26[(ga == A) & (Gm == m)] = dA(A, m, rr["pru"][0], float(gnew[s_].mean()))
        gnew = np.where(gsc, cf(gnew + kAp * dA26, glo), gp); print(f"PROD A: kA {kAp:.2f}, shift by apt " + " ".join(f"{A} {kAp*dA26[(ga==A)&gsc].mean():+.1f}" for A in GOODA), flush=True)
    NW = pl.DataFrame({"mid": G["mid"], "new": gnew, "sc": gsc})
    J = bj.join(NW, on="mid", how="left", maintain_order="left"); assert J["new"].null_count() == 0
    v0 = base[col].to_numpy(); um = J["unmatched"].to_numpy() == 1
    vn = np.where(um, v0, np.rint(J["new"].to_numpy())).astype(v0.dtype)
    out = base.with_columns(pl.Series(col, vn).cast(base[col].dtype)); assert out["MVT_ID_mvt"].equals(base["MVT_ID_mvt"])
    out.write_parquet(a.prod_out); d = vn.astype(float) - v0.astype(float)
    print(f"PROD wrote {a.prod_out}: vs base rows changed {(d != 0).sum():,}, RMS {np.sqrt((d**2).mean()):.2f}, mean {d.mean():+.3f}, max |d| {np.abs(d).max():.0f}; unmatched identical {bool((vn[um] == v0[um]).all())}", flush=True)
