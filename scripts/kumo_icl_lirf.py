#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""F4: runs Kumo Tabular in-context residual arms on (A) the frozen unm_LIRF rows, keeping the incumbent's per-row
p / taxi as inputs, and (B) the LTFM unm_other rows. Inputs: positional subcommand (PLUMB | ARM [ARM ...] | SUMMARY |
BUILD | SEALCHECK), env kumo_icl_lirf_frz_base (required base line for the arms and BUILD) and the DEV/SIZE/NEST/...
knobs below. Output: BUILD writes stages/kumo_icl_lirf_{tag}.parquet (kumo_icl_lirf_frz_cand); SEALCHECK writes the
sealed columns to a file; other subcommands only print gate numbers (computed on UNSEALED rows only).

usage: kumo_icl_lirf.py PLUMB | ARM [ARM ...] | SUMMARY | BUILD | SEALCHECK      ARM in F4 F4p T4 T4p F5 F5p
env: DEV (cuda|cpu), SIZE (medium), NEST (4), SEED (0), NT (lgb threads), PRC_THREADS, OUT (scratchpad/xtfm2/frz)
     kumo_icl_lirf_frz_base (base line file, required for the arms and BUILD), kumo_icl_lirf_frz_base_md5 (md5 prefix; default empty = no check),
     kumo_icl_lirf_frz_cand (output name pattern of BUILD, {tag} = arm; default stages/kumo_icl_lirf_{tag}.parquet),
     P2_MODE=new skips all md5 checks of the cached input tables and the base (they are period-specific;  fix 4)
Run with $KUMO_PY for F4/F4p/F5/F5p; .venv for T4/T4p/SUMMARY/BUILD/SEALCHECK."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, time, json, types, hashlib, shutil, subprocess, numpy as np, polars as pl

ROOT = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + "")
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scratchpad/xur")
t0 = time.time()
try:
    import lightgbm
except ImportError:
    sys.modules["lightgbm"] = types.ModuleType("lightgbm")
import unm_link_lever as X
from unm_resid_lib import FOLDS, GATE, RN

OUT = os.environ.get("OUT", "scratchpad/xtfm2/frz")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("DEV", "cuda")
SIZE = os.environ.get("SIZE", "medium")
NEST = int(os.environ.get("NEST", "4"))
SEED = int(os.environ.get("SEED", "0"))
NT = int(os.environ.get("NT", "4"))
BASE = os.environ.get("kumo_icl_lirf_frz_base", "stages/unmatched_link_a1.parquet")
BASE_MD5 = os.environ.get("kumo_icl_lirf_frz_base_md5", "")
CAND = os.environ.get("kumo_icl_lirf_frz_cand", "stages/kumo_icl_lirf_{tag}.parquet")
ids, y, isr, mon, apt, ms, ref, fo, cls, lk = (
    X.ids,
    X.y,
    X.isr,
    X.mon,
    X.apt,
    X.ms,
    X.ref,
    X.fo,
    X.cls,
    X.lk.astype(float),
)
LAB_ALL = (
    (~isr) & np.isfinite(y) & np.isfinite(ref) & (fo != "none")
)  # all unmatched classes (X.lab covers unm_other only)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


for f, h in (
    ("scratchpad/xhr/unm2025b.parquet", "95b296db"),
    ("scratchpad/xhr/unm_lirf_janjul_frozen.parquet", "b7ddf767"),
    ("scratchpad/xhr/unm2026.parquet", "79c633ab"),
    ("scratchpad/xhr2/fzr_rank.parquet", "6462a426"),
    (BASE, BASE_MD5),
):
    if h and os.environ.get("P2_MODE") != "new":
        assert md5(f).startswith(h), f"md5 {f}"
# ---------------- sealed mask (join on airport x date), UNS scoring set
SD = pl.read_parquet("data/sealed_days.parquet").select("ADEP_mvt", "day", "sealed")
Dk = (
    X.U.select(pl.col("ADEP_mvt"), pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    .with_row_index("i")
    .join(SD, on=["ADEP_mvt", "day"], how="left")
    .sort("i")
)
sealed = Dk["sealed"].fill_null(False).to_numpy().astype(bool)
assert int(Dk["sealed"].is_null().to_numpy()[LAB_ALL].sum()) == 0
UNS = LAB_ALL & ~sealed


# ---------------- frozen inputs
def joined(path, cols):
    D = (
        pl.read_parquet(path)
        .select(["MVT_ID_mvt"] + cols)
        .with_columns(pl.col("MVT_ID_mvt").cast(X.U.schema["MVT_ID_mvt"]))
    )
    return X.U.select("MVT_ID_mvt").join(D, on="MVT_ID_mvt", how="left", maintain_order="left")


P25 = joined("scratchpad/xhr/unm2025b.parquet", ["px", "px_p", "px_taxi", "cross", "regime"])
P26 = joined("scratchpad/xhr/unm2026.parquet", ["cross"])
FZR = joined("scratchpad/xhr2/fzr_rank.parquet", ["fzr_p", "fzr_taxi", "v62"])
TRU = joined("scratchpad/xhr/unm_lirf_janjul_frozen.parquet", ["fz", "fz_p"])
B98 = (
    X.U.select("MVT_ID_mvt")
    .join(
        pl.read_parquet(BASE)
        .rename({"TAXITIME_SEC_mvt": "v98"})
        .with_columns(pl.col("MVT_ID_mvt").cast(X.U.schema["MVT_ID_mvt"])),
        on="MVT_ID_mvt",
        how="left",
        maintain_order="left",
    )["v98"]
    .to_numpy()
    .astype(float)
)
cross = np.where(isr, P26["cross"].to_numpy().astype(float), P25["cross"].to_numpy().astype(float))
cross = np.nan_to_num(cross, nan=0.0)
px, px_p, px_taxi = (P25[c].to_numpy().astype(float) for c in ("px", "px_p", "px_taxi"))
fzr_p, fzr_taxi, v62 = (FZR[c].to_numpy().astype(float) for c in ("fzr_p", "fzr_taxi", "v62"))
fz, fz_p = TRU["fz"].to_numpy().astype(float), TRU["fz_p"].to_numpy().astype(float)
LIRF = cls == "unm_LIRF"
LT = (cls == "unm_other") & (apt == "LTFM")
# incumbent frames for scope A: proxy (2025 rows), TRUE frozen (janjul rows), 2026 (v98 = v62 frozen)
inc_px = px
p_px = px_p
taxi_px = px_taxi
has_true = LIRF & ~isr & np.isfinite(fz)
taxi_fz = np.where(fz_p <= 0.9, (fz - fz_p * ms) / np.maximum(1 - fz_p, 1e-6), 900.0)
inc_26 = B98
p_26 = fzr_p
taxi_26 = fzr_taxi
band = np.where(ms < 3600, 0, np.where(ms < 7200, 1, np.where(ms < 14400, 2, np.where(ms < 28800, 3, 4)))).astype(float)
COMMON_A = [
    ms,
    cross,
    band,
    X.col("hour"),
    X.col("dow"),
    lk,
    X.col("stand_since_arr_inblock"),
    X.col("stand_since_prev_dep"),
    X.col("arr_block_delay"),
    X.col("u_n"),
    X.col("u_ms_med"),
    X.col("u_ms_rank"),
    X.col("nbr_apt_m60_med"),
]


def XA(inc, p, taxi):
    return np.column_stack([inc, p, taxi, inc - ms] + COMMON_A).astype(np.float32)  # 17 cols


frz_arm_px = XA(inc_px, p_px, taxi_px)
frz_arm_fz = XA(fz, fz_p, taxi_fz)
frz_arm_26 = XA(inc_26, p_26, taxi_26)
# scope B: LTFM
C1_DAY = [
    "u_n",
    "u_n1h",
    "u_n2h",
    "u_ms_med",
    "u_ms_rank",
    "nbr_apt_m60_med",
    "nbr_apt_m60_max",
    "nbr_apt_m30_med",
    "arrx_off_day_med",
    "dreg_day_med",
    "dreg_day_late2h",
    "dreg_day_unm",
    "dreg_unm60",
]
thy = (X.U["cs_pre"] == "THY").cast(pl.Float64).fill_null(0.0).to_numpy()
XB = np.column_stack(
    [
        ref,
        ms,
        X.col("hour"),
        X.col("dow"),
        thy,
        X.col("vfr"),
        X.col("stand_since_arr_inblock"),
        X.col("stand_since_prev_dep"),
        X.col("arr_block_delay"),
    ]
    + [X.col(c) for c in C1_DAY]
).astype(np.float32)
# ---------------- weights
SLOT = np.where(np.isin(mon, [11, 12, 1, 2, 3, 4]), "W", "S")


def w_class(
    rows,
):  # FROZEN_WEIGHT_CLASS: (slot, lk) cells, w = n26/n25 (unsealed 2025 rows), fallback slot if n25 < 20; then per-fold rescale to the full (all-rows) weight sum
    w = np.zeros(len(y))
    r25 = rows & LAB_ALL
    r26 = rows & isr
    for s in ("W", "S"):
        for l in (0.0, 1.0):
            k25 = r25 & (SLOT == s) & (lk == l)
            n26 = int((r26 & (SLOT == s) & (lk == l)).sum())
            if k25.sum() >= 20:
                w[k25] = n26 / k25.sum()
            else:
                ks = r25 & (SLOT == s)
                w[k25] = int((r26 & (SLOT == s)).sum()) / max(int(ks.sum()), 1)
    for f in FOLDS:
        ka = r25 & (fo == f)
        ku = ka & UNS
        if ku.any():
            w[ku] *= w[ka].sum() / w[ku].sum()
    return w


W_A = w_class(LIRF)
W_B = X.W.copy()  # scope B uses UNM_LINK2's unm_other MW weights (rescaled to UNS as in kumo_icl_unmatched)
for f in FOLDS:
    ka = LAB_ALL & (fo == f)
    ku = UNS & (fo == f)
    if ku.any():
        W_B[ku] *= X.W[ka].sum() / X.W[ku].sum()


def w_u1(
    rows_true,
):  # frozen_weight_u1 U1: cells month x lk, w = n_rank / n_fold (unsealed janjul TRUE rows), fallback month if n_fold < 20
    w = np.zeros(len(y))
    r26 = LIRF & isr
    for mo in (1, 7):
        for l in (0.0, 1.0):
            kf = rows_true & UNS & (mon == mo) & (lk == l)
            nr = int((r26 & (mon == mo) & (lk == l)).sum())
            if kf.sum() >= 20:
                w[kf] = nr / kf.sum()
            else:
                w[kf] = int((r26 & (mon == mo)).sum()) / max(int((rows_true & UNS & (mon == mo)).sum()), 1)
    return w


# ---------------- learners
R_A = np.clip(y - inc_px, -3600, 3600)
R_B = np.clip(y - ref, -3600, 3600)


def lomo_months(rows):
    for m in list(range(1, 13)) + ["rank"]:
        te = (rows & ~isr & (mon == m)) if m != "rank" else (rows & isr)
        tr = rows & LAB_ALL & ((mon != m) if m != "rank" else True)
        if te.sum():
            yield m, tr, te


_model = None


def kumo_fit_predict(xc, yc, xq, chunk=4096):
    global _model
    import torch, sdm

    if _model is None:
        os.environ.setdefault("HF_HOME", f"{ROOT}/artifacts/xkumo/hf")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        torch.set_num_threads(int(os.environ.get("PRC_THREADS", "2")))
        _model = sdm.models.KumoTabular(task="regression", size=SIZE, device=DEV)
        print(f"kumo {SIZE} on {DEV} nest {NEST} seed {SEED}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    _model.fit(
        torch.as_tensor(xc, device=DEV),
        torch.as_tensor(yc.astype(np.float32)[:, None], device=DEV),
        num_estimators=NEST,
        generator=gen,
    )
    out = []
    for i in range(0, len(xq), chunk):
        p = _model.predict(torch.as_tensor(xq[i : i + chunk], device=DEV))
        q = p.numerical if hasattr(p, "numerical") else p
        out.append(q.float().mean(-1).cpu().numpy())
    _model.clear()
    return np.concatenate(out)


def lgb_fit_predict(xc, yc, xq):
    import lightgbm as lgb

    PS = dict(
        objective="l2",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=20.0,
        verbose=-1,
        num_threads=NT,
    )
    return np.mean([lgb.train(dict(PS, seed=s), lgb.Dataset(xc, yc), 300).predict(xq) for s in (0, 1, 2)], axis=0)


def run_lomo(learner, Xc_all, r, rows, Xq_all, Xq_true=None, true_rows=None, perm=False, tag=""):
    """g on the proxy/ref query rows of every month + ranking rows; g_true on the TRUE frozen janjul rows (same context as their month)."""
    g = np.full(len(y), np.nan)
    g_true = np.full(len(y), np.nan)
    rng = np.random.default_rng(SEED)
    for m, tr, te in lomo_months(rows):
        ti = np.where(tr)[0]
        yc = r[ti].copy()
        if perm:  # placebo: context targets permuted within month (no-information null)
            for mo in np.unique(mon[ti]):
                k = np.where(mon[ti] == mo)[0]
                yc[k] = yc[rng.permutation(k)]
        qi = np.where(te)[0]
        xq = Xq_all[qi]
        nq = len(qi)
        if Xq_true is not None and m in (1, 7):
            ti2 = np.where(true_rows & (mon == m))[0]
            xq = np.vstack([xq, Xq_true[ti2]])
        t1 = time.time()
        out = learner(Xc_all[ti], yc, xq)
        assert np.isfinite(out).all(), (tag, m)
        g[qi] = out[:nq]
        if Xq_true is not None and m in (1, 7):
            g_true[ti2] = out[nq:]
        print(
            f"  {tag} month {m}: ctx {len(ti):,} query {len(xq):,} mean(g) {out.mean():+.1f} sd {out.std():.1f} {time.time()-t1:.0f}s",
            flush=True,
        )
    return g, g_true


def finish(inc, g, rows, k, G):
    cand = inc.copy()
    ok = rows & np.isfinite(g)
    cand[ok] = inc[ok] + k * np.clip(g[ok], -G, G)
    cand = np.maximum(cand, 60.0)
    big = rows & (np.abs(cand - inc) > 7200)
    cand[big] = inc[big]
    return cand


# ---------------- scoring
def mw_table(cand, base, rows, W, tag):
    """Gate G1-G5 of scripts/unm_link_lever.py on UNS rows: per fold raw/row, raw w/o top-3, MW, MW w/o top-3; months < 0."""
    d = (cand - y) ** 2 - (base - y) ** 2
    res = {}
    print(f"\n== {tag}: Δ = arm − incumbent, UNS rows (MW board s² = Σ w·Δ / {RN})")
    print(f"{'fold':10s} {'n':>5s} {'chg':>5s} {'raw/row':>9s} {'raw w/o3':>9s} {'MW':>8s} {'MW w/o3':>8s}")
    for f in FOLDS:
        k = UNS & rows & (fo == f)
        dd = d[k]
        o = np.argsort(-np.abs(dd))
        wd = W[k] * dd
        o2 = np.argsort(-np.abs(wd))
        res[f] = (
            dd.mean(),
            np.delete(dd, o[:3]).mean(),
            wd.sum() / RN,
            np.delete(wd, o2[:3]).sum() / RN,
            int((cand[k] != base[k]).sum()),
        )
        print(
            f"{f:10s} {k.sum():5d} {res[f][4]:5d} {res[f][0]:+9,.0f} {res[f][1]:+9,.0f} {res[f][2]:+8.1f} {res[f][3]:+8.1f}"
        )
    mm = [(W[UNS & rows & (mon == m)] * d[UNS & rows & (mon == m)]).sum() / RN for m in range(1, 13)]
    print("per month MW:", " ".join(f"{m}:{v:+.1f}" for m, v in zip(range(1, 13), mm)))
    Gf = [res[f] for f in GATE]
    g1 = sum(r[2] < 0 for r in Gf) >= 3
    g2 = res["janjul"][2] <= 0
    g3 = sum(r[1] < 0 for r in Gf) >= 3
    g4 = sum(r[3] < 0 for r in Gf) >= 3
    g5 = sum(v < 0 for v in mm) >= 7
    mean4 = float(np.mean([r[2] for r in Gf]))
    kr = isr & rows & (cand != base)
    print(
        f"GATE {tag}: G1 {g1} G2 {g2} G3 {g3} G4 {g4} G5 {g5} ({sum(v < 0 for v in mm)}/12) | 4-fold mean MW {mean4:+.1f} | ranking rows changed {kr.sum()}, mean {np.mean((cand-base)[kr]) if kr.any() else 0:+.1f} rms {np.sqrt(np.mean((cand-base)[kr]**2)) if kr.any() else 0:.1f}"
    )
    return dict(
        per={f: res[f][2] for f in GATE},
        mean4=mean4,
        g15=bool(all([g1, g2, g3, g4, g5])),
        months_neg=int(sum(v < 0 for v in mm)),
    )


def u_table(cand_true, rows_true, tag, topk=20):
    """frozen_weight_u1 gate U on the TRUE frozen janjul rows (UNS): U1 weighted, boot CI, U2 Jan/Jul, U3 w/o top-3, U4 dark, U5 top-k tails."""
    k = rows_true & UNS
    d = (cand_true[k] - y[k]) ** 2 - (fz[k] - y[k]) ** 2
    w = w_u1(rows_true)[k]
    m = mon[k]
    l = lk[k]
    wd = w * d
    rng = np.random.default_rng(0)
    boots = np.array([wd[rng.integers(0, len(wd), len(wd))].sum() / RN for _ in range(2000)])
    o = np.argsort(-np.abs(d))
    ti = np.argsort(-np.abs(fz[k] - y[k]))[:topk]
    tc = np.argsort(-np.abs(cand_true[k] - y[k]))[:topk]
    mv = np.abs(cand_true[k] - fz[k]) >= 1
    u = dict(
        n=int(k.sum()),
        moved=int(mv.sum()),
        U1=wd.sum() / RN,
        boot95=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        raw=d.sum() / RN,
        U2_Jan=d[m == 1].sum() / RN,
        U2_Jul=d[m == 7].sum() / RN,
        U3=np.delete(d, o[:3]).sum() / RN,
        U4_dark=d[l == 0].sum() / RN,
        Jul_dark=d[(l == 0) & (m == 7)].sum() / RN,
        U5_inc=d[ti].sum() / RN,
        U5_cand=d[tc].sum() / RN,
        improved=float(np.mean(d[mv] < 0)) if mv.any() else float("nan"),
    )
    u["U_pass"] = bool(
        u["U1"] < 0
        and u["U2_Jan"] < 0
        and u["U2_Jul"] < 0
        and u["U3"] < 0
        and u["U4_dark"] < 0
        and u["U5_inc"] <= 0
        and u["U5_cand"] <= 0
    )
    print(
        f"  [TRUE frozen janjul {tag}] "
        + ", ".join(
            f"{a} {b:+.1f}" if isinstance(b, float) and a not in ("improved",) else f"{a} {b}" for a, b in u.items()
        )
    )
    return u


def plaus(cand, inc, rows, G, tag, base25=None):
    d = cand - inc
    res = {}
    for yr, s in (("2025", rows & UNS), ("2026", rows & isr)):
        chg = s & (np.abs(d) >= 1)
        q1 = float(np.mean(np.abs(d[chg]) >= 0.5 * G - 0.5)) if chg.any() else 0.0
        res[yr] = (int(chg.sum()), q1, float(np.sqrt(np.mean(d[s] ** 2))) if s.any() else 0.0)
        print(f"  {tag} {yr}: changed {res[yr][0]}, Q1 clip share {q1:.3f}, RMS move {res[yr][2]:.1f}")
    p = res["2026"][1] <= 0.40 and res["2026"][2] <= 2 * max(res["2025"][2], 1.0)
    print(f"  PLAUS {tag}: Q1 {res['2026'][1] <= 0.40} Q3 {res['2026'][2] <= 2 * max(res['2025'][2], 1.0)} -> {p}")
    return p


# ---------------- arms
def arm(name):
    if name in ("F4", "F4p", "T4", "T4p"):
        learner = kumo_fit_predict if name[0] == "F" else lgb_fit_predict
        g, gt = run_lomo(
            learner,
            frz_arm_px,
            R_A,
            LIRF & ~isr,
            frz_arm_px,
            Xq_true=frz_arm_fz,
            true_rows=has_true,
            perm=name.endswith("p"),
            tag=name,
        )  # 2025 months only; ranking below
        # proxy/2026 candidate: incumbent = px on 2025 rows, v98 on 2026 rows; TRUE candidate on janjul rows
        inc = np.where(isr, inc_26, inc_px)
        g26 = g.copy()
        # ranking query used frz_arm_px (proxy inputs) above -> redo the ranking rows with the frozen's own p/taxi inputs
        ti = np.where(LIRF & LAB_ALL)[0]
        yc = R_A[ti].copy()
        if name.endswith("p"):
            rng = np.random.default_rng(SEED)
            for mo in np.unique(mon[ti]):
                k = np.where(mon[ti] == mo)[0]
                yc[k] = yc[rng.permutation(k)]
        qi = np.where(LIRF & isr)[0]
        g26[qi] = learner(frz_arm_px[ti], yc, frz_arm_26[qi])
        print(
            f"  {name} ranking (frozen inputs): query {len(qi)} mean(g) {g26[qi].mean():+.1f} sd {g26[qi].std():.1f}",
            flush=True,
        )
        cand = finish(inc, g26, LIRF, 0.5, 1200.0)
        cand_true = finish(fz, gt, has_true, 0.5, 1200.0)
        return cand, g26, cand_true, gt
    if name in ("F5", "F5p"):
        Xq = XB
        if name == "F5p":
            Xq = np.column_stack([XB[:, :2], X.perm_within(XB[:, 2:], LT, np.random.default_rng(0))]).astype(np.float32)
        g, _ = run_lomo(kumo_fit_predict, Xq, R_B, LT, Xq, tag=name)
        return finish(ref, g, LT, 0.5, 600.0), g, None, None
    raise SystemExit(name)


SCOPE = {"F4": "A", "F4p": "A", "T4": "A", "T4p": "A", "F5": "B", "F5p": "B"}


def summary():
    R = {}
    for a in SCOPE:
        p = f"{OUT}/ARM_{a}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            R[a] = json.loads(str(z["res"]))
    print(f"\n######## KUMO_ICL_LIRF §7 SUMMARY (UNS) {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    for a, r in R.items():
        print(
            f"  {a:4s} LOMO 4-fold {r['mw']['mean4']:+7.1f} per fold "
            + " ".join(f"{v:+7.1f}" for v in r["mw"]["per"].values())
            + f" G1-G5 {r['mw']['g15']} months<0 {r['mw']['months_neg']}"
            + (
                f" | U1 {r['u']['U1']:+.1f} boot {tuple(round(x, 1) for x in r['u']['boot95'])} Jan {r['u']['U2_Jan']:+.1f} Jul {r['u']['U2_Jul']:+.1f} U3 {r['u']['U3']:+.1f} dark {r['u']['U4_dark']:+.1f} U5 {r['u']['U5_inc']:+.1f}/{r['u']['U5_cand']:+.1f} U_pass {r['u']['U_pass']}"
                if r.get("u")
                else ""
            )
            + f" | plaus {r['plaus']}"
        )
    verdict = {}
    for c, p_, K in (("F4", "F4p", 2), ("T4", "T4p", 2), ("F5", "F5p", 1)):
        if c not in R:
            continue
        r = R[c]
        m4 = r["mw"]["mean4"]
        bar = -(5.0 + {1: 0.0, 2: 0.56}[K] * 5.0)
        g6 = (p_ in R) and (R[p_]["mw"]["mean4"] - m4 >= 5.0) and ((not r.get("u")) or R[p_]["u"]["U1"] > r["u"]["U1"])
        ok = (
            r["mw"]["g15"]
            and g6
            and r["plaus"]
            and m4 <= bar
            and ((not r.get("u")) or (r["u"]["U_pass"] and r["u"]["U1"] <= bar))
        )
        verdict[c] = ok
        price = (r["u"]["U1"] if r.get("u") else m4) / (2 * 226.6)
        print(
            f"  {c}: G1-G5 {r['mw']['g15']} | G6 {g6} | haircut K={K} bar {bar:+.1f} ({m4:+.1f}{', U1 ' + format(r['u']['U1'], '+.1f') if r.get('u') else ''}) | "
            + (f"U1-U5 {r['u']['U_pass']} | " if r.get("u") else "")
            + f"plaus {r['plaus']} => {'PASS' if ok else 'FAIL'} (price 1x {price:+.3f} s)"
        )
    A = [a for a in ("F4", "T4") if verdict.get(a)]
    if len(A) == 2:
        A = [min(A, key=lambda a: R[a]["u"]["U1"])]
    print(f"  DECISION A (unm_LIRF): {A[0] if A else 'none'} | B (LTFM): {'F5' if verdict.get('F5') else 'none'}")
    json.dump(dict(verdict=verdict, A=A), open(f"{OUT}/verdict.json", "w"))


def build():
    V = json.load(open(f"{OUT}/verdict.json"))
    B = pl.read_parquet(BASE)
    base = B["TAXITIME_SEC_mvt"].to_numpy()
    pos = {v: i for i, v in enumerate(B["MVT_ID_mvt"].to_numpy().astype(float))}

    def write(arms, tag, scope_rows_list):
        out = base.copy()
        sc = np.zeros(len(base), bool)
        for a, rows in zip(arms, scope_rows_list):
            z = np.load(f"{OUT}/ARM_{a}.npz", allow_pickle=True)
            r = rows & isr
            ii = np.array([pos[v] for v in ids[r]])
            inc = z["inc"][r]
            new = np.maximum(np.rint(z["cand"][r]), 60).astype(base.dtype)
            assert (
                np.maximum(np.rint(inc), 60).astype(base.dtype) == base[ii]
            ).all(), f"{a}: base rows != scored incumbent"
            assert (np.abs(new.astype(float) - base[ii]) <= 7200).all()
            out[ii] = new
            sc[ii] = True
        assert (out[~sc] == base[~sc]).all()
        p = CAND.format(tag=tag)
        pl.DataFrame({"MVT_ID_mvt": B["MVT_ID_mvt"], "TAXITIME_SEC_mvt": out}).write_parquet(p)
        c = out != base
        print(
            f"wrote {p} md5 {md5(p)} | rows changed {int(c.sum())} mean {float((out.astype(float)-base)[c].mean()) if c.any() else 0:+.1f} RMS {float(np.sqrt(((out.astype(float)-base)[c]**2).mean())) if c.any() else 0:.1f} max {int(np.abs(out.astype(float)-base).max())}"
        )
        tmp = f"{OUT}/validate_check.parquet"
        shutil.copy(p, tmp)
        v = subprocess.run(
            [".venv/bin/python", "scripts/validate_submission.py", tmp, BASE], capture_output=True, text=True
        )
        print(v.stdout.strip(), v.stderr.strip()[-300:])
        os.remove(tmp)

    A = (
        V["A"][0] if V["A"] else ("F4" if os.path.exists(f"{OUT}/ARM_F4.npz") else None)
    )  # frozen-row policy: the file is built whatever the verdict, labelled
    if A:
        print(f"--- scope A isolate from {A} (verdict {'PASS' if V['A'] else 'FAIL - policy file'})")
        write([A], A, [LIRF])
    if V["verdict"].get("F5"):
        print("--- scope B isolate F5 (PASS)")
        write(["F5"], "F5", [LT])
    if A and V["A"] and V["verdict"].get("F5"):
        print("--- combined")
        write([A, "F5"], A + "F5", [LIRF, LT])


def sealcheck():
    lines = []
    for a in SCOPE:
        p = f"{OUT}/ARM_{a}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        rows = LIRF if SCOPE[a] == "A" else LT
        W = W_A if SCOPE[a] == "A" else W_B
        d = (z["cand"] - y) ** 2 - (z["inc"] - y) ** 2
        per = {
            f: float((W[LAB_ALL & sealed & rows & (fo == f)] * d[LAB_ALL & sealed & rows & (fo == f)]).sum() / RN)
            for f in GATE
        }
        s = f"{a} SEAL LOMO MW " + " ".join(f"{f} {v:+.1f}" for f, v in per.items())
        if SCOPE[a] == "A":
            k = has_true & sealed
            dt = (z["cand_true"][k] - y[k]) ** 2 - (fz[k] - y[k]) ** 2
            s += f" | SEAL TRUE janjul raw {dt.sum() / RN:+.1f} n {int(k.sum())}"
        lines.append(s)
    open(f"{OUT}/sealed_full.txt", "w").write("\n".join(lines) + "\n")
    print(f"sealed columns written to {OUT}/sealed_full.txt ({len(lines)} arms); not printed")


if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"kumo_icl_lirf: unm_LIRF 2025 {int((LIRF & ~isr).sum())} (UNS {int((LIRF & UNS).sum())}, TRUE janjul {int(has_true.sum())}, TRUE UNS {int((has_true & UNS).sum())}), 2026 {int((LIRF & isr).sum())} "
        f"(v98 == v62 on {int((B98[LIRF & isr] == v62[LIRF & isr]).sum())}) | LTFM 2025 {int((LT & ~isr).sum())} (UNS {int((LT & UNS).sum())}), 2026 {int((LT & isr).sum())} (v98 == ref on {int((B98[LT & isr] == np.maximum(np.rint(ref[LT & isr]), 60)).sum())}) | {time.time()-t0:.0f}s",
        flush=True,
    )
    if args == ["PLUMB"]:
        print(
            "XA cols",
            frz_arm_px.shape[1],
            "finite share (2025 LIRF)",
            np.isfinite(frz_arm_px[LIRF & ~isr]).mean(0).round(2),
            "| 2026 inputs finite",
            np.isfinite(frz_arm_26[LIRF & isr]).mean(0).round(2),
            "| TRUE inputs finite",
            np.isfinite(frz_arm_fz[has_true]).mean(0).round(2),
        )
        print(
            "proxy px finite on 2025 LIRF",
            np.isfinite(px[LIRF & ~isr]).mean().round(3),
            "| fz_p<=0.9 share",
            float(np.mean(fz_p[has_true] <= 0.9)).__round__(3),
            "| XB cols",
            XB.shape[1],
            "finite",
            np.isfinite(XB[LT & ~isr]).mean(0).round(2),
        )
        print(
            "W_A fold sums (UNS)",
            {f: round(float(W_A[LIRF & UNS & (fo == f)].sum())) for f in FOLDS},
            "| U1 weight sum",
            round(float(w_u1(has_true)[has_true & UNS].sum())),
            "(= 361 expected)",
        )
        print("LOMO sizes A:", {m: (int(tr.sum()), int(te.sum())) for m, tr, te in lomo_months(LIRF)})
        print("LOMO sizes B:", {m: (int(tr.sum()), int(te.sum())) for m, tr, te in lomo_months(LT)})
        sys.exit(0)
    if args == ["SUMMARY"]:
        summary()
        sys.exit(0)
    if args == ["BUILD"]:
        build()
        sys.exit(0)
    if args == ["SEALCHECK"]:
        sealcheck()
        sys.exit(0)
    for a in args:
        print(f"\n######## arm {a} [scope {SCOPE[a]}]", flush=True)
        cand, g, cand_true, gt = arm(a)
        if SCOPE[a] == "A":
            inc = np.where(isr, inc_26, inc_px)
            mw = mw_table(cand, inc, LIRF, W_A, f"{a} vs proxy incumbent (12-month LOMO)")
            u = u_table(cand_true, has_true, a)
            pl_ok = plaus(cand, inc, LIRF, 1200.0, a)
            res = dict(mw=mw, u=u, plaus=pl_ok)
        else:
            inc = ref
            mw = mw_table(cand, inc, LT, W_B, f"{a} vs ref")
            pl_ok = plaus(cand, inc, LT, 600.0, a)
            res = dict(mw=mw, plaus=pl_ok)
            cand_true = np.full(len(y), np.nan)
        np.savez_compressed(
            f"{OUT}/ARM_{a}.npz", ids=ids, cand=cand, inc=inc, g=g, cand_true=cand_true, res=json.dumps(res)
        )
        print(f"  saved {OUT}/ARM_{a}.npz | {time.time()-t0:.0f}s", flush=True)
    print("done", f"{time.time()-t0:.0f}s")
