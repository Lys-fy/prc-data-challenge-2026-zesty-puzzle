# SPDX-License-Identifier: GPL-3.0-only
"""unm_resid step 3: stacker S over [D, current members' fold OOF, clocks] on 2025 unmatched rows (ms <= 8 h), LOFO over the 6 month folds,
optional clock stage (residual LGB on linked rows), scoring vs the v53-era proxy (0.5 RA3 + 0.5 meta_oof_h_ctl, floor 60; janjul also the real v53 x10/x20).
env: DV=<D variant>  STK=lgb|lin  YCLIP=<s or 0>  CLK=0|1  MOVECAP=7200 (rows moving more than this vs baseline keep the baseline; the '>2 h class')
Writes scratchpad/xur/S_<tag>.npz with fold OOF + ranking prediction for all unmatched rows."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, numpy as np, polars as pl, lightgbm as lgb, pandas as pd

sys.path.insert(0, "scratchpad/xur")
from unm_resid_lib import *

DV = os.environ.get("DV", "L2")
STK = os.environ.get("STK", "lgb")
YCLIP = float(os.environ.get("YCLIP", "0"))
CLK = int(os.environ.get("CLK", "0"))
MOVECAP = float(os.environ.get("MOVECAP", "7200"))
NT = int(os.environ.get("NT", "4"))
TAG = os.environ.get(
    "TAG",
    ("x" + os.environ["APT_MEMBER"] + "_" if os.environ.get("APT_MEMBER") else "")
    + f"{DV}_{STK}_c{int(YCLIP)}_k{CLK}"
    + (f"_s{os.environ.get('SHR','0.5')}_g{os.environ.get('GCL','900')}" if STK in ("res", "resb") else ""),
)
SEEDS = int(os.environ.get("SSEEDS", "3"))
SHR = float(os.environ.get("SHR", "0.5"))
# ---- frame: all unmatched rows (2025 + ranking)
U = pl.read_parquet(
    "data/features_v9.parquet",
    columns=["MVT_ID_mvt", "is_rank", "unmatched", "ADEP_mvt", "y", "mvt_sched", "MVT_TIME_UTC_mvt"],
).filter(pl.col("unmatched") == 1)
for f, cs in [
    ("adsb", ["adsb_linked", "adsb_move", "adsb_lb"]),
    ("adsb2", ["adsb2_leave", "adsb2_at"]),
    ("adsb3", ["adsb3_mv_gs", "adsb3_mv_pos"]),
    ("adsb5", ["adsb5_push", "adsb5_mv2", "adsb5_first_moving"]),
]:
    U = U.join(
        pl.read_parquet(f"data/external/{f}_features.parquet", columns=["MVT_ID_mvt"] + cs), on="MVT_ID_mvt", how="left"
    )
inv = {m: f for f, ms in FOLDS.items() for m in ms}
U = U.with_columns(
    cls_expr(), lk=pl.col("adsb_linked").fill_null(0).cast(pl.Int8), mon=pl.col("MVT_TIME_UTC_mvt").dt.month()
).sort("MVT_ID_mvt")
U = U.with_columns(
    fold=pl.when(pl.col("is_rank") == 1)
    .then(pl.lit("rank"))
    .otherwise(pl.col("mon").replace_strict(inv, default="none"))
)
ids = U["MVT_ID_mvt"].to_numpy().astype(float)
y = U["y"].to_numpy().astype(float)
fo = U["fold"].to_numpy()
cl = U["cls"].to_numpy()
ms = U["mvt_sched"].to_numpy().astype(float)
apt = U["ADEP_mvt"].to_numpy().astype(str)
lk = U["lk"].to_numpy()
isr = fo == "rank"
# ---- D
z = np.load(f"scratchpad/xur/D_{DV}.npz", allow_pickle=True)
assert np.array_equal(z["ids"].astype(float), ids)
D = np.where(isr, z["rank"], z["oof"])


# ---- members: fold OOF (1 seed) on 2025 rows, production preds on ranking rows
def prod(pref, n, cols):
    acc = None
    for i in range(n):
        d = pl.DataFrame({"MVT_ID_mvt": ids}).join(
            pl.read_parquet(f"data/preds/{pref}_s{i}.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)),
            on="MVT_ID_mvt",
            how="left",
            maintain_order="left",
        )
        a = np.vstack([d[c].to_numpy().astype(float) for c in cols])
        acc = a if acc is None else acc + a
    return acc / n


M = {}
for fm, pm, n in [("queue_eb_ra3qa", "queue_eb_v47lqa", 6), ("meta_oof_h_ctl", "CATV17x", 14)]:
    pr = prod(pm, n, ["TAXITIME_SEC_mvt", "p_raw", "taxi", "art"])
    for j, nm in enumerate(["pred", "p", "taxi", "art"]):
        M[f"{fm}_{nm}"] = np.where(isr, pr[j], member(U, fm, row=1 + j))
M["NN"] = np.where(isr, prod("NNRA2", 2, ["TAXITIME_SEC_mvt"])[0], member(U, "NNA2", row=1))
# v53-era baseline: fold = 0.5 RA3 + 0.5 meta_oof_h_ctl (1 seed, the V47L / CATV17x recipes); janjul real v53 = V47L x10 + CATV17x x20
base = np.maximum(0.5 * member(U, "RA3") + 0.5 * member(U, "meta_oof_h_ctl"), 60.0)
v53j = np.maximum(
    0.5 * np.mean([member(U, f"V47L_s{i}") for i in range(10)], axis=0)
    + 0.5 * np.mean([member(U, f"CATV17x_s{i}") for i in range(20)], axis=0),
    60.0,
)
# ranking baseline = the unmatched-rule stage file
V62 = (
    pl.DataFrame({"MVT_ID_mvt": ids})
    .join(pl.read_parquet("stages/unmatched_rule.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left")[
        "TAXITIME_SEC_mvt"
    ]
    .to_numpy()
    .astype(float)
)
cur = np.maximum(0.5 * M["queue_eb_ra3qa_pred"] + 0.5 * M["meta_oof_h_ctl_pred"], 60.0)
base = np.where(np.isin(fo, SELF), cur, base)  # selection folds have no RA3 OOF: current-member mix as their reference
base = np.where(isr, V62, base)
# ---- stacker design
aptc = np.array([APTS.index(a) if a in APTS else -1 for a in apt], float)
feat = {"D": D, **M, "ms": ms, "apt": aptc, "lk": lk.astype(float)}
for c in [
    "adsb_move",
    "adsb_lb",
    "adsb2_leave",
    "adsb2_at",
    "adsb3_mv_gs",
    "adsb3_mv_pos",
    "adsb5_push",
    "adsb5_mv2",
    "adsb5_first_moving",
]:
    feat[c] = U[c].cast(pl.Float64).to_numpy()
GCL = float(os.environ.get("GCL", "900"))
FN = list(feat)
X = np.column_stack([feat[k] for k in FN]).astype(np.float32)
FNb = FN + ["base"]
Xb = np.column_stack([X, base]).astype(np.float32)
elig = np.nan_to_num(ms, nan=0) <= 28800  # unm_other + unm_LIRF type rows (exposed excluded)
lab = (
    np.isfinite(y)
    & ~isr
    & (fo != "none")
    & elig
    & np.isfinite(D)
    & np.isfinite(M["queue_eb_ra3qa_pred"])
    & np.isfinite(M["meta_oof_h_ctl_pred"])
    & np.isfinite(base)
)
yt = np.clip(y, 0, YCLIP) if YCLIP > 0 else y
PS = dict(
    objective=("huber" if STK == "hub" else "l2"),
    alpha=900.0,
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
LINF = ["D", "queue_eb_ra3qa_pred", "meta_oof_h_ctl_pred", "NN"]


def fit_pred(tr, te):
    if STK == "res":  # conservative residual stacker on the current-member mix: cur + SHR * g
        r = np.clip(yt - cur, -3600, 3600)
        p = np.zeros(te.sum())
        for s_ in range(SEEDS):
            m = lgb.train(
                dict(PS, seed=s_, min_data_in_leaf=200),
                lgb.Dataset(X[tr], r[tr], categorical_feature=[FN.index("apt")]),
                300,
            )
            p += m.predict(X[te]) / SEEDS
        return cur[te] + SHR * p, m
    if (
        STK == "resb"
    ):  # residual stacker on the v53-era baseline itself (fold: RA3+meta_oof_h_ctl mix, 511/612: current mix; ranking: v62): base + SHR * clip(g, +-GCL)
        r = np.clip(yt - base, -3600, 3600)
        p = np.zeros(te.sum())
        for s_ in range(SEEDS):
            m = lgb.train(
                dict(PS, seed=s_, min_data_in_leaf=200),
                lgb.Dataset(Xb[tr], r[tr], categorical_feature=[FNb.index("apt")]),
                300,
            )
            p += m.predict(Xb[te]) / SEEDS
        return base[te] + SHR * np.clip(p, -GCL, GCL), m
    if STK == "lin":
        A = np.column_stack([feat[k][tr] for k in LINF])
        w, *_ = np.linalg.lstsq(A, yt[tr], rcond=None)
        return np.column_stack([feat[k][te] for k in LINF]) @ w, w
    p = np.zeros(te.sum())
    for s in range(SEEDS):
        m = lgb.train(dict(PS, seed=s), lgb.Dataset(X[tr], yt[tr], categorical_feature=[FN.index("apt")]), 400)
        p += m.predict(X[te]) / SEEDS
    return p, m


S = np.full(len(y), np.nan)
for f in list(FOLDS) + ["rank"]:
    tr = lab & (fo != f)
    te = (fo == f) & elig & np.isfinite(D)
    S[te], mdl = fit_pred(tr, te)
    if f == "rank":
        if STK == "lin":
            print("linear weights", dict(zip(LINF, np.round(mdl, 3))))
        else:
            nms = FNb if STK == "resb" else FN
            imp = mdl.feature_importance("gain")
            print("stacker gain:", ", ".join(f"{nms[i]} {100*imp[i]/imp.sum():.0f}%" for i in np.argsort(-imp)[:12]))
if CLK:  # clock stage on linked rows: residual y - S, clocks + dd
    CC = [
        "adsb_move",
        "adsb_lb",
        "adsb2_leave",
        "adsb2_at",
        "adsb3_mv_gs",
        "adsb3_mv_pos",
        "adsb5_push",
        "adsb5_mv2",
        "adsb5_first_moving",
    ]
    Xc = np.column_stack([S, aptc] + [feat[c] for c in CC] + [S - feat[c] for c in CC]).astype(np.float32)
    r = np.clip(y - S, -1800, 1800)
    lkm = lk == 1
    S2 = S.copy()
    for f in list(FOLDS) + ["rank"]:
        tr = lab & lkm & (fo != f)
        te = (fo == f) & lkm & np.isfinite(S)
        g = np.zeros(te.sum())
        for s in range(SEEDS):
            m = lgb.train(
                dict(PS, seed=s, min_data_in_leaf=200), lgb.Dataset(Xc[tr], r[tr], categorical_feature=[1]), 300
            )
            g += m.predict(Xc[te]) / SEEDS
        S2[te] = S[te] + np.clip(g, -300, 300)
    S = S2
cand = np.maximum(S, 60.0)
okc = np.isfinite(cand) & np.isfinite(base)
moved_big = okc & (np.abs(cand - base) > MOVECAP)
APT_MEMBER = [
    a for a in os.environ.get("APT_MEMBER", "").split(",") if a
]  # airports kept at the baseline (round-2 variant, post-hoc: LTFM)
final = np.where(okc & ~moved_big & ~np.isin(apt, APT_MEMBER), cand, base)
# ---- scoring on unm_other fold rows
w = np.zeros(len(y))
R = frame(rank=True)
Fr = frame()
wf = rank_weights(Fr, R)
pos = {v: i for i, v in enumerate(ids)}
w[[pos[v] for v in Fr["MVT_ID_mvt"].to_numpy().astype(float)]] = wf
print(
    f"\n== {TAG}: unm_other fold rows; delta = candidate - v53-era proxy (negative = better); moved>{MOVECAP:.0f}s kept at baseline"
)
print(
    f"{'fold':10s} {'n':>5s} {'base MSE':>10s} {'cand MSE':>10s} {'raw d/row':>10s} {'w/o top3 d':>10s} {'rankw SSE d/RN':>14s} {'nbig':>4s}"
)
res = {}
for f in FOLDS:
    k = (fo == f) & (cl == "unm_other") & np.isfinite(y) & np.isfinite(base) & np.isfinite(final)
    eb = (base[k] - y[k]) ** 2
    ec = (final[k] - y[k]) ** 2
    d = ec - eb
    o = np.argsort(-np.abs(d))
    raw = d.mean()
    wo = np.delete(d, o[:3]).mean()
    rw = (w[k] * d).sum() / RN
    res[f] = (raw, wo, rw)
    print(
        f"{f:10s} {k.sum():5d} {eb.mean():10,.0f} {ec.mean():10,.0f} {raw:+10,.0f} {wo:+10,.0f} {rw:+14,.1f} {moved_big[k].sum():4d}"
    )
    if f == "janjul":
        e53 = (v53j[k] - y[k]) ** 2
        d53 = ec - e53
        o53 = np.argsort(-np.abs(d53))
        print(
            f"{'  vs real v53':10s} {k.sum():5d} {e53.mean():10,.0f} {ec.mean():10,.0f} {d53.mean():+10,.0f} {np.delete(d53, o53[:3]).mean():+10,.0f} {(w[k]*d53).sum()/RN:+14,.1f}"
        )
G = [res[f] for f in GATE]
g1 = sum(r[0] < 0 for r in G) >= 3
g2 = res["janjul"][0] < 0
g3 = sum(r[1] < 0 for r in G) >= 3
kg = np.isin(fo, GATE) & (cl == "unm_other") & np.isfinite(y) & np.isfinite(base)
eb = (base - y) ** 2
ec = (final - y) ** 2
tb = np.where(kg)[0][np.argsort(-eb[kg])[:20]]
tc = np.where(kg)[0][np.argsort(-ec[kg])[:20]]
tail_b = ec[tb].sum() - eb[tb].sum()
tail_c = ec[tc].sum() - eb[tc].sum()
g4 = (tail_b <= 0) and (tail_c <= 0)
print(f"top-20 rows by BASELINE error: SSE cand - base = {tail_b:+,.0f}; top-20 by CANDIDATE error: {tail_c:+,.0f}")
print("top-20 rows by baseline error (fold apt lk y base cand ms):")
for i in tb:
    print(f"   {fo[i]:9s} {apt[i]} {lk[i]} y {y[i]:7.0f} base {base[i]:6.0f} cand {final[i]:6.0f} ms {ms[i]:6.0f}")
print(
    f"GATE {TAG}: >=3/4 raw {g1} | janjul {g2} | w/o top3 >=3/4 {g3} | tail {g4} => {'PASS' if g1 and g2 and g3 and g4 else 'FAIL'}"
)
sel = [res[f][0] for f in SELF]
sw = [res[f][1] for f in SELF]
print(
    f"SELECTION {TAG} folds 511/612 (vs current-member mix) raw d/row: {sel[0]:+,.0f} {sel[1]:+,.0f} w/o top3 {sw[0]:+,.0f} {sw[1]:+,.0f} | rank-weighted sum over 4 gate folds {sum(r[2] for r in G):+.1f} board s2 per fold-equivalent"
)
# ranking side
kr = isr & (cl == "unm_other")
print(
    f"ranking unm_other rows {kr.sum()}, moved>{MOVECAP:.0f}: {moved_big[kr].sum()}, candidate finite {np.isfinite(cand[kr]).sum()}; mean change {np.mean(final[kr]-base[kr]):+.1f} rms {np.sqrt(np.mean((final[kr]-base[kr])**2)):.1f}"
)
np.savez_compressed(
    f"scratchpad/xur/S_{TAG}.npz",
    ids=ids,
    final=final,
    cand=cand,
    base=base,
    y=y,
    fold=fo,
    cls=cl,
    w=w,
    moved_big=moved_big,
    apt=apt,
    lk=lk,
)
