# SPDX-License-Identifier: GPL-3.0-only
"""Fits and scores pre-declared callsign-link lever arms (A1/A1p, B1-B3(p), C1/C1p, D1-D3, V1-V4) on unm_other rows,
month-nested, against a month-aware (MW) weighted gate, and writes each scored arm's candidate values.
Inputs: positional ARM names; scratchpad/xur2/frame.parquet, scratchpad/xur2/link.parquet (if present),
data/features_v9.parquet, data/external/dreg_features.parquet, scratchpad/xur/S_*.npz (arms B1/B2/D1/D2).
Output: scratchpad/xur2/ARM_<arm>.npz (ids, cand, ref, isr, uo, mean4, passed) per arm; "DIAG" prints a diagnostic
report instead of writing a file.

Callsign link of unmatched rows: score the pre-declared lever arms on the unm_other rows, month-nested, month-aware (MW) gate.
usage: unm_link_lever.py ARM [ARM ...]   arms: A1 A1p B1 B2 B3 B3p C1 C1p D1 D2 D3
Reference = frame.ref (C1 + unm_resid resb on non-LTFM, C1 on LTFM; janjul on the real C1). Candidates change only unm_other rows of the arm's scope.
Writes scratchpad/xur2/ARM_<arm>.npz (ids, cand for 2025 rows (nested) and ranking rows (fit on all 2025))."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, numpy as np, polars as pl, lightgbm as lgb

sys.path.insert(0, "scratchpad/xur")
from unm_resid_lib import *

RN = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)
K, G = 0.5, 600.0
NT = int(os.environ.get("NT", "4"))
U = pl.read_parquet("scratchpad/xur2/frame.parquet")
EX = {
    "nbr": ["nbr_apt_m60_med", "nbr_apt_m60_max", "nbr_apt_m30_med", "arrx_off_day_med", "arr_block_delay"],
}
F9 = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "hour", "dow"] + EX["nbr"])
DR = pl.read_parquet(
    "data/external/dreg_features.parquet",
    columns=["MVT_ID_mvt", "dreg_day_med", "dreg_day_late2h", "dreg_day_unm", "dreg_unm60"],
)
U = U.join(F9, on="MVT_ID_mvt", how="left").join(DR, on="MVT_ID_mvt", how="left")
LK = (
    pl.read_parquet("scratchpad/xur2/link.parquet")
    if os.path.exists("scratchpad/xur2/link.parquet")
    else pl.DataFrame(
        {
            "MVT_ID_mvt": [0.0],
            "apt": ["-"],
            "day": ["-"],
            "hex": [None],
            "n_ground": [0],
            "d_first_stand": [0.0],
            "dt_tko": [0.0],
        }
    ).cast({"MVT_ID_mvt": U["MVT_ID_mvt"].dtype})
)
tko = U["MVT_TIME_UTC_mvt"].dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)
U = U.with_columns(tko=tko).join(LK.drop("apt", "day"), on="MVT_ID_mvt", how="left")
for c in ["t_first_seen", "t_first_move", "t_hold_end", "t_at_stand", "t_land_prev"]:
    if c not in U.columns:
        U = U.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
for c in ["t_first_seen", "t_first_move", "t_hold_end", "t_at_stand", "t_land_prev"]:
    U = U.with_columns((pl.col("tko") - pl.col(c)).alias("L_" + c[2:]))
U = U.with_columns(lnk2=pl.col("hex").is_not_null(), day=pl.col("MVT_TIME_UTC_mvt").dt.date())
# day-state of the unmatched rows (label-free, both years)
U = U.join(
    U.group_by("ADEP_mvt", "day").agg(
        u_n=pl.len(),
        u_n1h=(pl.col("mvt_sched") > 3600).sum(),
        u_n2h=(pl.col("mvt_sched") > 7200).sum(),
        u_ms_med=pl.col("mvt_sched").median(),
    ),
    on=["ADEP_mvt", "day"],
    how="left",
)
U = U.with_columns(
    u_ms_rank=pl.col("mvt_sched").rank("average").over("ADEP_mvt", "day") / pl.col("u_n"),
    apron=pl.col("STAND_mvt").cast(pl.Utf8).str.slice(0, 1),
    vfr=(pl.col("FLIGHT_RULE_mvt") == "V").cast(pl.Float64),
)
ids = U["MVT_ID_mvt"].to_numpy().astype(float)
y = U["y"].to_numpy().astype(float)
fo = U["fold"].to_numpy()
mon = U["mon"].to_numpy()
isr = U["is_rank"].to_numpy() == 1
cls = U["cls"].to_numpy()
apt = U["ADEP_mvt"].to_numpy().astype(str)
ms = U["mvt_sched"].to_numpy().astype(float)
ref = U["ref"].to_numpy().astype(float)
lk = U["lk"].to_numpy()
ref = np.where(isr, U["v76"].to_numpy().astype(float), ref)
uo = cls == "unm_other"
ltfm = apt == "LTFM"
lab = ~isr & uo & np.isfinite(y) & np.isfinite(ref) & (fo != "none")
# ---- MW weights per fold: n_2026(apt|lnk|slot) / n_fold(same), fallback apt|slot, apt
slot = np.where(np.isin(mon, [11, 12, 1, 2, 3, 4]), "W", "S")


def mw_weights():
    w = np.zeros(len(y))
    keys = [
        lambda m: np.char.add(np.char.add(apt, "|"), np.char.add(lk.astype(str), np.char.add("|", slot))),
        lambda m: np.char.add(apt, np.char.add("|", slot)),
        lambda m: apt,
    ]
    for f in FOLDS:
        done = np.zeros(len(y), bool)
        s = lab & (fo == f)
        for kf in keys:
            kk = kf(None)
            rr = isr & uo
            for key in np.unique(kk[rr]):
                nr = np.sum(rr & (kk == key) & ~done)
                m = s & (kk == key)
                if nr == 0 or m.sum() < 20:
                    continue
                w[m & (w == 0)] += nr / m.sum()
                done[rr & (kk == key)] = True
    return w


W = mw_weights()


# ---- helpers
def lomo_lgb(Xall, r, rows, obj="l2", seeds=3, rounds=300, cats=()):
    """month-nested residual model: for each 2025 month fit on the other 11 months of `rows`, predict that month; ranking from all 12."""
    out = np.full(len(y), np.nan)
    PS = dict(
        objective=obj,
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
    tr_all = rows & lab
    for m in list(range(1, 13)) + ["rank"]:
        te = (rows & ~isr & (mon == m)) if m != "rank" else (rows & isr)
        tr = tr_all & (mon != m) if m != "rank" else tr_all
        if te.sum() == 0:
            continue
        p = np.zeros(te.sum())
        for s in range(seeds):
            mdl = lgb.train(dict(PS, seed=s), lgb.Dataset(Xall[tr], r[tr], categorical_feature=list(cats)), rounds)
            p += mdl.predict(Xall[te]) / seeds
        out[te] = p
    return out


def aptc():
    return np.array([APTS.index(a) if a in APTS else -1 for a in apt], float)


def col(c):
    return U[c].cast(pl.Float64).to_numpy()


def perm_within(X, rows, rng):
    """placebo: permute the rows of X within airport x month among `rows`"""
    X = X.copy()
    key = np.char.add(apt, mon.astype(str))
    for k in np.unique(key[rows]):
        ix = np.where(rows & (key == k))[0]
        X[ix] = X[rng.permutation(ix)]
    return X


def stand_enc(rows):
    """LOMO stand residual encoding: mean(y - ref) of the same airport x stand in other months, shrunk n/(n+20)."""
    st = U["STAND_mvt"].cast(pl.Utf8).fill_null("NA").to_numpy()
    key = np.char.add(apt, st)
    out = np.zeros(len(y))
    r = np.clip(y - ref, -1800, 1800)
    for m in list(range(1, 13)) + ["rank"]:
        te = (rows & ~isr & (mon == m)) if m != "rank" else (rows & isr)
        tr = rows & lab & ((mon != m) if m != "rank" else True)
        d = pl.DataFrame({"k": key[tr], "r": r[tr]}).group_by("k").agg(s=pl.col("r").sum(), n=pl.len())
        mp = dict(zip(d["k"].to_list(), (d["s"] / (d["n"] + 20)).to_list()))
        out[te] = [mp.get(k, 0.0) for k in key[te]]
    return out


# ---- arms
GC = {}


def arm(name):
    rng = np.random.default_rng(0)
    cand = ref.copy()
    if name in ("A1", "A1p"):
        rows = uo & ~ltfm & U["lnk2"].to_numpy()
        C = ["L_first_seen", "L_first_move", "L_hold_end", "L_at_stand", "L_land_prev"]
        Xc = np.column_stack(
            [col(c) for c in C] + [ref - col(c) for c in C] + [col("n_ground"), col("d_first_stand"), col("dt_tko")]
        )
        if name == "A1p":
            Xc = perm_within(Xc, rows, rng)
        X = np.column_stack([ref, aptc(), ms, Xc]).astype(np.float32)
        g = lomo_lgb(X, np.clip(y - ref, -3600, 3600), rows, cats=[1])
        ok = rows & np.isfinite(g)
        cand[ok] = ref[ok] + K * np.clip(g[ok], -G, G)
        if name == "A1":
            GC["g"], GC["ok"] = g, ok
    elif name in ("V1", "V2", "V3", "V4"):  # §8 variants of A1 (same g)
        if "g" not in GC:
            GC["A1"] = arm("A1")
        g, ok = GC["g"], GC["ok"]
        if name == "V4":
            cand = GC["A1"].copy()
            ls = col("L_at_stand")
            k4 = ok & np.isfinite(ls)
            cand[k4] = np.maximum(ls[k4], 60.0)
        else:
            kk, gg = {"V1": (1.0, 600.0), "V2": (0.5, 1200.0), "V3": (1.0, 1200.0)}[name]
            cand[ok] = ref[ok] + kk * np.clip(g[ok], -gg, gg)
    elif name in ("B1", "B2"):
        z = np.load(
            "scratchpad/xur/S_HUB_resb_c14400_k0_s0.5_g" + ("900" if name == "B1" else "300") + ".npz",
            allow_pickle=True,
        )
        assert np.array_equal(z["ids"].astype(float), ids)
        rows = uo & ltfm
        corr = z["final"] - z["base"]
        ok = rows & np.isfinite(corr)
        cand[ok] = ref[ok] + corr[ok]
    elif name in ("B3", "B3p"):
        rows = uo & ltfm
        Dm = np.column_stack(
            [
                col(c)
                for c in [
                    "u_n",
                    "u_n1h",
                    "u_n2h",
                    "u_ms_med",
                    "u_ms_rank",
                    "nbr_apt_m60_med",
                    "nbr_apt_m60_max",
                    "dreg_day_med",
                    "dreg_day_late2h",
                ]
            ]
        )
        if name == "B3p":  # day-state permuted across days within month
            Dm = perm_within(Dm, rows, rng)
        X = np.column_stack(
            [ref, ms, col("hour"), col("dow"), (U["cs_pre"] == "THY").cast(pl.Float64).to_numpy(), Dm]
        ).astype(np.float32)
        g = lomo_lgb(X, np.clip(y - ref, -3600, 3600), rows)
        ok = rows & np.isfinite(g)
        cand[ok] = ref[ok] + K * np.clip(g[ok], -G, G)
    elif name in ("C1", "C1p"):
        rows = uo & ~ltfm
        Dm = np.column_stack(
            [
                col(c)
                for c in [
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
            ]
            + [stand_enc(rows)]
        )
        if name == "C1p":
            Dm = perm_within(Dm, rows, rng)
        X = np.column_stack(
            [
                ref,
                aptc(),
                ms,
                lk.astype(float),
                col("vfr"),
                col("stand_since_arr_inblock"),
                col("stand_since_prev_dep"),
                col("arr_block_delay"),
                Dm,
            ]
        ).astype(np.float32)
        g = lomo_lgb(X, np.clip(y - ref, -3600, 3600), rows, cats=[1])
        ok = rows & np.isfinite(g)
        cand[ok] = ref[ok] + K * np.clip(g[ok], -G, G)
    elif name in ("D1", "D2"):
        z0 = np.load("scratchpad/xur/S_xLTFM_HUB_resb_c14400_k0_s0.5_g900.npz", allow_pickle=True)
        z = np.load(
            "scratchpad/xur/S_HUB_resb_c14400_k0_s" + ("1.0_g900" if name == "D1" else "0.5_g300") + ".npz",
            allow_pickle=True,
        )
        rows = uo & ~ltfm
        d = (z["final"] - z["base"]) - (z0["final"] - z0["base"])
        ok = rows & np.isfinite(d)
        cand[ok] = ref[ok] + d[ok]
    elif name == "D3":
        rows = uo & ~ltfm & (ms <= 10800)
        X = np.column_stack(
            [
                ref,
                aptc(),
                ms,
                ms - ref,
                lk.astype(float),
                col("vfr"),
                col("hour"),
                col("u_n"),
                col("u_n1h"),
                col("u_ms_med"),
                col("nbr_apt_m60_med"),
                col("stand_since_arr_inblock"),
                col("stand_since_prev_dep"),
            ]
        ).astype(np.float32)
        sub = (np.abs(y - ms) < 300).astype(float)
        p = np.full(len(y), np.nan)
        PS = dict(
            objective="binary",
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
        for m in list(range(1, 13)) + ["rank"]:
            te = (rows & ~isr & (mon == m)) if m != "rank" else (rows & isr)
            tr = rows & lab & ((mon != m) if m != "rank" else True)
            pp = np.zeros(te.sum())
            for s in range(3):
                pp += (
                    lgb.train(dict(PS, seed=s), lgb.Dataset(X[tr], sub[tr], categorical_feature=[1]), 300).predict(
                        X[te]
                    )
                    / 3
                )
            p[te] = pp
        ok = rows & np.isfinite(p)
        cand[ok] = p[ok] * ms[ok] + (1 - p[ok]) * ref[ok]
    cand = np.maximum(cand, 60.0)
    big = np.abs(cand - ref) > 7200
    cand[big] = ref[big]
    if name == "A1":
        GC["A1"] = cand.copy()
    return cand


def score(name, cand, base=None):
    base = ref if base is None else base
    e0 = (base - y) ** 2
    e1 = (cand - y) ** 2
    d = e1 - e0
    res = {}
    print(
        f"\n== {name}: Δ = arm − reference on unm_other 2025 rows (MW board s² = Σ w·Δ / {RN}); raw per row; changed rows"
    )
    print(
        f"{'fold':10s} {'n':>5s} {'chg':>5s} {'raw Δ/row':>10s} {'raw w/o3':>10s} {'MW':>8s} {'MW w/o3':>8s} {'old-rankw':>9s}"
    )
    wold = np.zeros(len(y))
    R = frame(rank=True)
    Fr = frame()
    wf = rank_weights(Fr, R)
    pos = {v: i for i, v in enumerate(ids)}
    wold[[pos[v] for v in Fr["MVT_ID_mvt"].to_numpy().astype(float)]] = wf
    for f in FOLDS:
        k = lab & (fo == f)
        dd = d[k]
        o = np.argsort(-np.abs(dd))
        wd = W[k] * dd
        o2 = np.argsort(-np.abs(wd))
        r = (
            dd.mean(),
            np.delete(dd, o[:3]).mean(),
            wd.sum() / RN,
            np.delete(wd, o2[:3]).sum() / RN,
            (wold[k] * dd).sum() / RN,
            int((cand[k] != ref[k]).sum()),
        )
        res[f] = r
        print(f"{f:10s} {k.sum():5d} {r[5]:5d} {r[0]:+10,.0f} {r[1]:+10,.0f} {r[2]:+8.1f} {r[3]:+8.1f} {r[4]:+9.1f}")
    mm = [(W[lab & (mon == m)] * d[lab & (mon == m)]).sum() / RN for m in range(1, 13)]
    print("per month MW:", " ".join(f"{m}:{v:+.1f}" for m, v in zip(range(1, 13), mm)))
    Gf = [res[f] for f in GATE]
    g1 = sum(r[2] < 0 for r in Gf) >= 3
    g2 = res["janjul"][2] <= 0
    g3 = sum(r[1] < 0 for r in Gf) >= 3
    g4 = sum(r[3] < 0 for r in Gf) >= 3
    g5 = sum(v < 0 for v in mm) >= 7
    kg = lab & np.isin(fo, GATE)
    tb = np.where(kg)[0][np.argsort(-e0[kg])[:20]]
    tc = np.where(kg)[0][np.argsort(-e1[kg])[:20]]
    mean4 = np.mean([r[2] for r in Gf])
    print(
        f"tail top-20 by ref err ΔSSE {e1[tb].sum()-e0[tb].sum():+,.0f}; by arm err {e1[tc].sum()-e0[tc].sum():+,.0f} | selection 511/612 MW {res['months511'][2]:+.1f} {res['months612'][2]:+.1f}"
    )
    kr = isr & (cand != base)
    print(
        f"GATE {name}: G1 {g1} G2 {g2} G3 {g3} G4 {g4} G5 {g5} ({sum(v < 0 for v in mm)}/12) | 4-fold mean MW {mean4:+.1f} board s² (≈ {mean4/(2*233.86):+.3f} s) | "
        f"ranking rows changed {kr.sum()}, mean {np.mean((cand-base)[kr]) if kr.any() else 0:+.1f} rms {np.sqrt(np.mean((cand-base)[kr]**2)) if kr.any() else 0:.1f}"
    )
    return mean4, all([g1, g2, g3, g4, g5])


def diag():
    """§8 label-free diagnosis: 2025 vs 2026 per airport on non-LTFM unm_other rows"""
    if "A1" not in GC:
        arm("A1")
    a1 = GC["A1"]
    g, ok = GC["g"], GC["ok"]
    ls = col("L_at_stand")
    lf = col("L_first_seen")
    print(
        "\n== §8 DIAG (non-LTFM unm_other): yr apt | n | linked | at-stand | |A1-ref| mean, q90 | g clipped ±600 share | median ref-(MVT-at_stand) | median ref-(MVT-first_seen) | ms median | ref median"
    )
    for yr, sel in (("2025", ~isr), ("2026", isr)):
        for ap in APTS + ["ALL"]:
            k = sel & uo & ~ltfm & ((apt == ap) if ap != "ALL" else True)
            if k.sum() == 0:
                continue
            kl = k & ok
            ka = kl & np.isfinite(ls)
            kf = kl & np.isfinite(lf)
            dA = np.abs(a1 - ref)[k]
            print(
                f"{yr} {ap:4s} | {k.sum():5d} | {kl.sum()/k.sum():.2f} | {ka.sum()/k.sum():.2f} | {dA.mean():6.1f} {np.quantile(dA, .9):6.0f} | "
                f"{np.mean(np.abs(g[kl]) >= 600) if kl.any() else np.nan:.2f} | {np.median(ref[ka]-ls[ka]) if ka.any() else np.nan:7.0f} | {np.median(ref[kf]-lf[kf]) if kf.any() else np.nan:7.0f} | "
                f"{np.median(ms[k]):6.0f} | {np.median(ref[k]):6.0f}"
            )
    k = ~isr & uo & ~ltfm & ok & np.isfinite(ls) & np.isfinite(y)
    print(
        "2025 linked with at-stand: e = y-(MVT-at_stand) quantiles 5/25/50/75/95:",
        np.round(np.quantile(y[k] - ls[k], [0.05, 0.25, 0.5, 0.75, 0.95])),
        "n",
        k.sum(),
        "| ref err rms %.0f, A1 err rms %.0f, V4 err rms %.0f (clip ±3600)"
        % tuple(np.sqrt(np.mean(np.clip(v[k] - y[k], -3600, 3600) ** 2)) for v in (ref, a1, np.maximum(ls, 60))),
    )


if __name__ == "__main__":
    print(
        "MW weight sum per fold:",
        {f: round(W[lab & (fo == f)].sum()) for f in FOLDS},
        "| ranking unm_other",
        int((isr & uo).sum()),
    )
    print(
        "UNM_LINK2-link coverage (unm_other, non-LTFM): 2025",
        round(U.filter(~pl.Series(isr) & pl.Series(uo & ~ltfm))["lnk2"].mean(), 3),
        "2026",
        round(U.filter(pl.Series(isr) & pl.Series(uo & ~ltfm))["lnk2"].mean(), 3),
    )
    for a in sys.argv[1:]:
        if a == "DIAG":
            diag()
            continue
        c = arm(a)
        if a.startswith("V"):
            print(f"\n(§8: {a} scored against A1)")
            m, ok = score(a, c, base=GC["A1"])
        else:
            m, ok = score(a, c)
        np.savez_compressed(
            f"scratchpad/xur2/ARM_{a}.npz", ids=ids, cand=c, ref=ref, isr=isr, uo=uo, mean4=m, passed=ok
        )
