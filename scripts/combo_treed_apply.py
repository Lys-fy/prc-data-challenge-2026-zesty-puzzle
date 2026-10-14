#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Applies Tree D: a LightGBM second step on top of the C4 correction (scripts/combo_c4_apply.py). Inputs: --target
(C4-corrected submission), --pre (pre-C4 submission), --fit-oof (stacked OOF prefix), fold-side member OOF (env
assembler_fl / assembler_fc). Output: --out (corrected submission); --out-d optionally writes a day-demeaned variant.

C4 = 11 day-demeaned state columns, beta per airport, x (1 - p) after, cf. Then a small LightGBM on e = y - C4 with inputs = the 11 C4 columns
+ N15 + adsb_linked + p + |L - C| + own residual (P - AOBT_3) + own excess (taxi - ref) + airport, fitted on the folds' stacked OOF and
applied to the ranking rows. Airports in --excl (default EGLL,EHAM) and unmatched rows are never changed.
Tree output clipped to +-CLIP (--clip, default 150 s), then cf; integer targets rounded; --out-d also writes the variant demeaned per airport x day.
Fold-side member OOF prefixes: env assembler_fl / assembler_fc (default runway_queue_ra3q / meta_oof_h_ctl).

usage: combo_treed_apply.py --target <C4-corrected submission> --pre <pre-C4 submission> --fit-oof <stacked OOF prefix> --out <out>
                    [--gate] [--save-oof <prefix>] [--dry] [--seeds 0,1] [--out-d <file for the demeaned variant>]
--gate: nested LOFO (C4 betas and tree both from the other three folds), increment over C4 and total over the L2 OOF.
Ranking side: C4 features from --pre as in scripts/combo_c4_apply.py (P = pre-C4 vector); the recomputed C4 is checked against --target.
"""

import os, sys, glob, argparse, hashlib, numpy as np, polars as pl
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)


def _solve(
    A, b
):  # 3 Oct 2026: guarded fallback; identical to np.linalg.solve unless it raises (LTFM months612: lk==0 -> 'u' cols = 'c' cols, singular on some LAPACK builds)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import daystate_lib as X
import lightgbm as lgb

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--target", required=True, help="C4-corrected submission parquet")
ap.add_argument("--pre", required=True, help="pre-C4 submission parquet")
ap.add_argument("--out", required=True, help="output corrected submission parquet")
ap.add_argument("--out-d", default=None, help="also write this day-demeaned variant")
ap.add_argument("--fit-oof", required=True, help="stacked OOF prefix the tree is fitted on")
ap.add_argument(
    "--feats",
    default="PR5,PR15,PRT5,PRT15,PRX5,PRX15,PAX15,PRT15u,PRX5u,PRX15u,PAX15u",
    help="comma list of C4 neighbour features",
)
ap.add_argument("--lgb", default="runway_queue_v47lq", help="LightGBM taxi-branch member name prefix")
ap.add_argument("--nlgb", type=int, default=6, help="number of LightGBM seeds to average")
ap.add_argument("--cat", default="CATV17x", help="CatBoost taxi-branch member name prefix")
ap.add_argument("--ncat", type=int, default=14, help="number of CatBoost seeds to average")
ap.add_argument("--clip", type=float, default=150.0, help="clip the tree output to +-CLIP seconds")
ap.add_argument("--seeds", default="0,1", help="comma list of tree training seeds to average")
ap.add_argument("--excl", default="EGLL,EHAM", help="comma list of airports left unchanged")
ap.add_argument("--save-oof", default=None, help="also save the fold OOF vectors under this prefix")
# XU: with --gate, save LOFO C4+tree[D] fold vector as <prefix>_<fold>.npy [MVT_ID, pred] (and <prefix>C4_<fold>.npy for C4 alone)
ap.add_argument("--gate", action="store_true", help="run the nested leave-one-fold-out gate instead of applying")
ap.add_argument("--dry", action="store_true", help="compute and report only, do not write --out")
ap.add_argument("--v53", default=None, help="protected-rows reference submission (target == v53 rows are checked)")
a = ap.parse_args()
FEATS = a.feats.split(",")
SEEDS = [int(s) for s in a.seeds.split(",")]
EXCL = a.excl.split(",") if a.excl else []
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
R = os.environ.get("daystate_root", os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
NT = int(os.environ.get("OMP_NUM_THREADS", 2))
PAR = dict(
    objective="l2",
    learning_rate=0.03,
    num_leaves=15,
    min_data_in_leaf=1000,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    num_threads=NT,
)
ROUNDS = 400
TS = ["MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]
C = [
    "MVT_ID_mvt",
    "ADEP_mvt",
    "RUNWAY_mvt",
    "is_rank",
    "is_dep",
    "unmatched",
    "y",
    "mvt_aobt3",
    "mvt_lobt",
    "mvt_eobt1",
    "mvt_sched",
    "ref_sr_med",
] + TS
F = pl.scan_parquet(f"{R}/data/features_v9.parquet").select(C).filter(pl.col("is_dep") == 1).collect(engine="streaming")
F = F.with_columns([pl.col(c).dt.epoch("s").cast(pl.Float64).alias(c) for c in TS]).rename({"MVT_TIME_UTC_mvt": "MVT"})
F = F.with_columns(
    unmatched=pl.when(pl.col("LOBT_flt").is_null()).then(1).otherwise(pl.col("unmatched")).cast(F.schema["unmatched"])
).drop("LOBT_flt", "AOBT_3_flt")
F = F.join(
    pl.read_parquet(
        os.environ.get("adsb_dir_override_adsb_dir", f"{R}/data/external") + "/adsb_features.parquet",
        columns=["MVT_ID_mvt", "adsb_linked"],
    ),
    on="MVT_ID_mvt",
    how="left",
).with_columns(lk=pl.col("adsb_linked").fill_null(0).cast(pl.Float64))
APTS = sorted(F["ADEP_mvt"].unique().to_list())
F = F.with_columns(_apt=pl.col("ADEP_mvt").replace_strict(APTS, list(range(len(APTS))), return_dtype=pl.Float64))


def find(prefix, fold):
    for d in os.environ.get("daystate_oofdirs", "").split(":") + [
        f"{R}/data/oof",
        f"{R}/scratchpad/xz",
        f"{R}/scratchpad/xl",
    ]:
        if not d:
            continue
        for pat in [f"{prefix}_{fold}.npy", f"{prefix}_{fold}_*.npy"]:
            g = sorted(glob.glob(f"{d}/{pat}"))
            if g:
                return g[0]
    raise FileNotFoundError(f"{prefix} {fold}")


def ld(p):
    x = np.load(p)
    return x[:, np.argsort(x[0])]


parts = []
for fold in X.FOLDS:
    l = ld(find(os.environ.get("assembler_fl", "runway_queue_ra3q"), fold))
    c = ld(find(os.environ.get("assembler_fc", "meta_oof_h_ctl"), fold))
    s = ld(find(a.fit_oof, fold))  # XU: fold members configurable
    D = pl.DataFrame({"MVT_ID_mvt": l[0], "L": l[1], "pL": l[2], "taxiL": l[3]}).join(
        pl.DataFrame({"MVT_ID_mvt": c[0], "Cb": c[1], "pC": c[2], "taxiC": c[3]}), on="MVT_ID_mvt", how="inner"
    )
    parts.append(
        D.join(pl.DataFrame({"MVT_ID_mvt": s[0], "P": s[1]}), on="MVT_ID_mvt", how="left").with_columns(
            fold=pl.lit(fold), stk_ok=pl.col("P").is_not_null()
        )
    )
T = F.filter(pl.col("is_rank") == 0).join(pl.concat(parts), on="MVT_ID_mvt", how="inner")
T = T.with_columns(taxi=0.5 * (pl.col("taxiL") + pl.col("taxiC")), p=0.5 * (pl.col("pL") + pl.col("pC")))
T = T.with_columns(P=pl.coalesce("P", X.cf_expr_col(0.5 * (pl.col("L") + pl.col("Cb")))))
print("fold rows", T.height, "with stacked P", T["stk_ok"].sum(), flush=True)


def feats(D):
    """nbr_c4_apply C4 features (day form) + tree context columns."""
    D = D.with_columns(
        _pr=(pl.col("P") - pl.col("mvt_aobt3")).clip(-1800, 1800),
        _prt=(pl.col("taxi") - pl.col("mvt_aobt3")).clip(-1800, 1800),
        _prx=(pl.col("taxi") - pl.col("ref_sr_med")).clip(-1800, 1800),
    )

    def f(d):
        t = d["MVT"].to_numpy().astype(float)
        o = {}
        for nm, c in [("PR", "_pr"), ("PRT", "_prt"), ("PRX", "_prx")]:
            v = d[c].to_numpy().astype(float)
            o[nm + "5"], _ = X.nbr_mean(t, v, -300, 300)
            o[nm + "15"], n = X.nbr_mean(t, v, -900, 900)
        o["N15"] = n.astype(float)
        o["PM15"], _ = X.nbr_mean(t, d["p"].to_numpy().astype(float), -900, 900)
        return o

    S = X.group_apply(D, ["ADEP_mvt", "RUNWAY_mvt"], "MVT", f, "")

    def fa(d):
        t = d["MVT"].to_numpy().astype(float)
        o = {}
        o["PAX15"], _ = X.nbr_mean(t, d["_prx"].to_numpy().astype(float), -900, 900)
        return o

    D = D.join(S, on="MVT_ID_mvt", how="left").join(
        X.group_apply(D, ["ADEP_mvt"], "MVT", fa, ""), on="MVT_ID_mvt", how="left"
    )
    cols = ["PR5", "PR15", "PRT5", "PRT15", "PRX5", "PRX15", "PM15", "PAX15"]
    D = D.with_columns(_day=pl.from_epoch(pl.col("MVT").cast(pl.Int64), time_unit="s").dt.strftime("%Y-%m-%d"))
    D = D.with_columns([pl.col(c).fill_nan(None) for c in cols])
    D = D.with_columns([(pl.col(c) - pl.col(c).mean().over(["ADEP_mvt", "_day"])).alias(c + "c") for c in cols])
    D = D.with_columns(
        [(pl.col(c + "c") * (1 - pl.col("lk"))).alias(c + "u") for c in ["PRT15", "PRX5", "PRX15", "PAX15"]]
    )
    return D.with_columns(_dis=(pl.col("L") - pl.col("Cb")).abs().clip(0, 1800), own=pl.col("_pr"), ownx=pl.col("_prx"))


FC = [f if f.endswith("u") else f + "c" for f in FEATS]
TR = FC + ["N15", "lk", "p", "_dis", "own", "ownx", "_apt"]
CATI = [TR.index("_apt")]
T = feats(T).with_columns(e=(pl.col("y") - pl.col("P")).clip(-1800, 1800))
apts = T["ADEP_mvt"].to_numpy()
Xall = T.select(FC).fill_nan(None).fill_null(0).to_numpy()
fo = T["fold"].to_numpy()
ok = ((T["unmatched"] == 0) & T["stk_ok"] & T["e"].is_not_null()).to_numpy()
ee = T["e"].fill_null(0).to_numpy()
ps = 1 - T["p"].fill_null(0.5).to_numpy()
Xt_all = T.select(TR).fill_nan(None).to_numpy().astype(np.float32)
yy = T["y"].to_numpy().astype(float)
applied = ok & ~np.isin(apts, EXCL)  # rows the tree may touch (fold side)


def c4_betas(mask):
    b = {}
    for apt in APTS:
        m = mask & (apts == apt)
        if m.sum() == 0:
            continue
        Xm = Xall[m]
        b[apt] = _solve(Xm.T @ Xm + 1e-6 * np.eye(len(FC)), Xm.T @ ee[m])
    return b


def c4_apply(b, Xf, ap_, psc):
    g = np.zeros(len(Xf))
    for apt, bb in b.items():
        g[ap_ == apt] = Xf[ap_ == apt] @ bb
    return g * psc


def cf_np(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo) - 3606, np.nan_to_num(lo) + 3606))
    return np.maximum(v, 60.0)


lobt = T["mvt_lobt"].to_numpy().astype(float)
Pv = T["P"].to_numpy().astype(float)


def fit_tree(trmask, c4v, seed):
    e2 = np.clip(yy[trmask] - c4v[trmask], -1800, 1800)
    return lgb.train(
        dict(PAR, seed=seed), lgb.Dataset(Xt_all[trmask], e2, categorical_feature=CATI, free_raw_data=True), ROUNDS
    )


def day_demean(g, mask, keys):
    """subtract the per-(airport, day) mean of g over the rows in mask; rows outside mask stay 0."""
    D = pl.DataFrame({"k": keys, "g": g, "msk": mask}).with_columns(
        m=pl.when(pl.col("msk")).then(pl.col("g")).otherwise(None).mean().over("k")
    )
    return np.where(mask, g - D["m"].fill_null(0).to_numpy(), 0.0)


akey_T = (T["ADEP_mvt"] + "|" + T["_day"]).to_numpy()

if a.gate:
    c4 = np.full(T.height, np.nan)
    gt = {s: np.zeros(T.height) for s in SEEDS}
    for fold in X.FOLDS:
        b = c4_betas(ok & (fo != fold))
        c4v = np.where(ok, cf_np(Pv + c4_apply(b, Xall, apts, ps), lobt), Pv)
        te = fo == fold
        c4[te] = c4v[te]
        for s in SEEDS:
            m = fit_tree(applied & (fo != fold) & ~np.isnan(yy), c4v, s)
            gt[s][te] = m.predict(Xt_all[te])
        print("fold", fold, "done", flush=True)
    G = {}
    for s in SEEDS:
        G[f"s{s}"] = np.where(applied, np.clip(gt[s], -a.clip, a.clip), 0.0)
    G["mean"] = np.where(applied, np.clip(np.mean([gt[s] for s in SEEDS], axis=0), -a.clip, a.clip), 0.0)
    G["D"] = day_demean(G["mean"], applied, akey_T)
    T = T.with_columns(_c4=pl.Series(c4))
    for k, g in G.items():
        T = T.with_columns(pl.Series(f"_t{k}", np.where(applied, cf_np(c4 + g, lobt), c4)))
    GG = T.filter(pl.col("stk_ok"))
    X.gate(GG, "_c4", base="P", label="C4 LOFO vs L2 OOF (should = XP -281/-404/-306/-483)")
    if a.save_oof:
        for fold in X.FOLDS:
            Z = GG.filter(pl.col("fold") == fold)
            np.save(
                f"{a.save_oof}_{fold}.npy",
                np.vstack([Z["MVT_ID_mvt"].to_numpy().astype(float), Z["_tD"].to_numpy().astype(float)]),
            )
            np.save(
                f"{a.save_oof}C4_{fold}.npy",
                np.vstack([Z["MVT_ID_mvt"].to_numpy().astype(float), Z["_c4"].to_numpy().astype(float)]),
            )
            print("saved", a.save_oof, fold, Z.height, flush=True)
    for k in G:
        X.gate(GG, f"_t{k}", base="_c4", label=f"tree[{k}] INCREMENT over C4 (clip {a.clip:g}, excl {EXCL})")
        X.gate(GG, f"_t{k}", base="P", label=f"C4 + tree[{k}] vs L2 OOF")
    # level shift of the (clipped, pre-cf) tree output on the fold rows it touches
    L = pl.DataFrame({"apt": apts, "day": T["_day"].to_numpy(), "g": G["mean"], "gD": G["D"], "app": applied}).filter(
        pl.col("app")
    )
    L = L.with_columns(mon=pl.col("day").str.slice(0, 7))
    for col in ["g", "gD"]:
        ld_ = L.group_by(["apt", "day"]).agg(pl.col(col).mean().alias("m"))
        lm = L.group_by(["apt", "mon"]).agg(pl.col(col).mean().alias("m"))
        print(
            f"fold-side level shift [{col}]: per apt x day mean|m| {ld_['m'].abs().mean():.2f} max|m| {ld_['m'].abs().max():.2f} | per apt x month max|m| {lm['m'].abs().max():.2f}"
        )
    print(
        L.group_by(["apt", "mon"])
        .agg(pl.col("g").mean().round(2).alias("mean_g"), pl.col("g").std().round(1).alias("sd_g"))
        .sort(["apt", "mon"])
    )
    # increment by airport x fold (raw dMSE / n_fold)
    nf = T.group_by("fold").len().rename({"len": "nf"})
    B = GG.filter(pl.col("y").is_not_null()).with_columns(
        d=(pl.col("_tmean") - pl.col("y")) ** 2 - (pl.col("_c4") - pl.col("y")) ** 2
    )
    B = (
        B.group_by(["fold", "ADEP_mvt"])
        .agg(pl.col("d").sum())
        .join(nf, on="fold")
        .with_columns(dmse=(pl.col("d") / pl.col("nf")).round(1))
    )
    print(B.pivot(on="fold", index="ADEP_mvt", values="dmse").sort("ADEP_mvt"), flush=True)
if a.dry:
    sys.exit(0)

# production: C4 betas + tree on all four folds
b = c4_betas(ok)
c4v = np.where(ok, cf_np(Pv + c4_apply(b, Xall, apts, ps), lobt), Pv)
models = [fit_tree(applied & ~np.isnan(yy), c4v, s) for s in SEEDS]
print("production trees fitted", flush=True)
V = pl.read_parquet(a.target)
V0 = pl.read_parquet(a.pre)
idc = "MVT_ID_mvt"
vc = [c for c in V.columns if c != idc][0]
assert V[idc].equals(V0[idc])


def seeds(prefix, n):
    fs = [f"{R}/data/preds/{prefix}_s{i}.parquet" for i in range(n)]
    S = (
        pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "p_raw", "taxi"]) for f in fs])
        .group_by("MVT_ID_mvt")
        .agg(pl.col("TAXITIME_SEC_mvt").mean(), pl.col("p_raw").mean(), pl.col("taxi").mean(), pl.len().alias("n"))
    )
    assert (S["n"] == n).all()
    return S.drop("n")


SL = seeds(a.lgb, a.nlgb).rename({"TAXITIME_SEC_mvt": "L", "p_raw": "pL", "taxi": "taxiL"})
SC = seeds(a.cat, a.ncat).rename({"TAXITIME_SEC_mvt": "Cb", "p_raw": "pC", "taxi": "taxiC"})
A = F.filter(pl.col("is_rank") == 1).join(
    V0.select(pl.col(idc).cast(pl.Float64), pl.col(vc).cast(pl.Float64).alias("P0")), on=idc, how="left"
)
A = A.join(V.select(pl.col(idc).cast(pl.Float64), pl.col(vc).cast(pl.Float64).alias("Q0")), on=idc, how="left")
assert A.height == NRANK and A["P0"].null_count() == 0 and A["Q0"].null_count() == 0
A = A.join(SL, on="MVT_ID_mvt", how="left").join(SC, on="MVT_ID_mvt", how="left")
A = A.with_columns(
    P=pl.col("P0"), taxi=0.5 * (pl.col("taxiL") + pl.col("taxiC")), p=0.5 * (pl.col("pL") + pl.col("pC"))
)
A = feats(A)
aa = A["ADEP_mvt"].to_numpy()
um = A["unmatched"].to_numpy() == 1
lo_A = A["mvt_lobt"].to_numpy().astype(float)
gC = c4_apply(b, A.select(FC).fill_nan(None).fill_null(0).to_numpy(), aa, 1 - A["p"].fill_null(0.5).to_numpy())
c4r = np.rint(cf_np(A["P0"].to_numpy() + gC, lo_A))
Q0 = A["Q0"].to_numpy()
print(
    "C4 recomputed vs --target on matched rows: identical %d / %d, max|diff| %.1f"
    % ((c4r == Q0)[~um].sum(), (~um).sum(), np.abs(c4r - Q0)[~um].max())
)
Xr = A.select(TR).fill_nan(None).to_numpy().astype(np.float32)
app_A = ~um & ~np.isin(aa, EXCL)
g = np.where(app_A, np.clip(np.mean([m.predict(Xr) for m in models], axis=0), -a.clip, a.clip), 0.0)
akey_A = (A["ADEP_mvt"] + "|" + A["_day"]).to_numpy()


def build(g, out_path, label):
    new = np.where(app_A, cf_np(Q0 + g, lo_A), Q0)
    B = A.select(idc, "ADEP_mvt", "_day").with_columns(new=pl.Series(new), dd=pl.Series(new - Q0), app=pl.Series(app_A))
    dd = B.filter(pl.col("app"))["dd"]
    print(
        f"[{label}] tree correction on touched rows: mean {dd.mean():.2f} sd {dd.std():.2f} p1 {dd.quantile(.01):.1f} p99 {dd.quantile(.99):.1f} max|g| {dd.abs().max():.1f} n {len(dd)}"
    )
    LD = B.filter(pl.col("app")).group_by(["ADEP_mvt", "_day"]).agg(pl.col("dd").mean().alias("m"))
    LM = (
        B.filter(pl.col("app"))
        .with_columns(mon=pl.col("_day").str.slice(0, 7))
        .group_by(["ADEP_mvt", "mon"])
        .agg(pl.col("dd").mean().round(2).alias("mean_g"), pl.col("dd").std().round(1).alias("sd_g"))
        .sort(["ADEP_mvt", "mon"])
    )
    print(
        f"[{label}] ranking level shift per apt x day: mean|m| {LD['m'].abs().mean():.3f} max|m| {LD['m'].abs().max():.2f} | per apt x month max|m| {LM['mean_g'].abs().max():.2f}"
    )
    print(LM)
    out = V.join(
        B.select(pl.col(idc).alias("_id"), "new"), left_on=pl.col(idc).cast(pl.Float64), right_on="_id", how="left"
    )
    out = out.with_columns(
        pl.coalesce(pl.col("new").round(0) if V.schema[vc].is_integer() else pl.col("new"), vc)
        .cast(V.schema[vc])
        .alias(vc)
    ).select(V.columns)
    uid = A.filter(pl.col("unmatched") == 1)[idc].to_list()
    chk = out.join(V, on=idc, suffix="_o").filter(pl.col(idc).cast(pl.Float64).is_in(uid))
    assert chk.height == len(uid) and (chk[vc] == chk[vc + "_o"]).all(), "unmatched rows changed"
    assert out.height == V.height and out[idc].equals(V[idc])
    ex = (
        out.join(V, on=idc, suffix="_o")
        .join(A.select(pl.col(idc).cast(V.schema[idc]), "ADEP_mvt"), on=idc)
        .filter(pl.col("ADEP_mvt").is_in(EXCL))
    )
    assert (ex[vc] == ex[vc + "_o"]).all(), "excluded airports changed"
    if a.v53:
        W = (
            V.join(pl.read_parquet(a.v53), on=idc, suffix="_53")
            .join(pl.read_parquet(a.pre), on=idc, suffix="_57")
            .join(out, on=idc, suffix="_n")
        )
        print(
            f"[{label}] rows where v59 == v57 == v53 changed in output:",
            W.filter(
                (pl.col(vc) == pl.col(vc + "_53"))
                & (pl.col(vc) == pl.col(vc + "_57"))
                & (pl.col(vc) != pl.col(vc + "_n"))
            ).height,
        )
    if os.path.exists(out_path):
        sys.exit(f"refusing to overwrite {out_path}")
    out.write_parquet(out_path)
    print(
        f"[{label}] wrote",
        out_path,
        out.shape,
        "md5",
        hashlib.md5(open(out_path, "rb").read()).hexdigest(),
        "| rows changed",
        out.join(V, on=idc, suffix="_o").filter(pl.col(vc) != pl.col(vc + "_o")).height,
        "| unmatched byte-identical",
        len(uid),
        flush=True,
    )


build(g, a.out, "raw")
if a.out_d:
    build(day_demean(g, app_A, akey_A), a.out_d, "demeaned apt x day")
