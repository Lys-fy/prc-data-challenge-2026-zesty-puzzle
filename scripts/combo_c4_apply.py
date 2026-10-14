#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Applies a C4 same-day neighbour correction to a candidate submission: day-demeaned state columns of the neighbouring
departures, a least-squares beta per airport, added to the matched ranking rows (level-neutral). Inputs: --target
(submission to correct), --fit-oof (OOF prefix), fold-side member OOF (env assembler_fl / assembler_fc). Output: --out.

Fold-side member OOF prefixes: env assembler_fl / assembler_fc (default runway_queue_ra3q / meta_oof_h_ctl), so that the fold-side taxi and p
branches match the production members (e.g. queue_eb_ra3qa / meta_oof_h_a1 for queue_eb_v47lqa / meta_oof_h_catv17xa1).
 (1) fold-side neighbour sets keep fold rows without a stacked OOF value (unmatched; P = cf(0.5 L + 0.5 C) member OOF), as on the
     ranking side where every departure is a neighbour; betas are fitted on matched rows with a stacked value only;
 (2) --gate: per-airport leave-one-fold-out fold gate of this code path (raw, without the top-3 rows, composition-weighted via gate_composition);
 (3) level-shift report per airport x day and per airport x month on the ranking set; protected rows (target == v53) are checked too.

usage: combo_c4_apply.py --target <submission.parquet> --out <out.parquet> --fit-oof <PREFIX> [--feats PR5,PR15,PRT5,PRT15,PRX5,PRX15,PM15]
                   [--lgb runway_queue_v47lq --nlgb 6 --cat CATV17x --ncat 14] [--dry]
Fold side (fit): frame = features_v9 departures (fold rows) + member OOF (runway_queue_ra3q_*, meta_oof_h_ctl_*: rows p, taxi) + final-vector OOF
  <PREFIX>_<fold>.npy (row 0 MVT_ID, row 1 prediction; janjul may be _months17), searched in data/oof and scratchpad/xz|xl.
Features (same airport x runway, |MVT_j - MVT_i| <= w, self excluded, neighbours without AOBT_3 / reference skipped):
  PRw  = mean clip(P_j - mvt_aobt3_j, +-1800)                (P = final vector: fold OOF when fitting, target when applying)
  PRTw = mean clip(taxi_j - mvt_aobt3_j, +-1800)             (taxi = 0.5 mean LightGBM taxi branch + 0.5 mean CatBoost taxi branch)
  PRXw = mean clip(taxi_j - ref_sr_med_j, +-1800)            (ref_sr_med = features_v9 LOMO stand x runway median of y)
  PM15 = mean p_j                                            (p = 0.5 pL + 0.5 pC)
  PAX15 = airport-wide (all runways) mean clip(taxi_j - ref_sr_med_j, +-1800), +-15 min
  <f>u  = centred <f> x (1 - adsb_linked)   (rows without an own ADS-B clock are smoothed more)
All features centred per airport x DAY (--demean day, default; strictly level-neutral) or per airport x calendar month. beta per airport = least squares of clip(y - P, +-1800) on the
features over the four folds' matched OOF rows. Applied to MATCHED ranking rows only, then cf (clip to mvt_lobt +- 3606, max 60).
Unmatched rows byte-identical (asserted). Writes only --out. Prints betas, correction stats per airport x month, md5."""

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

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--target", required=True, help="candidate submission parquet to correct")
ap.add_argument("--out", required=True, help="output corrected submission parquet")
ap.add_argument("--fit-oof", required=True, help="OOF prefix the per-airport betas are fitted on")
ap.add_argument(
    "--feats",
    default="PR5,PR15,PRT5,PRT15,PRX5,PRX15,PAX15,PRT15u,PRX5u,PRX15u,PAX15u",
    help="comma list of neighbour features",
)
ap.add_argument("--lgb", default="runway_queue_v47lq", help="LightGBM taxi-branch member name prefix")
ap.add_argument("--nlgb", type=int, default=6, help="number of LightGBM seeds to average")
ap.add_argument("--cat", default="CATV17x", help="CatBoost taxi-branch member name prefix")
ap.add_argument("--ncat", type=int, default=14, help="number of CatBoost seeds to average")
ap.add_argument("--dry", action="store_true", help="compute and report only, do not write --out")
ap.add_argument(
    "--demean", default="day", choices=["month", "day"], help="centre features per airport x day or per airport x month"
)
ap.add_argument("--gate", action="store_true", help="run the per-airport leave-one-fold-out gate instead of applying")
ap.add_argument("--pscale", action="store_true", help="scale the correction by the unmatched-probability complement")
ap.add_argument("--v53", default=None, help="protected-rows reference submission (target == v53 rows are checked)")
a = ap.parse_args()
FEATS = a.feats.split(",")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(250)
R = os.environ.get("daystate_root", os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
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
# frozen set = features_v9 unmatched OR no LOBT -> never corrected
print(
    "ranking rows unmatched:",
    F.filter((pl.col("is_rank") == 1) & (pl.col("unmatched") == 1)).height,
    "| + LOBT null:",
    F.filter((pl.col("is_rank") == 1) & ((pl.col("unmatched") == 1) | pl.col("LOBT_flt").is_null())).height,
    flush=True,
)
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
    s = ld(find(a.fit_oof, fold))  # XU: fold member branches configurable
    D = pl.DataFrame({"MVT_ID_mvt": l[0], "L": l[1], "pL": l[2], "taxiL": l[3]}).join(
        pl.DataFrame({"MVT_ID_mvt": c[0], "Cb": c[1], "pC": c[2], "taxiC": c[3]}), on="MVT_ID_mvt", how="inner"
    )
    D = D.join(pl.DataFrame({"MVT_ID_mvt": s[0], "P": s[1]}), on="MVT_ID_mvt", how="left").with_columns(
        fold=pl.lit(fold), stk_ok=pl.col("P").is_not_null()
    )
    print(fold, "member rows", l.shape[1], c.shape[1], "final-vector rows", s.shape[1], "joined", D.height, flush=True)
    parts.append(D)
O = pl.concat(parts)
T = F.filter(pl.col("is_rank") == 0).join(O, on="MVT_ID_mvt", how="inner")
T = T.with_columns(taxi=0.5 * (pl.col("taxiL") + pl.col("taxiC")), p=0.5 * (pl.col("pL") + pl.col("pC")))
T = T.with_columns(P=pl.coalesce("P", X.cf_expr_col(0.5 * (pl.col("L") + pl.col("Cb")))))
print("fold rows", T.height, "with stacked P", T["stk_ok"].sum(), "matched", (T["unmatched"] == 0).sum(), flush=True)


def feats(D):
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
            o[nm + "15"], _ = X.nbr_mean(t, v, -900, 900)
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
    if a.demean == "month":
        D = X.demean(D, cols)
        return D.with_columns(
            [(pl.col(c + "c") * (1 - pl.col("lk"))).alias(c + "u") for c in ["PRT15", "PRX5", "PRX15", "PAX15"]]
        )
    D = D.with_columns(_day=pl.from_epoch(pl.col("MVT").cast(pl.Int64), time_unit="s").dt.strftime("%Y-%m-%d"))
    D = D.with_columns([pl.col(c).fill_nan(None) for c in cols])
    D = D.with_columns([(pl.col(c) - pl.col(c).mean().over(["ADEP_mvt", "_day"])).alias(c + "c") for c in cols]).drop(
        "_day"
    )
    return D.with_columns(
        [(pl.col(c + "c") * (1 - pl.col("lk"))).alias(c + "u") for c in ["PRT15", "PRX5", "PRX15", "PAX15"]]
    )


FC = [f if f.endswith("u") else f + "c" for f in FEATS]
T = feats(T).with_columns(e=(pl.col("y") - pl.col("P")).clip(-1800, 1800))
betas = {}
for apt in sorted(T["ADEP_mvt"].unique().to_list()):
    M = T.filter(
        (pl.col("ADEP_mvt") == apt) & (pl.col("unmatched") == 0) & pl.col("stk_ok") & pl.col("e").is_not_null()
    )
    Xm = M.select(FC).fill_nan(None).fill_null(0).to_numpy()
    betas[apt] = _solve(Xm.T @ Xm + 1e-6 * np.eye(len(FC)), Xm.T @ M["e"].to_numpy())
print("feats", FEATS, "fit-oof", a.fit_oof)
print("beta:", {k: np.round(v, 4).tolist() for k, v in betas.items()}, flush=True)
# in-sample fold check of the fitted correction (not a gate: beta is fitted on all four folds)
g = np.zeros(T.height)
apts = T["ADEP_mvt"].to_numpy()
Xall = T.select(FC).fill_nan(None).fill_null(0).to_numpy()
for apt, b in betas.items():
    g[apts == apt] = Xall[apts == apt] @ b
if a.pscale:
    g = g * (
        1 - T["p"].fill_null(0.5).to_numpy()
    )  # XW refinement: correction acts on the taxi branch only (post, no refit)
T = T.with_columns(_g=pl.Series(g)).with_columns(
    _c=X.cf_expr_col(
        pl.when((pl.col("unmatched") == 0) & pl.col("stk_ok")).then(pl.col("P") + pl.col("_g")).otherwise(pl.col("P"))
    )
)
for fold in X.FOLDS:
    D = T.filter((pl.col("fold") == fold) & pl.col("y").is_not_null() & pl.col("stk_ok"))
    d = ((D["_c"] - D["y"]) ** 2 - (D["P"] - D["y"]) ** 2).to_numpy()
    n = T.filter(pl.col("fold") == fold).height
    print(
        (
            f"  in-sample fold {fold}: dMSE {d.sum()/n:+.1f} (w/o top3 {np.sort(d)[:-3].sum()/n:+.1f})"
            if False
            else f"  in-sample fold {fold}: dMSE {d.sum()/n:+.1f}"
        ),
        flush=True,
    )
if a.gate:
    # LOFO gate of this exact code path: beta per airport from the other three folds, scored on the held-out fold (after the L2 stacker)
    T = (
        T.join(F.select("MVT_ID_mvt", "adsb_linked"), on="MVT_ID_mvt", how="left")
        if "adsb_linked" not in T.columns
        else T
    )
    gl = np.zeros(T.height)
    fo = T["fold"].to_numpy()
    ok = ((T["unmatched"] == 0) & T["stk_ok"] & T["e"].is_not_null()).to_numpy()
    ee = T["e"].fill_null(0).to_numpy()
    for fold in X.FOLDS:
        for apt in betas:
            tr = (apts == apt) & (fo != fold) & ok
            te = (apts == apt) & (fo == fold)
            Xm = Xall[tr]
            b = _solve(Xm.T @ Xm + 1e-6 * np.eye(len(FC)), Xm.T @ ee[tr])
            gl[te] = Xall[te] @ b
    if a.pscale:
        gl = gl * (1 - T["p"].fill_null(0.5).to_numpy())
    T = T.with_columns(_gl=pl.Series(gl)).with_columns(
        _cl=X.cf_expr_col(
            pl.when((pl.col("unmatched") == 0) & pl.col("stk_ok"))
            .then(pl.col("P") + pl.col("_gl"))
            .otherwise(pl.col("P"))
        )
    )
    G = T.filter(pl.col("stk_ok"))
    X.gate(
        G,
        "_cl",
        base="P",
        label=f"LOFO gate {a.fit_oof} feats={a.feats} demean={a.demean} (raw (w/o top3) c=composition-weighted)",
    )
    TT = G.with_columns(
        mon=pl.from_epoch(pl.col("MVT").cast(pl.Int64), time_unit="s").dt.month(),
        d=(pl.col("_cl") - pl.col("y")) ** 2 - (pl.col("P") - pl.col("y")) ** 2,
    ).filter(pl.col("y").is_not_null())
    nf = T.group_by("fold").len().rename({"len": "nf"})
    TT = (
        TT.group_by(["fold", "ADEP_mvt", "mon"])
        .agg(pl.col("d").sum())
        .join(nf, on="fold")
        .with_columns(dmse=(pl.col("d") / pl.col("nf")).round(1))
    )
    print(TT.pivot(on="mon", index="ADEP_mvt", values="dmse", sort_columns=True).sort("ADEP_mvt"))
    print(
        "fold totals excl. LTFM:",
        TT.filter(pl.col("ADEP_mvt") != "LTFM").group_by("fold").agg(pl.col("dmse").sum().round(1)).sort("fold").rows(),
        flush=True,
    )
if a.dry:
    sys.exit(0)

# ranking side
V = pl.read_parquet(a.target)
idc = "MVT_ID_mvt"
vc = [c for c in V.columns if c != idc][0]


def seeds(prefix, n):
    fs = [f"{R}/data/preds/{prefix}_s{i}.parquet" for i in range(n)]
    for f in fs:
        assert os.path.exists(f), f
    S = (
        pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "p_raw", "taxi"]) for f in fs])
        .group_by("MVT_ID_mvt")
        .agg(pl.col("p_raw").mean(), pl.col("taxi").mean(), pl.len().alias("n"))
    )
    assert (S["n"] == n).all()
    return S.drop("n")


SL = seeds(a.lgb, a.nlgb).rename({"p_raw": "pL", "taxi": "taxiL"})
SC = seeds(a.cat, a.ncat).rename({"p_raw": "pC", "taxi": "taxiC"})
A = F.filter(pl.col("is_rank") == 1).join(
    V.select(pl.col(idc).cast(pl.Float64), pl.col(vc).cast(pl.Float64).alias("P0")), on=idc, how="left"
)
assert A.height == NRANK and A["P0"].null_count() == 0, A.height
A = A.join(SL, on="MVT_ID_mvt", how="left").join(SC, on="MVT_ID_mvt", how="left")
print(
    "ranking rows with member branches:",
    A.filter(pl.col("taxiL").is_not_null() & pl.col("taxiC").is_not_null()).height,
    "matched:",
    A.filter(pl.col("unmatched") == 0).height,
)
A = A.with_columns(
    P=pl.col("P0"), taxi=0.5 * (pl.col("taxiL") + pl.col("taxiC")), p=0.5 * (pl.col("pL") + pl.col("pC"))
)
A = feats(A)
apts = A["ADEP_mvt"].to_numpy()
Xr = A.select(FC).fill_nan(None).fill_null(0).to_numpy()
g = np.zeros(A.height)
for apt, b in betas.items():
    g[apts == apt] = Xr[apts == apt] @ b
if a.pscale:
    g = g * (1 - A["p"].fill_null(0.5).to_numpy())
g = np.where(A["unmatched"].to_numpy() == 0, g, 0.0)
A = A.with_columns(g=pl.Series(g)).with_columns(
    new=pl.when(pl.col("unmatched") == 1).then(pl.col("P0")).otherwise(X.cf_expr_col(pl.col("P0") + pl.col("g")))
)
dd = A.filter(pl.col("unmatched") == 0).select(pl.col("new") - pl.col("P0")).to_series()
print(
    "correction on matched rows: mean %.2f sd %.2f p1 %.1f p99 %.1f max|g| %.1f; rows changed %d"
    % (
        dd.mean(),
        dd.std(),
        dd.quantile(0.01),
        dd.quantile(0.99),
        dd.abs().max(),
        A.filter(pl.col("new") != pl.col("P0")).height,
    )
)
A = A.with_columns(
    _dd=pl.col("new") - pl.col("P0"),
    _day=pl.from_epoch(pl.col("MVT").cast(pl.Int64), time_unit="s").dt.strftime("%Y-%m-%d"),
)
LD = (
    A.filter(pl.col("unmatched") == 0)
    .group_by(["ADEP_mvt", "_day"])
    .agg(pl.col("_dd").mean().alias("m"), pl.len().alias("n"))
)
print(
    "level shift per airport x day (matched rows): mean|m| %.3f  max|m| %.2f  (n days %d)"
    % (LD["m"].abs().mean(), LD["m"].abs().max(), LD.height)
)
print(LD.sort(pl.col("m").abs(), descending=True).head(5))
print(
    A.with_columns(mon=pl.from_epoch(pl.col("MVT").cast(pl.Int64), time_unit="s").dt.month())
    .group_by(["ADEP_mvt", "mon"])
    .agg(
        (pl.col("new") - pl.col("P0")).mean().round(2).alias("mean_g"),
        (pl.col("new") - pl.col("P0")).std().round(1).alias("sd_g"),
    )
    .sort(["ADEP_mvt", "mon"])
)
out = V.join(
    A.select(pl.col(idc).alias("_id"), "new"), left_on=pl.col(idc).cast(pl.Float64), right_on="_id", how="left"
)
out = out.with_columns(
    pl.coalesce(pl.col("new").round(0) if V.schema[vc].is_integer() else pl.col("new"), vc).cast(V.schema[vc]).alias(vc)
).select(
    V.columns
)  # round, not truncate, for integer targets
print("target dtype", V.schema[vc])
um = A.filter(pl.col("unmatched") == 1)[idc].to_list()
chk = out.join(V, on=idc, suffix="_o").filter(pl.col(idc).cast(pl.Float64).is_in(um))
assert chk.height == len(um) and (chk[vc] == chk[vc + "_o"]).all(), "unmatched rows changed"
assert out.height == V.height and out[idc].equals(V[idc])
if a.v53:
    W = V.join(pl.read_parquet(a.v53), on=idc, suffix="_53").join(out, on=idc, suffix="_n")
    fz = W.filter(pl.col(vc) == pl.col(vc + "_53"))
    print(
        "rows with target == v53 (frozen set):",
        fz.height,
        "| changed in output:",
        fz.filter(pl.col(vc) != pl.col(vc + "_n")).height,
    )
out.write_parquet(a.out)
print(
    "wrote",
    a.out,
    out.shape,
    "md5",
    hashlib.md5(open(a.out, "rb").read()).hexdigest(),
    "| unmatched rows byte-identical:",
    len(um),
)
