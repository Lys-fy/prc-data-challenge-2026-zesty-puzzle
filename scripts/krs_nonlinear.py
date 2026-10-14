#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Production form of the end-of-chain KRS re-step, for the current ranking file and for new months (scripts/run_final.sh).
Two subcommands: `fit MODE` fits the prod coefficients and writes them to scratchpad/xkrsnl/prod/; `build MODE BASE
TW1_IDS OUT` re-steps BASE into OUT via krsnl_line() and isolate_delta.py. Inputs for `fit`/`build` and their env
overrides (ROWS, KPROD, SPPROD, V107COEF, COEFDIR) are listed below.

  fit   MODE                      fit the prod coefficients on ALL 6 folds' UNS rows (scratchpad/xmonths/rows.parquet + scratchpad/xkrsnl fold vectors)
                                   -> scratchpad/xkrsnl/prod/{coef.json, gbm_<MODE>.txt}.  MODE in A | B | AB.
  build MODE BASE TW1_IDS OUT     OUT = BASE (pre-KRS line, no KRS step) re-stepped by krsnl_line(), unmatched / exposed / TW1' rows kept
                                   (TW1' shift added via env TW1ADD="<base_tw1p>:<base>" exactly as krs_build.py), then isolate_delta.py
                                   (protections, LOBT law, floor 60, validator; env BASEMD5, VN). MODE in v107 (= v107's own tercile KRS, regression) | A | B | AB.
  env  ROWS (fit input, default scratchpad/xmonths/rows.parquet from scripts/months_rows_table.py)
       KPROD (default data/preds/KSBWPc100P_s0.parquet: MVT_ID_mvt, taxi)   SPPROD (default data/oof/KSBWPc100_prod.npy: rows id, taxi, spread)
       V107COEF (default scratchpad/xchain2/v106K_K2_KRS_prod_coef.json)     COEFDIR (default scratchpad/xkrsnl/prod)

The row function (no file I/O; for any month):
  krsnl_line(base, K, sp, apt, hour, month, mode, C) -> float line
    v107: base + beta_k (K - base), k = spread tercile (v107 coefficients)
    A   : L = v107 line; L + 0.5 * g(L, K - L, sp, apt, hour)        (g = LightGBM on clip(y - L, +-1800), all-6-fold fit)
    B   : base + beta_{month,k} (K - base)                            (month-kernel beta, sigma 1.5 months, one triple per calendar month 1..12)
    AB  : L = B line; L + 0.5 * g_AB(L, K - L, sp, apt, hour)
  Rows with K or sp missing keep base. apt = ADEP ICAO string, hour = UTC hour of MVT_TIME_UTC_mvt, month = its calendar month.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, subprocess, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
c = "TAXITIME_SEC_mvt"
NT = int(os.environ.get("NT", "6"))
CD = os.environ.get("COEFDIR", "scratchpad/xkrsnl/prod")
PRM = dict(
    objective="l2",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=2000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    seed=0,
    num_threads=NT,
)
NAMES = ["line", "KmL", "sp", "apt", "hour"]


def load_coef(mode):
    C = json.load(open(f"{CD}/coef.json")) if mode in ("A", "B", "AB") else {}
    v = json.load(open(os.environ.get("V107COEF", "scratchpad/xchain2/v106K_K2_KRS_prod_coef.json")))
    C["v107"] = dict(edges=v["edges"]["None"], beta=[v["beta"][str(k)] for k in range(3)])
    if mode in ("A", "AB"):
        import lightgbm as lgb

        C["gbm"] = lgb.Booster(model_file=f"{CD}/gbm_{mode}.txt")
    return C


def _g(C, L, K, sp, apt, hour):
    a = np.array([C["apts"].index(x) if x in C["apts"] else np.nan for x in apt], dtype=float)
    F = np.column_stack([L, K - L, sp, a, hour]).astype(np.float32)
    return C["gbm"].predict(F, num_threads=NT)


def krsnl_line(base, K, sp, apt, hour, month, mode, C):
    base = np.asarray(base, float)
    K = np.asarray(K, float)
    sp = np.asarray(sp, float)
    out = base.copy()
    h = np.isfinite(K) & np.isfinite(sp)
    if mode in ("v107", "A"):
        e = np.array(C["v107"]["edges"])
        b = np.array(C["v107"]["beta"])
        out[h] = base[h] + b[np.digitize(sp[h], e)] * (K[h] - base[h])
    elif mode in ("B", "AB"):
        e = np.array(C["B"]["edges"])
        mo = np.asarray(month, int)
        for m in np.unique(mo[h]):
            j = h & (mo == m)
            b = np.array(C["B"]["beta"][str(int(m))])
            out[j] = base[j] + b[np.digitize(sp[j], e)] * (K[j] - base[j])
    else:
        raise SystemExit(f"mode {mode}")
    if mode in ("A", "AB"):
        out[h] = out[h] + 0.5 * _g(C, out[h], K[h], sp[h], np.asarray(apt)[h], np.asarray(hour, float)[h])
    return out


def fit(mode):
    import lightgbm as lgb

    os.makedirs(CD, exist_ok=True)
    A = pl.read_parquet(os.environ.get("ROWS", "scratchpad/xmonths/rows.parquet"))
    FO = ["months39", "months28", "months410", "janjul", "months511", "months612"]
    y = A["y"].to_numpy()
    P = A["post"].to_numpy()
    Kv = A["K"].to_numpy()
    sp = A["sp"].to_numpy()
    mon = A["month"].to_numpy().astype(int)
    U = ~A["sealed"].fill_null(False).to_numpy()
    h = np.isfinite(Kv) & np.isfinite(sp)
    fm = U & h
    C = json.load(open(f"{CD}/coef.json")) if os.path.exists(f"{CD}/coef.json") else {}
    C["apts"] = sorted(set(A["apt"].to_list()))
    # B: month-kernel beta per calendar month (all 6 folds' UNS rows, plain weights, sigma 1.5)
    e = np.quantile(sp[fm], [1 / 3, 2 / 3])
    bk = np.digitize(sp, e)
    x = Kv - P
    r = y - P
    B = {}
    for M in range(1, 13):
        d = np.abs(mon - M) % 12
        d = np.minimum(d, 12 - d)
        kw = np.exp(-(d**2) / (2 * 1.5**2))
        B[str(M)] = [
            float((kw[j] * x[j] * r[j]).sum() / (kw[j] * x[j] * x[j]).sum()) for j in (fm & (bk == k) for k in range(3))
        ]
    C["B"] = dict(edges=e.tolist(), beta=B, sigma=1.5, fit="all 6 folds UNS rows, plain weights")
    print(
        "[fit] B edges",
        np.round(e, 2).tolist(),
        "| beta " + " ".join(f"{M}:{np.round(B[str(M)], 3).tolist()}" for M in range(1, 13)),
        flush=True,
    )
    if mode in ("A", "AB"):
        # base line for the GBM fit = the fold-honest line vector (A: the v107 fold line; AB: the fold B arm), as in the gate
        if mode == "A":
            L = A["v107"].to_numpy()
        else:
            ids = A["id"].to_numpy()
            fold = A["fold"].to_numpy()
            L = np.full(len(y), np.nan)
            for t in FO:
                z = np.load(f"scratchpad/xkrsnl/B_{t}.npy")
                o = np.argsort(z[0])
                m = fold == t
                pos = np.searchsorted(z[0][o], ids[m])
                L[m] = z[1][o][pos]
        a = A["apt"].replace_strict(C["apts"], list(range(len(C["apts"])))).cast(pl.Float64).to_numpy()
        F = np.column_stack([L, Kv - L, sp, a, A["hour"].cast(pl.Float64).to_numpy()]).astype(np.float32)
        idx = np.flatnonzero(fm)
        g = lgb.train(
            PRM,
            lgb.Dataset(
                F[idx],
                np.clip(y - L, -1800, 1800)[idx],
                feature_name=NAMES,
                categorical_feature=[3],
                free_raw_data=True,
            ),
            300,
        )
        g.save_model(f"{CD}/gbm_{mode}.txt")
        print(f"[fit] gbm_{mode} on {len(idx):,} rows saved", flush=True)
    C["meta"] = dict(note="", prm=PRM, shrink=0.5, features=NAMES)
    json.dump(C, open(f"{CD}/coef.json", "w"), indent=1)


def build(mode, BASE, TW, OUT):
    import clockresid_lib as L

    C = load_coef(mode)
    v = pl.read_parquet(BASE).with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
    K = pl.read_parquet(os.environ.get("KPROD", "data/preds/KSBWPc100P_s0.parquet")).select(
        mid=pl.col("MVT_ID_mvt").cast(pl.Float64), K="taxi"
    )
    z = np.load(os.environ.get("SPPROD", "data/oof/KSBWPc100_prod.npy"))
    S = pl.DataFrame({"mid": z[0].astype(float), "sp": z[2].astype(float)})
    J = v.join(K, on="mid", how="left", maintain_order="left").join(S, on="mid", how="left", maintain_order="left")
    Fr = L.frame(J["mid"].to_numpy())
    um = Fr["unmatched"].fill_null(1).to_numpy() == 1
    ap = Fr["ADEP_mvt"].cast(pl.Utf8).fill_null("?").to_numpy()
    ts = Fr["MVT_TIME_UTC_mvt"]
    hour = ts.dt.hour().cast(pl.Float64).to_numpy()
    month = ts.dt.month().fill_null(1).to_numpy()
    ex = (
        pl.read_parquet("scratchpad/xhr/exposed_rows.parquet", columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
        .cast(pl.Float64)
        .to_numpy()
    )
    tw = pl.read_parquet(TW)["MVT_ID_mvt"].cast(pl.Float64).to_numpy()
    mid = J["mid"].to_numpy()
    keep = um | np.isin(mid, ex) | np.isin(mid, tw)
    p = J[c].to_numpy().astype(float)
    new = krsnl_line(p, J["K"].to_numpy(), J["sp"].to_numpy(), ap, hour, month, mode, C)
    new = np.where(keep, p, new)
    if os.environ.get("TW1ADD"):
        a_, b_ = os.environ["TW1ADD"].split(":")
        T1 = (
            pl.read_parquet(a_)
            .join(pl.read_parquet(b_), on="MVT_ID_mvt", suffix="_b")
            .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64), s=(pl.col(c) - pl.col(c + "_b")).cast(pl.Float64))
        )
        sh = (
            J.select("mid")
            .join(T1.select("mid", "s"), on="mid", how="left", maintain_order="left")["s"]
            .fill_null(0.0)
            .to_numpy()
        )
        istw = np.isin(mid, tw)
        assert (sh[~istw] == 0).all()
        new = np.where(istw, p + sh, new)
        print(f"TW1ADD: shift on {int((sh != 0).sum())} TW1 rows", flush=True)
    print(
        f"KRS_NONLINEAR build {mode}: rows {len(p):,} kept {int(keep.sum())} | months {np.unique(month[~keep]).tolist()} | delta vs base mean {np.mean(new - p):+.3f} RMS {np.sqrt(np.mean((new - p) ** 2)):.2f}",
        flush=True,
    )
    A_ = OUT.replace(".parquet", "_A.parquet")
    v.select("MVT_ID_mvt").with_columns(pl.Series(c, np.rint(np.maximum(new, 60))).cast(v[c].dtype)).write_parquet(A_)
    r = subprocess.run([".venv/bin/python", "scripts/isolate_delta.py", OUT, A_, BASE, BASE], env=dict(os.environ))
    sys.exit(r.returncode)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2])
    elif cmd == "build":
        build(*sys.argv[2:6])
    else:
        raise SystemExit(__doc__)
