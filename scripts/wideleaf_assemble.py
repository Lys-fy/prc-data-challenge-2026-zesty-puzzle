#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Production pre-stage file of one chain: runs scripts/audited_assemble.py with the stacker source env STACKSRC (default scripts/wideleaf_stack.py).
If EXTRA contains xl, the eight ranking-side same-day state columns PR5c..PAX15c are added before the meta predicts; they are computed from
the production members SL x NSL (default queue_eb_v47lqa x 6, the production twin of the fold-side OL=queue_eb_ra3qa) and PCT x NCAT
(default CATV17x x 14) with the formulas of the fold side, which reads scratchpad/p13/prep_<fold>.parquet (6 folds).
If EXTRA contains xpr, the PRU daily inputs of the ranking rows are added (scripts/pru_lib.py).
If EXTRA contains xtd: the ranking-side extra member T = p*art + (1-p)*taxi of PXTD x NXTD (default wideleaf_a2p x 3) from data/preds.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
os.environ.setdefault("STACKSRC", "scripts/wideleaf_stack.py")
os.environ.setdefault("SL", "queue_eb_v47lqa")
S = open("scripts/audited_assemble.py").read()
assert S.count("gc = np.clip(meta.predict") == 1
if "xl" in os.environ.get("EXTRA", ""):
    S = S.replace("gc = np.clip(meta.predict", "_xdv_state(d, ids, m)\ngc = np.clip(meta.predict")
PRU_RANK_BLOCK = 'if "xpr" in extra_inputs:   # pru: ranking side, same code as the fold side; base = production v on matched rows (unmatched -> median proxy)\n    import pru_lib\n    Zr = pru_lib.feats(1, ids.astype(float)[m], v[m]); pru_lib.report(Zr, "rank2026"); Zr = Zr.sort("MVT_ID"); zi = Zr["MVT_ID"].to_numpy()\n    kk = np.searchsorted(zi, ids.astype(float)); assert np.array_equal(zi[kk], ids.astype(float))\n    for c in pru_lib.COLS: d[c] = Zr[c].to_numpy().astype(float)[kk]\n'
if "xpr" in os.environ.get("EXTRA", ""):
    S = S.replace(
        "gc = np.clip(meta.predict", PRU_RANK_BLOCK + "gc = np.clip(meta.predict"
    )  # P13: pru_assemble ranking-side block, verbatim
EXTRA1_BLOCK = (
    'if "xtd" in extra_inputs:   # WIDELEAF ( step 2): T = p*art + (1-p)*taxi of the production A2 member (seed mean of taxi, env PXTD x NXTD) on matched ranking rows, NaN elsewhere\n'
    '    NXTD = int(E("NXTD", "3")); iT, TX = avg(E("PXTD", "wideleaf_a2p"), NXTD, ["taxi"]); assert np.array_equal(iT, ids)\n'
    '    tx = np.where(m, TX[0], np.nan); d["T"] = p * art + (1 - p) * tx; d["TmL"] = d["T"] - Lv; print(f"target_diversity: T on {np.isfinite(d[\'T\']).sum():,} ranking rows ({NXTD} seeds)")\n'
)
if "xtd" in os.environ.get("EXTRA", ""):
    assert S.count('if "xapt" in extra_inputs:') == 1
    S = S.replace('if "xapt" in extra_inputs:', EXTRA1_BLOCK + 'if "xapt" in extra_inputs:')  # WIDELEAF

daystate_meta = ["PR5c", "PR15c", "PRT5c", "PRT15c", "PRX5c", "PRX15c", "PM15c", "PAX15c"]
FEATS = [c[:-1] for c in daystate_meta]


def nbr_mean(t, v, lo, hi):  # = icl_stacker_prep.nbr_mean
    ok = ~np.isnan(v)
    vv = np.where(ok, v, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(vv)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    a = np.searchsorted(t, t + lo)
    b = np.searchsorted(t, t + hi, side="right")
    s = cs[b] - cs[a] - vv
    n = cn[b] - cn[a] - ok
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def group_apply(D, keys, fn):  # = icl_stacker_prep.group_apply
    res = []
    for _, d in D.sort("MVT").group_by(keys, maintain_order=True):
        cols = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        cols.update(fn(d))
        res.append(pl.DataFrame(cols))
    return pl.concat(res)


def seeds(prefix, n):
    fs = [f"data/preds/{prefix}_s{i}.parquet" for i in range(n)]
    for f in fs:
        assert os.path.exists(f), f
    Z = (
        pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "p_raw", "taxi"]) for f in fs])
        .group_by("MVT_ID_mvt")
        .agg(
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).mean(),
            pl.col("p_raw").mean(),
            pl.col("taxi").mean(),
            pl.len().alias("n"),
        )
    )
    assert (Z["n"] == n).all()
    return Z.drop("n").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))


def _xdv_state(d, ids, m):
    FC = [
        "MVT_ID_mvt",
        "is_rank",
        "is_dep",
        "ADEP_mvt",
        "RUNWAY_mvt",
        "MVT_TIME_UTC_mvt",
        "mvt_lobt",
        "mvt_aobt3",
        "ref_sr_med",
    ]
    F = (
        pl.read_parquet("data/features_v9.parquet", columns=FC)
        .filter((pl.col("is_rank") == 1) & (pl.col("is_dep") == 1))
        .with_columns(
            pl.col("MVT_ID_mvt").cast(pl.Float64), MVT=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)
        )
    )
    SLp, NSL = os.environ.get("SL", "runway_queue_v47lq"), int(os.environ.get("NSL", "6"))
    L = seeds(SLp, NSL).rename({"TAXITIME_SEC_mvt": "Lp", "p_raw": "pLa", "taxi": "tL"})
    C = seeds(os.environ.get("PCT", "CATV17x"), int(os.environ.get("NCAT", "14"))).rename(
        {"TAXITIME_SEC_mvt": "Cp", "p_raw": "pCa", "taxi": "tC"}
    )
    A = F.join(L, on="MVT_ID_mvt", how="inner").join(C, on="MVT_ID_mvt", how="inner")
    assert A.height == F.height, (A.height, F.height)
    lo = A["mvt_lobt"].to_numpy().astype(float)
    v = 0.5 * A["Lp"].to_numpy() + 0.5 * A["Cp"].to_numpy()
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo) - 3606, np.nan_to_num(lo) + 3606))
    v = np.maximum(v, 60.0)
    A = A.with_columns(inc=pl.Series(v)).with_columns(
        _tx=0.5 * (pl.col("tL") + pl.col("tC")), _pm=0.5 * (pl.col("pLa") + pl.col("pCa"))
    )
    A = A.with_columns(
        _pr=(pl.col("inc") - pl.col("mvt_aobt3")).clip(-1800, 1800),
        _prt=(pl.col("_tx") - pl.col("mvt_aobt3")).clip(-1800, 1800),
        _prx=(pl.col("_tx") - pl.col("ref_sr_med")).clip(-1800, 1800),
    )

    def f_rwy(x):
        tt = x["MVT"].to_numpy()
        o = {}
        for nm, c in (("PR", "_pr"), ("PRT", "_prt"), ("PRX", "_prx")):
            vv = x[c].to_numpy().astype(float)
            o[nm + "5"] = nbr_mean(tt, vv, -300, 300)
            o[nm + "15"] = nbr_mean(tt, vv, -900, 900)
        o["PM15"] = nbr_mean(tt, x["_pm"].to_numpy().astype(float), -900, 900)
        return o

    A = A.join(group_apply(A, ["ADEP_mvt", "RUNWAY_mvt"], f_rwy), on="MVT_ID_mvt", how="left")
    A = A.join(
        group_apply(
            A,
            ["ADEP_mvt"],
            lambda x: {"PAX15": nbr_mean(x["MVT"].to_numpy(), x["_prx"].to_numpy().astype(float), -900, 900)},
        ),
        on="MVT_ID_mvt",
        how="left",
    )
    A = A.with_columns(_mo=pl.col("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m")).with_columns(
        [pl.col(c).fill_nan(None) for c in FEATS]
    )
    A = A.with_columns([(pl.col(c) - pl.col(c).mean().over(["ADEP_mvt", "_mo"])).alias(c + "c") for c in FEATS])
    R = pl.DataFrame({"MVT_ID_mvt": ids.astype(float)}).join(
        A.select(["MVT_ID_mvt"] + daystate_meta), on="MVT_ID_mvt", how="left", maintain_order="left"
    )
    assert R.height == len(ids)
    fold = pl.concat(
        [
            pl.read_parquet(f"scratchpad/p13/prep_{t}.parquet", columns=daystate_meta)
            for t in ("months39", "months28", "months410", "janjul", "months612", "months511")
        ]
    )
    for c in daystate_meta:
        d[c] = R[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
        x = d[c][m]
        fz = fold[c].cast(pl.Float64).drop_nulls().to_numpy()
        print(
            f"state {c}: ranking matched mean {np.nanmean(x):+.2f} sd {np.nanstd(x):.1f} null {np.isnan(x).mean():.3f} | fold mean {fz.mean():+.2f} sd {fz.std():.1f} null {fold[c].null_count()/fold.height:.3f}",
            flush=True,
        )


g = {"__file__": os.path.abspath("scripts/audited_assemble.py"), "__name__": "__main__", "_xdv_state": _xdv_state}
exec(compile(S, "xrt_assemble_p13", "exec"), g)
