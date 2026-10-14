#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the fold-side same-day state columns PR5c..PAX15c on all 6 folds (inputs of EXTRA xl of the chain stacker).
Reads env OL (default queue_eb_ra3qa) and OC (default meta_oof_h_ctl), the pooled chains' fold member OOF, plus
data/features_v9.parquet. Writes scratchpad/p13/prep_<fold>.parquet [MVT_ID_mvt + 8 columns] for all fold departures.

Neighbour means per airport x runway within +-5 / +-15 min, self excluded; PAX15 per airport; demeaned per airport x
calendar month."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OL, OC = os.environ.get("OL", "queue_eb_ra3qa"), os.environ.get("OC", "meta_oof_h_ctl")
OUT = "scratchpad/p13"
os.makedirs(OUT, exist_ok=True)
FO = ["months39", "months28", "months410", "janjul", "months612", "months511"]


def npy(p, t):
    c = (
        glob.glob(f"data/oof/{p}_{t}.npy")
        or glob.glob(f"data/oof/{p}_{t}_*.npy")
        or (glob.glob(f"data/oof/{p}_months17.npy") if t == "janjul" else [])
    )
    return np.load(sorted(c)[0])


def al(x, ids):
    o = np.argsort(x[0])
    k = np.searchsorted(x[0][o], ids)
    assert np.array_equal(x[0][o][k], ids)
    return x[1:, o][:, k]


def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606))
    return np.maximum(v, 60.0)


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


FC = [
    "MVT_ID_mvt",
    "is_rank",
    "is_dep",
    "ADEP_mvt",
    "RUNWAY_mvt",
    "MVT_TIME_UTC_mvt",
    "unmatched",
    "mvt_lobt",
    "mvt_aobt3",
    "ref_sr_med",
]
F = pl.read_parquet("data/features_v9.parquet", columns=FC).with_columns(
    MVT=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)
)
FEATS = ["PR5", "PR15", "PRT5", "PRT15", "PRX5", "PRX15", "PM15", "PAX15"]
daystate_meta = [c + "c" for c in FEATS]
for t in sys.argv[1:] or FO:
    L = npy(OL, t)
    ids = L[0]
    C = al(npy(OC, t), ids)
    A = (
        pl.DataFrame({"MVT_ID_mvt": ids, "Lp": L[1], "pLa": L[2], "tL": L[3], "Cp": C[0], "pCa": C[1], "tC": C[2]})
        .join(F.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)), on="MVT_ID_mvt", how="inner")
        .filter(pl.col("is_dep") == 1)
    )
    lo = A["mvt_lobt"].to_numpy().astype(float)
    A = A.with_columns(inc=pl.Series(cf(0.5 * A["Lp"].to_numpy() + 0.5 * A["Cp"].to_numpy(), lo)))
    A = A.with_columns(_tx=0.5 * (pl.col("tL") + pl.col("tC")), _pm=0.5 * (pl.col("pLa") + pl.col("pCa")))
    A = A.with_columns(
        _pr=(pl.col("inc") - pl.col("mvt_aobt3")).clip(-1800, 1800),
        _prt=(pl.col("_tx") - pl.col("mvt_aobt3")).clip(-1800, 1800),
        _prx=(pl.col("_tx") - pl.col("ref_sr_med")).clip(-1800, 1800),
    )

    def f_rwy(x):
        tt = x["MVT"].to_numpy()
        o = {}
        for nm, c in (("PR", "_pr"), ("PRT", "_prt"), ("PRX", "_prx")):
            v = x[c].to_numpy().astype(float)
            o[nm + "5"] = nbr_mean(tt, v, -300, 300)
            o[nm + "15"] = nbr_mean(tt, v, -900, 900)
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
    A.select(["MVT_ID_mvt"] + daystate_meta).write_parquet(f"{OUT}/prep_{t}.parquet")
    m = A["unmatched"] == 0
    print(
        t,
        "rows",
        A.height,
        "matched",
        int(m.sum()),
        " ".join(
            f"{c} sd {A.filter(m)[c].std():.1f} null {A.filter(m)[c].null_count()/max(int(m.sum()),1):.3f}"
            for c in daystate_meta
        ),
        flush=True,
    )
    if os.path.exists(
        f"scratchpad/xt/prep_{t}.parquet"
    ):  # info: agreement with icl_stacker_prep's runway_queue_ra3q-based state (different L member)
        R = pl.read_parquet(f"scratchpad/xt/prep_{t}.parquet", columns=["MVT_ID_mvt"] + daystate_meta).with_columns(
            pl.col("MVT_ID_mvt").cast(pl.Float64)
        )
        J = A.select(["MVT_ID_mvt"] + daystate_meta).join(R, on="MVT_ID_mvt", suffix="_xt")
        print(
            "  vs icl_stacker_prep (runway_queue_ra3q): rows",
            J.height,
            " ".join(
                f"{c} corr {np.corrcoef(*[J.drop_nulls([c, c + '_xt'])[k].to_numpy() for k in (c, c + '_xt')])[0,1]:.3f}"
                for c in daystate_meta[:3]
            ),
            flush=True,
        )
