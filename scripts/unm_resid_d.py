# SPDX-License-Identifier: GPL-3.0-only
"""unm_resid step 2: unmatched-row model. D = LightGBM on 2025 UNMATCHED rows only (ms<=8h; no NM-derived inputs exist on these rows).
LOFO over the 6 month folds; ranking prediction from a fit on all 12 months.
Selection folds: months511, months612 (never gated). Gate folds: months39, months28, months410, janjul.
Writes scratchpad/xur/D_<variant>.npz (fold OOF on all 2025 unmatched rows + ranking preds)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, time, numpy as np, polars as pl, lightgbm as lgb

sys.path.insert(0, "scratchpad/xur")
from unm_resid_lib import *

NT = int(os.environ.get("NT", "4"))
t0 = time.time()
EXT = ["adsb", "adsb2", "adsb3", "adsb5", "xba", "xe_queue", "durp", "dreg"]
U = pl.scan_parquet("data/features_v9.parquet").filter(pl.col("unmatched") == 1).collect()
for f in EXT:
    A = pl.read_parquet(f"data/external/{f}_features.parquet")
    U = U.join(A, on="MVT_ID_mvt", how="left")
U = U.with_columns(cls_expr(), mon=pl.col("MVT_TIME_UTC_mvt").dt.month()).sort("MVT_ID_mvt")
inv = {m: f for f, ms in FOLDS.items() for m in ms}
U = U.with_columns(
    fold=pl.when(pl.col("is_rank") == 1)
    .then(pl.lit("rank"))
    .otherwise(pl.col("mon").replace_strict(inv, default="none"))
)
PFX = (
    "rwy_dep_",
    "apt_dep_",
    "apt_arr_",
    "rwy_arr_",
    "rwy_heavy_",
    "sched_dep_",
    "eobt_dep_",
    "gap_prev",
    "gap_next",
    "queue_run",
    "run_len",
    "queue_pos",
    "prev_heavy",
    "prev2_heavy",
    "n_rwy_in_hour",
    "rwy_share_hr",
    "rwy_hr_n",
    "apt_hr_n",
    "ref_sr_",
    "ref_s_",
    "ref_r_",
    "ref_sh_",
    "te_op_",
    "te_cs_",
    "te_des_",
    "wx0_",
    "wx30_",
    "wx60_",
    "wxd_",
    "arrfb_",
    "arrart_",
    "arr_day_n",
    "arr_dayop_n",
    "arrlate_dayop_n",
    "nbr_",
    "te2_",
    "opdi_",
    "nbrx_",
    "arrx_",
    "durp_",
    "dreg_",
    "adsb_",
    "adsb2_",
    "adsb3_",
    "adsb5_",
    "xba_",
    "xe_",
)
BASE = [
    "hour",
    "dow",
    "sched_min",
    "mvt_sched",
    "cs_len",
    "cs_airline_style",
    "stand_since_arr_inblock",
    "stand_since_arr_land",
    "arr_block_delay",
    "stand_since_prev_dep",
]
DROP = ("te_fno_", "ref_sr_n", "ref_s_n", "ref_r_n", "ref_sh_n", "te_op_n", "te_cs_n", "te_des_n", "doy")
CATS = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "apt_rwy", "apt_stand", "FLIGHT_RULE_mvt", "cs_pre"]
T = U.filter(pl.col("is_rank") == 0)
num = [
    c
    for c in U.columns
    if (c in BASE or c.startswith(PFX))
    and not c.startswith(DROP)
    and U[c].dtype.is_numeric()
    and T[c].null_count() < 0.999 * T.height
]
X = U.select(num).to_pandas()
for c in CATS:
    X[c] = U[c].cast(pl.Utf8).fill_null("NA").to_pandas().astype("category")
cols = num + CATS
y = U["y"].to_numpy().astype(float)
fo = U["fold"].to_numpy()
cl = U["cls"].to_numpy()
ms = U["mvt_sched"].to_numpy().astype(float)
print(f"rows {U.height:,} features {len(cols)} ({time.time()-t0:.0f}s)", flush=True)
V = {
    "L2": dict(objective="l2"),
    "L2c4h": dict(objective="l2", clipy=14400),
    "HUB": dict(objective="huber", alpha=900.0),
}
only = os.environ.get("VARS")
V = {k: v for k, v in V.items() if not only or k in only.split(",")}
P = dict(
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=60,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    cat_smooth=20,
    cat_l2=10,
    max_cat_to_onehot=8,
    verbose=-1,
    num_threads=NT,
)
R = int(os.environ.get("ROUNDS", "700"))
SEEDS = int(os.environ.get("SEEDS", "2"))
trn_base = (fo != "rank") & (fo != "none") & np.isfinite(y) & (np.nan_to_num(ms, nan=0) <= 28800)
for vn, vc in V.items():
    vc = dict(vc)
    clipy = vc.pop("clipy", None)
    oof = np.full(U.height, np.nan)
    rk = np.full(U.height, np.nan)
    yt = np.clip(y, 0, clipy) if clipy else y
    for f in list(FOLDS) + ["rank"]:
        tr = trn_base & (fo != f)
        te = fo == f
        pr = np.zeros(te.sum())
        for s in range(SEEDS):
            m = lgb.train(dict(P, seed=s, **vc), lgb.Dataset(X[tr], yt[tr]), R)
            pr += m.predict(X[te]) / SEEDS
        if f == "rank":
            rk[te] = pr
            imp = m.feature_importance("gain")
            o = np.argsort(-imp)[:15]
            print("  top gain:", ", ".join(f"{cols[i]} {100*imp[i]/imp.sum():.0f}%" for i in o))
        else:
            oof[te] = pr
    msg = []
    for f in FOLDS:
        k = (fo == f) & (cl == "unm_other") & np.isfinite(y)
        e = (np.maximum(oof[k], 60) - y[k]) ** 2
        es = np.sort(e)[::-1]
        msg.append(f"{f} {e.mean():,.0f}/{es[3:].mean():,.0f}")
    print(vn, " | ".join(msg), f"({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(f"scratchpad/xur/D_{vn}.npz", ids=U["MVT_ID_mvt"].to_numpy(), oof=oof, rank=rk, fold=fo, cls=cl)
