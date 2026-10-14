#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Neural residual regressor (pytabkit RealMLP-TD or TabM) as a blend member: same target as train2 (y - AOBT_3 offset), same training rows
(non-substituted, non-ranking), same feature groups; writes OOF preds (reg + offset, clipped >= 60) to data/oof/<name>_<cv>.npy in the
train2 format (ids, pred, p=nan) and, with --predict-rank, ranking preds to data/preds/<name>.parquet.
usage: python scripts/nn_resid.py --name nn_forward --cv forward --features data/features_v6.parquet --feats base,...,opdi [--model realmlp|tabm] [--sample 200000] [--epochs 40]
"""

import argparse, json, os, sys, time, numpy as np, polars as pl, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train2 import CAT_BASE, cols_for

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--cv", default="forward")
ap.add_argument("--features", default="data/features_v6.parquet")
ap.add_argument("--feats", default="base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi")
ap.add_argument("--model", choices=["realmlp", "tabm"], default="realmlp")
ap.add_argument("--sample", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--predict-rank", action="store_true")
ap.add_argument(
    "--save-model",
    default="",
    help="directory to save the fitted model and its preprocessing, so phase 2 can reproduce the neural member rather than only its predictions",
)
ap.add_argument("--out", default="experiments/results_nn.jsonl")
ap.add_argument(
    "--side-tables",
    action="store_true",
    help="XR: join the train2 side tables (data/external/<group>_features.parquet) for the requested groups. Without it (the NNA/NNA2 behaviour) groups that live in side tables -- adsb, adsb2, adsb3, durp, dreg, xe, xba, ... -- are silently absent",
)
ap.add_argument(
    "--rank-only",
    action="store_true",
    help="SAVE_MODEL: skip the fold fit (and its OOF file); with --predict-rank only the production fit runs. Imputation statistics still come from the --cv fitting rows, as before",
)
ap.add_argument(
    "--load-model",
    default="",
    help="SAVE_MODEL: predict-only from a --save-model directory (model.pkl + prep.pkl): category lists, imputation fills and missing-indicator set are the saved ones; writes data/preds/<name>.parquet for the is_rank rows of the current --features table",
)
ap.add_argument("--drop-cols", default="", help="comma-separated column prefixes to drop (as train2 --drop-cols)")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
t0 = time.time()
schema = list(pl.read_parquet_schema(a.features).keys())
dummy = pl.DataFrame(schema={c: pl.Null for c in schema})
feats = list(CAT_BASE)
for g in a.feats.split(","):
    feats += cols_for(dummy, g)
SIDE = {
    "storm": "storm",
    "reg": "reg",
    "dur": "dur",
    "sarr": "sarr",
    "durp": "durp",
    "fbte": "fbte",
    "dreg": "dreg",
    "dreg2": "dreg2",
    "areg": "areg",
    "rreg": "rreg",
    "dreg3": "dreg3",
    "dreg3perm": "dreg3perm",
    "preg": "preg",
    "wxs": "wxs",
    "nbrs": "nbrs",
    "nbrsc": "nbrsc",
    "adsb": "adsb",
    "adsb2": "adsb2",
    "adsbF": "adsbF",
    "adsb3": "adsb3",
    "adsb4": "adsb4",
    "adsbU": "adsbU",
    "arrt": "arrt",
    "xe": "xe_queue",
    "xb39": "xb_m39",
    "xb28": "xb_m28",
    "xb410": "xb_m410",
    "xb17": "xb_m17",
    "xbP": "xb_prod",
    "xba": "xba",
}
sides = [g for g in a.feats.split(",") if g in SIDE] if a.side_tables else []
feats_main = [f for f in dict.fromkeys(feats) if f in schema]
meta = ["MVT_ID_mvt", "y", "m", "is_rank", "fb", "mvt_aobt3", "ref_sr_p10"]
d = pl.read_parquet(a.features, columns=list(dict.fromkeys(meta + feats_main)))
for g in sides:
    _f = f"{os.environ.get('adsb_dir_override_adsb_dir', 'data/external') if g in ('adsb', 'adsb2', 'adsb3') else 'data/external'}/{SIDE[g]}_features.parquet"  # ADSB_DIR_OVERRIDE: env override for the ADS-B tables (default unchanged)
    if not os.path.exists(_f):
        raise SystemExit(f"side table for group {g} missing: {_f}")
    d = d.join(pl.read_parquet(_f), on="MVT_ID_mvt", how="left", maintain_order="left")
_LOAD = a.load_model
_READS = [a.features] + [
    f"{os.environ.get('adsb_dir_override_adsb_dir', 'data/external') if g in ('adsb', 'adsb2', 'adsb3') else 'data/external'}/{SIDE[g]}_features.parquet"
    for g in sides
]
if _LOAD:  # SAVE_MODEL: predict-only -- the ranking rows are all that is needed (every statistic comes from prep.pkl)
    import pickle

    _PREP = pickle.load(open(f"{_LOAD}/prep.pkl", "rb"))
    _MANI = json.load(open(f"{_LOAD}/manifest.json"))
    _bad = [
        k
        for k in ("feats", "drop_cols", "side_tables", "model", "epochs", "seed", "cv", "sample")
        if _MANI["args"].get(k) != vars(a).get(k)
    ]
    if _bad:
        raise SystemExit(
            "load-model: args differ from the saved run: "
            + ", ".join(f"{k} saved {_MANI['args'].get(k)!r} now {vars(a).get(k)!r}" for k in _bad)
        )
    d = d.filter(pl.col("is_rank") == 1)
feats = list(CAT_BASE)
for g in a.feats.split(","):
    feats += cols_for(d, g)
feats = [f for f in dict.fromkeys(feats) if f in d.columns]
if a.drop_cols:
    _dp = tuple(x for x in a.drop_cols.split(",") if x)
    feats = [f for f in feats if not f.startswith(_dp)]
cats = [c for c in CAT_BASE if c in feats]
y = d["y"].to_numpy()
m = d["m"].to_numpy()
is_rank = (d["is_rank"] == 1).to_numpy()
fb = d["fb"].to_numpy() == 1
off = np.nan_to_num(d["mvt_aobt3"].to_numpy().astype(float), nan=0.0)
ids = d["MVT_ID_mvt"].to_numpy()
hold = {"forward": [11, 12], "janjul": [1, 7]}.get(a.cv) or [int(x) for x in a.cv.split(":")[1].split(",")]
val = (~is_rank) & np.isin(m, hold)
trn = (~is_rank) & ~np.isin(m, hold) if a.cv != "forward" else (~is_rank) & (m <= 10)
fit = trn & ~fb & (np.abs(y - off) < 7200)  # non-substituted rows, sane residual
X = d.select(feats).to_pandas()
del d
if _LOAD:
    for c in cats:
        _u = int((~X[c].isna() & ~X[c].isin(list(_PREP["cats"][c]))).sum())
        if _u:
            print(f"load-model: {c}: {_u:,} values not in the saved category list -> __na__", flush=True)
        X[c] = pd.Series(pd.Categorical(X[c], categories=_PREP["cats"][c]), index=X.index).fillna("__na__")
else:
    for c in cats:
        X[c] = X[c].astype("category").cat.add_categories(["__na__"]).fillna("__na__")
# pytabkit does not accept NaN in continuous columns: median-impute (train rows) + missing indicators for columns with > 1 % missing
num = [c for c in X.columns if c not in cats]
na = X[num].isna()
# Imputation statistics and missing-indicator selection must come from the FITTING rows only. Using all
# non-ranking rows (as this did until 14 Sep) lets the held-out months inform both -- a small but real leak,
# and this member's measured value is large enough that it must not rest on one.
_imp = fit
_fill = {}
_ind = []
if _LOAD:
    assert num == _PREP["num"], "load-model: numeric column list differs from the saved run"
    for c in num:
        if c in _PREP["ind"]:
            X[c + "_isna"] = na[c].astype("float32")
        X[c] = X[c].fillna(_PREP["fill"][c]).astype("float32")
else:
    for c in num:
        if na[c].any():
            if na.loc[_imp, c].mean() > 0.01:
                X[c + "_isna"] = na[c].astype("float32")
                _ind.append(c)
            _med = X.loc[_imp, c].median()
            X[c] = X[c].fillna(0.0 if np.isnan(_med) else _med).astype("float32")
            _fill[c] = 0.0 if np.isnan(_med) else float(_med)
        else:
            if a.save_model:
                _med = X.loc[_imp, c].median()
                _fill[c] = (
                    0.0 if np.isnan(_med) else float(_med)
                )  # SAVE_MODEL: fill for a NaN that only a new table has
            X[c] = X[c].astype("float32")
X = X.replace([np.inf, -np.inf], 0.0)
X["month_sin"] = np.sin(2 * np.pi * m / 12)
X["month_cos"] = np.cos(2 * np.pi * m / 12)
if _LOAD:
    X = X[_PREP["cols"]]
    ri = np.where(is_rank)[0]
    import torch

    t1 = time.time()
    model = pickle.load(open(f"{_LOAD}/model.pkl", "rb"))
    pr = np.maximum(model.predict(X.iloc[ri]) + off[ri], 60)
    os.makedirs("data/preds", exist_ok=True)
    _pf = f"data/preds/{a.name}.parquet"
    pl.DataFrame({"MVT_ID_mvt": ids[ri], "TAXITIME_SEC_mvt": pr}).write_parquet(_pf)
    import hashlib

    _pm = hashlib.md5(open(_pf, "rb").read()).hexdigest()
    res = dict(
        name=a.name,
        load_model=_LOAD,
        n_rank=int(len(ri)),
        rank_pred_mean=float(pr.mean()),
        pred_md5=_pm,
        same_as_saved=(_pm == _MANI.get("pred_md5")),
        predict_s=round(time.time() - t1, 1),
        secs=int(time.time() - t0),
    )
    json.dump(res, open(_pf + ".xsave_load.json", "w"), indent=1)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)
    sys.exit(0)
fi = np.where(fit)[0]
if a.sample and len(fi) > a.sample:
    fi = np.random.default_rng(a.seed).choice(fi, a.sample, replace=False)
ytr = (y - off)[fi]
print(f"[{time.time()-t0:.0f}s] train rows {len(fi)}, val rows {val.sum()}, feats {X.shape[1]}", flush=True)
import torch
from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

Model = RealMLP_TD_Regressor if a.model == "realmlp" else TabM_D_Regressor
model = Model(
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_epochs=a.epochs,
    random_state=a.seed,
    verbosity=0,
    n_threads=6,
)
if a.rank_only:  # SAVE_MODEL: no fold fit, no OOF file
    res = dict(name=a.name, cv=a.cv, model=a.model, rank_only=True, epochs=a.epochs, secs=int(time.time() - t0))
else:
    model.fit(X.iloc[fi], ytr)
    print(f"[{time.time()-t0:.0f}s] fitted", flush=True)
    vi = np.where(val)[0]
    pred = np.maximum(model.predict(X.iloc[vi]) + off[vi], 60)
    clean = ~fb[vi]
    rm = lambda mk: float(np.sqrt(((pred[mk] - y[vi][mk]) ** 2).mean()))
    res = dict(
        name=a.name,
        cv=a.cv,
        model=a.model,
        n_train=int(len(fi)),
        epochs=a.epochs,
        rmse_regonly_all=rm(np.ones(len(vi), bool)),
        rmse_regonly_clean=rm(clean),
        secs=int(time.time() - t0),
    )
    os.makedirs("data/oof", exist_ok=True)
    np.save(
        f"data/oof/{a.name}_{a.cv.replace(':', '').replace(',', '')}.npy",
        np.vstack([ids[vi], pred, np.full(len(vi), np.nan)]),
    )
if a.predict_rank:
    ri = np.where(is_rank)[0]
    fi2 = np.where((~is_rank) & ~fb & (np.abs(y - off) < 7200))[0]
    _tf = time.time()
    model.fit(X.iloc[fi2], (y - off)[fi2])
    pr = np.maximum(model.predict(X.iloc[ri]) + off[ri], 60)
    os.makedirs("data/preds", exist_ok=True)
    pl.DataFrame({"MVT_ID_mvt": ids[ri], "TAXITIME_SEC_mvt": pr}).write_parquet(f"data/preds/{a.name}.parquet")
    res["rank_pred_mean"] = float(pr.mean())
    if a.save_model or a.rank_only:
        res["rank_fit_s"] = round(time.time() - _tf, 1)  # SAVE_MODEL (default output unchanged)
if a.save_model:
    # phase-2 parity needs the FITTED member, not its predictions (SAVE_MODEL: + prep.pkl with the category lists, numeric fills,
    # missing-indicator set and final column order, so --load-model can predict a new table without refitting)
    import pickle, hashlib, platform, socket

    def _md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        return h.hexdigest()

    os.makedirs(a.save_model, exist_ok=True)
    _ts = time.time()
    with open(f"{a.save_model}/model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(f"{a.save_model}/prep.pkl", "wb") as fh:
        pickle.dump(
            dict(cats={c: X[c].cat.categories for c in cats}, num=num, ind=_ind, fill=_fill, cols=list(X.columns)), fh
        )
    _man = dict(
        script="nn_resid.py",
        script_md5=_md5(os.path.abspath(__file__)),
        argv=sys.argv,
        args=vars(a),
        host=socket.gethostname(),
        arch=platform.machine(),
        t_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        feats=list(X.columns),
        cats=cats,
        model=a.model,
        epochs=a.epochs,
        seed=a.seed,
        cv=a.cv,
        features=os.path.basename(a.features),
        n_train=int(len(fi2)) if a.predict_rank else int(len(fi)),
        predict_rank=a.predict_rank,
        reads=[
            dict(path=os.path.abspath(p), bytes=os.path.getsize(p), md5=_md5(p)) for p in _READS if os.path.exists(p)
        ],
        model_md5=_md5(f"{a.save_model}/model.pkl"),
        prep_md5=_md5(f"{a.save_model}/prep.pkl"),
        save_s=round(time.time() - _ts, 1),
        res=res,
    )
    if a.predict_rank:
        _man["pred_md5"] = _md5(f"data/preds/{a.name}.parquet")
    _man["save_bytes"] = int(sum(os.path.getsize(f"{a.save_model}/{f}") for f in os.listdir(a.save_model)))
    json.dump(_man, open(f"{a.save_model}/manifest.json", "w"), indent=1, default=str)
    print(
        f"[{time.time()-t0:.0f}s] saved fitted member to {a.save_model} ({_man['save_bytes'] / 1e6:.1f} MB)", flush=True
    )
with open(a.out, "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps(res), flush=True)
