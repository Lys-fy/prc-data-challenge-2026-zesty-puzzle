#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Save / load / rank-only hooks for the train2 family (train2.py and its wrappers train2_<variant>.py).

Inactive (and the trainer byte-identical to before) unless one of the three flags is given:
  --rank-only        skip the fold-CV fit (its boosters are replaced by zero-predicting dummies; the fold OOF file is NOT written).
                     The rank fit is unchanged: with --fixed-rounds N the rank stage uses int(N * 1.1) rounds exactly as before
                     (iters[g] = N comes from the fixed-rounds branch, not from the booster). Requires --fixed-rounds > 0 and --predict-rank.
  --save-model DIR   save every booster of the RANK stage (LightGBM -> lgb_KK.txt.gz, CatBoost -> cat_KK.cbm, in call order), the category
                     dictionaries of every categorical column (catmap.pkl), the rank-stage day-shift parameters, and manifest.json
                     (args, script md5, host/arch, library versions, md5 of every parquet read, call signatures, pred md5).
  --load-model DIR   predict-only: the script runs its normal data path on the CURRENT features table, but every rank-stage
                     lgb.train / CatBoost fit returns the saved model (after checking params, rounds and feature names), categoricals are
                     re-coded with the saved dictionaries (unseen -> NaN, which LightGBM routes like an unseen category), the day-shift
                     models / caps are the saved ones, the fold stage is skipped (implies --rank-only). Writes data/preds/<name>.parquet.
"""

import os, sys, json, time, gzip, hashlib, pickle, platform, socket
import numpy as np

ON = False
RANK_ONLY = False
SAVE = ""
LOAD = ""
PHASE = "fold"
_K = 0
_CALLS = []
_CATS = {}
_READS = []
_MAN = None
_CATMAP = None
_DS = {}
_DSK = {}
_T0 = time.time()
_A = None
ALLOWED_ARG_DIFFS = {"name", "out", "threads", "features", "save_model", "load_model", "rank_only", "predict_rank"}


def add_args(ap):
    ap.add_argument(
        "--rank-only",
        action="store_true",
        help="SAVE_MODEL: skip the fold-CV fit (rank fit unchanged; no fold OOF written)",
    )
    ap.add_argument(
        "--save-model",
        default="",
        help="SAVE_MODEL: directory to save the rank-stage boosters, category dictionaries and manifest",
    )
    ap.add_argument(
        "--load-model", default="", help="SAVE_MODEL: predict-only from a --save-model directory (implies --rank-only)"
    )


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _log(*x):
    print("[xsave %6.0fs]" % (time.time() - _T0), *x, flush=True)


class _Dummy:
    """fold-stage stand-in under --rank-only: predicts zeros, best_iteration 0 (only read in the early-stopping branch, which rank-only forbids)."""

    best_iteration = 0

    def predict(self, X, *a, **k):
        return np.zeros(X.shape[0])


def _jsonable(d):
    return json.loads(json.dumps(d, default=str, sort_keys=True))


def _lgb_sig(params, train_set, rounds):
    p = {k: v for k, v in dict(params).items() if k not in ("num_threads", "nthread", "num_thread", "n_jobs")}
    data = getattr(train_set, "data", None)
    feats = list(data.columns) if hasattr(data, "columns") else None
    n = int(data.shape[0]) if data is not None and hasattr(data, "shape") else None
    return dict(kind="lgb", params=_jsonable(p), rounds=int(rounds), feats=feats, n_rows=n)


def _check_sig(k, sig):
    rec = _MAN["calls"][k]
    if rec["kind"] != sig["kind"]:
        raise SystemExit(f"xsave load: call {k} kind {sig['kind']} != saved {rec['kind']}")
    for key in ("params", "rounds", "feats"):
        if rec.get(key) != sig.get(key):
            raise SystemExit(
                f"xsave load: call {k} {key} differs from the saved model:\n saved {str(rec.get(key))[:600]}\n now   {str(sig.get(key))[:600]}"
            )
    if rec.get("n_rows") != sig.get("n_rows"):
        _log(
            f"WARNING call {k}: training-row count now {sig.get('n_rows')} vs saved {rec.get('n_rows')} (model is loaded, not refitted)"
        )
    return rec


def _install_lgb():
    import lightgbm as lgb

    orig = lgb.train

    def train(params, train_set, num_boost_round=100, *args, **kw):
        global _K
        if PHASE == "fold":
            if RANK_ONLY:
                return _Dummy()
            return orig(params, train_set, num_boost_round, *args, **kw)
        k = _K
        _K += 1
        sig = _lgb_sig(params, train_set, num_boost_round)
        f = f"lgb_{k:02d}.txt.gz"
        if LOAD:
            rec = _check_sig(k, sig)
            t = time.time()
            with gzip.open(os.path.join(LOAD, rec["file"]), "rt") as fh:
                b = lgb.Booster(model_str=fh.read())
            _log(f"loaded {rec['file']} ({b.num_trees()} trees) in {time.time() - t:.1f}s")
            return b
        t = time.time()
        b = orig(params, train_set, num_boost_round, *args, **kw)
        fit_s = time.time() - t
        if SAVE:
            t = time.time()
            s = b.model_to_string(num_iteration=-1)
            with gzip.open(os.path.join(SAVE, f), "wt", compresslevel=6) as fh:
                fh.write(s)
            sig.update(
                file=f,
                fit_s=round(fit_s, 1),
                save_s=round(time.time() - t, 1),
                bytes=os.path.getsize(os.path.join(SAVE, f)),
                md5=md5(os.path.join(SAVE, f)),
                n_trees=b.num_trees(),
            )
            _CALLS.append(sig)
            _log(f"saved {f} ({b.num_trees()} trees, {sig['bytes'] / 1e6:.1f} MB) fit {fit_s:.0f}s")
        return b

    lgb.train = train


def _install_cat():
    import catboost

    C = catboost.CatBoostRegressor
    ofit = C.fit
    opred = C.predict

    def _sig(self, X):
        p = {
            k: v
            for k, v in self.get_params().items()
            if k not in ("thread_count", "task_type", "gpu_ram_part", "devices", "verbose")
        }
        feats = X.get_feature_names() if hasattr(X, "get_feature_names") else list(getattr(X, "columns", []))
        n = int(X.num_row()) if hasattr(X, "num_row") else int(len(X))
        return dict(kind="cat", params=_jsonable(p), rounds=int(p.get("iterations", 0)), feats=list(feats), n_rows=n)

    def fit(self, X, y=None, *args, **kw):
        global _K
        if PHASE == "fold":
            if RANK_ONLY:
                self._xs_dummy = True
                return self
            return ofit(self, X, y, *args, **kw)
        k = _K
        _K += 1
        sig = _sig(self, X)
        f = f"cat_{k:02d}.cbm"
        if LOAD:
            rec = _check_sig(k, sig)
            t = time.time()
            self.load_model(os.path.join(LOAD, rec["file"]))
            _log(f"loaded {rec['file']} in {time.time() - t:.1f}s")
            return self
        t = time.time()
        r = ofit(self, X, y, *args, **kw)
        fit_s = time.time() - t
        if SAVE:
            t = time.time()
            self.save_model(os.path.join(SAVE, f))
            sig.update(
                file=f,
                fit_s=round(fit_s, 1),
                save_s=round(time.time() - t, 1),
                bytes=os.path.getsize(os.path.join(SAVE, f)),
                md5=md5(os.path.join(SAVE, f)),
                n_trees=int(self.tree_count_),
            )
            _CALLS.append(sig)
            _log(f"saved {f} ({sig['bytes'] / 1e6:.1f} MB) fit {fit_s:.0f}s")
        return r

    def predict(self, data, *args, **kw):
        if getattr(self, "_xs_dummy", False):
            return np.zeros(data.num_row() if hasattr(data, "num_row") else len(data))
        return opred(self, data, *args, **kw)

    C.fit = fit
    C.predict = predict


def _install_reads():
    import polars as pl

    orig = pl.read_parquet

    def rp(source, *args, **kw):
        if isinstance(source, (str, os.PathLike)):
            _READS.append(os.path.abspath(str(source)))
        return orig(source, *args, **kw)

    pl.read_parquet = rp


def _install_dayshift():
    """rank-stage day-shift models / caps: recorded on save, replayed on load (fold-stage calls are left alone)."""
    mod = sys.modules.get("__main__")
    for fn in ("dayshift_model", "dayshift_model_pooled", "dayshift_caps"):
        if not hasattr(mod, fn):
            continue
        orig = getattr(mod, fn)

        def wrap(*args, _orig=orig, _fn=fn, **kw):
            if PHASE != "rank":
                return _orig(*args, **kw)
            i = _DSK.get(_fn, 0)
            _DSK[_fn] = i + 1
            key = f"{_fn}#{i}"
            if LOAD:
                if key not in _MAN.get("dayshift", {}):
                    raise SystemExit(f"xsave load: no saved day-shift parameters for {key}")
                v = _MAN["dayshift"][key]
                now = _orig(*args, **kw)
                if json.dumps(_jsonable(now), sort_keys=True) != json.dumps(v, sort_keys=True):
                    _log(f"NOTE {key}: recomputed on the current table differs from saved; using SAVED")
                return {k_: (tuple(x) if isinstance(x, list) else x) for k_, x in v.items()}
            r = _orig(*args, **kw)
            _DS[key] = _jsonable(r)
            return r

        setattr(mod, fn, wrap)


def setup(a, script):
    """call right after ap.parse_args()."""
    global ON, RANK_ONLY, SAVE, LOAD, _MAN, _CATMAP, _A
    if not (getattr(a, "rank_only", False) or getattr(a, "save_model", "") or getattr(a, "load_model", "")):
        return
    ON = True
    _A = a
    SAVE = a.save_model
    LOAD = a.load_model
    RANK_ONLY = bool(a.rank_only or LOAD)
    if SAVE and LOAD:
        raise SystemExit("xsave: --save-model and --load-model are exclusive")
    if not a.predict_rank:
        raise SystemExit("xsave: needs --predict-rank")
    if RANK_ONLY:
        if getattr(a, "fixed_rounds", 0) <= 0:
            raise SystemExit(
                "xsave: --rank-only/--load-model need --fixed-rounds > 0 (rank rounds come from the fold early stop otherwise)"
            )
        for bad in ("pfeat", "reuse_taxi"):
            if getattr(a, bad, False):
                raise SystemExit(f"xsave: --rank-only/--load-model not supported with --{bad.replace('_', '-')}")
    if getattr(a, "model", "lgbm") == "xgb":
        raise SystemExit("xsave: xgb not supported")
    if SAVE:
        os.makedirs(SAVE, exist_ok=True)
    if LOAD:
        _MAN = json.load(open(os.path.join(LOAD, "manifest.json")))
        with open(os.path.join(LOAD, "catmap.pkl"), "rb") as fh:
            _CATMAP = pickle.load(fh)
        if os.path.basename(script) != _MAN["script"]:
            raise SystemExit(f"xsave load: script {os.path.basename(script)} != saved {_MAN['script']}")
        now = _jsonable(vars(a))
        diffs = sorted(
            k for k in set(now) | set(_MAN["args"]) if now.get(k) != _MAN["args"].get(k) and k not in ALLOWED_ARG_DIFFS
        )
        if diffs:
            raise SystemExit(
                "xsave load: model-defining args differ from the saved run: "
                + ", ".join(f"{k}: saved {_MAN['args'].get(k)!r} now {now.get(k)!r}" for k in diffs)
            )
        if md5(script) != _MAN["script_md5"]:
            _log(
                f"WARNING script md5 {md5(script)} != saved {_MAN['script_md5']} (args and call signatures are checked)"
            )
    _install_lgb()
    _install_reads()
    _install_dayshift()
    if getattr(a, "model", "lgbm") == "cat":
        _install_cat()
    _log(
        f"mode save={SAVE or '-'} load={LOAD or '-'} rank_only={RANK_ONLY} host={socket.gethostname()} {platform.machine()}"
    )


def cat(s, c):
    """categorical column -> pandas categorical: today's dictionary (recorded for save) or the saved one (load)."""
    import pandas as pd

    if LOAD:
        cats = _CATMAP[c]
        out = pd.Series(pd.Categorical(s, categories=cats), index=s.index)
        unseen = int((out.isna() & s.notna()).sum())
        if unseen:
            _log(f"category {c}: {unseen:,} values not in the saved dictionary -> NaN")
        return out
    out = s.astype("category")
    _CATS[c] = out.cat.categories
    return out


def phase(p):
    global PHASE
    PHASE = p
    if ON:
        _log(f"phase {p}")


def finish(a, res, pred_path):
    if not ON:
        return
    import pandas as pd

    pred_path = str(pred_path)
    pm = md5(pred_path)
    P = pd.read_parquet(pred_path)
    info = dict(
        pred_file=pred_path,
        pred_md5=pm,
        n_rank=int(len(P)),
        pred_mean=float(P["TAXITIME_SEC_mvt"].mean()),
        rank_dayshift=res.get("rank_dayshift"),
        rank_gate_n_fit=res.get("rank_gate_n_fit"),
        rank_gate_n_rank=res.get("rank_gate_n_rank"),
        rank_p_fb_mean=res.get("rank_p_fb_mean"),
        iters=res.get("iters"),
        secs=round(time.time() - _T0, 1),
    )
    if SAVE:
        with open(os.path.join(SAVE, "catmap.pkl"), "wb") as fh:
            pickle.dump(dict(_CATS), fh)
        reads = []
        for r in dict.fromkeys(_READS):
            if os.path.exists(r):
                reads.append(dict(path=r, bytes=os.path.getsize(r), md5=md5(r)))
        import lightgbm

        vers = dict(
            python=platform.python_version(), lightgbm=lightgbm.__version__, numpy=np.__version__, pandas=pd.__version__
        )
        try:
            import catboost

            vers["catboost"] = catboost.__version__
        except Exception:
            pass
        script = os.path.abspath(sys.argv[0])
        man = dict(
            script=os.path.basename(script),
            script_md5=md5(script),
            argv=sys.argv,
            args=_jsonable(vars(a)),
            host=socket.gethostname(),
            arch=platform.machine(),
            versions=vers,
            t_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            rank_only=RANK_ONLY,
            calls=_CALLS,
            dayshift=_DS,
            catmap_md5=md5(os.path.join(SAVE, "catmap.pkl")),
            cat_cols={k: int(len(v)) for k, v in _CATS.items()},
            reads=reads,
            **info,
        )
        man["save_bytes"] = int(sum(os.path.getsize(os.path.join(SAVE, f)) for f in os.listdir(SAVE)))
        json.dump(man, open(os.path.join(SAVE, "manifest.json"), "w"), indent=1)
        _log(f"manifest written: {len(_CALLS)} models, {man['save_bytes'] / 1e6:.1f} MB, pred md5 {pm}")
    if LOAD:
        if _K != len(_MAN["calls"]):
            raise SystemExit(f"xsave load: {_K} rank-stage fits replayed but {len(_MAN['calls'])} saved")
        for k in ("rank_gate_n_fit", "rank_dayshift"):
            if _jsonable(info.get(k)) != _MAN.get(k):
                _log(f"NOTE {k} now {str(info.get(k))[:300]} vs saved {str(_MAN.get(k))[:300]}")
        same = pm == _MAN.get("pred_md5")
        _log(
            f"load done: pred md5 {pm} ({'== saved run' if same else 'differs from saved run (expected on a new table)'}) secs {info['secs']}"
        )
        json.dump(
            dict(load_dir=LOAD, same_as_saved=same, **_jsonable(info)),
            open(pred_path + ".xsave_load.json", "w"),
            indent=1,
        )
