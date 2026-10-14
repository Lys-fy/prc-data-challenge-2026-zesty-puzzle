#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Pipeline outputs as Kumo inputs -- an adapter on an UNCHANGED km_engine.py (pattern of km_resid.py / km_adapters_ctx.py).
Knobs, all OFF by default. With none set NOTHING is patched, no spec key is added, and a run is bit-identical to `python km_engine.py`.
 (1) FEED_SIDE=<name|path.parquet>   SIDE COLUMNS = label-free processed per-flight tables from the pipeline (default file
     $BUNDLE/data/payload/side/<name>.parquet; key column MVT_ID_mvt, every other column numeric), joined on the frame ids (meta__MVT_ID_mvt) and
     appended to X. Their indices are appended to EVERY instance's column subset (columns(i, fold) = the engine's own subset + all side columns), so an
     arm differs from its control (the same spec without FEED_SIDE, e.g. KSBWPc100) only by the side columns. FEED_SIDE_COLS=<a,b,...> restricts the
     columns (default: all). Rows without a side row get NaN (Kumo's missing value, as in the 412). Label-free by construction: the table is read, never fitted.
 (2) FEED_STK=<chain>   STACKER MODE (Kumo as an L2 augment): the frame of an airport is the T2x meta-input dump of that chain
     ($BUNDLE/data/payload/stk/stk_<chain>_<fold>.npz for the 4 gate folds, written by scripts/kstk_run.sh dump): rows = the matched rows of the 4 folds,
     X = the meta inputs (out-of-fold member columns; NaN kept unless FEED_STK_NAN0=1), offset = v41 (the chain's pre-meta base, as the GBM meta),
     context target = clip(y - v41, +-FEED_STK_CLIP) (default 1800, the T2x screen rule), every instance reads ALL meta columns. Each row's month key is the first held
     month of its fold, so the unchanged km_engine.split puts the query fold's rows in the query and the OTHER THREE folds' rows (finite y and v41) in the
     context: exactly the GBM meta's cross-fit (screen_l2_extra METAFIT=all). Output taxi = v41 + gK (gK = taxi - v41).
     Only the 4 gate folds exist in this mode (no prod: production needs the chain stacker's own dump).
Every active knob is recorded in KE.S["feed"] (the meta of every unit).
usage: python km_adapters_feed.py <unit> ... | --from-list list.txt [--claim]       (units as km_engine: inst:<fold>:<APT>:<i>  agg:<fold>:<APT>)
       python km_adapters_feed.py --selftest            numpy only: side join/append, stacker frame + split semantics on synthetic dumps
       python km_adapters_feed.py --selftest --real     + the stacker frames from the real dumps in $FEED_STK_DIR (split counts per airport x fold)"""
import os, sys, json, time, numpy as np
E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__)); BUNDLE = os.path.abspath(os.path.join(HERE, ".."))
FEED_SIDE = E("FEED_SIDE", ""); FEED_SIDE_COLS = E("FEED_SIDE_COLS", "")
FEED_STK = E("FEED_STK", ""); FEED_STK_DIR = E("FEED_STK_DIR", os.path.join(BUNDLE, "data", "payload", "stk"))
FEED_STK_CLIP = float(E("FEED_STK_CLIP", "1800")); FEED_STK_NAN0 = int(E("FEED_STK_NAN0", "0"))
FO4 = ["months39", "months28", "months410", "janjul"]
HOLD = {"janjul": [1, 7], "months28": [2, 8], "months39": [3, 9], "months410": [4, 10], "months511": [5, 11], "months612": [6, 12], "prod": None}   # = km_task.HOLD
assert not (FEED_SIDE and FEED_STK), "KSTK_FEED: FEED_SIDE and FEED_STK are separate arms; set one"

# ---------------- pure helpers (numpy) shared by the live patches and the self test
def side_path(v):
    return v if v.endswith(".parquet") and os.path.exists(v) else os.path.join(BUNDLE, "data", "payload", "side", f"{v}.parquet")
def side_table(ids_key, M):
    """(sorted ids, values [n, C] float32) from a key vector and a value matrix (duplicate keys refused)."""
    o = np.argsort(ids_key, kind="stable"); k = ids_key[o]; assert (np.diff(k) > 0).all(), "side table: duplicate MVT_ID_mvt"
    return k, M[o].astype(np.float32)
def side_lookup(tab, ids):
    """[n, C] side values for the frame ids (NaN where the table has no row) + hit mask."""
    k, V = tab; j = np.clip(np.searchsorted(k, ids), 0, len(k) - 1); hit = k[j] == ids
    out = np.full((len(ids), V.shape[1]), np.nan, np.float32); out[hit] = V[j[hit]]; return out, hit
def side_frame(F, tab):
    vals, hit = side_lookup(tab, F["ids"]); G = dict(F); G["X"] = np.hstack([F["X"], vals]); G["side_hit"] = float(hit.mean()); return G
def stk_frame(dumps, apt, nan0=False):
    """Stacker frame of one airport from the 4 fold dumps {fold: dict(apt, y, v41, X[, mid])} (all folds, FO4 order).
    m = first held month of the row's fold (so km_engine.split selects query = that fold, context = the other folds); row = global position in the
    chain's FO4 concatenation (unique; agg locates rows by it); ids = MVT ids when the dump has them, else -(1 + row)."""
    parts, base = [], 0
    for t in FO4:
        z = dumps[t]; sel = np.where(z["apt"] == apt)[0]; n = len(z["y"])
        ids = z["mid"][sel].astype(float) if "mid" in z else -(1.0 + base + sel)
        parts.append(dict(X=z["X"][sel].astype(np.float32), y=z["y"][sel].astype(float), off=z["v41"][sel].astype(float), m=np.full(len(sel), HOLD[t][0]),
                          ids=ids, row=(base + sel).astype(np.int64)))
        base += n
    F = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if nan0: F["X"] = np.nan_to_num(F["X"], nan=0.0, posinf=0.0, neginf=0.0)
    n = len(F["y"]); F["rank"] = np.zeros(n, bool); F["clean"] = np.isfinite(F["y"]) & np.isfinite(F["off"]); F["lk"] = np.ones(n, bool)
    F["off"] = np.where(np.isfinite(F["off"]), F["off"], 0.0); return F
def split_like_engine(F, fold):
    """The logic of km_engine.split, for the self test only (the live adapter calls the engine's own)."""
    m, y, rank, clean = F["m"], F["y"], F["rank"], F["clean"]
    if HOLD[fold] is None: val = rank; trn = ~rank & clean & np.isfinite(y)
    else: val = ~rank & np.isin(m, HOLD[fold]); trn = ~rank & ~np.isin(m, HOLD[fold]) & clean & np.isfinite(y)
    return np.where(trn)[0], np.where(val)[0]
def load_dumps(chain, d=None):
    d = d or FEED_STK_DIR; out = {}
    for t in FO4:
        p = f"{d}/stk_{chain}_{t}.npz"
        if not os.path.exists(p): p = f"{d}/t2x_{chain}_{t}.npz"            # the T2x screen payload layout (no mid) is accepted for checks
        assert os.path.exists(p), f"KSTK_FEED: stacker dump missing {d}/stk_{chain}_{t}.npz (scripts/kstk_run.sh dump, then copy into data/payload/stk)"
        z = np.load(p, allow_pickle=False); out[t] = {k: z[k] for k in z.files}
    cols = [list(map(str, out[t]["cols"])) for t in FO4]; assert all(c == cols[0] for c in cols), "KSTK_FEED: meta columns differ between fold dumps"
    return out, cols[0]

def selftest(real=False):
    rng = np.random.default_rng(0)
    # (1) side columns
    n = 2000; ids = np.sort(rng.choice(10 ** 7, n, replace=False)).astype(float)
    F = dict(X=rng.standard_normal((n, 412)).astype(np.float32), ids=ids, y=rng.standard_normal(n), off=np.zeros(n), clean=np.ones(n, bool))
    keep = rng.random(n) < 0.6; key = ids[keep]; V = rng.normal(0, 1, (keep.sum(), 5)); perm = rng.permutation(len(key))
    tab = side_table(key[perm], V[perm]); G = side_frame(F, tab)
    assert G["X"].shape == (n, 417) and np.array_equal(G["X"][:, :412], F["X"]) and np.allclose(G["X"][keep, 412:], V.astype(np.float32))
    assert np.isnan(G["X"][~keep, 412:]).all() and abs(G["side_hit"] - keep.mean()) < 1e-12
    try: side_table(np.array([1.0, 1.0]), np.zeros((2, 1))); raise AssertionError("duplicate keys accepted")
    except AssertionError as ex: assert "duplicate" in str(ex)
    cols = np.sort(rng.choice(412, 100, replace=False)); cs = np.concatenate([cols, 412 + np.arange(5)]); assert G["X"][:, cs].shape == (n, 105)
    # (2) stacker frame + engine split semantics on synthetic 4-fold dumps
    D = {}
    for j, t in enumerate(FO4):
        m = 3000 + 100 * j; apt = rng.choice(np.array(["LSZH", "EDDM", "LTFM"]), m); y = rng.normal(900, 200, m); y[rng.random(m) < 0.01] = np.nan
        D[t] = dict(apt=apt, y=y, v41=y + rng.normal(0, 50, m), X=rng.standard_normal((m, 43)).astype(np.float32), mid=(j * 10 ** 6 + np.arange(m)).astype(float))
    for apt in ("LSZH", "LTFM"):
        S = stk_frame(D, apt)
        assert len(np.unique(S["row"])) == len(S["row"]) and len(np.unique(S["ids"])) == len(S["ids"])
        for t in FO4:
            ti, qi = split_like_engine(S, t); fold_rows = np.isin(S["ids"], D[t]["mid"])
            assert np.array_equal(np.sort(qi), np.where(fold_rows)[0]), (apt, t, "query != the fold's rows")
            assert not fold_rows[ti].any(), (apt, t, "a query-fold row in the context")
            other = ~fold_rows & np.isfinite(S["y"]); assert np.array_equal(np.sort(ti), np.where(other)[0]), (apt, t, "context != other folds' finite rows")
            assert len(qi) == int((D[t]["apt"] == apt).sum())
        r = np.clip(S["y"] - S["off"], -FEED_STK_CLIP, FEED_STK_CLIP); assert np.nanmax(np.abs(r)) <= FEED_STK_CLIP
    print(f"SELFTEST OK: side join/append (hit {G['side_hit']:.3f}, NaN where missing, dup keys refused), stacker frame + split (query = fold rows, context = "
          f"other 3 folds' finite rows, unique rows/ids) | FEED_STK_CLIP {FEED_STK_CLIP}", flush=True)
    if real:
        chains = [c for c in ("Kq", "NNA3") if any(os.path.exists(f"{FEED_STK_DIR}/{p}_{c}_janjul.npz") for p in ("stk", "t2x"))]
        assert chains, f"--real: no dumps in {FEED_STK_DIR}"
        for c in chains:
            Dr, cols = load_dumps(c); apts = sorted(set(np.concatenate([Dr[t]["apt"] for t in FO4]).tolist())); tot = {t: 0 for t in FO4}
            for apt in apts:
                S = stk_frame(Dr, apt)
                for t in FO4:
                    ti, qi = split_like_engine(S, t); assert not np.isin(S["m"][ti], HOLD[t]).any() and np.isin(S["m"][qi], HOLD[t]).all(); tot[t] += len(qi)
            for t in FO4: assert tot[t] == len(Dr[t]["y"]), (c, t, tot[t], len(Dr[t]["y"]))
            S = stk_frame(Dr, "LTFM"); ti, qi = split_like_engine(S, "janjul")
            print(f"REAL {c}: {len(cols)} meta cols ({', '.join(cols[:6])}, ...), {len(apts)} airports; query rows per fold == dump rows {tot}; "
                  f"LTFM janjul context {len(ti):,} (whole pool of the other 3 folds) query {len(qi):,}; mid {'yes' if 'mid' in Dr['janjul'] else 'no (kumo_member_screen layout)'}", flush=True)

if __name__ == "__main__" and "--selftest" in sys.argv: selftest(real="--real" in sys.argv); sys.exit(0)

# ---------------- live mode: wrap the engine (torch + sdm are imported by km_engine, not above)
sys.path.insert(0, HERE)
import km_engine as KE
K = KE.K; FEED = {}; _cache = {}
if FEED_SIDE:
    import polars as pl
    p = side_path(FEED_SIDE); assert os.path.exists(p), f"KSTK_FEED: side table missing {p} (copy the side table into data/payload/side)"
    _d = pl.read_parquet(p); _names = [c for c in _d.columns if c != "MVT_ID_mvt"]
    if FEED_SIDE_COLS: _want = FEED_SIDE_COLS.split(","); _miss = [c for c in _want if c not in _names]; assert not _miss, f"kstk_feed_side_cols not in {p}: {_miss}"; _names = _want
    _TAB = side_table(_d["MVT_ID_mvt"].to_numpy().astype(float), _d.select(_names).cast(pl.Float32).fill_nan(None).to_numpy().astype(np.float32)); del _d
    _orig_frame, _orig_columns = K.frame, KE.columns
    def frame(apt):
        if apt not in _cache:
            _cache.clear(); G = side_frame(_orig_frame(apt), _TAB); _cache[apt] = G
            print(f"  feed[side {os.path.basename(p)}] {apt}: +{len(_names)} cols, side row on {G['side_hit']:.3f} of {len(G['ids']):,} frame rows", flush=True)
        return _cache[apt]
    def columns(i, fold):
        cols, src = _orig_columns(i, fold); return np.concatenate([cols, KE.D + np.arange(len(_names))]).astype(int), f"{src}+side{len(_names)}"
    K.frame = frame; KE.columns = columns
    FEED["side"] = dict(file=os.path.basename(p), cols=_names, n=len(_names))
if FEED_STK:
    _DUMPS, _SCOLS = load_dumps(FEED_STK); _orig_fit = KE.fit
    def frame(apt):
        if apt not in _cache:
            _cache.clear(); F = stk_frame(_DUMPS, apt, nan0=bool(FEED_STK_NAN0)); assert len(F["y"]) > 0, f"KSTK_FEED stk: no rows for {apt}"; _cache[apt] = F
            print(f"  feed[stk {FEED_STK}] {apt}: {len(F['y']):,} rows x {F['X'].shape[1]} meta cols, clean {F['clean'].mean():.4f}", flush=True)
        return _cache[apt]
    def columns(i, fold):
        assert fold in FO4, f"KSTK_FEED stk: fold {fold} not dumped (4 gate folds only)"; return np.arange(len(_SCOLS)), "stk-all"
    def fit(xc, yc, i): return _orig_fit(xc, np.clip(yc, -FEED_STK_CLIP, FEED_STK_CLIP), i)
    K.frame = frame; KE.columns = columns; KE.fit = fit
    FEED["stk"] = dict(chain=FEED_STK, dir=os.path.basename(FEED_STK_DIR.rstrip("/")), cols=_SCOLS, clip=FEED_STK_CLIP, nan0=FEED_STK_NAN0)
if FEED: KE.S["feed"] = FEED; print(f"km_adapters_feed active: {json.dumps(FEED)[:400]}", flush=True)

if __name__ == "__main__":
    import socket
    a = sys.argv[1:]; claim = "--claim" in a; a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list": a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(f"km_adapters_feed: knobs {json.dumps(FEED)[:300] if FEED else 'none (native engine)'} | engine spec {KE.S}", flush=True)
    t0 = time.time(); n = 0
    for spec in a:
        if claim:
            lk = f"{KE.RUN}/locks/{KE.S['name']}_{spec.replace(':', '_')}"; os.makedirs(f"{KE.RUN}/locks", exist_ok=True)
            try: os.mkdir(lk)
            except FileExistsError: continue
            open(f"{lk}/owner", "w").write(f"{socket.gethostname()} {os.getpid()}\n")
        try: r = KE.run(spec)
        except BaseException:
            if claim: import shutil; shutil.rmtree(lk, ignore_errors=True)
            raise
        n += r is not None
    print(f"KM_FEED_DONE units run {n} | {time.time() - t0:.0f}s", flush=True)
