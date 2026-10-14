#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Adapts km_engine.py to add Kumo as a chain-stacker augment (gK for KB = 0.7 gM + 0.3 gK inside the production chain
stacker), covering the 6 chain folds and the production rows; the underlying km_engine.py is unchanged. Inputs:
positional units as km_engine (inst:<fold>:<APT>:<i>, agg:<fold>:<APT>) or --from-list list.txt, env KSTK_CHAIN
(Kq|NNA3; unset = no patch, bit-identical to km_engine.py), KSTK_DIR, KSTK_CLIP, KSTK_JJCTX. Output taxi = offset + gK
(agg unit); scripts/kstk_gk.py turns these into data/oof/KSTK6<chain>_<fold>.npy and the prod file.

Knob KSTK_CHAIN=<Kq|NNA3> (OFF by default: with it unset NOTHING is patched and a run is bit-identical to `python km_engine.py`).
FRAME of an airport = the chain stacker's OWN meta frame (scripts/kstk_stack.py KSTK_DUMP on the 6 chain folds + scripts/kstk_assemble.py KSTK_DUMP on
  the matched ranking rows), files $KSTK_DIR/stk_<chain>_<fold>.npz (fold in months39 months28 months410 months612 months511 janjul) and stk_<chain>_prod.npz:
  X = the 43 meta inputs (out-of-fold member columns on fold rows, production member columns on ranking rows; NaN kept), offset = v41 (fold) / v (prod)
  = the chain's pre-meta base, context target = clip(y - v41, +-KSTK_CLIP) (default 1800, the T2x screen rule / km_adapters_feed.py), every instance reads ALL meta columns.
  Month key of a fold row = the first held month of its fold; prod rows are rank rows (y NaN).
CONTEXT = the chain GBM meta's own training set (chainvar_stack METAFIT=pairs, oof_build_folds=6; audited_assemble FIT=PAIRS):
  pair fold f (months39/28/410/612/511) <- the other 4 pairs; janjul <- the 5 pairs; prod <- the 5 pairs. janjul rows are NEVER context
  (KSTK_JJCTX=1 would add them for the pair folds and prod = the KSTK_FEED METAFIT=all form; not used). Implemented by wrapping km_engine.split.
Output taxi = offset + gK (agg); scripts/kstk_gk.py turns the instance files / member files into data/oof/KSTK6<chain>_<fold>.npy and the prod file.
usage: python km_adapters_kstk.py <unit> ... | --from-list list.txt [--claim]        units as km_engine (inst:<fold>:<APT>:<i>, agg:<fold>:<APT>; fold may be prod)
       python km_adapters_kstk.py --selftest [--real]   numpy only: frame + split semantics on synthetic dumps [+ the real dumps in $KSTK_DIR]
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, time, numpy as np

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.abspath(os.path.join(HERE, ".."))
KSTK_CHAIN = E("KSTK_CHAIN", "")
KSTK_DIR = E("KSTK_DIR", os.path.join(BUNDLE, "data", "payload", "stk6"))
KSTK_CLIP = float(E("KSTK_CLIP", "1800"))
KSTK_JJCTX = int(E("KSTK_JJCTX", "0"))
FO6 = ["months39", "months28", "months410", "months612", "months511", "janjul"]
UNITS = FO6 + ["prod"]
HOLD = {
    "janjul": [1, 7],
    "months28": [2, 8],
    "months39": [3, 9],
    "months410": [4, 10],
    "months511": [5, 11],
    "months612": [6, 12],
    "prod": None,
}  # = km_task.HOLD


def load_dumps(chain, d=None, need_prod=True):
    d = d or KSTK_DIR
    out = {}
    for t in FO6 + (["prod"] if need_prod else []):
        p = f"{d}/stk_{chain}_{t}.npz"
        assert os.path.exists(
            p
        ), f"KSTK_ASSEMBLE: dump missing {p} (scripts/kstk_run.sh dump on the CPU host, rsync into data/payload/stk6)"
        z = np.load(p, allow_pickle=False)
        out[t] = {k: z[k] for k in z.files}
    cols = [list(map(str, out[t]["cols"])) for t in out]
    assert all(c == cols[0] for c in cols), "KSTK_ASSEMBLE: meta columns differ between dumps"
    return out, cols[0]


def kstk_frame(dumps, apt):
    """Frame of one airport: the 6 fold dumps then prod (if present). row = global position in that concatenation (unique; agg locates rows by it)."""
    parts, base = [], 0
    for t in [x for x in UNITS if x in dumps]:
        z = dumps[t]
        sel = np.where(z["apt"] == apt)[0]
        n = len(z["mid"])
        pr = t == "prod"
        parts.append(
            dict(
                X=z["X"][sel].astype(np.float32),
                y=(np.full(len(sel), np.nan) if pr else z["y"][sel].astype(float)),
                off=z["v41"][sel].astype(float),
                m=np.full(len(sel), 0 if pr else HOLD[t][0]),
                ids=z["mid"][sel].astype(float),
                row=(base + sel).astype(np.int64),
                rank=np.full(len(sel), pr),
                jj=np.full(len(sel), t == "janjul"),
            )
        )
        base += n
    F = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    F["clean"] = np.isfinite(F["off"]) & (F["rank"] | np.isfinite(F["y"]))
    F["lk"] = np.ones(len(F["y"]), bool)
    F["off"] = np.where(np.isfinite(F["off"]), F["off"], 0.0)
    return F


def split_engine(F, fold):  # verbatim km_engine.split (self test; live mode wraps the engine's own)
    m, y, rank, clean = F["m"], F["y"], F["rank"], F["clean"]
    if HOLD[fold] is None:
        val = rank
        trn = ~rank & clean & np.isfinite(y)
    else:
        val = ~rank & np.isin(m, HOLD[fold])
        trn = ~rank & ~np.isin(m, HOLD[fold]) & clean & np.isfinite(y)
    return np.where(trn)[0], np.where(val)[0]


def pairs_split(F, fold, base_split, jjctx=None):
    """KSTK_ASSEMBLE context rule on top of the engine split: janjul rows never in the context (= the GBM meta's pairs protocol) unless KSTK_JJCTX=1."""
    ti, qi = base_split(F, fold)
    jj = KSTK_JJCTX if jjctx is None else jjctx
    if not jj:
        ti = ti[~F["jj"][ti]]
    return ti, qi


def selftest(real=False):
    rng = np.random.default_rng(0)
    D = {}
    for j, t in enumerate(UNITS):
        n = 2000 + 50 * j
        apt = rng.choice(np.array(["LSZH", "EDDM", "LTFM"]), n)
        y = rng.normal(900, 200, n)
        y[rng.random(n) < 0.01] = np.nan
        if t == "prod":
            y = None
        D[t] = dict(
            apt=apt,
            v41=rng.normal(900, 200, n),
            X=rng.standard_normal((n, 43)).astype(np.float32),
            mid=(j * 10**6 + np.arange(n)).astype(float),
        )
        if y is not None:
            D[t]["y"] = y
    for apt in ("LSZH", "LTFM"):
        S = kstk_frame(D, apt)
        assert len(np.unique(S["row"])) == len(S["row"]) and len(np.unique(S["ids"])) == len(S["ids"])
        for t in UNITS:
            ti, qi = pairs_split(S, t, split_engine, 0)
            own = np.isin(S["ids"], D[t]["mid"])
            assert np.array_equal(np.sort(qi), np.where(own)[0]), (apt, t, "query != the unit's rows")
            assert not S["rank"][ti].any() and not S["jj"][ti].any(), (apt, t, "prod or janjul row in the context")
            want = ~S["rank"] & ~S["jj"] & ~own & np.isfinite(S["y"])
            assert np.array_equal(np.sort(ti), np.where(want)[0]), (apt, t, "context != the GBM meta's training folds")
            ti1, _ = pairs_split(S, t, split_engine, 1)
            assert S["jj"][ti1].any() == (t != "janjul"), (
                apt,
                t,
                "KSTK_JJCTX=1 must add janjul rows to every non-janjul unit",
            )
        r = np.clip(S["y"] - S["off"], -KSTK_CLIP, KSTK_CLIP)
        assert np.nanmax(np.abs(r)) <= KSTK_CLIP
    print(
        f"SELFTEST OK: 6 folds + prod frame, unique rows/ids, query = the unit's rows, context = pairs protocol (janjul + prod rows never context; "
        f"janjul <- 5 pairs, pair <- 4 pairs, prod <- 5 pairs) | KSTK_CLIP {KSTK_CLIP}",
        flush=True,
    )
    if real:
        for c in ("Kq", "NNA3"):
            if not os.path.exists(f"{KSTK_DIR}/stk_{c}_janjul.npz"):
                print(f"REAL {c}: no dumps in {KSTK_DIR}")
                continue
            Dr, cols = load_dumps(c, need_prod=os.path.exists(f"{KSTK_DIR}/stk_{c}_prod.npz"))
            apts = sorted(set(np.concatenate([Dr[t]["apt"] for t in Dr]).tolist()))
            tot = {t: 0 for t in Dr}
            ctx = {}
            for apt in apts:
                S = kstk_frame(Dr, apt)
                for t in Dr:
                    ti, qi = pairs_split(S, t, split_engine, 0)
                    tot[t] += len(qi)
                    ctx[(t, apt)] = len(ti)
                    assert not S["jj"][ti].any() and not S["rank"][ti].any()
            for t in Dr:
                assert tot[t] == len(Dr[t]["mid"]), (c, t, tot[t], len(Dr[t]["mid"]))
            print(
                f"REAL {c}: {len(cols)} meta cols, {len(apts)} airports, units {list(Dr)}; query rows == dump rows {tot}; "
                f"LTFM context janjul {ctx.get(('janjul', 'LTFM'), 0):,} months28 {ctx.get(('months28', 'LTFM'), 0):,} prod {ctx.get(('prod', 'LTFM'), 0):,}",
                flush=True,
            )


if __name__ == "__main__" and "--selftest" in sys.argv:
    selftest(real="--real" in sys.argv)
    sys.exit(0)

# ---------------- live mode (torch + sdm are imported by km_engine)
sys.path.insert(0, HERE)
import km_engine as KE

K = KE.K
_cache = {}
if KSTK_CHAIN:
    _DUMPS, _SCOLS = load_dumps(KSTK_CHAIN, need_prod=os.path.exists(f"{KSTK_DIR}/stk_{KSTK_CHAIN}_prod.npz"))
    _orig_fit, _orig_split = KE.fit, KE.split

    def frame(apt):
        if apt not in _cache:
            _cache.clear()
            F = kstk_frame(_DUMPS, apt)
            assert len(F["y"]) > 0, f"KSTK_ASSEMBLE: no rows for {apt}"
            _cache[apt] = F
            print(
                f"  kstk[{KSTK_CHAIN}] {apt}: {len(F['y']):,} rows x {F['X'].shape[1]} meta cols (prod rows {int(F['rank'].sum()):,}, janjul rows {int(F['jj'].sum()):,})",
                flush=True,
            )
        return _cache[apt]

    def columns(i, fold):
        assert fold in _DUMPS, f"KSTK_ASSEMBLE: unit fold {fold} not dumped"
        return np.arange(len(_SCOLS)), "kstk-all"

    def fit(xc, yc, i):
        return _orig_fit(xc, np.clip(yc, -KSTK_CLIP, KSTK_CLIP), i)

    def split(F, fold):
        return pairs_split(F, fold, _orig_split)

    K.frame = frame
    KE.columns = columns
    KE.fit = fit
    KE.split = split
    KE.S["kstk"] = dict(
        chain=KSTK_CHAIN,
        dir=os.path.basename(KSTK_DIR.rstrip("/")),
        cols=_SCOLS,
        clip=KSTK_CLIP,
        jjctx=KSTK_JJCTX,
        units=list(_DUMPS),
    )
    print(f"km_adapters_kstk active: {json.dumps(KE.S['kstk'])[:300]}", flush=True)

if __name__ == "__main__":
    import socket

    a = sys.argv[1:]
    claim = "--claim" in a
    a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list":
        a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(f"km_adapters_kstk: chain {KSTK_CHAIN or 'none (native engine)'} | engine spec {KE.S}", flush=True)
    t0 = time.time()
    n = 0
    for spec in a:
        if claim:
            lk = f"{KE.RUN}/locks/{KE.S['name']}_{spec.replace(':', '_')}"
            os.makedirs(f"{KE.RUN}/locks", exist_ok=True)
            try:
                os.mkdir(lk)
            except FileExistsError:
                continue
            open(f"{lk}/owner", "w").write(f"{socket.gethostname()} {os.getpid()}\n")
        try:
            r = KE.run(spec)
        except BaseException:
            if claim:
                import shutil

                shutil.rmtree(lk, ignore_errors=True)
            raise
        n += r is not None
    print(f"KM_KSTK_DONE units run {n} | {time.time() - t0:.0f}s", flush=True)
