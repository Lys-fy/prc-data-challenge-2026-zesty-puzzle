#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Residual-target adapter for km_engine.py -- km_engine.py itself is NOT edited ( the clean patch is
proposed in). This module imports km_engine unchanged and patches three functions at module level:
  1. km_task.frame(apt)  -> the same frame with `off` replaced by the fold's OFFSET TABLE (the fold-honest nested L of:
                            $BUNDLE/data/resid/<RESID_L>_<fold>.npy = [MVT_ID, off], joined on MVT_ID) and, with OFFCOL=1, X extended by one column = off.
                            Rows without an L value keep mvt_aobt3, are flagged clean=False (never in a context) and counted (expected: none).
  2. km_engine.columns   -> the engine's column subset + index 412 (the L column) when OFFCOL=1.
  3. km_engine.fit       -> the same fit with the context target clipped to +-RCLIP (default 1800 s; 0 = off).
So the context target becomes clip(y - L_nest, +-1800), every instance reads L as an input, and the member output is taxi = L_nest + g_hat.
Everything else (slices, column views, seeds, sharding/resume, trimmed mean, spread/g_sd, memlog + guard, claiming, npz format) is km_engine's,
and collect.py / the screens run unchanged.
env (besides km_engine's NAME, SIZE, CAP, NCOL, COLMODE, CORE, NINST, EIN, AMP, SEED, ...): RESID_L=kumo_resid_adapter_lnest  OFFCOL=1  RCLIP=1800  RESID_DIR ($BUNDLE/data/resid)
usage: python km_resid.py <unit> ... | --from-list list.txt [--claim]     (units as km_engine: inst:<fold>:<APT>:<i>  agg:<fold>:<APT>)
       python km_resid.py --selftest                                       (numpy-only check of the frame/column patch on a synthetic frame; no torch)
"""

import os, sys, numpy as np

E = os.environ.get
RESID_L = E("RESID_L", "kumo_resid_adapter_lnest")
OFFCOL = int(E("OFFCOL", "1"))
RCLIP = float(E("RCLIP", "1800"))
RESID_DIR = E("RESID_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "resid"))
CUR = {"fold": None}
_tab = {}
_cache = {}


def offsets(fold):
    """[ids_sorted, off_sorted] of the fold's offset table (prod: fold 'prod')."""
    if fold not in _tab:
        _tab.clear()
        p = f"{RESID_DIR}/{RESID_L}_{fold}.npy"
        assert os.path.exists(
            p
        ), f"offset table missing: {p} (copy the fold's offset table into the bundle; not part of the final model)"
        a = np.load(p)
        o = np.argsort(a[0])
        _tab[fold] = (a[0][o], a[1][o].astype(float))
    return _tab[fold]


def lookup(tab, ids):
    i, v = tab
    k = np.clip(np.searchsorted(i, ids), 0, len(i) - 1)
    hit = i[k] == ids
    out = np.full(len(ids), np.nan)
    out[hit] = v[k[hit]]
    return out


def patch_frame(F, off, offcol):
    """New frame dict: off replaced where available; rows without an L value keep mvt_aobt3 and are excluded from contexts; X gets the L column."""
    miss = ~np.isfinite(off)
    G = dict(F)
    G["off"] = np.where(miss, F["off"], off)
    G["clean"] = F["clean"] & ~miss
    G["n_noL"] = int(miss.sum())
    if offcol:
        G["X"] = np.hstack([F["X"], np.where(miss, np.nan, off).astype(np.float32)[:, None]])
    return G


def selftest():
    rng = np.random.default_rng(0)
    n = 1000
    ids = np.sort(rng.choice(10**6, n, replace=False)).astype(float)
    F = dict(
        X=rng.standard_normal((n, 412)).astype(np.float32),
        off=np.zeros(n),
        clean=np.ones(n, bool),
        ids=ids,
        y=rng.standard_normal(n),
    )
    off_true = rng.normal(900, 100, n)
    keep = rng.random(n) > 0.01  # 1 % of rows without an L value
    tab_ids = ids[keep]
    perm = rng.permutation(len(tab_ids))
    tab = (tab_ids[perm], off_true[keep][perm])  # unsorted table on purpose
    o = np.argsort(tab[0])
    tab = (tab[0][o], tab[1][o])
    off = lookup(tab, F["ids"])
    assert np.allclose(off[keep], off_true[keep]) and np.isnan(off[~keep]).all()
    G = patch_frame(F, off, 1)
    assert G["X"].shape == (n, 413) and np.allclose(G["X"][keep, 412], off_true[keep].astype(np.float32))
    assert (
        np.isnan(G["X"][~keep, 412]).all()
        and (G["off"][~keep] == 0).all()
        and (~G["clean"][~keep]).all()
        and G["clean"][keep].all()
        and G["n_noL"] == int((~keep).sum())
    )
    cols = np.append(np.arange(100), 412)
    assert G["X"][:, cols].shape == (n, 101)
    yc = np.array([-5000.0, 0.0, 5000.0])
    assert np.array_equal(np.clip(yc, -RCLIP, RCLIP), [-1800.0, 0.0, 1800.0])
    print(
        f"SELFTEST OK: lookup + frame patch + column append + clip (RESID_L {RESID_L}, OFFCOL {OFFCOL}, RCLIP {RCLIP}, RESID_DIR {RESID_DIR})"
    )


if __name__ == "__main__" and "--selftest" in sys.argv:
    selftest()
    sys.exit(0)

# ---------------- live mode: wrap the engine (torch + sdm are imported here, not above)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import km_engine as KE

K = KE.K
_orig_frame = K.frame
_orig_columns = KE.columns
_orig_fit = KE.fit
KE.S.update(resid_l=RESID_L, offcol=OFFCOL, rclip=RCLIP)  # recorded in every unit's meta.spec


def frame(apt):
    fold = CUR["fold"]
    assert fold, "km_resid: fold unknown (frame called outside a unit)"
    key = (fold, apt)
    if key not in _cache:
        _cache.clear()
        F = _orig_frame(apt)
        off = lookup(offsets(fold), F["ids"])
        G = patch_frame(F, off, OFFCOL)
        _cache[key] = G
        print(
            f"  resid[{RESID_L}] {fold} {apt}: L offset on {len(off) - G['n_noL']:,}/{len(off):,} rows (no L: {G['n_noL']}), off mean {np.nanmean(off):.1f} sd {np.nanstd(off):.1f} | "
            f"X {G['X'].shape[1]} cols | context target clip +-{RCLIP:.0f}",
            flush=True,
        )
    return _cache[key]


def columns(i, fold):
    cols, src = _orig_columns(i, fold)
    return (np.append(cols, KE.D), src + "+L") if OFFCOL else (cols, src)


def fit(xc, yc, i):
    return _orig_fit(xc, np.clip(yc, -RCLIP, RCLIP) if RCLIP > 0 else yc, i)


K.frame = frame
KE.columns = columns
KE.fit = fit

if __name__ == "__main__":
    import time, socket

    a = sys.argv[1:]
    claim = "--claim" in a
    a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list":
        a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(f"km_resid: RESID_L {RESID_L} OFFCOL {OFFCOL} RCLIP {RCLIP} dir {RESID_DIR} | engine spec {KE.S}", flush=True)
    t0 = time.time()
    n = 0
    for spec in a:
        p = spec.split(":")
        assert p[0] in ("inst", "agg"), f"km_resid runs inst/agg units only ({spec})"
        CUR["fold"] = p[1]
        _cache.clear()
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
    print(f"KM_RESID_DONE units run {n} | {time.time() - t0:.0f}s", flush=True)
