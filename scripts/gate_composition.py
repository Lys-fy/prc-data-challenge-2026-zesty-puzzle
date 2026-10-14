#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Composition-aware gate report (N-gate,-43). Label-free reweighting of a candidate's per-row fold dMSE to the
2026 ranking set's cell shares, reported PER FOLD next to the raw fold number (never pooled across folds).

Cells: airport x link (adsb_linked) x observable class {M_main, M_gate, unm_other, unm_LIRF, exposed}.
  M_gate  = matched with AOBT_3 < 300 or > 2200, EOBT_1 delay > 1 h, or |LOBT - AOBT_3| > 2400
  exposed = unmatched with MVT - SCHED > 8 h; unm_LIRF = other unmatched at LIRF; unm_other = other unmatched.
Weight of a fold row = n_rank(cell) / n_fold(cell). A cell with < 40 fold rows falls back to airport x class, then
link x class; exposed uses LIRF / non-LIRF only. Only covariates are counted (no ranking labels are used).
Protected rows (unm_LIRF, exposed) are frozen in production, so their dSE is set to 0 unless --no-freeze.

Series spec: a '+'-joined list of [w*]PREFIX with {fold} substituted, each PREFIX a data/oof/<PREFIX>.npy array of
shape (k, n) with row 0 = MVT_ID and row 1 = completed prediction. Default post-processing: cf (clip to mvt_lobt +- 3606
where present, then max 60); --no-cf to skip.

usage: gate_composition.py --base "0.5*RA3_{fold}+0.5*RAC2_{fold}" --cand "RA3_{fold}" [--folds janjul_months17,...]
       [--out file.csv] [--no-cf] [--no-freeze]
"""

import os, sys, argparse, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RN = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)
MIN = 40
DEF_FOLDS = "months39_months39,months28_months28,months410_months410,janjul_months17"


def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606))
    return np.maximum(v, 60.0)


def load_series(spec, fold):
    ids = None
    acc = None
    for term in spec.split("+"):
        term = term.strip()
        w = 1.0
        if "*" in term:
            w, term = term.split("*", 1)
            w = float(w)
        a = np.load(f"{ROOT}/data/oof/{term.format(fold=fold)}.npy")
        a = a[:, np.argsort(a[0])]
        if ids is None:
            ids = a[0]
            acc = w * a[1]
        else:
            assert np.array_equal(ids, a[0]), f"ID mismatch in {term}"
            acc = acc + w * a[1]
    return ids, acc


def classify(D):
    lo = D["mvt_lobt"].to_numpy().astype(float)
    ao = D["mvt_aobt3"].to_numpy().astype(float)
    eo = D["mvt_eobt1"].to_numpy().astype(float)
    ms = np.nan_to_num(D["mvt_sched"].to_numpy().astype(float), nan=0.0)
    u = D["unmatched"].to_numpy() == 1
    apt = D["ADEP_mvt"].to_numpy().astype(str)
    stale = np.abs(np.nan_to_num(lo - np.nan_to_num(ao, nan=0.0), nan=0.0)) > 2400
    gate = ~np.isnan(ao) & (
        (np.nan_to_num(ao, nan=-1.0) < 300)
        | (np.nan_to_num(ao, nan=-1.0) > 2200)
        | (np.nan_to_num(eo, nan=0.0) > 3600)
        | stale
    )
    exp_ = u & (ms > 28800)
    uL = u & ~exp_ & (apt == "LIRF")
    uo = u & ~exp_ & ~uL
    obs = np.where(
        exp_, "exposed", np.where(uL, "unm_LIRF", np.where(uo, "unm_other", np.where(gate, "M_gate", "M_main")))
    )
    lk = np.where(np.nan_to_num(D["adsb_linked"].to_numpy().astype(float), nan=0) == 1, "lnk", "none")
    return apt, lk, obs


def weights(apt, lk, obs, rcnt):
    """ranking count per (apt|lk|obs) -> per-fold-row weights, with the fallbacks. Returns w, fallback tags."""
    fk = np.char.add(np.char.add(np.char.add(np.char.add(apt, "|"), lk), "|"), obs)
    w = np.zeros(len(apt))
    used = {}
    for key, nr in rcnt.items():
        a_, l_, o_ = key.split("|")
        if o_ == "exposed":
            sel = (obs == "exposed") & ((apt == "LIRF") if a_ == "LIRF" else (apt != "LIRF"))
            tag = f"exposed|{'LIRF' if a_ == 'LIRF' else 'nonLIRF'}"
        else:
            sel = fk == key
            tag = key
            if sel.sum() < MIN:
                sel = (apt == a_) & (obs == o_)
                tag = f"{a_}|*|{o_}"
            if sel.sum() < MIN:
                sel = (obs == o_) & (lk == l_)
                tag = f"*|{l_}|{o_}"
        if sel.sum() == 0:
            used.setdefault("UNCOVERED", [0, 0])
            used["UNCOVERED"][0] += nr
            continue
        used.setdefault(tag, [0, int(sel.sum()), sel])
        used[tag][0] += nr
    for tag, v in used.items():
        if tag == "UNCOVERED":
            continue
        nr, nf, sel = v
        w[sel] += nr / nf
    fb = sorted(t for t in used if "*" in t or t.startswith("exposed") or t == "UNCOVERED")
    return w, fb, used.get("UNCOVERED", [0])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--folds", default=DEF_FOLDS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cf", action="store_true")
    ap.add_argument("--no-freeze", action="store_true")
    a = ap.parse_args()
    cols = ["MVT_ID_mvt", "is_rank", "y", "ADEP_mvt", "unmatched", "mvt_lobt", "mvt_aobt3", "mvt_eobt1", "mvt_sched"]
    F = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=cols).join(
        pl.read_parquet(f"{ROOT}/data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"]),
        on="MVT_ID_mvt",
        how="left",
    )
    Rk = F.filter(pl.col("is_rank") == 1)
    ra, rl, ro = classify(Rk)
    rk = np.char.add(np.char.add(np.char.add(np.char.add(ra, "|"), rl), "|"), ro)
    k_, c_ = np.unique(rk, return_counts=True)
    rcnt = dict(zip(k_.tolist(), c_.tolist()))
    assert sum(rcnt.values()) == RN
    print(
        f"base {a.base}\ncand {a.cand}\nranking linked share {np.mean(rl == 'lnk'):.3f} | cf {'off' if a.no_cf else 'on'} | protected rows "
        f"{'scored' if a.no_freeze else 'frozen (dSE=0)'}"
    )
    print(
        f"{'fold':22s} {'n':>8s} {'lnk%':>5s} | {'raw dMSE':>9s} {'raw w/o top3':>12s} | {'comp dMSE':>9s} {'comp w/o top3':>13s} | {'board s2':>8s} {'board s':>7s} | fallbacks"
    )
    rows = []
    for fold in a.folds.split(","):
        ib, pb = load_series(a.base, fold)
        ic, pc = load_series(a.cand, fold)
        common = np.intersect1d(ib, ic)
        pb = pb[np.searchsorted(ib, common)]
        pc = pc[np.searchsorted(ic, common)]
        D = (
            pl.DataFrame({"MVT_ID_mvt": common, "b": pb, "c": pc})
            .join(F, on="MVT_ID_mvt", how="left")
            .filter(pl.col("y").is_not_null() & (pl.col("is_rank") == 0))
        )
        y = D["y"].to_numpy().astype(float)
        lo = D["mvt_lobt"].to_numpy().astype(float)
        b = D["b"].to_numpy()
        c = D["c"].to_numpy()
        if not a.no_cf:
            b = cf(b, lo)
            c = cf(c, lo)
        apt, lk, obs = classify(D)
        d = (c - y) ** 2 - (b - y) ** 2
        if not a.no_freeze:
            d = np.where((obs == "unm_LIRF") | (obs == "exposed"), 0.0, d)
        w, fb, unc = weights(apt, lk, obs, rcnt)
        top = np.argsort(-np.abs(d))[:3]
        keep = np.ones(len(d), bool)
        keep[top] = False
        raw = d.mean()
        raw3 = d[keep].mean()
        comp = (w * d).sum() / w.sum()
        comp3 = (w * d)[keep].sum() / w[keep].sum()
        board = comp  # comp is already a per-ranking-row mean = board s2 (sum w = ranking rows covered)
        print(
            f"{fold:22s} {len(d):8,d} {100*np.mean(lk=='lnk'):5.1f} | {raw:+9.1f} {raw3:+12.1f} | {comp:+9.1f} {comp3:+13.1f} | {board:+8.1f} "
            f"{board/483.5:+7.3f} | {len(fb)} ({', '.join(fb[:4])}{'...' if len(fb) > 4 else ''}){' UNCOVERED ' + str(unc) if unc else ''}"
        )
        # per-airport composition contributions (board s2), for reading where the shift comes from
        per = {}
        for ap_ in np.unique(apt):
            k = apt == ap_
            per[ap_] = ((w * d)[k].sum() / w.sum(), d[k].sum() / len(d))
        rows.append(
            dict(
                fold=fold,
                n=len(d),
                linked=float(np.mean(lk == "lnk")),
                raw=raw,
                raw_wo3=raw3,
                comp=comp,
                comp_wo3=comp3,
                **{f"comp_{k}": v[0] for k, v in per.items()},
                **{f"raw_{k}": v[1] for k, v in per.items()},
            )
        )
    T = pl.DataFrame(rows)
    print("\nper-airport contribution, composition-weighted (raw in brackets), s2 per row:")
    apts = sorted({c[5:] for c in T.columns if c.startswith("comp_") and c not in ("comp_wo3",)})
    for r in T.iter_rows(named=True):
        print(
            f"  {r['fold']:22s} "
            + " ".join(f"{ap_} {r.get('comp_'+ap_, 0):+6.1f}({r.get('raw_'+ap_, 0):+6.1f})" for ap_ in apts)
        )
    wins_raw = int((T["raw"] < 0).sum())
    wins_comp = int((T["comp"] < 0).sum())
    print(
        f"\nwins raw {wins_raw}/{T.height}, composition {wins_comp}/{T.height}; janjul raw {T.filter(pl.col('fold').str.starts_with('janjul'))['raw'].to_list()} "
        f"comp {T.filter(pl.col('fold').str.starts_with('janjul'))['comp'].to_list()}"
    )
    if a.out:
        T.write_csv(a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
