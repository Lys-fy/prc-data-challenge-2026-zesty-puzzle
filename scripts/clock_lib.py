#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Shared helpers for the clock-family diagnosis / reweighting.
src(F)            anchor clock source per row (lgb_adsb_linked definition): 0 adsb2_leave, 1 adsb3_mv_gs, 2 adsb_move, 3 none (AOBT_3 / nothing),
                  each clock only inside [0, 7200] s. Label-free.
rank_frame()      ranking rows (all 344,841) as clockresid_lib.frame + src + gate_composition class
cells(F)          string cell key apt|lnk|class|src (class = gate_composition observable class)
cell_weights      fold-row weights n_rank(cell)/n_fold(cell) with fallbacks (apt|lnk|class|src -> apt|lnk|class -> lnk|class|src -> lnk|class),
                  min 40 fold rows per cell. Sum of weights = ranking rows covered.
SRC = names."""

import os, sys, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
import clockresid_lib as L, gate_composition as GC

SRC = ["leave", "mv_gs", "move", "none"]
RN = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)
MIN = 40


def src(F):
    def ok(c):
        v = F[c].cast(pl.Float64).to_numpy()
        return np.isfinite(v) & (v >= 0) & (v <= 7200)

    s = np.full(F.height, 3)
    for k, c in reversed(list(enumerate(["adsb2_leave", "adsb3_mv_gs", "adsb_move"]))):
        s = np.where(ok(c), k, s)
    return s


def rank_frame():
    R = pl.read_parquet(f"{ROOT}/data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank"]).filter(
        pl.col("is_rank") == 1
    )
    return L.frame(R["MVT_ID_mvt"].to_numpy())


def keys(F, s=None):
    apt, lk, obs = GC.classify(F)
    s = src(F) if s is None else s
    sn = np.array(SRC)[s]
    sn = np.where(lk == "lnk", sn, "-")  # source only meaningful on linked rows
    return apt, lk, obs, sn


def cell_weights(fk, rk, matched_only=True):
    """fk, rk: tuples (apt, lk, obs, src) arrays for fold rows / ranking rows. Returns w (fold rows), covered ranking rows."""
    fa, fl, fo, fs = fk
    ra, rl, ro, rs = rk
    if matched_only:
        m = np.char.startswith(ro.astype(str), "M_")
        ra, rl, ro, rs = ra[m], rl[m], ro[m], rs[m]
    J = lambda *a: np.array(["|".join(x) for x in zip(*a)])
    levels = [
        (J(fa, fl, fo, fs), J(ra, rl, ro, rs)),
        (J(fa, fl, fo), J(ra, rl, ro)),
        (J(fl, fo, fs), J(rl, ro, rs)),
        (J(fl, fo), J(rl, ro)),
    ]
    w = np.zeros(len(fa))
    done = np.zeros(len(ra), bool)
    cov = 0
    for fkey, rkey in levels:
        uk, inv = np.unique(rkey[~done], return_inverse=True)
        rc = np.bincount(inv)
        fu, fc = np.unique(fkey, return_counts=True)
        fcnt = dict(zip(fu.tolist(), fc.tolist()))
        idx = np.where(~done)[0]
        for j, k in enumerate(uk.tolist()):
            nf = fcnt.get(k, 0)
            if nf < MIN:
                continue
            w[fkey == k] += rc[j] / nf
            done[idx[inv == j]] = True
            cov += rc[j]
    return w, cov, int((~done).sum())


# ---- month_weight: month-aware cells. slot W = months {11,12,1,2,3,4}, S = {5..10}; ranking Jan 2026 -> W, Jul 2026 -> S.
def slot(F):
    m = F["MVT_TIME_UTC_mvt"].dt.month().to_numpy()
    return np.where(np.isin(m, [11, 12, 1, 2, 3, 4]), "W", "S")


def cell_weights_mw(fk, rk, fslot, rslot, matched_only=True, levels_out=None):
    """month-aware cell_weights: apt|lnk|class|src|slot -> apt|lnk|class|slot -> apt|lnk|slot -> month-blind chain of cell_weights.
    fk, rk as cell_weights; fslot/rslot = slot arrays. levels_out (dict) collects ranking rows covered per level."""
    fa, fl, fo, fs = fk
    ra, rl, ro, rs = rk
    rt = np.asarray(rslot)
    ft = np.asarray(fslot)
    if matched_only:
        m = np.char.startswith(ro.astype(str), "M_")
        ra, rl, ro, rs, rt = ra[m], rl[m], ro[m], rs[m], rt[m]
    J = lambda *a: np.array(["|".join(x) for x in zip(*a)])
    levels = [
        ("L1 apt|lnk|cls|src|slot", J(fa, fl, fo, fs, ft), J(ra, rl, ro, rs, rt)),
        ("L2 apt|lnk|cls|slot", J(fa, fl, fo, ft), J(ra, rl, ro, rt)),
        ("L3 apt|lnk|slot", J(fa, fl, ft), J(ra, rl, rt)),
        ("L4 apt|lnk|cls|src", J(fa, fl, fo, fs), J(ra, rl, ro, rs)),
        ("L5 apt|lnk|cls", J(fa, fl, fo), J(ra, rl, ro)),
        ("L6 lnk|cls|src", J(fl, fo, fs), J(rl, ro, rs)),
        ("L7 lnk|cls", J(fl, fo), J(rl, ro)),
    ]
    w = np.zeros(len(fa))
    done = np.zeros(len(ra), bool)
    cov = 0
    for name, fkey, rkey in levels:
        if done.all():
            break
        uk, inv = np.unique(rkey[~done], return_inverse=True)
        rc = np.bincount(inv)
        fu, fc = np.unique(fkey, return_counts=True)
        fcnt = dict(zip(fu.tolist(), fc.tolist()))
        idx = np.where(~done)[0]
        lv = 0
        for j, k in enumerate(uk.tolist()):
            nf = fcnt.get(k, 0)
            if nf < MIN:
                continue
            w[fkey == k] += rc[j] / nf
            done[idx[inv == j]] = True
            cov += rc[j]
            lv += rc[j]
        if levels_out is not None:
            levels_out[name] = levels_out.get(name, 0) + lv
    return w, cov, int((~done).sum())


# ---- month_weight / : disruption-aware weights. D(apt, day) = share of the airport-day's matched DEP rows with mvt_lobt > 3600.
DEDGES = [0.01, 0.02, 0.05, 0.10, 0.20]
_DMAP = None


def dbin(ids):
    global _DMAP
    if _DMAP is None:
        G = (
            pl.read_parquet(
                f"{ROOT}/data/features_v9.parquet",
                columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "unmatched", "MVT_TIME_UTC_mvt", "mvt_lobt"],
            )
            .filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("unmatched") == 0))
            .with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date())
        )
        G = G.join(
            G.group_by("ADEP_mvt", "day").agg(D=(pl.col("mvt_lobt") > 3600).mean()), on=["ADEP_mvt", "day"], how="left"
        )
        _DMAP = pl.DataFrame({"MVT_ID_mvt": G["MVT_ID_mvt"].cast(pl.Float64), "D": G["D"].cast(pl.Float64)})
    D = (
        pl.DataFrame({"MVT_ID_mvt": np.asarray(ids, dtype=float)})
        .join(_DMAP, on="MVT_ID_mvt", how="left", maintain_order="left")["D"]
        .fill_null(0.0)
        .to_numpy()
    )
    return np.digitize(D, DEDGES)


def cell_weights_mwd(fk, rk, fslot, rslot, fD, rD, matched_only=True, info=None):
    """MW weights re-allocated over D-bins inside each apt|lnk|slot group to the 2026 D-bin mix; thin bins (< MIN fold rows)
    pass their 2026 rows to the next-lower bin; groups with thin bin 0 or no 2026 rows keep w_MW. Returns w, w_MW."""
    wm = cell_weights_mw(fk, rk, fslot, rslot, matched_only)[0]
    ra, rl, rt, rDD = rk[0], rk[1], np.asarray(rslot), np.asarray(rD)
    if matched_only:
        m = np.char.startswith(rk[2].astype(str), "M_")
        ra, rl, rt, rDD = ra[m], rl[m], rt[m], rDD[m]
    fg = np.array(["|".join(x) for x in zip(fk[0], fk[1], np.asarray(fslot))])
    rg = np.array(["|".join(x) for x in zip(ra, rl, rt)])
    fD = np.asarray(fD)
    w = wm.copy()
    nb = len(DEDGES) + 1
    kept = 0
    for g in np.unique(fg):
        fm = fg == g
        n26 = np.bincount(rDD[rg == g], minlength=nb).astype(float)
        nf = np.bincount(fD[fm], minlength=nb)
        if n26.sum() == 0:
            kept += 1
            continue
        for b in range(nb - 1, 0, -1):
            if nf[b] < MIN:
                n26[b - 1] += n26[b]
                n26[b] = 0
        if nf[0] < MIN:
            kept += 1
            continue
        tot = wm[fm].sum()
        for b in range(nb):
            cm = fm & (fD == b)
            sw = wm[cm].sum()
            if cm.any() and sw > 0:
                w[cm] = wm[cm] * (n26[b] / n26.sum()) * tot / sw
    if info is not None:
        info["groups_kept_MW"] = kept
    return w, wm
