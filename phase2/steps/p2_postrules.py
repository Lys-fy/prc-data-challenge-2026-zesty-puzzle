#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fits or applies two post-chain rules, standfloor (stand-exclusivity floor) and offblock_pull (late-off-block
overlay), via a parameter file rather than candidate files tied to one ranking set.
`fit --fold <prefix> --x4 <X4_fold.parquet> --out params.json` fits per-airport rule parameters on 2025 fold
vectors and row weights, writing params.json.
`apply --in <parquet> --out <parquet> --params params.json [--only xsx|xpk]` applies the fitted rules to matched
ranking rows of --in (unmatched rows stay byte-identical) and writes --out.

the post-chain rules as fit / apply with a parameter file (instead of candidate files keyed to one ranking set).
Rules: standfloor (stand-exclusivity floor) and offblock_pull (late-NM-off-block overlay). Parameters are fitted on 2025 only (fold vectors +
ranking-mix row weights of the X4 fold frame).

  fit   --fold <prefix of the chain's (B)-stage fold vectors, e.g. scratchpad/xpr/pru_pxxb> --x4 <X4_fold.parquet> --out params.json
        standfloor: per airport arm (W, b, k) from the grid W{600,900} x b{0,300} x k{.5,1} (+ none), fit on all 12 months (full_choice).
        offblock_pull: per airport arm from {px thr 5400/7200 x k .25/.5/1, sh thr 5400/7200 (shift fitted, shrunk n/(n+50))}, fit on all 12
             months ON TOP of the standfloor month-nested fold vectors.
  apply --in <submission-like parquet> --out <parquet> --params params.json [--only xsx|xpk]
        matched ranking rows only; unmatched rows byte-identical; rint; LOBT clip +-3606 and 60 s floor (cf)."""

import os, sys, json, argparse, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
FOLDS = ["months39", "months28", "months410", "janjul", "months612", "months511"]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
RN = 344841  # only a constant divisor of the fit objective (does not change any argmin)
GRID = [(W, b, k) for W in (600, 900) for b in (0, 300) for k in (0.5, 1.0)]
ARMS = [("px", t, k) for t in (5400, 7200) for k in (0.25, 0.5, 1.0)] + [("sh", t, 0) for t in (5400, 7200)]
N = lambda D, c: D[c].to_numpy().astype(float)
ep = lambda c: pl.col(c).dt.replace_time_zone(None).dt.epoch("s").cast(pl.Float64)


def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo) - 3606, np.nan_to_num(lo) + 3606))
    return np.maximum(v, 60.0)


# ---------------- standfloor
def anchor(ssai, ssal=None, e1=False):
    a = np.where(np.isfinite(ssai) & (ssai < 7200), ssai, np.nan)
    if e1:
        a = np.where(np.isnan(a) & np.isfinite(ssal) & (ssal < 7200), ssal - 180.0, a)
    return a


def xsx_arm(p, a, lo, g):
    W, b, k = g
    sel = np.isfinite(a) & (a - p > 0) & (a - p < W)
    return cf(np.where(sel, p + k * (a + b - p), p), lo)


# ---------------- offblock_pull
def xpk_apply(arm, pp, al, lo_, s=0.0):
    f, t, k = arm
    sel = np.nan_to_num(al, nan=-1e9) > t
    if f == "px":
        return cf(np.where(sel & np.isfinite(lo_), pp + k * (np.maximum(lo_, 0) - pp), pp), lo_)
    return cf(np.where(sel, pp + s, pp), lo_)


def fit(a):
    F = pl.read_parquet(a.x4, columns=["MVT_ID_mvt", "y", "MVT_TIME_UTC_mvt", "fo", "ok", "w", "apt", "mvt_lobt"])
    G = (
        pl.read_parquet(
            "data/features_v9.parquet", columns=["MVT_ID_mvt", "stand_since_arr_inblock", "AOBT_3_flt", "LOBT_flt"]
        )
        .with_columns(MVT_ID_mvt=pl.col("MVT_ID_mvt").cast(pl.Float64), al=ep("AOBT_3_flt") - ep("LOBT_flt"))
        .drop("AOBT_3_flt", "LOBT_flt")
    )
    PO = pl.concat([pl.DataFrame({"MVT_ID_mvt": (z := np.load(f"{a.fold}_{t}.npy"))[0], "p": z[1]}) for t in FOLDS])
    D = (
        F.join(PO, on="MVT_ID_mvt", how="left", maintain_order="left")
        .join(G, on="MVT_ID_mvt", how="left", maintain_order="left")
        .filter(pl.col("ok") & pl.col("p").is_not_null())
        .with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int32))
    )
    y = N(D, "y")
    p = N(D, "p")
    w = N(D, "w")
    lo = N(D, "mvt_lobt")
    apt = D["apt"].to_numpy().astype(str)
    mon = D["month"].to_numpy()
    ssai = N(D, "stand_since_arr_inblock")
    al = N(D, "al")
    r = p - y
    A0 = anchor(ssai)
    # standfloor production choice
    xsx = {}
    for A in APTS:
        m = (apt == A) & np.isfinite(A0) & (A0 - p > 0) & (A0 - p < 900)
        best, bv = None, 0.0
        for g in GRID:
            v = xsx_arm(p[m], A0[m], lo[m], g)
            s_ = (w[m] * ((v - y[m]) ** 2 - r[m] ** 2)).sum()
            if s_ < bv:
                best, bv = g, s_
        xsx[A] = {"arm": list(best) if best else None, "obj": bv / RN / 6}
    # standfloor month-nested fold vectors = the base of the offblock_pull fit
    cand = np.isfinite(A0) & (A0 - p > 0) & (A0 - p < 900)
    idx = np.where(cand)[0]
    pi, yi, wi, loi, Ai, Mi, ai = p[idx], y[idx], w[idx], lo[idx], apt[idx], mon[idx], A0[idx]
    ri = pi - yi
    base = {g: xsx_arm(pi, ai, loi, g) for g in GRID}
    pn = p.copy()
    for m_ in range(1, 13):
        for A in APTS:
            tr = (Mi != m_) & (Ai == A)
            te = (Mi == m_) & (Ai == A)
            best, bv = None, 0.0
            for g in GRID:
                s_ = (wi[tr] * ((base[g][tr] - yi[tr]) ** 2 - ri[tr] ** 2)).sum()
                if s_ < bv:
                    best, bv = g, s_
            if best:
                pn[idx[te]] = base[best][te]
    # offblock_pull production choice on the standfloor-nested vectors (all of 2025)
    r2 = pn - y
    xpk = {}

    def shift_fit(m):
        n = w[m].sum()
        return 0.0 if n <= 0 else float(-(w[m] * r2[m]).sum() / n * n / (n + 50))

    for A in APTS:
        m = (apt == A) & (np.nan_to_num(al, nan=-1e9) > 5400)
        best, bv, bs_ = None, 0.0, 0.0
        for arm in ARMS:
            s = shift_fit(m & (np.nan_to_num(al, nan=-1e9) > arm[1])) if arm[0] == "sh" else 0.0
            e = (w[m] * ((xpk_apply(arm, pn[m], al[m], lo[m], s) - y[m]) ** 2 - r2[m] ** 2)).sum()
            if e < bv:
                best, bv, bs_ = arm, e, s
        xpk[A] = {"arm": list(best) if best else None, "shift": bs_, "obj": bv / RN / 6}
    P = {
        "xsx": xsx,
        "xpk": xpk,
        "fitted_on": {"fold": a.fold, "x4": a.x4, "rows": int(len(p))},
        "source": "phase2/steps/p2_postrules.py fit",
    }
    json.dump(P, open(a.out, "w"), indent=1)
    print("standfloor:", " ".join(f"{A}:{tuple(v['arm']) if v['arm'] else 'none'}" for A, v in xsx.items()))

    def fmt(v):
        if not v["arm"]:
            return "none"
        return str(tuple(v["arm"])) + (" s=%+.3f" % v["shift"] if v["arm"][0] == "sh" else "")

    print("offblock_pull:", " ".join(f"{A}:{fmt(v)}" for A, v in xpk.items()))


def apply(a):
    P = json.load(open(a.params))
    base = pl.read_parquet(getattr(a, "in"))
    col = "TAXITIME_SEC_mvt"
    R = (
        pl.read_parquet(
            "data/features_v9.parquet",
            columns=[
                "MVT_ID_mvt",
                "ADEP_mvt",
                "is_rank",
                "unmatched",
                "mvt_lobt",
                "stand_since_arr_inblock",
                "AOBT_3_flt",
                "LOBT_flt",
            ],
        )
        .filter(pl.col("is_rank") == 1)
        .with_columns(
            mid=pl.col("MVT_ID_mvt").cast(pl.Float64),
            ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8),
            al=ep("AOBT_3_flt") - ep("LOBT_flt"),
        )
        .drop("MVT_ID_mvt")
    )
    J = base.with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64)).join(
        R, on="mid", how="left", maintain_order="left"
    )
    assert J.height == base.height and J["unmatched"].null_count() == 0, "ids of --in are not the period's ranking rows"
    v0 = base[col].to_numpy()
    um = J["unmatched"].to_numpy() == 1
    lo = N(J, "mvt_lobt")
    ap = J["ADEP_mvt"].to_numpy().astype(str)
    cur = v0.astype(float)
    rep = {}
    if a.only in ("", "xsx"):
        a26 = anchor(N(J, "stand_since_arr_inblock"))
        new = cur.copy()
        for A in APTS:
            g = P["xsx"][A]["arm"]
            if not g:
                continue
            m = (~um) & (ap == A)
            new[m] = xsx_arm(cur[m], a26[m], lo[m], tuple(g))
        nv = np.where(um, v0, np.rint(new))
        rep["xsx"] = int((nv != cur).sum())
        cur = nv.astype(float)
    if a.only in ("", "xpk"):
        al = N(J, "al")
        new = cur.copy()
        for A in APTS:
            arm = P["xpk"][A]["arm"]
            if not arm:
                continue
            m = (~um) & (ap == A)
            new[m] = xpk_apply(tuple(arm), cur[m], al[m], lo[m], P["xpk"][A]["shift"])
        nv = np.where(um, v0, np.rint(new))
        rep["xpk"] = int((nv != cur).sum())
        cur = nv
    out = base.with_columns(pl.Series(col, cur.astype(v0.dtype)).cast(base.schema[col]))
    assert (
        out["MVT_ID_mvt"].equals(base["MVT_ID_mvt"])
        and out.schema == base.schema
        and bool((out[col].to_numpy()[um] == v0[um]).all())
    )
    out.write_parquet(a.out)
    print("post-rules applied: rows changed", rep, "->", a.out)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sp = ap.add_subparsers(dest="cmd", required=True)
f = sp.add_parser("fit")
f.add_argument("--fold", required=True, help="prefix of the chain's (B)-stage fold vector .npy files")
f.add_argument("--x4", required=True, help="X4_fold.parquet with fold rows and weights")
f.add_argument("--out", required=True, help="output params.json path")
g = sp.add_parser("apply")
g.add_argument("--in", required=True, help="submission-like input parquet")
g.add_argument("--out", required=True, help="output parquet path")
g.add_argument("--params", required=True, help="params.json written by the fit subcommand")
g.add_argument("--only", default="", help="restrict to one rule: xsx or xpk (default: apply both)")
a = ap.parse_args()
{"fit": fit, "apply": apply}[a.cmd](a)
