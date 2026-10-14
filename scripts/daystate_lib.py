# SPDX-License-Identifier: GPL-3.0-only
"""XL helpers: frame load, per-fold gate (as nbrsmooth_lib/gate_composition), linear LOFO screen, neighbour means, S1 features."""
import os, sys, numpy as np, polars as pl
NRANK = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
XL = os.environ.get("daystate_dir", os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + "/scratchpad/xl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_composition as GC
FOLDS = ["months39", "months28", "months410", "janjul"]

def load(cols=None):
    return pl.read_parquet(f"{XL}/frame.parquet", columns=cols)

_RC = None
def rank_counts():
    global _RC
    if _RC is None:
        Rk = load(["is_rank", "is_dep", "ADEP_mvt", "unmatched", "mvt_lobt", "mvt_aobt3", "mvt_eobt1", "mvt_sched", "adsb_linked"]).filter(
            (pl.col("is_rank") == 1) & (pl.col("is_dep") == 1))
        assert Rk.height == NRANK, Rk.height
        ra, rl, ro = GC.classify(Rk)
        rk = np.char.add(np.char.add(np.char.add(np.char.add(ra, "|"), rl), "|"), ro)
        k_, c_ = np.unique(rk, return_counts=True); _RC = dict(zip(k_.tolist(), c_.tolist()))
    return _RC

def gate(F, cand, base="inc", label="", freeze=True, verbose=True):
    rc = rank_counts(); out = []
    for fold in FOLDS:
        D = F.filter((pl.col("fold") == fold) & pl.col("y").is_not_null() & pl.col(cand).is_not_null())
        y = D["y"].to_numpy(); b = D[base].to_numpy(); c = D[cand].to_numpy()
        apt, lk, obs = GC.classify(D)
        d = (c - y) ** 2 - (b - y) ** 2
        if freeze: d = np.where((obs == "unm_LIRF") | (obs == "exposed"), 0.0, d)
        n_all = F.filter(pl.col("fold") == fold).height
        w, _, _ = GC.weights(apt, lk, obs, rc)
        top = np.argsort(-np.abs(d))[:3]; keep = np.ones(len(d), bool); keep[top] = False
        raw = d.sum() / n_all; raw3 = d[keep].sum() / n_all
        comp = (w * d).sum() / w.sum(); comp3 = (w * d)[keep].sum() / w[keep].sum()
        out.append((fold, raw, raw3, comp, comp3, int((d != 0).sum())))
    if verbose:
        print(f"[{label}] " + " | ".join(f"{f} {r:+7.1f} ({r3:+7.1f}) c{c:+7.1f}" for f, r, r3, c, c3, n in out)
              + f" | mean {np.mean([o[1] for o in out]):+7.1f} wins {sum(o[1] < 0 for o in out)}/4", flush=True)
    return out

def cf_expr(c):
    return pl.when(pl.col("mvt_lobt").is_null()).then(pl.col(c)).otherwise(
        pl.min_horizontal(pl.max_horizontal(pl.col(c), pl.col("mvt_lobt") - 3606), pl.col("mvt_lobt") + 3606)).clip(60, None)

def lin(F, feats, label, base="inc", CL=1800.0, per_apt=True, verbose=True, ret=False, extra_mask=None):
    """cand = base + sum_k beta_{apt,k} x_k (NaN->0) on matched rows; beta LOFO least squares on e = clip(y-base)."""
    F = F.with_columns(_e=(pl.col("y") - pl.col(base)).clip(-CL, CL))
    outs = []; betas = {}
    apts = F["ADEP_mvt"].unique().sort().to_list() if per_apt else [None]
    for fold in FOLDS:
        for apt in apts:
            sel = (pl.col("ADEP_mvt") == apt) if apt else pl.lit(True)
            Tr = F.filter(sel & (pl.col("fold") != fold) & (pl.col("unmatched") == 0) & pl.col("_e").is_not_null())
            Te = F.filter(sel & (pl.col("fold") == fold))
            X = Tr.select(feats).fill_nan(None).fill_null(0).to_numpy().astype(float)
            b = np.linalg.solve(X.T @ X + 1e-6 * np.eye(len(feats)), X.T @ Tr["_e"].to_numpy())
            betas[(fold, apt)] = b
            g = Te.select(feats).fill_nan(None).fill_null(0).to_numpy().astype(float) @ b
            g = np.where(Te["unmatched"].to_numpy() == 0, g, 0.)
            outs.append(Te.select("MVT_ID_mvt").with_columns(_g=pl.Series(g)))
    G = pl.concat(outs)
    F2 = F.join(G, on="MVT_ID_mvt", how="left").with_columns(_c=pl.col(base) + pl.col("_g").fill_null(0))
    F2 = F2.with_columns(_c=cf_expr("_c"))
    r = gate(F2, "_c", base=base, label=label, verbose=verbose)
    return (r, F2.drop("_e"), betas) if ret else r

def nbr_mean(t, v, lo, hi, excl=True):
    """t sorted. mean of v over j with t_j in [t_i+lo, t_i+hi], excluding i. NaN v ignored."""
    ok = ~np.isnan(v); vv = np.where(ok, v, 0.)
    cs = np.concatenate([[0.], np.cumsum(vv)]); cn = np.concatenate([[0], np.cumsum(ok)])
    a = np.searchsorted(t, t + lo); b = np.searchsorted(t, t + hi, side="right")
    s = cs[b] - cs[a]; n = cn[b] - cn[a]
    if excl: s = s - vv; n = n - ok
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n

def group_nbr(D, keys, tcol, vcol, windows, prefix):
    """neighbour means of vcol by keys over time tcol windows [(name, lo, hi)] -> DataFrame MVT_ID_mvt + cols"""
    res = []
    for _, d in D.sort(tcol).group_by(keys, maintain_order=True):
        t = d[tcol].to_numpy().astype(float); v = d[vcol].to_numpy().astype(float)
        cols = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        for nm, lo, hi in windows:
            cols[prefix + nm], _ = nbr_mean(t, v, lo, hi)
        res.append(pl.DataFrame(cols))
    return pl.concat(res)

def demean(F, cols):
    """centre per airport x calendar month over rows with a value (level-neutral form of S1)."""
    F = F.with_columns(_mo=pl.from_epoch(pl.col("MVT").cast(pl.Int64), time_unit="s").dt.strftime("%Y-%m"))
    F = F.with_columns([pl.col(c).fill_nan(None) for c in cols])
    F = F.with_columns([(pl.col(c) - pl.col(c).mean().over(["ADEP_mvt", "_mo"])).alias(c + "c") for c in cols])
    return F.drop("_mo")

def s1_feats(D):
    """S1 (XZ): PR5/PR15 = same airport x runway neighbour mean of clip(inc - mvt_aobt3, +-1800) by MVT, self excluded."""
    D = D.with_columns(_pr=(pl.col("inc") - pl.col("mvt_aobt3")).clip(-1800, 1800))
    S = group_nbr(D, ["ADEP_mvt", "RUNWAY_mvt"], "MVT", "_pr", [("5", -300, 300), ("15", -900, 900)], "PR")
    return S

def fold_deps():
    return load().filter((pl.col("is_dep") == 1) & pl.col("fold").is_not_null())


# ---------------- XL additions ----------------
def exp_nbr(t, v, tau, w=None, excl=True):
    """t sorted (s). Exponentially weighted mean of v over j != i: weights w_j*exp(-|t_i-t_j|/tau). NaN v ignored.
    Returns (weighted mean, total weight)."""
    n = len(t); ok = ~np.isnan(v)
    if w is None: w = np.ones(n)
    w = np.where(ok, w, 0.); vv = np.where(ok, v, 0.) * w
    Sf = np.zeros(n); Wf = np.zeros(n); Sb = np.zeros(n); Wb = np.zeros(n)
    for i in range(1, n):
        d = np.exp(-(t[i] - t[i - 1]) / tau)
        Sf[i] = d * (Sf[i - 1] + vv[i - 1]); Wf[i] = d * (Wf[i - 1] + w[i - 1])
    for i in range(n - 2, -1, -1):
        d = np.exp(-(t[i + 1] - t[i]) / tau)
        Sb[i] = d * (Sb[i + 1] + vv[i + 1]); Wb[i] = d * (Wb[i + 1] + w[i + 1])
    S = Sf + Sb; W = Wf + Wb
    return np.where(W > 1e-9, S / np.maximum(W, 1e-9), np.nan), W

def group_apply(D, keys, tcol, fn, prefix):
    """fn(d: DataFrame sorted by tcol) -> dict name->array. Returns MVT_ID_mvt + prefixed columns."""
    res = []
    for _, d in D.sort(tcol).group_by(keys, maintain_order=True):
        cols = {"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy()}
        for k, a in fn(d).items(): cols[prefix + k] = a
        res.append(pl.DataFrame(cols))
    return pl.concat(res)

def nbr_wmean(t, v, w, lo, hi, excl=True):
    """window-weighted mean with weights w (NaN v ignored)."""
    ok = ~np.isnan(v); ww = np.where(ok, w, 0.); vv = np.where(ok, v, 0.) * ww
    cs = np.concatenate([[0.], np.cumsum(vv)]); cw = np.concatenate([[0.], np.cumsum(ww)])
    a = np.searchsorted(t, t + lo); b = np.searchsorted(t, t + hi, side="right")
    s = cs[b] - cs[a]; W = cw[b] - cw[a]
    if excl: s = s - vv; W = W - ww
    return np.where(W > 1e-9, s / np.maximum(W, 1e-9), np.nan), W

def nbr_rank(t, v, lo, hi):
    """percentile rank of v_i among neighbours j (window), NaN if < 3 neighbours. O(n*k) python-free via sorting per row is costly;
    use a simple loop over rows with numpy slices (windows are short)."""
    n = len(t); out = np.full(n, np.nan)
    a = np.searchsorted(t, t + lo); b = np.searchsorted(t, t + hi, side="right")
    for i in range(n):
        s = v[a[i]:b[i]]; s = s[~np.isnan(s)]
        m = len(s) - (0 if np.isnan(v[i]) else 1)
        if m >= 3 and not np.isnan(v[i]):
            out[i] = ((s < v[i]).sum() + 0.5 * ((s == v[i]).sum() - 1)) / m
    return out

def cf_expr_col(e):
    """cf on an expression: clip to mvt_lobt +- 3606 where present, then max(60)."""
    return pl.when(pl.col("mvt_lobt").is_null()).then(e).otherwise(
        pl.min_horizontal(pl.max_horizontal(e, pl.col("mvt_lobt") - 3606), pl.col("mvt_lobt") + 3606)).clip(60, None)
