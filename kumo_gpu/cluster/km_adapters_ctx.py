#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Engineered contexts for km_engine.py -- km_engine.py is NOT edited. Like km_adapters_xlit.py this module imports the engine
unchanged and patches context/fit/predict ONLY when a knob is set in the env; with every knob unset nothing is patched, no spec key is added, and a run is
bit-identical to `python km_engine.py` (same functions, same seeds, same column views). Every arm keeps the engine's FIXED budget CAP rows per fit, the same
query rows, the same instances (column subsets columns(i, fold), generator seeds) as KME8c100 / KNN8c100, so only the CHOICE of context rows changes.
Fold honesty (all arms): every context row comes from the engine's clean TRAINING pool of the fold (KE.split: held months excluded, rank rows excluded,
clean, finite y); selection reads pool FEATURES and pool LABELS (training months) only; the query fold's labels are never read. Arm (d) reads the nested
fold-honest L table data/resid/kumo_resid_adapter_lnest_<fold>.npy.
  (a) CTX_MIX=knn:0.4,temporal:0.25,recent:0.15,tail:0.2   COMPOSITE context per query cluster (CTX_QPC query rows per k-means cluster, as xlit KNN):
      knn      = nearest pool rows in the instance's own standardized column space (xlit's rule);
      temporal = 'temporal analogs': pool rows whose CTX_TKEYS (default hour,n_rwy_in_hour = hour-of-day band +- CTX_HBAND h circular, and the runway
                 configuration proxy = number of runways in use that hour) match the cluster's queries, any month; uniform among the matches;
      recent   = pool rows nearest in calendar months to the held months (engine's month_dist, ties random; prod: recency);
      tail     = target-stratified rows: CTX_TAIL_BINS quantile bins of the pool target y - off, equal allocation per bin (covers the taxi-time tail);
      uniform  = the instance's own uniform slice (explicit share, and the filler for duplicates so every fit has exactly CAP rows).
      Parts are de-duplicated in this priority order; the shortfall is filled from the instance's uniform slice, then from the rest of the pool.
  (b) CTX_PROTO=1 [CTX_PROTO_K=1024]  DIVERSITY / PROTOTYPE selection (cheap TACO/BAPS-style): k-means with K clusters over the pool in the instance's
      standardized column space (GPU), then CAP real rows allocated equally over the clusters (water-filling: dense regions are thinned, sparse regions
      kept) with a seeded random pick inside each cluster -> a de-duplicated context with real labels; query independent (one fit per instance).
  (c) CTX_XAPT=all|LTFM,EHAM,...  [CTX_XAPT_GIB=24 CTX_XAPT_COL=1]  CROSS-AIRPORT POOLED context for data-poor airports: the airport's WHOLE clean pool
      (needs CAP >= pool, i.e. CAP=1000000) + donor rows from the other airports' clean training pools of the SAME fold (same held months excluded),
      sized by the memory formula n_total*(4+C)*512 B <= CTX_XAPT_GIB, split equally over the donors; CTX_XAPT_COL=1 appends one column
      'airport code' (own = 0, donors 1..) to context and query rows so the model can condition on the airport. Control: the own-airport whole pool
      (KSBWPc100, same views for instances 0..NINST-1). KUMO_CTX_VARIANT variants: CTX_XAPT_MASK=<names or prefix*,...> blanks those columns (NaN)
      in the DONOR rows only (c-mask: other airports' rows without their NM / own-clock columns); CTX_XAPT_RAW=1 fits the plain taxi-out target y
      instead of y - off for every context row and subtracts the query's own offset afterwards, so the member output stays taxi = off + g (c-raw);
      CTX_XAPT_LOAO=1 drops the own pool: the airport is predicted from the other airports only (leave-one-airport-out).
  (d) CTX_HARD=<share> [CTX_HARD_POW=1 RESID_L=kumo_resid_adapter_lnest]  HARD-EXAMPLE rows: share*CAP pool rows sampled without replacement with probability
      proportional to |y - L_nest|^POW (the base L model's fold-honest nested OOF error on the context rows), the rest = the instance's uniform slice.
Every active knob is recorded in spec.ctx (meta of every unit); nothing is written when no knob is set. ctx_rows in the instance npz stay the own-pool rows.
usage: python km_adapters_ctx.py <unit> ... | --from-list list.txt [--claim]     (units as km_engine: inst:<fold>:<APT>:<i>  agg:<fold>:<APT>)
       python km_adapters_ctx.py --selftest        numpy only (torch optional): shares, water-filling, tail/hard/temporal selection, exact budget, memory formula
"""

import os, sys, math, json, time, numpy as np

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
CTX_MIX = E("CTX_MIX", "")
CTX_QPC = int(E("CTX_QPC", "1024"))
CTX_ITERS = int(E("CTX_ITERS", "15"))
CTX_TKEYS = E("CTX_TKEYS", "hour,n_rwy_in_hour")
CTX_HBAND = int(E("CTX_HBAND", "1"))
CTX_TAIL_BINS = int(E("CTX_TAIL_BINS", "20"))
CTX_PROTO = int(E("CTX_PROTO", "0"))
CTX_PROTO_K = int(E("CTX_PROTO_K", "1024"))
CTX_HARD = float(E("CTX_HARD", "0") or 0)
CTX_HARD_POW = float(E("CTX_HARD_POW", "1"))
RESID_L = E("RESID_L", "kumo_resid_adapter_lnest")
RESID_DIR = E("RESID_DIR", os.path.join(HERE, "..", "data", "resid"))
CTX_XAPT = E("CTX_XAPT", "")
CTX_XAPT_GIB = float(E("CTX_XAPT_GIB", "24"))
CTX_XAPT_COL = int(E("CTX_XAPT_COL", "1"))
CTX_XAPT_MASK = E("CTX_XAPT_MASK", "")
CTX_XAPT_RAW = int(E("CTX_XAPT_RAW", "0"))
CTX_XAPT_LOAO = int(E("CTX_XAPT_LOAO", "0"))  # KUMO_CTX_VARIANT variants of (c)
ALL_APTS = "LTFM EHAM EGLL LFPG EDDF LEMD LEBL EDDM LIRF LSZH".split()
PARTS = ("knn", "temporal", "recent", "tail", "uniform")
CUR = {"fold": None, "apt": None, "inst": None}
CX = {}


# ---------------- pure helpers (numpy; torch only where noted) -- shared by the live patches and the self test
def parse_mix(s):
    """'knn:0.4,temporal:0.25,...' -> dict part -> share (shares sum to 1 within 1e-6; parts from PARTS)."""
    mix = {}
    for kv in s.split(","):
        k, v = kv.split(":")
        assert k in PARTS, f"CTX_MIX part {k} not in {PARTS}"
        mix[k] = float(v)
    assert abs(sum(mix.values()) - 1) < 1e-6, f"CTX_MIX shares sum to {sum(mix.values())}, not 1"
    return mix


def counts(cap, mix):
    """Integer row counts per part summing exactly to cap (largest remainder)."""
    raw = {k: cap * v for k, v in mix.items()}
    c = {k: int(math.floor(x)) for k, x in raw.items()}
    rest = cap - sum(c.values())
    for k in sorted(raw, key=lambda k: raw[k] - c[k], reverse=True)[:rest]:
        c[k] += 1
    return c


def water_fill(sizes, n):
    """Equal allocation of n over groups capped by their sizes (water-filling) -> int array; sum == min(n, sum(sizes))."""
    sizes = np.asarray(sizes, dtype=int)
    a = np.zeros(len(sizes), dtype=int)
    n = min(int(n), int(sizes.sum()))
    free = np.ones(len(sizes), bool)
    while n > 0 and free.any():
        q = n // free.sum()
        q = max(q, 1)
        take = np.minimum(np.where(free, q, 0), sizes - a)
        if take.sum() > n:  # the last round: spread the remainder one by one over the free groups
            idx = np.where(take > 0)[0][:n]
            take = np.zeros_like(take)
            take[idx] = 1
        a += take
        n -= int(take.sum())
        free &= a < sizes
    return a


def grouped_pick(lab, n, rng):
    """Indices of n rows spread equally over the groups lab (water-filled), random inside each group."""
    u, inv = np.unique(lab, return_inverse=True)
    sizes = np.bincount(inv)
    alloc = water_fill(sizes, n)
    out = []
    for g in range(len(u)):
        if alloc[g] > 0:
            idx = np.where(inv == g)[0]
            out.append(rng.choice(idx, alloc[g], replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def tail_strata(r, bins):
    """Quantile-bin label of each finite pool target r (NaN -> -1, never chosen)."""
    f = np.isfinite(r)
    lab = np.full(len(r), -1)
    q = np.quantile(r[f], np.linspace(0, 1, bins + 1)[1:-1])
    lab[f] = np.searchsorted(q, r[f], side="right")
    return lab


def hard_sample(absr, n, pw, rng):
    """n indices sampled WITHOUT replacement with p ~ absr**pw (Gumbel top-k); rows with absr NaN or 0 are never chosen."""
    w = np.where(np.isfinite(absr) & (absr > 0), np.abs(absr) ** pw, 0.0)
    ok = w > 0
    n = min(int(n), int(ok.sum()))
    key = np.full(len(w), -np.inf)
    key[ok] = np.log(w[ok]) + rng.gumbel(size=int(ok.sum()))
    return np.argsort(-key, kind="stable")[:n]


def temporal_mask(Kp, Kq, hband):
    """Pool rows matching the queries' temporal keys: column 0 = hour (circular distance <= hband to any query hour), other columns = value in the
    query set; NaN never matches."""
    hq = np.unique(Kq[:, 0][np.isfinite(Kq[:, 0])]).astype(int) % 24
    hp = Kp[:, 0]
    ok = np.isfinite(hp)
    m = np.zeros(len(Kp), bool)
    if len(hq):
        d = np.abs((hp[ok].astype(int) % 24)[:, None] - hq[None, :])
        d = np.minimum(d, 24 - d)
        m[ok] = (d <= hband).any(1)
    for c in range(1, Kp.shape[1]):
        vq = np.unique(Kq[:, c][np.isfinite(Kq[:, c])])
        pc = Kp[:, c]
        m &= np.isfinite(pc) & np.isin(pc, vq)
    return m


def fill_budget(parts, cap, fallback, n_pool):
    """parts = [(indices in priority order, n wanted), ...] -> exactly cap distinct indices (< n_pool): each part contributes its first n NEW rows, the
    shortfall comes from fallback (in order), then from the remaining pool rows. Returns (ctx, taken per part, n_fallback).
    """
    seen = np.zeros(n_pool, bool)
    out = []
    taken = []
    for idx, n in parts:
        idx = np.asarray(idx, dtype=int)
        new = idx[~seen[idx]]
        _, first = np.unique(new, return_index=True)
        new = new[np.sort(first)][: int(n)]
        seen[new] = True
        out.append(new)
        taken.append(len(new))
    need = cap - sum(taken)
    nf = 0
    if need > 0:
        fb = np.asarray(fallback, dtype=int)
        fb = fb[~seen[fb]]
        _, first = np.unique(fb, return_index=True)
        fb = fb[np.sort(first)][:need]
        seen[fb] = True
        out.append(fb)
        nf = len(fb)
        need -= nf
    if need > 0:
        rest = np.where(~seen)[0][:need]
        out.append(rest)
        nf += len(rest)
    ctx = np.concatenate(out).astype(int)
    assert len(ctx) == min(cap, n_pool) and len(np.unique(ctx)) == len(ctx), (len(ctx), cap, n_pool)
    return ctx, taken, nf


def xapt_budget(gib, ncol, n_own):
    """Donor rows that fit the memory formula n*(4+C)*512 B <= gib GiB next to the own pool (C = instance columns incl. the airport code column)."""
    return max(0, int(gib * 2**30 // ((4 + ncol) * 512)) - int(n_own))


def robust_scale(P):
    med = np.nanmedian(P, 0)
    q75, q25 = np.nanpercentile(P, [75, 25], 0)
    s = (q75 - q25) / 1.349
    sd = np.nanstd(P, 0)
    s = np.where(s > 0, s, sd)
    keep = np.isfinite(s) & (s > 0) & np.isfinite(med)
    return med, s, keep


def zkeys(X, med, s, keep, clip=5.0):
    Z = (X[:, keep] - med[keep]) / s[keep]
    Z = np.nan_to_num(Z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(Z, -clip, clip).astype(np.float32)


def kmeans(Z, nc, seed, iters):
    """Seeded Lloyd k-means (torch, any device), farthest-point init -> labels [n]."""
    import torch

    n = Z.shape[0]
    nc = max(1, min(nc, n))
    g = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=g))
    C = Z[first : first + 1].clone()
    dmin = torch.cdist(Z, C).squeeze(1)
    for _ in range(nc - 1):
        j = int(dmin.argmax())
        C = torch.cat([C, Z[j : j + 1]], 0)
        dmin = torch.minimum(dmin, torch.cdist(Z, Z[j : j + 1]).squeeze(1))
    lab = torch.zeros(n, dtype=torch.long, device=Z.device)
    for _ in range(iters):
        lab = torch.cdist(Z, C).argmin(1)
        C = torch.zeros_like(C).index_add_(0, lab, Z)
        cnt = torch.bincount(lab, minlength=nc).clamp(min=1).unsqueeze(1).to(C.dtype)
        C = C / cnt
    return lab


def nearest(Zp, p2, Zq, n_take, blk=32768):
    import torch

    q2 = (Zq**2).sum(1)
    dmin = torch.empty(Zp.shape[0], device=Zp.device, dtype=torch.float32)
    for a in range(0, Zp.shape[0], blk):
        b = min(a + blk, Zp.shape[0])
        d = p2[a:b, None] - 2.0 * (Zp[a:b] @ Zq.T) + q2[None, :]
        dmin[a:b] = d.min(1).values
    n_take = min(int(n_take), Zp.shape[0])
    return torch.topk(dmin, n_take, largest=False).indices, dmin


def resid_table(fold):
    """[ids_sorted, L_sorted] of data/resid/<RESID_L>_<fold>.npy (KUMO_RESID_ADAPTER nested L; = the fold-honest offset table)."""
    p = f"{RESID_DIR}/{RESID_L}_{fold}.npy"
    assert os.path.exists(p), f"CTX_HARD needs the nested L table {p}"
    a = np.load(p)
    o = np.argsort(a[0])
    return a[0][o], a[1][o].astype(float)


def lookup(tab, ids):
    i, v = tab
    k = np.clip(np.searchsorted(i, ids), 0, len(i) - 1)
    hit = i[k] == ids
    out = np.full(len(ids), np.nan)
    out[hit] = v[k[hit]]
    return out


def mask_cols(names, spec):
    """Positions in names matched by spec = comma list of exact names or prefixes ending in '*'."""
    pats = [p for p in spec.split(",") if p]
    return np.array(
        [j for j, n in enumerate(names) if any(n == p or (p.endswith("*") and n.startswith(p[:-1])) for p in pats)],
        dtype=int,
    )


# ---------------- self test (numpy only; the torch parts run when torch is importable)
def selftest():
    rng = np.random.default_rng(0)
    mix = parse_mix("knn:0.4,temporal:0.25,recent:0.15,tail:0.2")
    c = counts(20000, mix)
    assert sum(c.values()) == 20000 and c == dict(knn=8000, temporal=5000, recent=3000, tail=4000), c
    c = counts(60001, parse_mix("knn:0.5,uniform:0.5"))
    assert sum(c.values()) == 60001, c
    a = water_fill([5, 100, 100, 1], 50)
    assert a.sum() == 50 and a[0] == 5 and a[3] == 1 and abs(a[1] - a[2]) <= 1, a
    a = water_fill([3, 3], 100)
    assert a.tolist() == [3, 3]
    lab = rng.integers(0, 50, 20000)
    lab[:15000] = 0  # one dense cluster = 75 % of the rows
    pick = grouped_pick(lab, 2000, rng)
    assert len(pick) == 2000 and len(np.unique(pick)) == 2000 and (lab[pick] == 0).mean() < 0.1, (lab[pick] == 0).mean()
    r = rng.normal(size=10000) ** 3
    r[:20] = np.nan
    st = tail_strata(r, 20)
    assert st.min() == -1 and st.max() == 19 and (st[:20] == -1).all()
    tp = grouped_pick(st[st >= 0], 2000, rng)
    tp = np.where(st >= 0)[0][tp]
    q = np.quantile(r[np.isfinite(r)], [0.05, 0.95])
    share = ((r[tp] < q[0]) | (r[tp] > q[1])).mean()
    assert (
        0.08 < share < 0.12
    ), share  # equal bins: the 10 % tails get 10 % (a uniform draw would too, but the far tail bins are guaranteed)
    absr = np.abs(rng.normal(size=10000))
    absr[:10] = np.nan
    absr[10:20] = 0
    h = hard_sample(absr, 1000, 1.0, rng)
    assert (
        len(h) == 1000 and len(np.unique(h)) == 1000 and h.min() >= 20 and absr[h].mean() > 1.5 * np.nanmean(absr[20:])
    ), absr[h].mean()
    Kp = np.column_stack([rng.integers(0, 24, 5000), rng.integers(1, 4, 5000)]).astype(float)
    Kp[:3] = np.nan
    Kq = np.array([[6, 2.0], [7, 2.0]])
    m = temporal_mask(Kp, Kq, 1)
    assert (
        m.sum() > 0
        and not m[:3].any()
        and set(np.unique(Kp[m, 0]).astype(int)) <= {5, 6, 7, 8}
        and (Kp[m, 1] == 2).all()
    )
    m23 = temporal_mask(Kp, np.array([[23, 1.0]]), 1)
    assert set(np.unique(Kp[m23, 0]).astype(int)) <= {22, 23, 0}  # circular hour band
    parts = [(np.array([5, 5, 7, 9, 11]), 3), (np.array([7, 9, 13, 15]), 2), (np.arange(100, 110), 4)]
    ctx, taken, nf = fill_budget(parts, 12, np.arange(20, 30), 1000)
    assert (
        taken == [3, 2, 4]
        and nf == 3
        and ctx[:3].tolist() == [5, 7, 9]
        and ctx[3:5].tolist() == [13, 15]
        and set(ctx[9:]) == {20, 21, 22}
    ), (ctx, taken, nf)
    ctx2, _, nf2 = fill_budget([(np.arange(50), 50)], 60, np.arange(50), 55)
    assert len(ctx2) == 55 and nf2 == 5  # pool smaller than cap: the whole pool
    assert xapt_budget(24, 101, 105930) == int(24 * 2**30 // (105 * 512)) - 105930 and xapt_budget(1, 100, 10**6) == 0
    n_fit = 105930 + xapt_budget(24, 101, 105930)
    assert n_fit * 105 * 512 <= 24 * 2**30 < (n_fit + 1) * 105 * 512
    tab = (np.array([1.0, 3.0, 5.0]), np.array([10.0, 30.0, 50.0]))
    assert np.array_equal(np.nan_to_num(lookup(tab, np.array([3.0, 4.0, 5.0])), nan=-1), [30, -1, 50])
    assert (
        mask_cols(["mvt_aobt3", "hour", "mvt_lobt", "aobt3_sched"], "mvt_*,aobt3_sched").tolist() == [0, 2, 3]
        and len(mask_cols(["hour"], "")) == 0
    )
    msg = f"SELFTEST OK: mix/counts | water_fill | grouped_pick (dense cluster share {(lab[pick] == 0).mean():.2f}) | tail strata | hard_sample (mean |r| {absr[h].mean():.2f} vs {np.nanmean(absr[20:]):.2f}) | temporal mask | fill_budget exact | xapt formula ({n_fit:,} rows at 24 GiB x 101 cols)"
    try:
        import torch

        P = rng.normal(size=(3000, 8))
        med, s, keep = robust_scale(P)
        Zp = torch.as_tensor(zkeys(P, med, s, keep))
        p2 = (Zp**2).sum(1)
        idx, _ = nearest(Zp, p2, torch.as_tensor(zkeys(P[40:50] + 1e-3, med, s, keep)), 10, blk=500)
        assert set(idx.tolist()) == set(range(40, 50))
        lab = kmeans(torch.as_tensor(zkeys(np.vstack([P[:60] + 40, P[60:120] - 40]), med, s, keep)), 2, 1, 10).numpy()
        assert len(np.unique(lab[:60])) == 1 and lab[0] != lab[60]
        gp = grouped_pick(kmeans(Zp, 64, 0, 5).numpy(), 500, rng)
        assert len(np.unique(gp)) == 500
        msg += " | torch: nearest/kmeans/prototype pick OK"
    except ImportError:
        msg += " | (torch not installed: nearest/kmeans not exercised)"
    print(msg)


if __name__ == "__main__" and "--selftest" in sys.argv:
    selftest()
    sys.exit(0)

# ---------------- live mode: wrap the engine (torch + sdm are imported here, not above)
sys.path.insert(0, HERE)
import torch
import km_engine as KE

K = KE.K
DEV = KE.DEV
_orig_fit, _orig_predict, _orig_context = KE.fit, KE.predict, KE.context
assert (
    sum(bool(x) for x in (CTX_MIX, CTX_PROTO, CTX_XAPT, CTX_HARD)) <= 1
), "KUMO_CONTEXT_ADAPTER: set ONE of CTX_MIX / CTX_PROTO / CTX_XAPT / CTX_HARD per run"
assert not (
    any((CTX_MIX, CTX_PROTO, CTX_XAPT, CTX_HARD)) and (int(E("KNN", "0") or 0) or KE.S["noise"] > 0)
), "KUMO_CONTEXT_ADAPTER arms do not compose with xlit KNN=1 or NOISE>0"


def set_cur(spec):
    p = spec.split(":")
    CUR["fold"] = p[1] if len(p) > 1 else None
    CUR["apt"] = p[2] if len(p) > 2 else None
    CUR["inst"] = int(p[3]) if p[0] == "inst" and len(p) > 3 else None


def feat_idx(names):
    idx = {c: j for j, c in enumerate(K.FEATS)}
    miss = [n for n in names if n not in idx]
    assert not miss, f"KUMO_CONTEXT_ADAPTER keys not in the 412: {miss}"
    return np.array([idx[n] for n in names])


def rng_of(i, *extra):
    return np.random.default_rng([KE.S["seed"], 243, int(i), *[int(x) for x in extra]])


def locate_shard(F, qi, cols, xq):
    """Position of this query shard inside qi (the engine shards qi in order; resumed parts are skipped, so match the first and last row)."""
    X = F["X"]
    n = len(xq)
    for p0 in range(0, len(qi), KE.S["shard"]):
        if (
            p0 + n <= len(qi)
            and np.array_equal(X[qi[p0]][cols], xq[0], equal_nan=True)
            and np.array_equal(X[qi[p0 + n - 1]][cols], xq[-1], equal_nan=True)
        ):
            return p0
    raise AssertionError("KUMO_CONTEXT_ADAPTER: query shard not found in the fold's query rows")


# --- (b) prototypes and (d) hard examples: query independent -> only KE.context changes (one fit per instance, engine's fit/predict untouched)
if CTX_PROTO or CTX_HARD:
    _resid = {}

    def context_ctx(F, ti, i, fold):
        cap = KE.S["cap"]
        i = int(i)
        if len(ti) <= cap:
            return ti
        uni = np.searchsorted(ti, _orig_context(F, ti, i, fold))
        rng = rng_of(i)
        t0 = time.time()
        if CTX_PROTO:
            cols, _ = KE.columns(i, fold)
            P = F["X"][ti][:, cols]
            med, s, keep = robust_scale(P)
            Z = torch.as_tensor(zkeys(P, med, s, keep), device=DEV)
            nc = min(CTX_PROTO_K, max(2, len(ti) // 8))
            lab = kmeans(Z, nc, seed=KE.S["seed"] * 1000 + i, iters=CTX_ITERS).cpu().numpy()
            del Z
            sel = grouped_pick(lab, cap, rng)
            ctx, taken, nf = fill_budget([(sel, cap)], cap, uni, len(ti))
            sizes = np.bincount(lab)
            print(
                f"  ctx PROTO inst {i}: pool {len(ti):,} -> {nc} clusters (sizes med {int(np.median(sizes))}, max {sizes.max()}), {taken[0]:,} prototype rows + {nf} filler of CAP {cap:,} | {time.time() - t0:.1f}s",
                flush=True,
            )
        else:
            if fold not in _resid:
                _resid.clear()
                _resid[fold] = resid_table(fold)
            L = lookup(_resid[fold], F["ids"][ti])
            r = F["y"][ti] - L
            n_h = int(round(CTX_HARD * cap))
            h = hard_sample(np.abs(r), n_h, CTX_HARD_POW, rng)
            ctx, taken, nf = fill_budget([(h, n_h), (uni, cap - n_h)], cap, uni, len(ti))
            print(
                f"  ctx HARD inst {i}: pool {len(ti):,} (L missing {int((~np.isfinite(L)).sum())}) -> {taken[0]:,} hard rows (mean |y-L| {np.abs(r[h]).mean():.0f} s vs pool {np.nanmean(np.abs(r)):.0f} s) + {taken[1]:,} uniform + {nf} filler of CAP {cap:,}",
                flush=True,
            )
        return ti[ctx]

    KE.context = context_ctx
    CX.update(
        dict(proto=dict(k=CTX_PROTO_K, iters=CTX_ITERS))
        if CTX_PROTO
        else dict(hard=dict(share=CTX_HARD, pow=CTX_HARD_POW, table=RESID_L))
    )

# --- (a) composite contexts per query cluster: context = whole pool handed to fit, the real contexts are built inside predict (xlit KNN pattern)
if CTX_MIX:
    MIX = parse_mix(CTX_MIX)

    class Composite:
        def __init__(self, F, fold, i, xc, yc):
            self.F, self.fold, self.i, self.xc, self.yc = F, fold, int(i), xc, yc
            self.ti, self.qi = KE.split(F, fold)
            cap = KE.S["cap"]
            self.n = counts(cap, MIX)
            self.cols, _ = KE.columns(self.i, fold)
            assert xc.shape[1] == len(self.cols) and len(xc) == len(
                self.ti
            ), "KUMO_CONTEXT_ADAPTER MIX: fit received something other than the pool"
            self.uni = np.searchsorted(self.ti, _orig_context(F, self.ti, self.i, fold))
            rng = rng_of(self.i)
            if self.n.get("knn"):
                self.med, self.s, self.keep = robust_scale(xc)
                self.Zp = torch.as_tensor(zkeys(xc, self.med, self.s, self.keep), device=DEV)
                self.p2 = (self.Zp**2).sum(1)
            if self.n.get("temporal"):
                tk = feat_idx(CTX_TKEYS.split(","))
                self.TKp = F["X"][self.ti][:, tk]
                self.TKq = F["X"][self.qi][:, tk]
            if self.n.get("recent"):
                key = KE.month_dist(F, self.ti, fold) + rng.random(len(self.ti))
                self.recent = np.argsort(key, kind="stable")[: 2 * self.n["recent"]]
            if self.n.get("tail"):
                st = tail_strata(yc, CTX_TAIL_BINS)
                ok = np.where(st >= 0)[0]
                self.tail = ok[grouped_pick(st[ok], min(2 * self.n["tail"], len(ok)), rng)]
            self.n_fit = 0
            self.stat = {k: 0 for k in PARTS}
            self.stat["filler"] = 0
            self.dloc = []

        def clear(self):
            self.Zp = self.p2 = None

    def context_mix(F, ti, i, fold):
        CUR["fold"] = fold
        CUR["inst"] = int(i)
        return ti

    def fit_mix(xc, yc, i):
        fold = CUR["fold"]
        assert fold, "KUMO_CONTEXT_ADAPTER MIX: fold unknown"
        sel = Composite(K.frame(CUR["apt"]), fold, i, xc, yc)
        print(
            f"  ctx MIX inst {i}: pool {len(sel.ti):,} rows, budget {sel.n} of CAP {KE.S['cap']:,}, keys {CTX_TKEYS} band +-{CTX_HBAND}h, {CTX_QPC} queries/cluster",
            flush=True,
        )
        return sel

    def predict_mix(sel, xq, ch0=None):
        cap = KE.S["cap"]
        n = len(xq)
        p0 = locate_shard(sel.F, sel.qi, sel.cols, xq)
        ch = ch0
        if len(sel.ti) <= cap:  # pool fits the budget: nothing to choose (engine semantics)
            m = _orig_fit(sel.xc, sel.yc, sel.i)
            g, qs, ch = _orig_predict(m, xq, ch)
            m.clear()
            return g, qs, ch
        Zq = (
            torch.as_tensor(zkeys(xq, sel.med, sel.s, sel.keep), device=DEV)
            if sel.n.get("knn")
            else torch.as_tensor(np.nan_to_num(xq[:, :8]), device=DEV)
        )
        nc = max(1, int(round(n / CTX_QPC)))
        lab = kmeans(Zq, nc, seed=KE.S["seed"] * 1000 + sel.i * 7919 + p0, iters=CTX_ITERS).cpu().numpy()
        g = np.empty(n)
        qs = np.empty((n, len(K.QLEV)), np.float32)
        for c in np.unique(lab):
            q = np.where(lab == c)[0]
            rng = rng_of(sel.i, p0, c)
            parts = []
            for part in PARTS:
                k = sel.n.get(part, 0)
                if k <= 0:
                    continue
                if part == "knn":
                    loc, dmin = nearest(sel.Zp, sel.p2, Zq[torch.as_tensor(q, device=DEV)], k)
                    loc = loc.cpu().numpy()
                    sel.dloc.append(float(dmin[torch.as_tensor(loc, device=DEV)].clamp(min=0).sqrt().mean()))
                    parts.append((loc, k))
                elif part == "temporal":
                    cand = np.where(temporal_mask(sel.TKp, sel.TKq[p0 + q], CTX_HBAND))[0]
                    parts.append((rng.permutation(cand)[: 2 * k], k))
                elif part == "recent":
                    parts.append((sel.recent, k))
                elif part == "tail":
                    parts.append((sel.tail, k))
                else:
                    parts.append((sel.uni, k))
            ctx, taken, nf = fill_budget(parts, cap, sel.uni, len(sel.ti))
            for part, t in zip([p for p in PARTS if sel.n.get(p, 0) > 0], taken):
                sel.stat[part] += t
            sel.stat["filler"] += nf
            m = _orig_fit(sel.xc[ctx], sel.yc[ctx], sel.i)
            gg, qq, ch = _orig_predict(m, xq[q], ch)
            m.clear()
            g[q] = gg
            qs[q] = qq
            sel.n_fit += 1
        tot = max(1, sel.n_fit * cap)
        comp = " ".join(f"{k} {v / tot:.2f}" for k, v in sel.stat.items() if v)
        print(
            f"    ctx MIX shard @{p0}: {n:,} queries -> {len(np.unique(lab))} clusters, {sel.n_fit} fits so far | realised shares {comp} | knn key dist {np.mean(sel.dloc[-len(np.unique(lab)):]) if sel.dloc else float('nan'):.2f}",
            flush=True,
        )
        return g, qs, ch

    KE.context, KE.fit, KE.predict = context_mix, fit_mix, predict_mix
    CX["mix"] = dict(
        shares=MIX, qpc=CTX_QPC, tkeys=CTX_TKEYS, hband=CTX_HBAND, tail_bins=CTX_TAIL_BINS, iters=CTX_ITERS
    )

# --- (c) cross-airport pooled context: own whole pool (engine) + donor rows appended inside fit; airport code column on context and query rows
if CTX_XAPT:
    _don = {}

    def donors_of(apt):
        return [a for a in (ALL_APTS if CTX_XAPT == "all" else CTX_XAPT.split(",")) if a != apt]

    def donor_pool(apt, fold):
        """Clean training pool of a donor airport for this fold (same held months excluded): X (412 cols, float32), y - off, off; cached per process."""
        key = (apt, fold)
        if key not in _don:
            Fd = K.frame(apt)
            td, _ = KE.split(Fd, fold)
            _don[key] = (
                Fd["X"][td].copy(),
                (Fd["y"][td] - Fd["off"][td]).astype(np.float64),
                Fd["off"][td].astype(np.float64),
            )
            K._frames.clear()
        return _don[key]

    def context_xapt(F, ti, i, fold):
        assert len(ti) <= KE.S["cap"], f"CTX_XAPT needs CAP >= the own pool ({len(ti):,}); use CAP=1000000"
        CUR["fold"] = fold
        CUR["inst"] = int(i)
        return ti

    def fit_xapt(xc, yc, i):
        fold, apt, i = CUR["fold"], CUR["apt"], int(i)
        assert fold and apt
        cols, _ = KE.columns(i, fold)
        assert xc.shape[1] == len(cols)
        F = K.frame(apt)
        ti, _ = KE.split(F, fold)
        dons = donors_of(apt)
        ncol = len(cols) + CTX_XAPT_COL
        n_own = 0 if CTX_XAPT_LOAO else len(xc)
        budget = xapt_budget(CTX_XAPT_GIB, ncol, n_own)
        t0 = time.time()
        mpos = mask_cols([K.FEATS[j] for j in cols], CTX_XAPT_MASK) if CTX_XAPT_MASK else np.array([], dtype=int)
        pools = [donor_pool(a, fold) for a in dons]
        alloc = water_fill([len(p[1]) for p in pools], budget)
        Xs, ys, code = (
            ([], [], [])
            if CTX_XAPT_LOAO
            else ([xc], [yc + F["off"][ti] if CTX_XAPT_RAW else yc], [np.zeros(len(xc), np.float32)])
        )
        for j, (a, (Xd, yd, od)) in enumerate(zip(dons, pools)):
            if alloc[j] <= 0:
                continue
            pick = rng_of(i, j).choice(len(yd), alloc[j], replace=False)
            Xj = Xd[pick][:, cols]
            if len(mpos):
                Xj = Xj.copy()
                Xj[:, mpos] = np.nan
            Xs.append(Xj)
            ys.append(yd[pick] + od[pick] if CTX_XAPT_RAW else yd[pick])
            code.append(np.full(alloc[j], j + 1, np.float32))
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        assert len(X) > 0, "CTX_XAPT: empty context (budget too small for LOAO?)"
        if CTX_XAPT_COL:
            X = np.hstack([X, np.concatenate(code)[:, None]])
        est = len(X) * (4 + ncol) * 512 / 2**30
        print(
            f"  ctx APT_MEMBER inst {i} {apt}: own {n_own:,}{' (LOAO: excluded)' if CTX_XAPT_LOAO else ''} + donors {int(alloc.sum()):,} from {len(dons)} airports ({', '.join(f'{a} {n}' for a, n in zip(dons, alloc))}) = {len(X):,} rows x {X.shape[1]} cols"
            f", masked donor cols {len(mpos)}, target {'raw y' if CTX_XAPT_RAW else 'y - off'}, formula {est:.1f} GiB (cap {CTX_XAPT_GIB}) | prep {time.time() - t0:.0f}s",
            flush=True,
        )
        K.frame(apt)  # restore the frame cache for the engine (agg / next unit)
        return _orig_fit(X, y, i)

    def predict_xapt(mdl, xq, ch0=None):
        xin = np.hstack([xq, np.zeros((len(xq), 1), np.float32)]) if CTX_XAPT_COL else xq
        g, qs, ch = _orig_predict(mdl, xin, ch0)
        if (
            CTX_XAPT_RAW
        ):  # the model predicted raw taxi-out; the engine adds off afterwards, so hand back g - off(query)
            F = K.frame(CUR["apt"])
            _, qi = KE.split(F, CUR["fold"])
            p0 = locate_shard(F, qi, KE.columns(CUR["inst"], CUR["fold"])[0], xq)
            off = F["off"][qi[p0 : p0 + len(xq)]]
            g = g - off
            qs = (qs - off[:, None]).astype(np.float32)
        return g, qs, ch

    KE.context, KE.fit, KE.predict = context_xapt, fit_xapt, predict_xapt
    CX["xapt"] = dict(
        donors=CTX_XAPT,
        gib=CTX_XAPT_GIB,
        code_col=CTX_XAPT_COL,
        mask=CTX_XAPT_MASK,
        raw=CTX_XAPT_RAW,
        loao=CTX_XAPT_LOAO,
    )

if CX:
    KE.S["ctx"] = CX
    print(f"km_adapters_ctx active: {json.dumps(CX)[:400]}", flush=True)

if __name__ == "__main__":
    import socket

    a = sys.argv[1:]
    claim = "--claim" in a
    a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list":
        a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(f"km_adapters_ctx: knobs {json.dumps(CX) if CX else 'none (native engine)'} | engine spec {KE.S}", flush=True)
    t0 = time.time()
    n = 0
    for spec in a:
        set_cur(spec)
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
    print(f"KM_CTX_DONE units run {n} | {time.time() - t0:.0f}s", flush=True)
