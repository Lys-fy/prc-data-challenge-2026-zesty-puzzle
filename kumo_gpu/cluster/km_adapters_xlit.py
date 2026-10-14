#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Adapters from published Kumo Tabular / TabPFN techniques for km_engine.py -- km_engine.py is NOT edited. This module imports
km_engine unchanged (so cluster/km_adapters.py's LOGN_CLAMP / ROPE_PI still apply) and installs extra patches ONLY when a knob is set in the env;
with every knob unset nothing is patched, no spec key is added, and a run is bit-identical to `python km_engine.py` (same functions, same seeds).
  KNN=1          RETRIEVAL CONTEXTS (LoCalPFN 2406.05207; CRUMB 2606.11473; TabDPT-Turbo 2608.01400 ablation; item 1). The instance no
                 longer fits ONE slice of the pool: each shard of query rows is k-means-clustered (KNN_QPC query rows per cluster, default 1024) and
                 every cluster gets its own context = the (1-KNN_GLOBAL)*CAP pool rows nearest to the cluster's queries (min over the cluster of the
                 standardized Euclidean distance in the KEY space) + KNN_GLOBAL*CAP (0.25) rows of the instance's own uniform slice (KumoRFM-2
                 "local + global"). KEY space = KNN_KEYS: 'cols' (default) = the instance's own column subset (every view retrieves in the space it
                 reads), or a comma list of feature names that must lie inside the instance's columns (force them in with the engine's EXTRACOLS;
                 e.g. the delay-band keys of the pocket view, lever 2). Fold-honest by construction: the pool is the engine's clean
                 training-month pool and retrieval reads features only. Cost: one CAP-row fit per cluster instead of one per instance.
  RR_ROT=1       VENDOR-RECIPE DIVERSITY FIX: sdm's default recipe chooses the numeric transform per ensemble member by round_robin(member_id % 3)
                 (Identity / PowerTransform / RobustScale+ClipSoft). With EIN=1 every engine instance is member 0 => all instances use Identity.
                 RR_ROT rotates the round robin by the instance index, restoring the vendor's per-estimator transform mix across our instances.
  TOPK_ATTN=<k>  TOP-K SPARSE ATTENTION on the ICL query->context path only (that path is the only GQA attention: 2 KV heads for query rows, medium
                 and large), applied when the context has more than TOPK_MIN keys (default k). Exact logits are formed per block of TOPK_QB (128)
                 query rows, the k largest keys are kept, and the dropped attention mass is NOT thrown away: TOPK_COMP=mean (default) gives it to the
                 mean value of the dropped keys (SparQ 2312.04985 / ClusterAttention 2608.26965: in tabular ICL the excluded-mean bias dominates the
                 error of sparse attention); TOPK_COMP=none renormalises over the k keys (the team's raw TOPK_ATTN form). With k >= #keys the
                 result equals dense attention up to rounding. Information arm for the attention-fading question; needs no training.
  LOGN_TAU=<t>   temperature on the LogScale output (q * a_h * log n * t) for attentions with more than LOGN_TAU_MIN (1000) keys (SSMax 2501.19399's
                 scale, swept at inference); composes with km_adapters' LOGN_CLAMP. Row blocks (<= 416 keys) are untouched.
  WEIGHTS=<path> load a fine-tuned state dict into the model (template fields {size} {fold} {apt}; both the KumoTabular wrapper's and the raw
                 regressor.pt key layout are accepted; sha256 recorded) so fine-tuned members run through the engine's views, aggregation and npz
                 (GPU_SCALE_CHECK P5). The original vendor file loaded through WEIGHTS is bit-identical to native (smoke E).
Every active knob is recorded in spec.xlit (meta of every unit); nothing is written when no knob is set.
usage: python km_adapters_xlit.py <unit> ... | --from-list list.txt [--claim]      (units as km_engine: inst:<fold>:<APT>:<i>  agg:<fold>:<APT>)
       python km_adapters_xlit.py --selftest        numpy + torch CPU, no sdm, no data: kNN selection, top-k == dense when k >= KV, compensation identities
       DEV=cpu CUDA_VISIBLE_DEVICES= python km_adapters_xlit.py --smoke [outdir]
                                                    build machine .venv-kumo, CPU only: the real medium model on a synthetic 300-row context, one fresh process per
                                                    knob setting; knobs unset == native bit-identical; TOPK k>=KV ~ dense; WEIGHTS=<vendor file> bit-identical;
                                                    LOGN_TAU=1 bit-identical; KNN end to end vs a uniform context of the same size (RMSE, information).
"""

import os, sys, math, json, time, hashlib, numpy as np

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
KNN = int(E("KNN", "0"))
KNN_QPC = int(E("KNN_QPC", "1024"))
KNN_GLOBAL = float(E("KNN_GLOBAL", "0.25"))
KNN_KEYS = E("KNN_KEYS", "cols")
KNN_ITERS = int(E("KNN_ITERS", "15"))
RR_ROT = int(E("RR_ROT", "0"))
TOPK = int(E("TOPK_ATTN", "0"))
TOPK_MIN = int(E("TOPK_MIN", str(TOPK)))
TOPK_QB = int(E("TOPK_QB", "128"))
TOPK_COMP = E("TOPK_COMP", "mean")
LOGN_TAU = float(E("LOGN_TAU", "0") or 0)
LOGN_TAU_MIN = float(E("LOGN_TAU_MIN", "1000"))
WEIGHTS = E("WEIGHTS", "")
CUR = {"fold": None, "apt": None, "inst": None}
XL = {}


# ---------------- pure helpers (numpy / torch only; shared by the live patches and the self test)
def robust_scale(P):
    """Per-column robust location/scale of the pool keys: median and IQR/1.349 (std where the IQR is 0); keep = columns with a finite positive scale."""
    med = np.nanmedian(P, 0)
    q75, q25 = np.nanpercentile(P, [75, 25], 0)
    s = (q75 - q25) / 1.349
    sd = np.nanstd(P, 0)
    s = np.where(s > 0, s, sd)
    keep = np.isfinite(s) & (s > 0) & np.isfinite(med)
    return med, s, keep


def zkeys(X, med, s, keep, clip=5.0):
    """Standardized key matrix: (x - med) / s on the kept columns, NaN -> 0 (= the median), clipped to +-clip."""
    Z = (X[:, keep] - med[keep]) / s[keep]
    Z = np.nan_to_num(Z, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(Z, -clip, clip).astype(np.float32)


def kmeans(Zq, nc, seed, iters):
    """Seeded Lloyd k-means on the query keys (torch, any device) -> labels [n]; empty clusters simply disappear."""
    import torch

    n = Zq.shape[0]
    nc = max(1, min(nc, n))
    g = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=g))
    C = Zq[first : first + 1].clone()
    dmin = torch.cdist(Zq, C).squeeze(1)
    for _ in range(nc - 1):  # farthest-point init: the next centroid is the row farthest from the chosen ones
        j = int(dmin.argmax())
        C = torch.cat([C, Zq[j : j + 1]], 0)
        dmin = torch.minimum(dmin, torch.cdist(Zq, Zq[j : j + 1]).squeeze(1))
    lab = torch.zeros(n, dtype=torch.long, device=Zq.device)
    for _ in range(iters):
        lab = torch.cdist(Zq, C).argmin(1)
        for c in range(nc):
            m = lab == c
            if m.any():
                C[c] = Zq[m].mean(0)
    return lab


def nearest(Zp, p2, Zq, n_take, blk=32768):
    """Indices of the n_take pool rows with the smallest min-over-queries squared distance (blocked over the pool; [blk, nq] temporaries)."""
    import torch

    q2 = (Zq**2).sum(1)
    dmin = torch.empty(Zp.shape[0], device=Zp.device, dtype=torch.float32)
    for a in range(0, Zp.shape[0], blk):
        b = min(a + blk, Zp.shape[0])
        d = p2[a:b, None] - 2.0 * (Zp[a:b] @ Zq.T) + q2[None, :]
        dmin[a:b] = d.min(1).values
    n_take = min(int(n_take), Zp.shape[0])
    idx = torch.topk(dmin, n_take, largest=False).indices
    return idx, dmin


def topk_core(q, k, v, scale, kk, qb, comp):
    """q [B,Hq,Q,C], k/v [B,Hkv,KV,C] -> [B,Hq,Q,C]. Exact logits per block of qb queries; keep the kk largest keys per (head, query).
    comp='mean': weights = exp(logit - logsumexp(all)) (the kept keys' share of the FULL softmax, sum alpha) and the dropped share 1-alpha goes to
    the mean value of the dropped keys; comp='none': softmax over the kept keys only. kk >= KV reproduces dense attention up to rounding.
    """
    import torch

    B, Hq, Q, C = q.shape
    Hkv, KV = k.shape[1], k.shape[2]
    rep = Hq // Hkv
    kk = min(int(kk), KV)
    scale = scale or 1.0 / math.sqrt(C)
    if rep > 1:
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
    wd = torch.float64 if q.dtype == torch.float64 else torch.float32
    kt = k.transpose(-1, -2)
    vsum = v.to(wd).sum(2)
    out = torch.empty_like(q)
    for a in range(0, Q, qb):
        qblk = q[:, :, a : a + qb]
        nq = qblk.shape[2]
        sf = (torch.matmul(qblk, kt) * scale).to(wd)  # [B,Hq,nq,KV]
        top, idx = sf.topk(kk, dim=-1)  # [B,Hq,nq,kk]
        if comp == "mean":
            w = torch.exp(top - torch.logsumexp(sf, -1, keepdim=True))
        else:
            w = torch.softmax(top, -1)
        del sf
        vg = torch.gather(v.unsqueeze(2).expand(B, Hq, nq, KV, C), 3, idx.unsqueeze(-1).expand(B, Hq, nq, kk, C)).to(
            wd
        )  # [B,Hq,nq,kk,C]
        o = torch.einsum("bhqk,bhqkc->bhqc", w, vg)
        if comp == "mean" and KV > kk:
            alpha = w.sum(-1, keepdim=True)
            vrest = (vsum.unsqueeze(2) - vg.sum(3)) / float(KV - kk)
            o = o + (1.0 - alpha) * vrest
        out[:, :, a : a + qb] = o.to(out.dtype)
        del vg, w, idx, top
    return out


# ---------------- self test (no sdm, no data)
def selftest():
    import torch

    rng = np.random.default_rng(0)
    # 1. key standardization + nearest: the nearest rows of a cluster are its own members' neighbours
    P = rng.normal(size=(5000, 12))
    P[:, 3] = 1.0
    P[rng.random(P.shape) < 0.02] = np.nan  # one constant column -> dropped
    med, s, keep = robust_scale(P)
    assert keep.sum() == 11 and not keep[3], keep
    Zp = torch.as_tensor(zkeys(P, med, s, keep))
    p2 = (Zp**2).sum(1)
    assert torch.isfinite(Zp).all() and Zp.abs().max() <= 5
    Qc = P[100:110] + 1e-3
    Zq = torch.as_tensor(zkeys(Qc, med, s, keep))
    idx, dmin = nearest(Zp, p2, Zq, 10, blk=777)
    assert set(idx.tolist()) == set(range(100, 110)), sorted(idx.tolist())
    lab = kmeans(torch.as_tensor(zkeys(np.vstack([P[:50] + 50, P[50:100] - 50]), med, s, keep)), 2, seed=1, iters=10)
    assert len(torch.unique(lab[:50])) == 1 and len(torch.unique(lab[50:])) == 1 and lab[0] != lab[50]
    # 2. top-k attention == dense (GQA 8 query heads / 2 kv heads) when k >= KV, both compensation modes; exact convexity otherwise
    B, Hq, Hkv, Q, KV, C = 2, 8, 2, 37, 300, 16
    q = torch.randn(B, Hq, Q, C, dtype=torch.float64)
    k = torch.randn(B, Hkv, KV, C, dtype=torch.float64)
    v = torch.randn(B, Hkv, KV, C, dtype=torch.float64)
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    for comp in ("mean", "none"):
        o = topk_core(q, k, v, None, 1000, 16, comp)
        d = (o - dense).abs().max().item()
        assert d < 1e-9, (comp, d)
    o32 = topk_core(q, k, v, None, 32, 16, "mean")
    o32n = topk_core(q, k, v, None, 32, 16, "none")
    dd = (o32 - dense).abs().max().item()
    dn = (o32n - dense).abs().max().item()
    assert dd > 0 and dn > 0 and torch.isfinite(o32).all()
    vc = torch.ones_like(v) * 3.0
    oc = topk_core(q, k, vc, None, 32, 16, "mean")
    assert (oc - 3.0).abs().max().item() < 1e-9  # constant values: exact for any k
    # 3. the GQA detector used in the live patch: query heads != kv heads <=> the ICL query->context path (medium/large)
    assert q.size(1) != k.size(1)
    print(
        f"SELFTEST OK: keys/nearest/kmeans | topk(k>=KV) == dense (max diff {d:.1e}) | topk32 mean/none differ from dense by {dd:.3f}/{dn:.3f} (expected > 0) | constant-value exactness OK"
    )


if __name__ == "__main__" and "--selftest" in sys.argv:
    selftest()
    sys.exit(0)

# ---------------- live mode: wrap the engine (torch + sdm are imported here, not above)
sys.path.insert(0, HERE)
import torch
import km_engine as KE

K = KE.K
DEV = KE.DEV
_orig_fit, _orig_predict, _orig_context, _orig_model = KE.fit, KE.predict, KE.context, K.model


def set_cur(spec):
    p = spec.split(":")
    CUR["fold"] = p[1] if len(p) > 1 else None
    CUR["apt"] = p[2] if len(p) > 2 else None
    CUR["inst"] = int(p[3]) if p[0] == "inst" and len(p) > 3 else None


# --- RR_ROT: rotate the recipe's round-robin transform choice by the instance index
if RR_ROT:
    from sdm.processing.common import choice as _C

    _orig_draw = _C.Choice._draw_option_ids

    def _draw(self, ensemble_table, *, generator=None):
        ids = _orig_draw(self, ensemble_table, generator=generator)
        if self.method == "round_robin" and CUR["inst"]:
            ids = tuple((m + CUR["inst"]) % len(self.options) for m in ids)
        return ids

    _C.Choice._draw_option_ids = _draw
    XL["rr_rot"] = 1

    def fit_rr(xc, yc, i):
        CUR["inst"] = int(i)
        return _orig_fit(xc, yc, i)

    KE.fit = fit_rr

# --- LOGN_TAU: temperature on LogScale for long attentions (GatedLogScale inherits through super().forward)
if LOGN_TAU:
    from sdm.nn import scaling as _S

    _orig_ls = _S.LogScale.forward

    def _ls(self, query, *, key_len):
        out = _orig_ls(self, query, key_len=key_len)
        n = float(key_len.max()) if isinstance(key_len, torch.Tensor) else float(key_len)
        return out * LOGN_TAU if n > LOGN_TAU_MIN else out

    _S.LogScale.forward = _ls
    XL["logn_tau"] = LOGN_TAU
    XL["logn_tau_min"] = LOGN_TAU_MIN

# --- TOPK_ATTN: top-k sparse attention with dropped-mass compensation on the GQA (query -> cached context) path
if TOPK > 0:
    from sdm.nn import attention as _A

    _orig_sdpa = _A.SDPA.forward

    def _sdpa(self, query, key, value, seqused_key_value=None, attn_mask=None):
        if (
            query.numel() == 0
            or seqused_key_value is not None
            or attn_mask is not None
            or query.size(-2) == key.size(-2)
            or key.size(-3) <= TOPK_MIN
        ):
            return _orig_sdpa(self, query, key, value, seqused_key_value, attn_mask)
        batch_shape = torch.broadcast_shapes(query.size()[:-3], key.size()[:-3], value.size()[:-3])
        if self.query_scaling is not None:
            query = self.query_scaling(query, key_len=key.size(-3))
        qs, ks, vs = query.size()[-3:], key.size()[-3:], value.size()[-3:]
        q = query.expand(batch_shape + qs).reshape(-1, *qs).transpose(-3, -2)
        k = key.expand(batch_shape + ks).reshape(-1, *ks).transpose(-3, -2)
        v = value.expand(batch_shape + vs).reshape(-1, *vs).transpose(-3, -2)
        out = topk_core(q, k, v, self.scale, TOPK, TOPK_QB, TOPK_COMP).transpose(-3, -2)
        return out.view(batch_shape + out.size()[-3:])

    _A.SDPA.forward = _sdpa
    XL["topk_attn"] = TOPK
    XL["topk_min"] = TOPK_MIN
    XL["topk_comp"] = TOPK_COMP
    XL["topk_qb"] = TOPK_QB

# --- WEIGHTS: fine-tuned state dict into the (cached) model; re-loaded when the model object or the templated path changes
if WEIGHTS:
    _loaded = {"key": None}

    def model_w(size):
        mdl = _orig_model(size)
        path = WEIGHTS.format(size=size, fold=CUR["fold"], apt=CUR["apt"])
        key = (id(mdl), path)
        if _loaded["key"] != key:
            sd = torch.load(path, map_location=K.DEV, weights_only=True)
            try:
                mdl.load_state_dict(sd, strict=True)
                how = "model"
            except Exception:
                subs = list(mdl.models.values())
                assert len(subs) == 1, "WEIGHTS: ambiguous task modules"
                subs[0].load_state_dict(sd, strict=True)
                how = "regressor"
            mdl.eval()
            _loaded["key"] = key
            sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            XL["weights"] = dict(path=path, sha256=sha, how=how)
            print(f"  xlit WEIGHTS loaded ({how}) {path} sha256 {sha[:16]}", flush=True)
        return mdl

    K.model = model_w
    XL.setdefault("weights", dict(path=WEIGHTS))

# --- KNN: retrieval contexts (whole pool handed to fit; per-cluster fits inside predict)
if KNN:

    class Retrieval:
        """What fit() returns in KNN mode: the instance's whole pool in its column space + standardized keys + the uniform slice (global share)."""

        def __init__(self, xc, yc, i, keypos, uni):
            self.xc, self.yc, self.i, self.keypos, self.uni = xc, yc, int(i), keypos, uni
            self.med, self.s, self.keep = robust_scale(xc[:, keypos])
            self.Zp = torch.as_tensor(zkeys(xc[:, keypos], self.med, self.s, self.keep), device=DEV)
            self.p2 = (self.Zp**2).sum(1)
            self.n_fit = 0
            self.rows_fit = 0
            self.d_loc = []

        def clear(self):
            self.Zp = self.p2 = None

    def context_knn(F, ti, i, fold):
        """The instance 'context' is the whole clean pool; the real contexts are chosen per query cluster inside predict()."""
        CUR["fold"] = fold
        CUR["inst"] = int(i)
        return ti

    def fit_knn(xc, yc, i):
        fold = CUR["fold"]
        assert fold, "xlit KNN: fold unknown (fit called outside a unit)"
        F = K.frame(CUR["apt"])
        ti, _ = KE.split(F, fold)
        assert len(xc) == len(
            ti
        ), f"xlit KNN: fit received {len(xc)} rows, the pool has {len(ti)} (context patch not applied?)"
        cols, _ = KE.columns(int(i), fold)
        if KNN_KEYS == "cols":
            keypos = np.arange(len(cols))
        else:
            idx = {c: j for j, c in enumerate(K.FEATS)}
            cl = list(cols)
            miss = [n for n in KNN_KEYS.split(",") if idx.get(n) not in cl]
            assert not miss, f"xlit KNN_KEYS not in this instance's columns (add them with EXTRACOLS): {miss}"
            keypos = np.array([cl.index(idx[n]) for n in KNN_KEYS.split(",")])
        uni_rows = _orig_context(F, ti, int(i), fold)
        uni = np.searchsorted(ti, uni_rows)
        assert np.array_equal(ti[uni], uni_rows)
        rm = Retrieval(xc, yc, i, keypos, uni)
        print(
            f"  xlit KNN inst {i}: pool {len(ti):,} rows, keys {int(rm.keep.sum())}/{len(keypos)} cols ({KNN_KEYS}), per cluster {int(round((1 - KNN_GLOBAL) * KE.S['cap'])):,} local + {int(round(KNN_GLOBAL * KE.S['cap'])):,} global of CAP {KE.S['cap']:,}",
            flush=True,
        )
        return rm

    def predict_knn(mdl, xq, ch0=None):
        cap = KE.S["cap"]
        n = len(xq)
        n_glob = int(round(KNN_GLOBAL * cap))
        n_loc = cap - n_glob
        Zq = torch.as_tensor(zkeys(xq[:, mdl.keypos], mdl.med, mdl.s, mdl.keep), device=DEV)
        nc = max(1, int(round(n / KNN_QPC)))
        lab = kmeans(Zq, nc, seed=KE.S["seed"] * 1000 + mdl.i * 7919 + n, iters=KNN_ITERS).cpu().numpy()
        g = np.empty(n)
        qs = np.empty((n, len(K.QLEV)), np.float32)
        ch = ch0
        for c in np.unique(lab):
            qi = np.where(lab == c)[0]
            loc, dmin = nearest(mdl.Zp, mdl.p2, Zq[torch.as_tensor(qi, device=DEV)], n_loc)
            loc = loc.cpu().numpy()
            glob = mdl.uni[~np.isin(mdl.uni, loc)][:n_glob]
            ctx = np.concatenate([loc, glob]).astype(int)
            m = _orig_fit(mdl.xc[ctx], mdl.yc[ctx], mdl.i)
            gg, qq, ch = _orig_predict(m, xq[qi], ch)
            m.clear()
            g[qi] = gg
            qs[qi] = qq
            mdl.n_fit += 1
            mdl.rows_fit += len(ctx)
            mdl.d_loc.append(float(dmin[torch.as_tensor(loc, device=DEV)].clamp(min=0).sqrt().mean()))
        print(
            f"    xlit KNN shard: {n:,} queries -> {len(np.unique(lab))} clusters, {mdl.n_fit} fits so far, mean local key distance {np.mean(mdl.d_loc[-len(np.unique(lab)):]):.2f}",
            flush=True,
        )
        return g, qs, ch

    KE.context = context_knn
    KE.fit = fit_knn
    KE.predict = predict_knn
    XL["knn"] = dict(qpc=KNN_QPC, glob=KNN_GLOBAL, keys=KNN_KEYS, iters=KNN_ITERS)

if XL:
    KE.S["xlit"] = XL
    print(f"km_adapters_xlit active: {json.dumps(XL)[:400]}", flush=True)


# ---------------- CPU smoke (build machine .venv-kumo): one fresh process per knob setting, compared by the driver
def _smoke_one(out):
    """Synthetic problem (seeded): pool P rows x Dm cols, nonlinear target; fits with the engine's fit/predict (patched or not) -> npy [1+7, Q]."""
    import sdm

    rng = np.random.default_rng(0)
    P, Q, Dm = int(E("SMK_P", "300")), 60, 40
    X = rng.normal(size=(P + Q, Dm)).astype(np.float32)
    y = 100 * np.sin(X[:, 0]) + 50 * X[:, 1] * X[:, 2] + 30 * X[:, 3] + rng.normal(0, 10, P + Q)
    X[rng.random(X.shape) < 0.03] = np.nan
    torch.manual_seed(0)
    mdl = sdm.models.KumoTabular(task="regression", size="medium", device=DEV)
    K._models.clear()
    K._models["medium"] = mdl
    cap = int(E("SMK_CAP", str(P)))
    KE.S.update(cap=cap, amp="off", seed=0, ncol=Dm, colmode="all", ninst=1, size="medium")
    KE.D = Dm
    F = dict(
        X=X,
        y=y,
        m=np.full(P + Q, 3),
        rank=np.zeros(P + Q, bool),
        clean=np.ones(P + Q, bool),
        off=np.zeros(P + Q),
        lk=np.ones(P + Q, bool),
        ids=np.arange(P + Q, dtype=float),
        row=np.arange(P + Q),
    )
    F["m"][P:] = 1
    K.frame = lambda apt: F
    set_cur("inst:janjul:SYN:" + E("SMK_INST", "0"))
    ti, qi = KE.split(F, "janjul")
    i = int(E("SMK_INST", "0"))
    k = KE.context(F, ti, i, "janjul")
    t0 = time.time()
    m = KE.fit(X[k], y[k] - F["off"][k], i)
    g, qs, _ = KE.predict(m, X[qi])
    m.clear()
    rm = float(np.sqrt(((g - y[qi]) ** 2).mean()))
    np.save(out, np.vstack([g, qs.T]))
    print(
        f"SMOKE_ONE ctx {len(k)} (cap {cap}) queries {len(qi)} rmse {rm:.2f} | {time.time() - t0:.0f}s | xlit {json.dumps(XL)[:200]}",
        flush=True,
    )


def _smoke(outdir):
    import subprocess, glob

    os.makedirs(outdir, exist_ok=True)
    base = dict(os.environ, DEV="cpu", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    for kname in ("KNN", "RR_ROT", "TOPK_ATTN", "LOGN_TAU", "WEIGHTS", "LOGN_CLAMP", "ROPE_PI"):
        base.pop(kname, None)
    w = glob.glob(f"{os.environ.get('HF_HOME', '')}/hub/models--nvidia--Kumo-Tabular/snapshots/*/medium/regressor.pt")
    runs = [
        ("A_native", {}),
        ("B_native_again", {}),
        ("C_topk_k_ge_KV", dict(TOPK_ATTN="1000", TOPK_MIN="100", TOPK_COMP="mean")),
        ("D_topk32_mean", dict(TOPK_ATTN="32", TOPK_MIN="100", TOPK_COMP="mean")),
        ("D2_topk32_none", dict(TOPK_ATTN="32", TOPK_MIN="100", TOPK_COMP="none")),
        ("F_tau1", dict(LOGN_TAU="1.0", LOGN_TAU_MIN="100")),
        ("F2_tau1.1", dict(LOGN_TAU="1.1", LOGN_TAU_MIN="100")),
        ("G_rr_rot_inst1", dict(RR_ROT="1", SMK_INST="1")),
        ("G0_inst1", dict(SMK_INST="1")),
        ("H0_uniform100", dict(SMK_CAP="100")),
        ("H_knn100", dict(KNN="1", SMK_CAP="100", KNN_QPC="20", KNN_GLOBAL="0.25")),
    ]
    if w:
        runs.insert(5, ("E_weights_vendor", dict(WEIGHTS=w[0])))
    else:
        print("smoke: vendor regressor.pt not found under HF_HOME -> WEIGHTS check skipped", flush=True)
    res = {}
    for name, env in runs:
        out = f"{outdir}/{name}.npy"
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--smoke-one", out],
            env=dict(base, **env),
            capture_output=True,
            text=True,
        )
        tail = [l for l in r.stdout.splitlines() if l.startswith("SMOKE_ONE") or l.startswith("km_adapters_xlit")]
        print(f"[{name}] rc {r.returncode} {time.time() - t0:.0f}s | " + " | ".join(tail)[:300], flush=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True)
            continue
        res[name] = np.load(out)

    def diff(a, b):
        return float(np.abs(res[a][0] - res[b][0]).max()) if a in res and b in res else float("nan")

    def rmse(a):
        return float(np.sqrt(((res[a][0] - _truth()) ** 2).mean())) if a in res else float("nan")

    lines = [
        f"B vs A (determinism, knobs unset): max|diff| {diff('B_native_again', 'A_native'):.3e}  -> PASS iff 0",
        f"C vs A (TOPK k >= KV, mean comp): max|diff| {diff('C_topk_k_ge_KV', 'A_native'):.3e}  -> PASS iff < 0.05 s",
        f"D vs A (TOPK 32 of 300, mean): {diff('D_topk32_mean', 'A_native'):.3f} s | D2 (none): {diff('D2_topk32_none', 'A_native'):.3f} s  -> expected > 0",
        f"E vs A (WEIGHTS = vendor file): max|diff| {diff('E_weights_vendor', 'A_native'):.3e}  -> PASS iff 0",
        f"F vs A (LOGN_TAU 1.0): {diff('F_tau1', 'A_native'):.3e} -> PASS iff 0 | F2 (1.1): {diff('F2_tau1.1', 'A_native'):.3f} s -> expected > 0",
        f"G (RR_ROT inst 1) vs G0 (inst 1, native): {diff('G_rr_rot_inst1', 'G0_inst1'):.3f} s -> expected > 0 (PowerTransform instead of Identity)",
        f"H (KNN cap 100) rmse {rmse('H_knn100'):.2f} vs H0 (uniform cap 100) rmse {rmse('H0_uniform100'):.2f} vs A (whole pool 300) {rmse('A_native'):.2f}  (information)",
    ]
    print("SMOKE SUMMARY\n  " + "\n  ".join(lines), flush=True)
    open(f"{outdir}/SMOKE_SUMMARY.txt", "w").write("\n".join(lines) + "\n")
    ok = (
        diff("B_native_again", "A_native") == 0
        and diff("C_topk_k_ge_KV", "A_native") < 0.05
        and (not w or diff("E_weights_vendor", "A_native") == 0)
        and diff("F_tau1", "A_native") == 0
    )
    print("SMOKE " + ("PASS" if ok else "CHECK"), flush=True)


def _truth():
    rng = np.random.default_rng(0)
    P, Q, Dm = int(E("SMK_P", "300")), 60, 40
    X = rng.normal(size=(P + Q, Dm)).astype(np.float32)
    y = 100 * np.sin(X[:, 0]) + 50 * X[:, 1] * X[:, 2] + 30 * X[:, 3] + rng.normal(0, 10, P + Q)
    return y[P:]


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--smoke-one":
        _smoke_one(a[1])
        sys.exit(0)
    if a and a[0] == "--smoke":
        _smoke(a[1] if len(a) > 1 else os.path.join(KE.RUN, "xlit_smoke"))
        sys.exit(0)
    import socket

    claim = "--claim" in a
    a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list":
        a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(
        f"km_adapters_xlit: knobs {json.dumps(XL) if XL else 'none (native engine)'} | engine spec {KE.S}", flush=True
    )
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
    print(f"KM_XLIT_DONE units run {n} | {time.time() - t0:.0f}s", flush=True)
