#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""KCATL: one Kumo Tabular instance of the log-target KCAT member on (fold, airport).
Everything except the target is the KCAT8 spec (scripts/kumo_kcat_member.py): medium model, the airport's whole 2025 pool as context,
100 columns (importance core 50) plus the 8 declared categorical columns, seed 0, fp16. Instance i therefore uses the same contexts,
columns and generators as KCAT8 instance i.
Target log1p(y). Point prediction = mean over the 999 predicted quantiles of expm1(q), with q clipped to the context's [min, max] log
target: the distributional mean, which carries its own back-transform correction. The 7 stored summary quantiles are inverted the same way.
Outputs stay in the engine's offset space (g = taxi - off), so the engine aggregation (kumo_kcat_member.py agg) applies unchanged.
usage: python kumo_kcat_logtarget_member.py <fold>:<APT>:<i> ...   -> $RUN/results/engine/KCATL/<fold>_<APT>/inst<i>.npz
env: as kumo_kcat_member.py (PAYLOAD, RUN, CODES, ...); KCATL_COOL = seconds of GPU cool-down between units (default 0);
     SMOKE_ROWS = n: CPU code-path smoke on n context and n query rows (use a separate RUN)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, json, numpy as np

ROOT = os.environ.get("PRC_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "kumo_gpu", "cluster"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kumo_kcat_member as XC  # patches KE.frame / KE.columns / KE.fit for the 8 declared categoricals
import km_engine as KE, torch

K = KE.K
DEV = KE.DEV


def predict(mdl, xq, h):
    """h(q [n, 999] torch, rows) -> per-row point; also keeps the 7 summary quantiles in predict.q. The chunk halves on OOM."""
    out, qs, i, ch = [], [], 0, K.CHUNK0
    while i < len(xq):
        try:
            xb = torch.as_tensor(xq[i : i + ch], device=DEV)
            with K.amp_ctx(KE.S["amp"]):
                p = mdl.predict(XC.table(xb))
            q = (p.numerical if hasattr(p, "numerical") else p).float()
            while q.dim() > 2:
                q = q[0]
            assert q.shape == (len(xb), 999), q.shape
            out.append(h(q, slice(i, i + len(xb))).cpu().numpy().astype(np.float64))
            qs.append(q[:, K.QIDX].cpu().numpy().astype(np.float64))
            i += ch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            assert ch > 64
            ch //= 2
            print(f"    OOM -> CHUNK {ch}", flush=True)
    predict.q = np.concatenate(qs)
    return np.concatenate(out)


def inst(fold, apt, i):
    i = int(i)
    out = f"{K.RUN}/results/engine/KCATL/{fold}_{apt}/inst{i}.npz"
    if os.path.exists(out):
        print(f"  skip {out}", flush=True)
        return False
    F = K.frame(apt)
    X = F["X"]
    ti, qi = KE.split(F, fold)
    k = KE.context(F, ti, i, fold)
    cols, csrc = KE.columns(i, fold)
    SM = int(os.environ.get("SMOKE_ROWS", "0"))
    if SM:
        k = k[np.random.default_rng(0).choice(len(k), SM, replace=False)]
        qi = qi[:SM]
    y, off = F["y"], F["off"]
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = np.log1p(np.maximum(y[k], 0.0))
    lo, hi = float(t.min()), float(t.max())
    mdl = KE.fit(X[k][:, cols], t, i)
    pt = predict(mdl, X[qi][:, cols], lambda q, s: torch.expm1(q.clamp(lo, hi)).mean(-1))
    mdl.clear()
    g = pt - off[qi]
    assert np.isfinite(g).all(), (fold, apt, i)
    qq = (np.expm1(np.clip(predict.q, lo, hi)) - off[qi][:, None]).astype(np.float32)
    tt = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    ok = F["clean"][qi] & np.isfinite(y[qi])
    rm = float(np.sqrt(((off[qi] + g - y[qi])[ok] ** 2).mean())) if ok.any() else float("nan")
    K.save_atomic(
        out,
        row=F["row"][qi],
        id=F["ids"][qi],
        g=g,
        q=qq,
        cols=cols,
        ctx_rows=F["row"][k],
        meta=json.dumps(
            dict(
                spec=dict(KE.S, name="KCATL", target="log1p", inverse="mean expm1(clip q)"),
                inst=i,
                fold=fold,
                apt=apt,
                ctx=len(k),
                ncol=len(cols),
                colsrc=csrc,
                query=len(qi),
                fit_s=tt,
                pred_s=0.0,
                peak_gib=mem,
                rmse_clean=rm,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                torch=torch.__version__,
            )
        ),
    )
    print(
        f"  KCATL {fold} {apt} #{i}: ctx {len(k):,} cols {len(cols)} query {len(qi):,} rmse_clean {rm:.2f} | {tt:.0f}s peak {mem:.1f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    print(f"KCATL spec {KE.S} PAYLOAD {K.PAYLOAD} RUN {K.RUN}", flush=True)
    COOL = float(os.environ.get("KCATL_COOL", "0"))
    ran = False
    for u in sys.argv[1:]:
        if ran and COOL > 0:
            time.sleep(COOL)
        ran = inst(*u.split(":")) is not False
    print("KCATL_DONE", flush=True)
