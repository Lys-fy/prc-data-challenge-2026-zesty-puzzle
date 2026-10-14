#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Runs the Kumo Tabular "FM engine": an ensemble of many small in-distribution instances, sharded and resumable, the
same code on the build machine, under Slurm, or on any GPU machine. Inputs: positional unit(s) (inst:<fold>:<APT>:<i>,
agg:<fold>:<APT>, bench:<fold>:<APT>, parity16) or --from-list list.txt, plus the env ENSEMBLE SPEC below (NAME
required). Writes $RUN/results/engine/<NAME>/... instance npz and $RUN/results/member/<NAME>_<fold>_<APT>.npz (member
output, consumed by collect.py). Builds on KUMO_CLUSTER_SETUP's km_task.py (payload frames, model cache, atomic writes, output format).

ENSEMBLE SPEC (env): NAME (required, e.g. KME8c100), SIZE medium|large, CAP 20000 (context rows per instance; 60000 = the end of Kumo's pretraining
  envelope; CAP=1000000 = the WHOLE pool, 105-270k rows per airport-fold, S-B proof arms), NCOL 412 (columns per instance; <= 100 = the pretraining envelope), COLMODE imp|rand|all, CORE 50 (imp: top-CORE columns by LGB gain
  in every instance, the other NCOL-CORE filled round-robin from a seeded permutation of the rest so every column is read by some instance),
  NINST 8 (instances per airport-fold), EIN 1 (Kumo estimators inside one instance; the vendor recipe's column shuffle / transform / sign flip),
  AMP fp16|bf16|off, SEED 0, SHARD 8192 (query rows per resumable part file), TRIM 0.2 (trimmed-mean fraction over instances), QMAX (bench only),
  CTXMODE uniform|recent|mixed (recent = nearest-month-first slices, KUMO_CTXMODE_RECENT (3); mixed = half uniform, half recent),
  NOISE 0 (ablation only: that many pure N(0,1) columns appended to every instance's inputs, independent per row),
  EXTRACOLS (ablation only: file or comma list of real feature names appended to every instance's subset, e.g. data/km_drift20.txt).
INSTANCE i = (context slice i, column subset i, generator seed SEED*1000+i). Context slices follow the vendor adapter rule (consecutive slices of
  concatenated seeded permutations of the airport-fold's CLEAN training pool = the other months of 2025 at that airport: in-distribution rows);
  if the pool is <= CAP every instance sees the whole pool. Target = y - mvt_aobt3 (kumo_member_screen arm M's offset); NaN kept.
AGGREGATION (agg unit): per row, TRIM-trimmed mean over instances of the instance point g_i (= mean of its 999 quantiles) and of each of the 7
  stored quantile levels; taxi = offset + g; spread = q90 - q10 of the aggregated quantiles and g_sd = sd over instances (both kept as features).
  Output = km_task.py's member npz (row, id, taxi, g, q, qlev, meta) + spread, g_sd, g_inst, so collect.py / the screens work unchanged.
UNITS:  inst:<fold>:<APT>:<i>   -> $RUN/results/engine/<NAME>/<fold>_<APT>/inst<i>.npz   (parts inst<i>.p<k>.npz while running; resume skips done parts)
        agg:<fold>:<APT>        -> $RUN/results/member/<NAME>_<fold>_<APT>.npz           (needs all NINST instances)
        bench:<fold>:<APT>      one instance (i = 0) on the first QMAX query rows, no files but a line in $RUN/bench.jsonl (pilot a)
        parity16                km_task profile shared20k with AMP=fp16 on janjul x LSZH vs the fp32 GPU workstation reference (payload ref/) -> RMS / max / corr
usage:  python km_engine.py <unit> ... | --from-list list.txt [--claim]      env PAYLOAD, RUN, MEMLOG (file; one line per minute), MEMGUARD_GIB.
Safety: fp16, estimator_batch_size=1, SDM_CHUNK_MEMORY_FRACTION=0.02 on the build machine (0.05 = vendor default on cluster GPUs),
  expandable_segments, memlog + guard. Long-context knobs LOGN_CLAMP / ROPE_PI: cluster/km_adapters.py (off unless set; recorded in spec.adapters).
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, json, math, socket, numpy as np

os.environ.setdefault("SDM_CHUNK_MEMORY_FRACTION", "0.02")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memlog

memlog.start()
import km_task as K  # payload frames, model cache, amp_ctx, save_atomic, PROFILES, HOLD, QLEV/QIDX
import torch, km_adapters

E = os.environ.get
S = dict(
    name=E("NAME", ""),
    size=E("SIZE", "medium"),
    cap=int(E("CAP", "20000")),
    ncol=int(E("NCOL", "412")),
    colmode=E("COLMODE", "imp"),
    core=int(E("CORE", "50")),
    ninst=int(E("NINST", "8")),
    ein=int(E("EIN", "1")),
    amp=E("AMP", "fp16"),
    seed=int(E("SEED", "0")),
    shard=int(E("SHARD", "8192")),
    trim=float(E("TRIM", "0.2")),
    ctxmode=E("CTXMODE", "uniform"),
    noise=int(E("NOISE", "0")),
    adapters=km_adapters.apply(),
)
QMAX = int(E("QMAX", "4096"))
DEV = K.DEV
RUN = K.RUN
PAYLOAD = K.PAYLOAD
BDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
D = len(K.FEATS)


def importance(fold):
    """Column order by LightGBM gain, computed on the fold's TRAINING months only (shipped tables in data/); pooled file as fallback.
    SHAP_IMP=1: prefer the exact-TreeSHAP ranking (stage/importance_shap.py) when present; OFF by default (byte-identical).
    """
    cand = ((f"{BDIR}/km_importance_412_shap_{fold}.tsv",) if E("SHAP_IMP") else ()) + (
        f"{BDIR}/km_importance_412_{fold}.tsv",
        f"{BDIR}/km_importance_412_all.tsv",
    )
    for f in cand:
        if os.path.exists(f):
            rows = [l.split("\t") for l in open(f) if l.strip() and not l.startswith("#")]
            idx = {c: i for i, c in enumerate(K.FEATS)}
            order = [idx[r[0]] for r in rows if r[0] in idx]
            assert len(order) == D, (f, len(order))
            return np.array(order), f
    raise SystemExit(f"COLMODE=imp needs {BDIR}/km_importance_412_<fold>.tsv (shipped in kumo_gpu/data/)")


def columns(i, fold):
    """Column subset of instance i (sorted indices into FEATS)."""
    if S["ncol"] >= D or S["colmode"] == "all":
        return np.arange(D), "all"
    if S["colmode"] == "rand":
        return np.sort(np.random.default_rng(S["seed"] * 1000 + 777 + i).choice(D, S["ncol"], replace=False)), "rand"
    order, src = importance(fold)
    extra = extracols()
    if len(extra):
        order = np.array([j for j in order if j not in set(extra.tolist())])
        src += f"+extra{len(extra)}"
    core = order[: S["core"]]
    rest = order[S["core"] :]
    k = S["ncol"] - S["core"]
    perm = np.random.default_rng(S["seed"] + 1).permutation(rest)
    reps = math.ceil((i + 1) * k / len(perm)) + 1
    fill = np.concatenate([perm] * reps)[i * k : (i + 1) * k]
    return np.sort(np.concatenate([core, fill, extra]).astype(int)), os.path.basename(src)


def extracols():
    """EXTRACOLS = file (first token per non-# line) or comma list of feature names appended to every instance's subset (ablation (e): drifted real columns)."""
    v = E("EXTRACOLS", "")
    if not v:
        return np.array([], dtype=int)
    names = (
        [l.split()[0] for l in open(v) if l.strip() and not l.startswith("#")] if os.path.exists(v) else v.split(",")
    )
    idx = {c: i for i, c in enumerate(K.FEATS)}
    miss = [n for n in names if n not in idx]
    assert not miss, f"EXTRACOLS not in the 412: {miss}"
    return np.array([idx[n] for n in names], dtype=int)


def month_dist(F, ti, fold):
    """Calendar distance (months) of each pool row to the held-out months (prod: to Jan 2026, i.e. recency within 2025)."""
    m = F["m"][ti].astype(int)
    held = K.HOLD[fold] or [13]
    return np.min([np.abs(m - h) for h in held], axis=0)


def context(F, ti, i, fold):
    """CTXMODE uniform (default): vendor adapter rule, instance i = slice i of concatenated seeded permutations of the clean training pool.
    recent (KUMO_CTXMODE_RECENT (3)): pool ordered by calendar distance to the held months (ties shuffled), instance i = the i-th nearest slice.
    mixed: even instances uniform (slice i//2), odd instances recent (slice i//2). Whole pool if it fits CAP."""
    cap = S["cap"]
    mode = S["ctxmode"]
    if len(ti) <= cap:
        return ti
    j = i // 2 if mode == "mixed" else i
    rec = mode == "recent" or (mode == "mixed" and i % 2 == 1)
    rng = np.random.default_rng(S["seed"])
    if rec:
        dist = month_dist(F, ti, fold)
        key = dist + rng.random(len(ti))
        order = ti[np.argsort(key, kind="stable")]
        reps = math.ceil((j + 1) * cap / len(ti))
        perm = np.concatenate([order] * reps)
        return perm[j * cap : (j + 1) * cap]
    reps = math.ceil(S["ninst"] * cap / len(ti))
    perm = np.concatenate([rng.permutation(ti) for _ in range(reps)])
    return perm[j * cap : (j + 1) * cap]


def split(F, fold):
    m, y, rank, clean = F["m"], F["y"], F["rank"], F["clean"]
    if K.HOLD[fold] is None:
        val = rank
        trn = ~rank & clean & np.isfinite(y)
    else:
        val = ~rank & np.isin(m, K.HOLD[fold])
        trn = ~rank & ~np.isin(m, K.HOLD[fold]) & clean & np.isfinite(y)
    return np.where(trn)[0], np.where(val)[0]


def fit(xc, yc, i):
    mdl = K.model(S["size"])
    gen = torch.Generator(device=DEV).manual_seed(S["seed"] * 1000 + i)
    with K.amp_ctx(S["amp"]):
        mdl.fit(
            torch.as_tensor(xc, device=DEV),
            torch.as_tensor(yc.astype(np.float32)[:, None], device=DEV),
            num_estimators=S["ein"],
            estimator_batch_size=1,
            generator=gen,
        )
    return mdl


def predict(mdl, xq, ch0=None):
    """mean of the 999 quantiles + the 7 summary quantiles; the query chunk halves on OOM (predictions are chunk-invariant)."""
    out, qs, i, ch = [], [], 0, ch0 or K.CHUNK0
    while i < len(xq):
        try:
            xb = torch.as_tensor(xq[i : i + ch], device=DEV)
            with K.amp_ctx(S["amp"]):
                p = mdl.predict(xb)
            q = (p.numerical if hasattr(p, "numerical") else p).float()
            while q.dim() > 2:
                assert q.shape[0] == 1, q.shape
                q = q[0]
            assert q.shape[-1] == 999 and q.shape[0] == len(xb), q.shape
            out.append(q.mean(-1).cpu().numpy())
            qs.append(q[:, K.QIDX].cpu().numpy())
            i += ch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            assert ch > 64, "OOM at 64 query rows: the context does not fit"
            ch //= 2
            print(f"    OOM -> CHUNK {ch}", flush=True)
    return np.concatenate(out), np.concatenate(qs).astype(np.float32), ch


def inst(fold, apt, i, bench=False):
    assert S["name"] or bench, "NAME required"
    i = int(i)
    out = f"{RUN}/results/engine/{S['name']}/{fold}_{apt}/inst{i}.npz"
    if not bench and os.path.exists(out):
        print(f"  skip {out} (done)", flush=True)
        return None
    memlog.STATE["unit"] = f"{'bench' if bench else 'inst'}:{S['name']}:{fold}:{apt}:{i}"
    F = K.frame(apt)
    ti, qi = split(F, fold)
    k = context(F, ti, i, fold)
    cols, csrc = columns(i, fold)
    if bench:
        qi = qi[:QMAX]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    def X(
        rows, cols, tag
    ):  # instance inputs; NOISE pure-noise columns appended (ablation d), deterministic per (seed, instance, tag)
        A = F["X"][rows][:, cols]
        if S["noise"] > 0:
            A = np.hstack(
                [
                    A,
                    np.random.default_rng([S["seed"], i, abs(hash(tag)) % 2**31]).standard_normal(
                        (len(rows), S["noise"]), dtype=np.float32
                    ),
                ]
            )
        return A

    mdl = fit(X(k, cols, "ctx"), F["y"][k] - F["off"][k], i)
    torch.cuda.synchronize()
    tf = time.time() - t0
    n = len(qi)
    parts = []
    ch = None
    t1 = time.time()
    done = 0
    for p0 in range(0, n, S["shard"]):
        pf = f"{out[:-4]}.p{p0 // S['shard']}.npz"
        if not bench and os.path.exists(pf):
            parts.append(pf)
            continue
        g, qs, ch = predict(mdl, X(qi[p0 : p0 + S["shard"]], cols, f"q{p0}"), ch)
        done += len(g)
        if bench:
            parts.append((g, qs))
            continue
        K.save_atomic(pf, g=g, q=qs)
        parts.append(pf)
    torch.cuda.synchronize()
    tp = time.time() - t1
    mdl.clear()
    mem = torch.cuda.max_memory_allocated() / 2**30
    if bench:
        g = np.concatenate([p[0] for p in parts])
        ok = F["clean"][qi] & np.isfinite(F["y"][qi])
        e = F["off"][qi][ok] + g[ok] - F["y"][qi][ok]
        rec = dict(
            fold=fold,
            apt=apt,
            size=S["size"],
            cap=len(k),
            ncol=len(cols),
            ein=S["ein"],
            amp=S["amp"],
            query=n,
            fit_s=round(tf, 1),
            pred_s=round(tp, 1),
            ms_per_row=round(1000 * tp / n, 2),
            peak_gib=round(mem, 2),
            rmse_clean=round(float(np.sqrt((e**2).mean())), 1),
            gpu=torch.cuda.get_device_name(0),
            host=socket.gethostname(),
        )
        print("BENCH " + json.dumps(rec), flush=True)
        open(f"{RUN}/bench.jsonl", "a").write(json.dumps(rec) + "\n")
        return rec
    G = [np.load(p) for p in parts]
    g = np.concatenate([z["g"] for z in G])
    qs = np.concatenate([z["q"] for z in G])
    assert len(g) == n and np.isfinite(g).all(), (fold, apt, i)
    ok = F["clean"][qi] & np.isfinite(F["y"][qi])
    e = F["off"][qi][ok] + g[ok] - F["y"][qi][ok]
    rm = float(np.sqrt((e**2).mean())) if ok.any() else float("nan")
    K.save_atomic(
        out,
        row=F["row"][qi],
        id=F["ids"][qi],
        g=g,
        q=qs,
        cols=cols,
        ctx_rows=F["row"][k],
        meta=json.dumps(
            dict(
                spec=S,
                inst=i,
                fold=fold,
                apt=apt,
                ctx=len(k),
                ncol=len(cols),
                colsrc=csrc,
                query=n,
                rows_predicted_now=done,
                fit_s=tf,
                pred_s=tp,
                peak_gib=mem,
                rmse_clean=rm,
                gpu=torch.cuda.get_device_name(0),
                host=socket.gethostname(),
                torch=torch.__version__,
            )
        ),
    )
    for p in parts:
        os.remove(p)
    print(
        f"  inst[{S['name']}] {fold} {apt} #{i}: ctx {len(k):,} cols {len(cols)} ({csrc}) query {n:,} rmse_clean {rm:.1f} | fit {tf:.0f}s pred {tp:.0f}s "
        f"({1000*tp/max(done,1):.1f} ms/row) peak {mem:.1f} GiB",
        flush=True,
    )
    return dict(rows=n, secs=tf + tp, peak=mem)


def trimmed_mean(A, trim):
    """A [n_inst, ...] -> per-coordinate trimmed mean over axis 0 (int(trim*n) dropped from each tail; = sdm AverageEstimators)."""
    n = A.shape[0]
    c = int(trim * n)
    B = np.sort(A, axis=0)
    return B[c : n - c].mean(0) if n - 2 * c > 0 else B.mean(0)


def agg(fold, apt):
    assert S["name"], "NAME required"
    out = f"{RUN}/results/member/{S['name']}_{fold}_{apt}.npz"
    if os.path.exists(out):
        print(f"  skip {out} (done)", flush=True)
        return None
    d = f"{RUN}/results/engine/{S['name']}/{fold}_{apt}"
    fs = [f"{d}/inst{i}.npz" for i in range(S["ninst"])]
    miss = [f for f in fs if not os.path.exists(f)]
    assert not miss, f"agg {fold} {apt}: {len(miss)} instances missing (e.g. {miss[0]})"
    Z = [np.load(f) for f in fs]
    row, ids = Z[0]["row"], Z[0]["id"]
    for z in Z[1:]:
        assert np.array_equal(z["row"], row)
    F = K.frame(apt)
    srt = np.argsort(F["row"], kind="stable")
    loc = srt[np.searchsorted(F["row"], row, sorter=srt)]  # 3 Oct fix: row = global features_v9 _row, not a frame index
    assert np.array_equal(F["row"][loc], row), f"agg {fold} {apt}: rows not in the frame"
    off = F["off"][loc]
    G = np.stack([z["g"] for z in Z])
    Q = np.stack([z["q"] for z in Z])  # [n, R], [n, R, 7]
    g = trimmed_mean(G, S["trim"])
    q = trimmed_mean(Q, S["trim"])
    taxi = off + g
    qa = q + off[:, None].astype(np.float32)
    lev = list(K.QLEV)
    spread = (qa[:, lev.index(0.90)] - qa[:, lev.index(0.10)]).astype(np.float32)
    gsd = G.std(0).astype(np.float32)
    ok = F["clean"][loc] & np.isfinite(F["y"][loc])
    y = F["y"][loc]
    rm = float(np.sqrt(((taxi - y)[ok] ** 2).mean()))
    rmi = [float(np.sqrt(((off + G[j] - y)[ok] ** 2).mean())) for j in range(len(Z))]
    metas = [json.loads(str(z["meta"])) for z in Z]
    K.save_atomic(
        out,
        row=row,
        id=ids,
        taxi=taxi,
        g=g,
        q=qa,
        qlev=np.array(K.QLEV),
        spread=spread,
        g_sd=gsd,
        g_inst=G.astype(np.float32),
        n_inst=len(Z),
        meta=json.dumps(
            dict(
                fold=fold,
                apt=apt,
                profile=dict(engine=S),
                spec=S,
                query=len(row),
                rmse_clean=rm,
                rmse_inst=rmi,
                seed=S["seed"],
                nest=S["ein"],
                size=S["size"],
                secs=sum(m["fit_s"] + m["pred_s"] for m in metas),
                peak_gib=max(m["peak_gib"] for m in metas),
                gpu=metas[0]["gpu"],
                host=socket.gethostname(),
            )
        ),
    )
    print(
        f"  agg[{S['name']}] {fold} {apt}: {len(Z)} inst, rows {len(row):,} | rmse_clean ensemble {rm:.1f} vs instances {min(rmi):.1f}..{max(rmi):.1f} (mean {np.mean(rmi):.1f}) "
        f"| spread med {np.median(spread):.0f}s g_sd med {np.median(gsd):.1f}s",
        flush=True,
    )
    return dict(rows=len(row))


def parity16():
    import glob

    ref = np.load(glob.glob(f"{PAYLOAD}/ref/parity_XKT_KM_janjul_*.npy")[0])
    apt = os.path.basename(glob.glob(f"{PAYLOAD}/ref/parity_*.npy")[0])[-8:-4]
    memlog.STATE["unit"] = f"parity16:janjul:{apt}"
    prof = dict(K.PROFILES["shared20k"], amp="fp16")
    out = f"{RUN}/results/parity/shared20k_fp16_janjul_{apt}.npz"
    if os.path.exists(out):
        os.remove(out)
    r = K.member("janjul", apt, prof=prof, out=out)
    z = np.load(out)
    o = np.argsort(z["id"])
    zt = z["taxi"][o]
    ro = np.argsort(ref[0])
    assert np.array_equal(z["id"][o], ref[0][ro])
    d = zt - ref[1][ro]
    c = np.corrcoef(zt, ref[1][ro])[0, 1]
    rms = float(np.sqrt((d**2).mean()))
    ok = rms < 2.0 and c > 0.9995
    print(
        f"PARITY16 {'PASS' if ok else 'CHECK'} (shared20k profile, fp16 vs the fp32 GPU workstation file): {len(d):,} rows | RMS diff {rms:.3f} s | max |diff| {np.abs(d).max():.2f} s | corr {c:.6f} "
        f"| {r['rows']/r['secs']:.0f} query rows/s, peak {r['peak']:.1f} GiB",
        flush=True,
    )
    open(f"{RUN}/parity16.txt", "w").write(
        f"{'PASS' if ok else 'CHECK'} rms {rms:.3f} max {np.abs(d).max():.2f} corr {c:.6f} rows_s {r['rows']/r['secs']:.0f} peak {r['peak']:.1f}\n"
    )
    return r


def run(spec):
    p = spec.split(":")
    if p[0] == "inst":
        return inst(p[1], p[2], p[3])
    if p[0] == "bench":
        return inst(p[1], p[2], 0, bench=True)
    if p[0] == "agg":
        return agg(p[1], p[2])
    if p[0] == "parity16":
        return parity16()
    raise SystemExit(f"bad unit {spec}")


if __name__ == "__main__":
    a = sys.argv[1:]
    claim = "--claim" in a
    a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list":
        a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(
        f"engine spec {S} | QMAX {QMAX} | SDM_CHUNK_MEMORY_FRACTION {os.environ['SDM_CHUNK_MEMORY_FRACTION']} | {os.environ['PYTORCH_CUDA_ALLOC_CONF']} | memlog {E('MEMLOG')}",
        flush=True,
    )
    t0 = time.time()
    n = 0
    for spec in a:
        if claim:
            lk = f"{RUN}/locks/{S['name']}_{spec.replace(':', '_')}"
            os.makedirs(f"{RUN}/locks", exist_ok=True)
            try:
                os.mkdir(lk)
            except FileExistsError:
                continue
            open(f"{lk}/owner", "w").write(f"{socket.gethostname()} {os.getpid()}\n")
        try:
            r = run(spec)
        except BaseException:
            if claim:
                import shutil

                shutil.rmtree(lk, ignore_errors=True)
            raise
        n += r is not None
    print(f"KM_ENGINE_DONE units run {n} | {time.time()-t0:.0f}s | {memlog.line()[0]}", flush=True)
