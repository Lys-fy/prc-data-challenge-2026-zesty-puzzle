#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""GPU worker for the Kumo Tabular jobs on a cluster or a GPU machine. One process runs a LIST of task units with resume;
a unit whose output file exists is skipped, outputs are written atomically (tmp + os.replace), so a killed job just restarts.

ARM-M PROFILES (env PROFILE; library pinned at sdm ee6acd402, weights v1.0.0 = bd7fa122):
  shared20k    medium, 20k context SHARED by 4 estimators, proportional month x adsb_linked stratified draw (as in kumo_km_member.py), fp32.
          = the GPU workstation runs. Used by the PARITY gate; reproduces the workstation member exactly.        NAME kumo_member_screen_km
  m20e8   PRIMARY. medium, E=8 estimators, each with its OWN disjoint 20k context slice (vendor adapter rule: consecutive slices of
          concatenated seeded permutations of the clean training pool -> x_context [E, cap, D], fit(num_estimators=None)), fp16 autocast,
          vendor default recipe (latin column shuffles, numeric-transform round robin, FlipSign) and readout (sorted quantiles,
          20 %-trimmed mean over members, mean of the 999 quantiles).                                                NAME kumo_member_screen_km8
  m50e8   medium, E=8, disjoint 50k slices, fp16.                                                                     NAME kumo_member_screen_km50
  l50e16  large,  E=16, disjoint 50k slices, fp16 (vendor default E for large; H200 141 GB safest).                  NAME kumo_member_screen_kl50
  l50e8   large,  E=8,  disjoint 50k slices, fp16 (the 80 GB fallback; the brief's "E = 4-8" large form).             NAME kumo_member_screen_kl50e8
  Overrides: SIZE, CAP, NEST, AMP (fp16|bf16|off), NAME. Safe-memory settings are always on: estimator_batch_size=1,
  SDM_CHUNK_MEMORY_FRACTION=0.02, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, peak GPU memory logged per unit.

DISTILLATION OUTPUT ( #2, Nums AI 2610.01435): every member unit stores, besides the point prediction, the teacher's quantile
summary (q05 q10 q25 q50 q75 q90 q95 of the ensembled 999-quantile vector). The 6 CV folds put every 2025 row in exactly one held-out fold
whose context never contains that row's months (OOF by month), and `prod` covers the 2026 rows, so collect.py can write a single teacher table
for a CPU LightGBM student. SYNTH=<frac> (default 0) also labels synthetic query rows (cluster/synth.py, regenerable from the frame + seed).

Task units (see tasks.py):
  member <fold> <APT>        query = ALL rows of the fold (janjul|months28|months39|months410|months511|months612) at that airport, or for
                             `prod` all 2026 ranking rows (context pool = all 12 training months).  -> results/member/<NAME>_<fold>_<APT>.npz
  t2x <chain> <fold> <APT>   the T2x screen exactly as pre-declared (kumo_t2x_screen2.py `kumo`: medium, shared random 20k, 4 est, fp32; the
                             screen rule was pre-declared on that construction, so the t2x step does not take PROFILE). -> results/t2x/gK_*.npz
  parity                     profile shared20k on janjul x LSZH vs the GPU workstation output (payload ref/), then one unit of the active PROFILE on the same
                             cell for timing + peak memory; prints rows/s and the projected GPU-h for the whole job.
usage: python km_task.py <unit-spec> ...     | python km_task.py --from-list list.txt [--claim]
env:   PAYLOAD, RUN, HF_HOME ($PAYLOAD/vendor/hf), PROFILE (m20e8), SEED 0, CHUNK 4096 (query rows per predict; halved on OOM, chunk-invariant)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, json, socket, math, numpy as np

os.environ.setdefault("SDM_CHUNK_MEMORY_FRACTION", "0.02")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
E = os.environ.get
PAYLOAD = os.path.abspath(
    E("PAYLOAD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "payload"))
)
RUN = os.path.abspath(E("RUN", "run"))
os.environ.setdefault("HF_HOME", f"{PAYLOAD}/vendor/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
PROFILES = {
    "shared20k": dict(size="medium", cap=20000, nest=4, amp="off", ctx="shared_strat", name="kumo_member_screen_km"),
    "m20e8": dict(size="medium", cap=20000, nest=8, amp="fp16", ctx="disjoint", name="kumo_member_screen_km8"),
    "m50e8": dict(size="medium", cap=50000, nest=8, amp="fp16", ctx="disjoint", name="kumo_member_screen_km50"),
    "l50e16": dict(size="large", cap=50000, nest=16, amp="fp16", ctx="disjoint", name="kumo_member_screen_kl50"),
    "l50e8": dict(size="large", cap=50000, nest=8, amp="fp16", ctx="disjoint", name="kumo_member_screen_kl50e8"),
}  # 80 GB fallback for l50e16
PNAME = E("PROFILE", "m20e8")
P = dict(PROFILES[PNAME])
for k, ev, f in (
    ("size", "SIZE", str),
    ("cap", "CAP", int),
    ("nest", "NEST", int),
    ("amp", "AMP", str),
    ("name", "NAME", str),
):
    if E(ev):
        P[k] = f(E(ev))
SEED = int(E("SEED", "0"))
CHUNK0 = int(E("CHUNK", "4096"))
DEV = E("DEV", "cuda")
SYNTH = float(E("SYNTH", "0"))
HOLD = {
    "janjul": [1, 7],
    "months28": [2, 8],
    "months39": [3, 9],
    "months410": [4, 10],
    "months511": [5, 11],
    "months612": [6, 12],
    "prod": None,
}
FO4 = ["months39", "months28", "months410", "janjul"]
QLEV = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QIDX = [int(round(q * 1000)) - 1 for q in QLEV]
import torch, sdm, polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth

try:
    import km_adapters

    _ADAPT = km_adapters.apply()
except Exception as _ex:
    _ADAPT = {"error": str(_ex)}
    print(f"km_adapters not applied: {_ex}", flush=True)
try:
    import memlog

    memlog.start()  # MEMLOG=<file> -> one memory line per minute + host-memory guard (MEMGUARD_GIB)
except Exception as _ex:
    print(f"memlog not started: {_ex}", flush=True)
torch.set_num_threads(int(E("PRC_THREADS", "4")))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
FEATS = (
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "km_features_412.txt")).read().split()
)
assert len(FEATS) == 412
_models = {}


def model(size):
    if size not in _models:
        _models.clear()
        torch.cuda.empty_cache()
        _models[size] = sdm.models.KumoTabular(task="regression", size=size, device=DEV)
        print(
            f"[{socket.gethostname()}] torch {torch.__version__} cuda {torch.version.cuda} sdm {getattr(sdm, '__version__', '?')} {torch.cuda.get_device_name(0)} "
            f"cap {torch.cuda.get_device_capability(0)} | model {size} | SDM_CHUNK_MEMORY_FRACTION {os.environ['SDM_CHUNK_MEMORY_FRACTION']}",
            flush=True,
        )
    return _models[size]


def amp_ctx(amp):
    if amp == "off":
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


_q3d = {"ok": None}


def kumo(xc, yc, xq, size, nest, amp, per_est):
    """xc [R, D] shared (fit num_estimators=nest) or [E, cap, D] per estimator (fit num_estimators=None). Returns (mean of 999 q, summary q)."""
    mdl = model(size)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    with amp_ctx(amp):
        if per_est:
            mdl.fit(
                torch.as_tensor(xc, device=DEV),
                torch.as_tensor(yc.astype(np.float32)[..., None], device=DEV),
                num_estimators=None,
                estimator_batch_size=1,
                generator=gen,
            )
        else:
            mdl.fit(
                torch.as_tensor(xc, device=DEV),
                torch.as_tensor(yc.astype(np.float32)[:, None], device=DEV),
                num_estimators=nest,
                estimator_batch_size=1,
                generator=gen,
            )
    out, qs, i, ch = [], [], 0, CHUNK0
    while i < len(xq):
        try:
            xb = torch.as_tensor(xq[i : i + ch], device=DEV)
            with amp_ctx(amp):
                if per_est and _q3d["ok"] is not False:  # vendor adapter: query expanded over the estimator axis
                    try:
                        p = mdl.predict(xb.unsqueeze(0).expand(nest, *xb.shape))
                        _q3d["ok"] = True
                    except (ValueError, RuntimeError, AssertionError) as ex:
                        if _q3d["ok"] is True or isinstance(ex, torch.cuda.OutOfMemoryError):
                            raise
                        _q3d["ok"] = False
                        print(
                            f"    3-D query not accepted ({type(ex).__name__}: {str(ex)[:120]}) -> 2-D query",
                            flush=True,
                        )
                        p = mdl.predict(xb)
                else:
                    p = mdl.predict(xb)
            q = (p.numerical if hasattr(p, "numerical") else p).float()
            while q.dim() > 2:
                assert q.shape[0] == 1, q.shape
                q = q[0]  # reduced output [R, 999]
            assert q.shape[-1] == 999 and q.shape[0] == len(xb), q.shape
            out.append(q.mean(-1).cpu().numpy())
            qs.append(q[:, QIDX].cpu().numpy())
            i += ch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            assert ch > 64, "OOM even at 64 query rows: the context itself does not fit"
            ch //= 2
            print(f"    OOM -> CHUNK {ch}", flush=True)
    mdl.clear()
    return np.concatenate(out), np.concatenate(qs).astype(np.float32)


def save_atomic(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


_frames = {}


def frame(apt):
    if apt not in _frames:
        _frames.clear()
        d = pl.read_parquet(f"{PAYLOAD}/member/km_{apt}.parquet")
        X = d.select(FEATS).to_numpy().astype(np.float32)  # nulls -> NaN (as kumo_km_member.py)
        _frames[apt] = dict(
            X=X,
            y=d["meta__y"].to_numpy().astype(float),
            m=d["meta__m"].to_numpy(),
            rank=d["meta__is_rank"].to_numpy() == 1,
            off=np.nan_to_num(d["meta__mvt_aobt3"].to_numpy().astype(float), nan=0.0),
            clean=d["meta__fb_pru"].to_numpy() != 1,
            lk=d["meta__adsb_linked"].fill_null(0).to_numpy() > 0,
            ids=d["meta__MVT_ID_mvt"].to_numpy().astype(float),
            row=d["_row"].to_numpy(),
        )
    return _frames[apt]


def contexts(F, ti, prof):
    """shared20k: kumo_km_member.py's shared stratified draw. disjoint: vendor adapter rule (each estimator its own consecutive slice of concatenated seeded permutations)."""
    rng = np.random.default_rng(SEED)
    cap, nest = prof["cap"], prof["nest"]
    if prof["ctx"] == "shared_strat":
        frac = min(1.0, cap / len(ti))
        cell = F["m"][ti] * 2 + F["lk"][ti]
        pick = []
        for c in np.unique(cell):
            ci = ti[cell == c]
            k = int(round(frac * len(ci)))
            pick.append(rng.choice(ci, min(k, len(ci)), replace=False))
        return np.concatenate(pick), False
    if len(ti) <= cap:
        return ti, False  # whole pool fits: one shared context, nest estimators (adapter does the same)
    reps = math.ceil(nest * cap / len(ti))
    perm = np.concatenate([rng.permutation(ti) for _ in range(reps)])[: nest * cap]
    return perm.reshape(nest, cap), True


def member(fold, apt, prof=None, out=None):
    prof = prof or P
    out = out or f"{RUN}/results/member/{prof['name']}_{fold}_{apt}.npz"
    if os.path.exists(out):
        print(f"  skip {out} (done)", flush=True)
        return None
    try:
        memlog.STATE["unit"] = f"member:{prof['name']}:{fold}:{apt}"
    except Exception:
        pass
    F = frame(apt)
    m, y, rank, clean = F["m"], F["y"], F["rank"], F["clean"]
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    if HOLD[fold] is None:
        val = rank
        trn = ~rank & clean & np.isfinite(y)
    else:
        val = ~rank & np.isin(m, HOLD[fold])
        trn = ~rank & ~np.isin(m, HOLD[fold]) & clean & np.isfinite(y)
    ti = np.where(trn)[0]
    qi = np.where(val)[0]
    k, per_est = contexts(F, ti, prof)
    xq = F["X"][qi]
    srow = sbase = None
    if SYNTH > 0:
        xs, sbase, srow = synth.make(F["X"], qi, SYNTH, seed=SEED, key=f"{fold}:{apt}")
        xq = np.vstack([xq, xs])
    g, qs = kumo(F["X"][k], y[k] - F["off"][k], xq, prof["size"], prof["nest"], prof["amp"], per_est)
    assert np.isfinite(g).all(), (fold, apt)
    n = len(qi)
    taxi = F["off"][qi] + g[:n]
    ok = clean[qi] & np.isfinite(y[qi])
    e = taxi[ok] - y[qi][ok]
    dt = time.time() - t1
    rm = float(np.sqrt((e**2).mean())) if ok.any() else float("nan")
    mem = torch.cuda.max_memory_allocated() / 2**30
    extra = (
        {}
        if SYNTH <= 0
        else dict(synth_base_row=F["row"][sbase], synth_g=g[n:], synth_q=qs[n:], synth_off=F["off"][sbase])
    )
    save_atomic(
        out,
        row=F["row"][qi],
        id=F["ids"][qi],
        taxi=taxi,
        g=g[:n],
        q=qs[:n] + F["off"][qi][:, None].astype(np.float32),
        qlev=np.array(QLEV),
        ctx_md5=__import__("hashlib").md5(np.ascontiguousarray(F["row"][k]).tobytes()).hexdigest(),
        **extra,
        meta=json.dumps(
            dict(
                fold=fold,
                apt=apt,
                profile=prof,
                ctx_shape=list(np.shape(k)),
                per_est=per_est,
                query=n,
                synth=len(xq) - n,
                rmse_clean=rm,
                secs=dt,
                peak_gib=mem,
                gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,
                seed=SEED,
                host=socket.gethostname(),
                q3d=_q3d["ok"],
                chunk_frac=os.environ["SDM_CHUNK_MEMORY_FRACTION"],
                adapters=_ADAPT,
            )
        ),
    )
    print(
        f"  member[{prof['name']}] {fold} {apt}: ctx {tuple(np.shape(k))} query {n:,}{f' +synth {len(xq)-n:,}' if SYNTH > 0 else ''} clean-row rmse {rm:.1f} "
        f"| {dt:.0f}s ({len(xq)/dt:.0f} rows/s) peak {mem:.1f} GiB",
        flush=True,
    )
    return dict(rows=len(xq), secs=dt, peak=mem)


_t2x = {}


def t2x(chain, fold, apt):  # = kumo_t2x_screen2.py `kumo` (pre-declared): medium, shared random 20k, 4 estimators, fp32
    out = f"{RUN}/results/t2x/gK_{chain}_{fold}_{apt}.npz"
    if os.path.exists(out):
        print(f"  skip {out} (done)", flush=True)
        return None
    if chain not in _t2x:
        _t2x.clear()
        Z = {t: dict(np.load(f"{PAYLOAD}/t2x/t2x_{chain}_{t}.npz")) for t in FO4}
        for z in Z.values():
            z["X"] = np.nan_to_num(z["X"], nan=0.0, posinf=0.0, neginf=0.0)
            z["r"] = np.clip(z["y"] - z["v41"], -1800, 1800)
        _t2x[chain] = Z
    Z = _t2x[chain]
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    xc = np.vstack([Z[x]["X"][Z[x]["apt"] == apt] for x in FO4 if x != fold])
    yc = np.concatenate([Z[x]["r"][Z[x]["apt"] == apt] for x in FO4 if x != fold])
    k = np.random.default_rng(SEED).choice(len(yc), min(20000, len(yc)), replace=False)
    qi = np.where(Z[fold]["apt"] == apt)[0]
    gK, _ = kumo(xc[k], yc[k], Z[fold]["X"][qi], "medium", 4, "off", False)
    assert np.isfinite(gK).all()
    dt = time.time() - t1
    mem = torch.cuda.max_memory_allocated() / 2**30
    save_atomic(
        out,
        pos=qi,
        gK=gK,
        meta=json.dumps(
            dict(
                chain=chain,
                fold=fold,
                apt=apt,
                ctx=len(k),
                query=len(qi),
                secs=dt,
                peak_gib=mem,
                gpu=torch.cuda.get_device_name(0),
            )
        ),
    )
    print(
        f"  t2x {chain} {fold} {apt}: ctx {len(k):,} query {len(qi):,} sd(gK) {gK.std():.1f} | {dt:.0f}s ({len(qi)/dt:.0f} rows/s) peak {mem:.1f} GiB",
        flush=True,
    )
    return dict(rows=len(qi), secs=dt)


# totals for the projections: member query rows (6 folds + prod), units, t2x query rows
N_Q, N_UNITS, N_T2X = 2_429_888, 70, 2 * 1_373_000


def parity():
    import glob

    ref = np.load(glob.glob(f"{PAYLOAD}/ref/parity_XKT_KM_janjul_*.npy")[0])
    apt = os.path.basename(glob.glob(f"{PAYLOAD}/ref/parity_*.npy")[0])[-8:-4]
    os.makedirs(f"{RUN}/results/parity", exist_ok=True)
    out = f"{RUN}/results/parity/shared20k_janjul_{apt}.npz"
    if os.path.exists(out):
        os.remove(out)
    r = member("janjul", apt, prof=PROFILES["shared20k"], out=out)
    z = np.load(out)
    o = np.argsort(z["id"])
    zi, zt = z["id"][o], z["taxi"][o]
    ro = np.argsort(ref[0])
    assert np.array_equal(zi, ref[0][ro]), "parity: row set differs from the GPU workstation output"
    d = zt - ref[1][ro]
    c = np.corrcoef(zt, ref[1][ro])[0, 1]
    ok = np.sqrt((d**2).mean()) < 2.0 and c > 0.9995
    print(
        f"PARITY {'PASS' if ok else 'CHECK'} (profile shared20k vs GPU workstation file): {len(d):,} rows | RMS diff {np.sqrt((d ** 2).mean()):.3f} s | max |diff| {np.abs(d).max():.2f} s | corr {c:.6f}",
        flush=True,
    )
    per_row_ref = r["secs"] / (4 * (r["rows"] + 20000))
    print(
        f"RATE shared20k: {r['rows']/r['secs']:.0f} query rows/s, peak {r['peak']:.1f} GiB -> t2x (fp32, ~3.3x cheaper per row) ~{N_T2X / (3.3 * r['rows'] / r['secs']) / 3600:.1f} GPU-h",
        flush=True,
    )
    if PNAME != "shared20k":
        out2 = f"{RUN}/results/parity/{P['name']}_janjul_{apt}.npz"
        if os.path.exists(out2):
            os.remove(out2)
        r2 = member("janjul", apt, prof=P, out=out2)
        z2 = np.load(out2)
        c2 = np.corrcoef(z2["taxi"][np.argsort(z2["id"])], zt)[0, 1]
        tot = (
            r2["secs"] * (N_Q + N_UNITS * P["cap"]) / (r2["rows"] + P["cap"]) / 3600
        )  # unit time scales with (query + per-estimator context) rows
        print(
            f"RATE {PNAME} ({P}): {r2['rows']/r2['secs']:.0f} query rows/s, peak {r2['peak']:.1f} GiB, corr with shared20k {c2:.4f} -> projected arm M {tot:.1f} GPU-h "
            f"(+~10 % load overhead){' x (1+SYNTH)' if SYNTH > 0 else ''}",
            flush=True,
        )
    return None


def run(spec):
    p = spec.split(":")
    if p[0] == "member":
        return member(p[1], p[2])
    if p[0] == "t2x":
        return t2x(p[1], p[2], p[3])
    if p[0] == "parity":
        return parity()
    raise SystemExit(f"bad unit {spec}")


if __name__ == "__main__":
    a = sys.argv[1:]
    claim = "--claim" in a
    a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list":
        a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(
        f"profile {PNAME}: {P} | SYNTH {SYNTH} | SDM_CHUNK_MEMORY_FRACTION {os.environ['SDM_CHUNK_MEMORY_FRACTION']} | {os.environ['PYTORCH_CUDA_ALLOC_CONF']}",
        flush=True,
    )
    t0 = time.time()
    n = 0
    for spec in a:
        if (
            claim
        ):  # multi-worker on one box: first worker to mkdir the lock owns the unit (a stale lock of a dead worker is removed by hand)
            lk = f"{RUN}/locks/{P['name']}_{spec.replace(':', '_')}"
            os.makedirs(f"{RUN}/locks", exist_ok=True)
            try:
                os.mkdir(lk)
            except FileExistsError:
                continue
            open(f"{lk}/owner", "w").write(
                f"{socket.gethostname()} {os.getpid()} {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
            )
        try:
            r = run(spec)
        except BaseException:
            if claim:
                import shutil

                shutil.rmtree(lk, ignore_errors=True)  # release the unit so another worker / the next run retries it
            raise
        n += r is not None
    print(f"KM_TASK_DONE units run {n} | {time.time()-t0:.0f}s", flush=True)
