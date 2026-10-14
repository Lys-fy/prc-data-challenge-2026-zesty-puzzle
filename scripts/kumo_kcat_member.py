#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""KCAT8 stage C: K250 engine + 8 raw categorical columns (C8) appended to every instance.
CATMODE=cat  -> passed as declared categoricals (TableTensor categorical block: AlignCategories + AddCategoryCounts(50) + ToNumerical)
CATMODE=float-> the same codes as float columns (null -> NaN) = our all-float practice. Everything else = km_engine (K250 spec via env).
Usage: python kumo_kcat_member.py <units...> | --from-list f   (same units as km_engine.py).  CATSMOKE=1: synthetic CPU check of the cat path only.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np

C8 = [
    "STAND_mvt",
    "RUNWAY_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "ADES_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
]
if os.environ.get("CATCOLS"):
    C8 = os.environ["CATCOLS"].split(
        ","
    )  # stage D1: C8 + FLIGHT_mvt, prev_wtc, SCHED_TOD15, STANDxSH (needs CODES built with EXT=1)
MODE = os.environ.get("CATMODE", "cat")
NC = len(C8)
import torch
from sdm import TableTensor
from sdm.stype import Stype
from sdm.tensor.categorical import CategoricalTensor


def table(x):
    """x [R, D_num + NC] float tensor whose last NC columns are integer codes (-1 = null) -> TableTensor."""
    xn, xc = x[:, :-NC], x[:, -NC:].round().to(torch.int64)
    return TableTensor(
        columns={Stype.numerical: [f"n{i}" for i in range(xn.shape[1])], Stype.categorical: C8},
        numerical=xn,
        categorical=CategoricalTensor.from_tensor(xc).to(x.device),
    )


if os.environ.get("CATSMOKE"):
    import sdm, time

    g = torch.Generator().manual_seed(0)
    n = 3000
    xn = torch.randn(n + 500, 20, generator=g)
    cat = torch.randint(0, 80, (n + 500, NC), generator=g).float()
    eff = torch.randn(80, generator=g) * 30
    y = (20 * xn[:, 0] + eff[cat[:, 0].long()] + 5 * torch.randn(n + 500, generator=g))[:, None]
    x = torch.cat([xn, cat], 1)
    m = sdm.models.KumoTabular(task="regression", size="medium", device="cpu")
    for mode in ("cat", "float"):
        t = time.time()
        xc, xq = (table(x[:n]), table(x[n:])) if mode == "cat" else (x[:n], x[n:])
        m.fit(xc, y[:n], num_estimators=1, estimator_batch_size=1, generator=torch.Generator().manual_seed(0))
        p = m.predict(xq)
        q = (p.numerical if hasattr(p, "numerical") else p).float()
        while q.dim() > 2:
            q = q[0]
        print(
            f"CATSMOKE {mode}: rmse {((q.mean(-1) - y[n:, 0]) ** 2).mean().sqrt().item():.2f} ({time.time()-t:.0f}s)",
            flush=True,
        )
    sys.exit(0)
if (
    os.environ.get("DEV") == "cpu"
):  # CPU fallback (no CUDA): km_task/km_engine/memlog call torch.cuda.* for logging only
    torch.set_num_threads(int(os.environ.get("PRC_THREADS", "4")))
    for _n, _f in dict(
        empty_cache=lambda *a, **k: None,
        reset_peak_memory_stats=lambda *a, **k: None,
        synchronize=lambda *a, **k: None,
        get_device_name=lambda *a, **k: "cpu",
        get_device_capability=lambda *a, **k: (0, 0),
        max_memory_allocated=lambda *a, **k: 0,
        memory_allocated=lambda *a, **k: 0,
        memory_reserved=lambda *a, **k: 0,
    ).items():
        setattr(torch.cuda, _n, _f)
sys.path.insert(
    0, os.environ.get("KMCODE", os.path.join(os.environ.get("PRC_ROOT", os.getcwd()), "kumo_gpu", "cluster"))
)
import polars as pl
import km_engine as KE

K = KE.K
D = KE.D
_frame0 = K.frame
_codes = {}


def codes(apt):
    if apt not in _codes:
        v9 = pl.read_parquet(
            os.path.join(os.environ.get("PRC_ROOT", os.getcwd()), "data", "features_v9.parquet"),
            columns=["ADEP_mvt", "MVT_ID_mvt"] + C8,
        ).with_row_index("_row")
        _codes.clear()
        _codes[apt] = v9
    return _codes[apt]


def frame(apt):
    new = apt not in K._frames
    F = _frame0(apt)
    if new or F["X"].shape[1] == D:
        cd = os.environ.get("CODES")
        if cd:  # precomputed by kumo_kcat_codes.py (same rule as below)
            z = pl.read_parquet(f"{cd}/kcat_codes_{apt}.parquet")
            assert np.array_equal(z["_row"].to_numpy(), F["row"]), apt
            assert np.array_equal(z["MVT_ID_mvt"].cast(pl.Float64).to_numpy(), F["ids"]), apt
            Cm = np.stack([z[c].to_numpy().astype(np.float32) for c in C8], 1)
        else:
            v9 = codes(apt)
            sub = v9[F["row"].tolist()]
            assert np.array_equal(
                sub["MVT_ID_mvt"].cast(pl.Float64).to_numpy(), F["ids"]
            ), f"{apt}: _row is not the features_v9 row index"
            assert (sub["ADEP_mvt"] == apt).all()
            cols = []
            for c in C8:
                s = sub[c].cast(pl.Utf8)
                vals = np.sort(s.drop_nulls().unique().to_numpy())
                lut = {v: i for i, v in enumerate(vals)}
                cols.append(np.array([lut[v] if v is not None else -1 for v in s.to_list()], dtype=np.float32))
            Cm = np.stack(cols, 1)
        if MODE == "float":
            Cm = np.where(Cm < 0, np.nan, Cm).astype(np.float32)
        F["X"] = np.hstack([F["X"][:, :D], Cm])
        print(
            f"  C8 appended for {apt} ({MODE}): cardinalities {[int(np.nanmax(Cm[:, j])) + 1 for j in range(NC)]}",
            flush=True,
        )
    return F


K.frame = frame
_columns0 = KE.columns


def columns(i, fold):
    c, src = _columns0(i, fold)
    return np.concatenate([c, np.arange(D, D + NC)]).astype(int), src + f"+C8{MODE}"


KE.columns = columns
if (
    KE.S["ctxmode"] == "season1"
):  # stage D4: same-airport context restricted to months within circular distance 1 of the held-out months (whole subset)
    _ctx0 = KE.context

    def context(F, ti, i, fold):
        held = K.HOLD[fold] or [1]
        m = F["m"][ti].astype(int)
        d = np.min([np.minimum(np.abs(m - h), 12 - np.abs(m - h)) for h in held], axis=0)
        sub = ti[d <= 1]
        if i == 0:
            print(
                f"  season1 context {fold}: {len(sub):,} of {len(ti):,} pool rows (months {sorted(set(m[d <= 1].tolist()))})",
                flush=True,
            )
        return sub if len(sub) <= KE.S["cap"] else _ctx0(F, sub, i, fold)

    KE.context = context
if MODE == "cat":
    _fit0 = KE.fit

    def fit(xc, yc, i):
        mdl = K.model(KE.S["size"])
        gen = torch.Generator(device=KE.DEV).manual_seed(KE.S["seed"] * 1000 + i)
        with K.amp_ctx(KE.S["amp"]):
            mdl.fit(
                table(torch.as_tensor(xc, device=KE.DEV)),
                torch.as_tensor(yc.astype(np.float32)[:, None], device=KE.DEV),
                num_estimators=KE.S["ein"],
                estimator_batch_size=1,
                generator=gen,
            )
        return mdl

    KE.fit = fit
    _inst0 = KE.inst

    def inst(*a, **k):
        _pred = KE.predict

        def predict(mdl, xq, ch0=None):
            p0 = mdl.predict
            mdl.predict = lambda xb: p0(table(xb))
            try:
                return _pred(mdl, xq, ch0)
            finally:
                mdl.predict = p0

        KE.predict = predict
        try:
            return _inst0(*a, **k)
        finally:
            KE.predict = _pred

    KE.inst = inst
if __name__ == "__main__":
    a = sys.argv[1:]
    claim = "--claim" in a
    a = [x for x in a if x != "--claim"]
    if a and a[0] == "--from-list":
        a = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
    print(
        f"kumo_kcat_member CATMODE {MODE} spec {KE.S} PAYLOAD {K.PAYLOAD} RUN {K.RUN} CODES {os.environ.get('CODES')}",
        flush=True,
    )
    for spec in a:
        if claim:  # one mkdir lock per unit (several workers); run agg units afterwards in one process
            lk = f"{K.RUN}/locks/{KE.S['name']}_{spec.replace(':', '_')}"
            os.makedirs(f"{K.RUN}/locks", exist_ok=True)
            try:
                os.mkdir(lk)
            except FileExistsError:
                continue
        try:
            KE.run(spec)
        except BaseException:
            if claim:
                import shutil

                shutil.rmtree(lk, ignore_errors=True)
            raise
    print("kcat8_cat_done", flush=True)
