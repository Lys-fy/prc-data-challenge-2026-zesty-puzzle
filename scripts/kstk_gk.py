#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Collects the Kumo chain-stacker run (kumo_gpu/cluster/km_adapters_kstk.py, NAME KSTK6<chain>) into the gK files
read by scripts/kstk_stack.py / kstk_assemble.py. Inputs: positional <RUN dir> and chain names (default Kq NNA3), env
KSTK_DIR (default kumo_gpu/data/payload/stk6), KSTK_NAME (default KSTK6), reading
<RUN>/results/engine/KSTK6<chain>/<unit>_<APT>/inst<i>.npz. Writes data/oof/KSTK6<chain>_<fold>.npy = [id, taxi] (6
chain folds) and data/preds/KSTK6<chain>P.npy = [id, taxi] (matched ranking rows), plus md5 lines on stdout.

No dependence on the box's agg unit; re-aggregates exactly as km_engine.agg (TRIM-trimmed mean of the instance points
g over instances), offset = the dump's v41 (fold) / v (prod) looked up by MVT id.
VALIDATION per cell: each instance's clean-row RMSE recomputed here must equal the rmse_clean in its meta (|diff| < 0.05 s; prod: NaN == NaN), and the
cell's query ids must be exactly the dump's rows of that airport. Any failure -> nothing written for that chain.
Also prints the stack-level INFO (never deciding): per fold mse of cf(v41 + clip(gM)) vs cf(v41 + clip(0.7 gM + 0.3 gK)), corr(gK, gM), sd.
usage: kstk_gk.py <RUN dir> [Kq NNA3]      env KSTK_DIR (default kumo_gpu/data/payload/stk6), KSTK_NAME (default KSTK6)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, json, hashlib, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
RUN = sys.argv[1]
CH = sys.argv[2:] or ["Kq", "NNA3"]
KD = os.environ.get("KSTK_DIR", f"{ROOT}/kumo_gpu/data/payload/stk6")
NM = os.environ.get("KSTK_NAME", "KSTK6")
FO6 = ["months39", "months28", "months410", "months612", "months511", "janjul"]
UNITS = FO6 + ["prod"]


def tm(A, trim):
    n = A.shape[0]
    c = int(trim * n)
    B = np.sort(A, axis=0)
    return B[c : n - c].mean(0) if n - 2 * c > 0 else B.mean(0)


def cf(v, lo):
    v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606))
    return np.maximum(v, 60.0)


md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
for c in CH:
    name = f"{NM}{c}"
    out = {}
    ok_all = True
    for t in UNITS:
        z = np.load(f"{KD}/stk_{c}_{t}.npz", allow_pickle=False)
        mid, apt, v41 = z["mid"].astype(float), z["apt"], z["v41"].astype(float)
        y = z["y"].astype(float) if t != "prod" else np.full(len(mid), np.nan)
        taxi = np.full(len(mid), np.nan)
        o = np.argsort(mid)
        for a in sorted(set(apt.tolist())):
            d = f"{RUN}/results/engine/{name}/{t}_{a}"
            fs = sorted(glob.glob(f"{d}/inst*.npz"))
            if not fs:
                print(f"{name} {t} {a}: NO instance files in {d}")
                ok_all = False
                continue
            Z = [np.load(f) for f in fs]
            metas = [json.loads(str(q["meta"])) for q in Z]
            S = metas[0]["spec"]
            if len(Z) != S["ninst"]:
                print(f"{name} {t} {a}: {len(Z)}/{S['ninst']} instances")
                ok_all = False
                continue
            ids = Z[0]["id"].astype(float)
            for q in Z[1:]:
                assert np.array_equal(q["id"].astype(float), ids)
            sel = np.where(apt == a)[0]
            if not np.array_equal(np.sort(ids), np.sort(mid[sel])):
                print(f"{name} {t} {a}: query ids != dump rows")
                ok_all = False
                continue
            k = o[np.searchsorted(mid[o], ids)]
            off = v41[k]
            yy = y[k]
            okr = np.isfinite(yy)
            G = np.stack([q["g"] for q in Z])
            rmi = [
                float(np.sqrt(((off + G[j] - yy)[okr] ** 2).mean())) if okr.any() else float("nan")
                for j in range(len(Z))
            ]
            dv = max(
                0.0 if (np.isnan(r) and np.isnan(m_["rmse_clean"])) else abs(r - m_["rmse_clean"])
                for r, m_ in zip(rmi, metas)
            )
            if not dv < 0.05:
                print(f"{name} {t} {a}: VALIDATION FAIL max |rmse_inst - meta| {dv:.3f}")
                ok_all = False
                continue
            taxi[k] = off + tm(G, S["trim"])
        if np.isnan(taxi).any():
            print(f"{name} {t}: {int(np.isnan(taxi).sum()):,} rows without gK")
            ok_all = False
            continue
        out[t] = (mid, taxi, v41, y, z["gM"].astype(float), z["lobt"].astype(float))
    if not ok_all:
        print(f"kstk_assemble_gk {c}: FAILED, nothing written")
        continue
    for t, (mid, taxi, v41, y, gM, lobt) in out.items():
        p = f"data/preds/{name}P.npy" if t == "prod" else f"data/oof/{name}_{t}.npy"
        np.save(p, np.vstack([mid, taxi]))
        gK = taxi - v41
        info = f"corr(gK, gM) {np.corrcoef(gK, gM)[0, 1]:.3f} sd gK {gK.std():.1f} gM {gM.std():.1f} |gK|>150 {np.mean(np.abs(gK) > 150):.4f}"
        if t != "prod":
            ok = np.isfinite(y)
            b = cf(v41 + np.clip(gM, -150, 150), lobt)
            kb = cf(v41 + np.clip(0.7 * gM + 0.3 * gK, -150, 150), lobt)
            info += f" | INFO stack mse gM {np.mean((b - y)[ok] ** 2):.1f} KB {np.mean((kb - y)[ok] ** 2):.1f} d {np.mean((kb - y)[ok] ** 2) - np.mean((b - y)[ok] ** 2):+.1f}"
        print(f"{md5(p)}  {p}  rows {len(mid):,} | {info}", flush=True)
    print(f"kstk_assemble_gk {c}: OK (6 folds + prod)", flush=True)
