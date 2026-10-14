#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""RERUN_PATCH 3 Oct: OFF-BOX re-aggregation of km_engine instance files (results/engine/<NAME>/<fold>_<APT>/inst<i>.npz) into member files
(results/member/<NAME>_<fold>_<APT>.npz, km_engine.agg's format). Reason: the night run's agg units crashed (km_engine.py agg indexed the
airport frame with global _row ids). numpy + polars only (no torch/sdm): the frame meta (offset, y, clean) is read from the payload parquet,
for km_resid arms (spec carries resid_l) the offset is the fold's L table exactly as km_resid.patch_frame (rows without L keep mvt_aobt3, clean=False).
VALIDATION per cell: each instance's RMSE on clean rows, recomputed here from the parquet offsets, must equal the rmse_clean the GPU unit wrote
into its meta (|diff| < 0.05 s) -> proves row mapping, offset, y and clean mask; otherwise the cell FAILS and nothing is written.
usage: python reagg_local.py <RUNDIR> <PAYLOAD_DIR> <RESID_DIR> [NAME ...]   (default: every NAME under RUNDIR/results/engine)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, json, numpy as np, polars as pl

RUN, PAY, RES = sys.argv[1:4]
ONLY = sys.argv[4:]
QLEV = None


def tm(A, trim):
    n = A.shape[0]
    c = int(trim * n)
    B = np.sort(A, axis=0)
    return B[c : n - c].mean(0) if n - 2 * c > 0 else B.mean(0)


_fr = {}


def frame(apt):
    if apt not in _fr:
        _fr.clear()
        d = pl.read_parquet(
            f"{PAY}/member/km_{apt}.parquet",
            columns=["_row", "meta__mvt_aobt3", "meta__y", "meta__fb_pru", "meta__MVT_ID_mvt"],
        )
        _fr[apt] = dict(
            row=d["_row"].to_numpy(),
            off=np.nan_to_num(d["meta__mvt_aobt3"].to_numpy().astype(float), nan=0.0),
            y=d["meta__y"].to_numpy().astype(float),
            clean=d["meta__fb_pru"].to_numpy() != 1,
            ids=d["meta__MVT_ID_mvt"].to_numpy().astype(float),
        )
    return _fr[apt]


def Ltab(name, fold):
    a = np.load(f"{RES}/{name}_{fold}.npy")
    o = np.argsort(a[0])
    return a[0][o], a[1][o].astype(float)


def lookup(tab, ids):
    i, v = tab
    k = np.clip(np.searchsorted(i, ids), 0, len(i) - 1)
    hit = i[k] == ids
    out = np.full(len(ids), np.nan)
    out[hit] = v[k[hit]]
    return out


_stk = {}


def stk_F(chain, apt):
    if "ns" not in _stk:
        src = (
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster", "km_adapters_feed.py"))
            .read()
            .split("# ---------------- live mode")[0]
        )
        src = src.replace('if __name__ == "__main__" and "--selftest" in sys.argv', "if False")
        ns = {"__file__": os.path.join(PAY, "..", "..", "cluster", "km_adapters_feed.py")}
        exec(src, ns)
        _stk["ns"] = ns
    ns = _stk["ns"]
    if chain not in _stk:
        _stk[chain] = ns["load_dumps"](chain, os.path.join(PAY, "stk"))[0]
    F = ns["stk_frame"](_stk[chain], apt)
    F["clean"] = F["clean"]
    return F


names = ONLY or sorted(os.listdir(f"{RUN}/results/engine"))
os.makedirs(f"{RUN}/results/member", exist_ok=True)
nfail = nok = 0
for NAME in names:
    for d in sorted(glob.glob(f"{RUN}/results/engine/{NAME}/*_*")):
        fold, apt = os.path.basename(d).rsplit("_", 1)
        out = f"{RUN}/results/member/{NAME}_{fold}_{apt}.npz"
        fs = sorted(glob.glob(f"{d}/inst*.npz"), key=lambda f: int(f.split("inst")[-1][:-4]))
        if not fs:
            continue
        m0 = json.loads(str(np.load(fs[0])["meta"]))
        S = m0["spec"]
        ninst = S["ninst"]
        if len(fs) < ninst or any(not os.path.exists(f"{d}/inst{i}.npz") for i in range(ninst)):
            print(f"{NAME} {fold} {apt}: INCOMPLETE {len(fs)}/{ninst} -> skip")
            continue
        if os.path.exists(out):
            print(f"{NAME} {fold} {apt}: exists -> skip")
            continue
        Z = [np.load(f"{d}/inst{i}.npz") for i in range(ninst)]
        row, ids = Z[0]["row"], Z[0]["id"]
        for z in Z[1:]:
            assert np.array_equal(z["row"], row)
        fd = S.get("feed") or {}
        if fd.get(
            "stk"
        ):  # KSTK_FEED stacker arm: frame = km_adapters_feed.stk_frame on the chain's fold dumps (helpers exec'd; no torch)
            F = stk_F(fd["stk"] if isinstance(fd["stk"], str) else fd["stk"].get("chain"), apt)
        else:
            F = frame(apt)
        srt = np.argsort(F["row"], kind="stable")
        loc = srt[np.searchsorted(F["row"], row, sorter=srt)]
        assert np.array_equal(F["row"][loc], row) and np.array_equal(
            F["ids"][loc], ids
        ), f"{NAME} {fold} {apt}: row/id mismatch"
        off = F["off"][loc]
        clean = F["clean"][loc]
        if S.get("resid_l"):
            L = lookup(Ltab(S["resid_l"], fold), ids)
            miss = ~np.isfinite(L)
            off = np.where(miss, off, L)
            clean = clean & ~miss
        y = F["y"][loc]
        ok = clean & np.isfinite(y)
        G = np.stack([z["g"] for z in Z])
        Q = np.stack([z["q"] for z in Z])
        metas = [json.loads(str(z["meta"])) for z in Z]
        rmi = [float(np.sqrt(((off + G[j] - y)[ok] ** 2).mean())) for j in range(ninst)]
        dv = max(
            (0.0 if (np.isnan(a) and np.isnan(m["rmse_clean"])) else abs(a - m["rmse_clean"]))
            for a, m in zip(rmi, metas)
        )  # prod: no labels -> nan == nan
        if not dv < 0.05:
            print(f"{NAME} {fold} {apt}: VALIDATION FAIL max |rmse_inst - meta| {dv:.3f}")
            nfail += 1
            continue
        trim = S["trim"]
        g = tm(G, trim)
        q = tm(Q, trim)
        taxi = off + g
        qa = q + off[:, None].astype(np.float32)
        lev = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        assert q.shape[1] == 7
        spread = (qa[:, lev.index(0.90)] - qa[:, lev.index(0.10)]).astype(np.float32)
        gsd = G.std(0).astype(np.float32)
        rm = float(np.sqrt(((taxi - y)[ok] ** 2).mean()))
        tmp = out + ".tmp.npz"
        np.savez(
            tmp,
            row=row,
            id=ids,
            taxi=taxi,
            g=g,
            q=qa,
            qlev=np.array(lev),
            spread=spread,
            g_sd=gsd,
            g_inst=G.astype(np.float32),
            n_inst=ninst,
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
                    host=metas[0]["host"],
                    reagg="RERUN_PATCH reagg_local.py 3 Oct",
                    ctx=metas[0]["ctx"],
                    ncol=metas[0]["ncol"],
                )
            ),
        )
        os.replace(tmp, out)
        nok += 1
        print(
            f"{NAME} {fold} {apt}: {ninst} inst rows {len(row):,} ctx {metas[0]['ctx']:,} | rmse_clean ens {rm:.1f} inst {min(rmi):.1f}..{max(rmi):.1f} | validation max diff {dv:.4f}",
            flush=True,
        )
print(f"REAGG_LOCAL written {nok} fail {nfail}")
