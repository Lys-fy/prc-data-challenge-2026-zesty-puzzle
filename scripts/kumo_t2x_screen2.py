#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""kumo_member_screen (v2: harness screen_l2_extra.py, the CAPACITY_XTD raw-column bug fixed): Kumo Tabular as an L2 augment of the CURRENT T2x chains (XT construction moved onto today's stack).
Harness = the screen_l2_extra.py setup exec'd unchanged (chain_stack source, base arm xtd:wideleaf_retune_a2, EXTRA lnk,xlk,xdk,xapt,spr,raw,xpr,xtd, METAFIT=all,
4-fold cross-fit, chains Kq and NNA3, meta seeds 0,1,2, clip +-150). Three steps:
  dump   (.venv)   per chain x fold: the meta input matrix, y, v41 (= the chain's pre-meta base b), meta corrections gM (seeds 0,1,2 = the base)
                   and gS (seeds 3,4,5, the seed control) -> scratchpad/xkt/t2x/<chain>_<fold>.npz
  kumo   (.venv-kumo) per chain x fold x airport: Kumo Tabular medium, context = the airport's rows of the other 3 folds (random 20k, seed 0),
                   4 estimators, target clip(y - v41, +-1800), inputs = the meta inputs (NaN -> 0 as XT) -> gK (mean of the 999 quantiles)
  score  (.venv)   pred(g) = cf(v41 + clip(g, +-150)); arms per chain, then 0.5 Kq + 0.5 NNA3 on common rows:
                   base = gM | KB = 0.7 gM + 0.3 gK | SEED = 0.7 gM + 0.3 gS | SHR = 0.7 gM (shrink-only control) -> month_weight_rescore --mwd + decision.
usage: kumo_t2x_screen2.py dump|kumo|score"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, re, subprocess, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/scratchpad/xkt/t2x"
os.makedirs(OUT, exist_ok=True)
FO = ["months39", "months28", "months410", "janjul"]
CH = ["Kq", "NNA3"]
step = sys.argv[1]
t0 = time.time()
if step == "dump":
    os.environ["wideleaf_folds"] = ",".join(FO)
    src = open(f"{ROOT}/scripts/screen_l2_extra.py").read().split("for arm in sys.argv[1:]:")[0]
    H = {"__file__": f"{ROOT}/scripts/screen_l2_extra.py", "__name__": "xkt"}
    exec(src, H)
    label, env = H["parse"]("base=xtd:wideleaf_retune_a2")
    os.environ.pop("EXTRA1_OOF", None)
    for cfg, ce in H["CFG"].items():
        os.environ.update(env)
        os.environ.update(ce)
        g = {
            "__file__": os.path.abspath("scripts/chain_stack.py"),
            "__name__": "xf",
            "wideleaf_retune_f": H["FOLDS"],
            "CAPACITY_COLS": ["T", "TmL"],
        }
        exec(H["code"], g)
        cols = g["BASEC"] + g["EXTRA"]["L2"] + sum((g["EXTRA"][x] for x in g["extra_inputs"]), [])
        D = g["D"]
        lgb, pd = g["lgb"], g["pd"]
        print(
            f"{cfg}: {len(cols)} cols {cols} | MSEEDS {g['MSEEDS']} clip {g['CLIP_LO']}/{g['CLIP_HI']} METAFIT {g['METAFIT']} | {time.time()-t0:.0f}s",
            flush=True,
        )
        for t in g["ALL"]:
            tags = [x for x in g["ALL"] if x != t]
            ds = lgb.Dataset(
                pd.concat([g["X"](D[x], cols) for x in tags]), np.concatenate([D[x]["y"] - D[x]["v41"] for x in tags])
            )
            xq = g["X"](D[t], cols)
            gM = np.mean([lgb.train(g["prm"](s), ds, g["MROUNDS"]).predict(xq) for s in g["MSEEDS"]], axis=0)
            gS = np.mean([lgb.train(g["prm"](s + 3), ds, g["MROUNDS"]).predict(xq) for s in g["MSEEDS"]], axis=0)
            ref = g["cf"](D[t]["v41"] + np.clip(gM, -g["CLIP_LO"], g["CLIP_HI"]), D[t]["mvt_lobt"])
            np.savez(
                f"{OUT}/{cfg}_{t}.npz",
                mid=D[t]["mid"],
                apt=D[t]["apt"],
                y=D[t]["y"],
                v41=D[t]["v41"],
                lobt=D[t]["mvt_lobt"],
                X=xq.to_numpy(dtype=np.float32),
                cols=np.array(xq.columns),
                gM=gM,
                gS=gS,
            )
            print(
                f"  {cfg} {t}: rows {len(gM):,} base mse {np.mean((ref - D[t]['y']) ** 2):.1f} | L mse {np.mean((D[t]['L'] - D[t]['y']) ** 2):.1f} | {time.time()-t0:.0f}s",
                flush=True,
            )
elif step == "kumo":
    import torch, sdm

    os.environ.setdefault("HF_HOME", f"{ROOT}/artifacts/xkumo/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(int(os.environ.get("PRC_THREADS", "2")))
    DEV = "cuda"
    CTX, NEST, SEED, CHUNK = 20000, 4, 0, 4096
    model = sdm.models.KumoTabular(task="regression", size="medium", device=DEV)
    print(f"torch {torch.__version__} sdm {sdm.__version__} {torch.cuda.get_device_name(0)}", flush=True)

    def kumo(xc, yc, xq):
        gen = torch.Generator(device=DEV).manual_seed(SEED)
        model.fit(
            torch.as_tensor(xc, device=DEV),
            torch.as_tensor(yc.astype(np.float32)[:, None], device=DEV),
            num_estimators=NEST,
            generator=gen,
        )
        out = []
        for i in range(0, len(xq), CHUNK):
            p = model.predict(torch.as_tensor(xq[i : i + CHUNK], device=DEV))
            q = p.numerical if hasattr(p, "numerical") else p
            assert q.shape[-1] == 999
            out.append(q.float().mean(-1).cpu().numpy())
        model.clear()
        return np.concatenate(out)

    for cfg in CH:
        Z = {t: dict(np.load(f"{OUT}/{cfg}_{t}.npz", allow_pickle=True)) for t in FO}
        for z in Z.values():
            z["X"] = np.nan_to_num(z["X"], nan=0.0, posinf=0.0, neginf=0.0)
            z["r"] = np.clip(z["y"] - z["v41"], -1800, 1800)
        for t in FO:
            gK = np.full(len(Z[t]["y"]), np.nan)
            for a in sorted(set(Z[t]["apt"].tolist())):
                xc = np.vstack([Z[x]["X"][Z[x]["apt"] == a] for x in FO if x != t])
                yc = np.concatenate([Z[x]["r"][Z[x]["apt"] == a] for x in FO if x != t])
                k = np.random.default_rng(SEED).choice(len(yc), min(CTX, len(yc)), replace=False)
                qi = Z[t]["apt"] == a
                t1 = time.time()
                gK[qi] = kumo(xc[k], yc[k], Z[t]["X"][qi])
                print(
                    f"  {cfg} {t} {a}: ctx {len(k):,} query {qi.sum():,} sd(gK) {gK[qi].std():.1f} {time.time()-t1:.0f}s",
                    flush=True,
                )
            assert np.isfinite(gK).all()
            np.save(f"{OUT}/gK_{cfg}_{t}.npy", gK)
            print(f"{cfg} {t} done | {time.time()-t0:.0f}s", flush=True)
elif step == "score":

    def cf(v, lo):
        v = np.where(np.isnan(lo), v, np.clip(v, np.nan_to_num(lo, nan=0.0) - 3606, np.nan_to_num(lo, nan=0.0) + 3606))
        return np.maximum(v, 60.0)

    ARMS = ["base", "KB", "SEED", "SHR"]
    for t in FO:
        P = {}
        for cfg in CH:
            z = np.load(f"{OUT}/{cfg}_{t}.npz", allow_pickle=True)
            gK = np.load(f"{OUT}/gK_{cfg}_{t}.npy")
            gM, gS = z["gM"], z["gS"]
            G = {"base": gM, "KB": 0.7 * gM + 0.3 * gK, "SEED": 0.7 * gM + 0.3 * gS, "SHR": 0.7 * gM}
            P[cfg] = (z["mid"], {a: cf(z["v41"] + np.clip(G[a], -150, 150), z["lobt"]) for a in ARMS}, z["y"])
            print(
                f"  {cfg} {t}: "
                + " ".join(f"{a} mse {np.mean((P[cfg][1][a] - z['y']) ** 2):.1f}" for a in ARMS)
                + f" | corr(gK, gM) {np.corrcoef(gK, gM)[0, 1]:.3f}",
                flush=True,
            )
        (i1, p1, _), (i2, p2, _) = P["Kq"], P["NNA3"]
        c = np.intersect1d(i1, i2)
        a1 = np.argsort(i1)
        a2 = np.argsort(i2)
        for a in ARMS:
            m1 = p1[a][a1][np.searchsorted(i1[a1], c)]
            m2 = p2[a][a2][np.searchsorted(i2[a2], c)]
            np.save(f"{OUT}/mix_{a}_{t}.npy", np.vstack([c, 0.5 * m1 + 0.5 * m2]))
    S = f"{OUT}/mix"
    pairs = [
        f"KB=npy:{S}_KB~npy:{S}_base",
        f"SEED=npy:{S}_SEED~npy:{S}_base",
        f"SHR=npy:{S}_SHR~npy:{S}_base",
        f"KBvsSEED=npy:{S}_KB~npy:{S}_SEED",
        f"KBvsSHR=npy:{S}_KB~npy:{S}_SHR",
    ]
    r = subprocess.run(
        [f"{ROOT}/.venv/bin/python", f"{ROOT}/scripts/month_weight_rescore.py", "--mwd"] + pairs,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(r.stdout, r.stderr[-2000:], flush=True)
    num = r"\s*([+-]?\s*\d+\.\d)"
    R = {}
    for line in r.stdout.splitlines():
        mm = re.match(r"\[(\w+)\] (.*)", line)
        if mm:
            for f in re.finditer(
                r"(months\d+|janjul)\s+" + num + r"\s*\[" + num + r"\]\s*s" + num + r"\s*m" + num + r"\s*m3" + num,
                mm.group(2),
            ):
                R.setdefault(mm.group(1), {})[f.group(1)] = [float(x.replace(" ", "")) for x in f.groups()[1:]]
    K, SD, SH = R["KB"], R["SEED"], R["SHR"]  # [raw, raw w/o top-3, s, m, m3]
    pf = {t: K[t][3] <= -20 and K[t][3] <= min(SD[t][3], SH[t][3]) - 20 and K[t][1] < 0 and K[t][4] < 0 for t in FO}
    ok = sum(pf.values()) >= 3 and K["janjul"][3] <= 0
    print(f"per-fold {pf} | DECISION T2x Kumo augment: {'PASS' if ok else 'FAIL'}", flush=True)
