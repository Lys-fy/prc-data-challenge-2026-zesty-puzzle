#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""FINE-TUNE NVIDIA Kumo Tabular on the 2025 member task (kumo_member_screen arm M form) per fold, then
infer exactly as scripts/kumo_km_member.py so the same T2x member screen compares FT vs the zero-shot KM member.
Protocol (per fold f): TRAIN = clean training-month rows of the other 8 months (kumo_km_member's `trn`); VAL = all rows of 5 % of TRAIN's airport-days
(seeded) -- held out from every fine-tune batch, used only for early stopping; the held fold is never touched before inference.
Batch = one airport (sampled proportional to rows): CTX_FT context rows + Q_FT query rows from TRAIN minus VAL, loss = pinball over the 999 fixed
quantile levels on the raw target r = y - mvt_aobt3 (sdm's forward() inverse-transforms the target, so outputs are in seconds), divided by
sd(r_context) for scale-free gradients; AdamW lr 1e-5 (all parameters, as examples/tabular/finetune.py), grad clip 1.0, bf16 autocast.
Eval every EVAL steps: RMSE on VAL queries with a 20k-row TRAIN context per airport; keep the best state -> artifacts/xtfm2/ft_<SIZE>_<fold>.pt.
Inference = kumo_km_member.py's loop with the fine-tuned weights -> data/oof/<NAME>_<fold>.npy = [id, taxi].
usage: kumo_icl_finetune.py SMOKE | ft <fold> | infer <fold>        ($KUMO_PY, GPU)
env: SIZE (small), STEPS (600), CTX_FT (4096), Q_FT (1024), LR (1e-5), EVAL (100), VALFRAC (0.05), SEED (0), CTX (20000, inference), NEST (4), NAME (kumo_icl_lirf_ft)
Not smoke-tested on a GPU: run SMOKE first."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
t0 = time.time()
os.environ.setdefault("HF_HOME", f"{ROOT}/artifacts/xkumo/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
SIZE = os.environ.get("SIZE", "small")
STEPS = int(os.environ.get("STEPS", "600"))
CTX_FT = int(os.environ.get("CTX_FT", "4096"))
Q_FT = int(os.environ.get("Q_FT", "1024"))
LR = float(os.environ.get("LR", "1e-5"))
EVAL = int(os.environ.get("EVAL", "100"))
VALFRAC = float(os.environ.get("VALFRAC", "0.05"))
SEED = int(os.environ.get("SEED", "0"))
CTX = int(os.environ.get("CTX", "20000"))
NEST = int(os.environ.get("NEST", "4"))
NAME = os.environ.get("NAME", "kumo_icl_lirf_ft")
DEV = os.environ.get("DEV", "cuda")
AMP = os.environ.get("AMP", "bf16")
os.makedirs(f"{ROOT}/artifacts/xtfm2", exist_ok=True)
import torch, sdm, sdm.processing as sp

torch.set_num_threads(int(os.environ.get("PRC_THREADS", "2")))
torch.manual_seed(SEED)
LEVELS = torch.linspace(0.001, 0.999, 999, device=DEV)


def pinball(pred, target):  # pred [R, 999] raw seconds, target [R]
    diff = target[:, None] - pred
    return torch.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean()


def train_recipe(model):
    r = model.default_recipe()
    r.target = sp.StypeDispatch(
        categorical=[sp.AlignCategories(), sp.ShuffleCategories(method="shift")], numerical=sp.Standardize()
    )
    r.output = sp.Identity()
    return r


def tt(x):  # numpy [R, D] float32 (NaN allowed) -> TableTensor with numerical columns
    import pandas as pd

    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    return sdm.TableTensor.from_pandas(df, stypes={c: "numerical" for c in df.columns}, device=DEV)


def ty(v):
    import pandas as pd

    df = pd.DataFrame({"y": v.astype(np.float32)})
    return sdm.TableTensor.from_pandas(df, stypes={"y": "numerical"}, device=DEV)


def ft_step(model, recipe, opt, xc, yc, xq, yq):
    model.train()
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(AMP == "bf16" and DEV.startswith("cuda"))):
        out = model(x_context=tt(xc), y_context=ty(yc), x_query=tt(xq), recipe=recipe)[0]
    pred = out.numerical.float()
    assert pred.shape[-1] == 999, pred.shape
    loss = pinball(pred, torch.as_tensor(yq, device=DEV).float()) / max(float(yc.std()), 1.0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


def predict(model, xc, yc, xq, chunk=4096):
    model.eval()
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    model.fit(
        torch.as_tensor(xc, device=DEV),
        torch.as_tensor(yc.astype(np.float32)[:, None], device=DEV),
        num_estimators=NEST,
        generator=gen,
    )
    out = []
    for i in range(0, len(xq), chunk):
        p = model.predict(torch.as_tensor(xq[i : i + chunk], device=DEV))
        q = p.numerical if hasattr(p, "numerical") else p
        out.append(q.float().mean(-1).cpu().numpy())
    model.clear()
    return np.concatenate(out)


if (
    sys.argv[1] == "SMOKE"
):  # synthetic: fine-tune 20 steps on a nonlinear target, check that train loss falls and prediction still works
    model = sdm.models.KumoTabular(task="regression", size=SIZE, device=DEV)
    rec = train_recipe(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6000, 12)).astype(np.float32)
    yv = (50 * x[:, 0] + 20 * x[:, 1] * x[:, 2] + rng.normal(0, 10, 6000)).astype(np.float32)
    x[rng.random(x.shape) < 0.05] = np.nan
    g0 = predict(model, x[:2000], yv[:2000], x[5000:])
    print(f"zero-shot rmse {np.sqrt(((g0 - yv[5000:]) ** 2).mean()):.2f}", flush=True)
    L = []
    for s in range(20):
        i = rng.permutation(5000)
        L.append(ft_step(model, rec, opt, x[i[:1024]], yv[i[:1024]], x[i[1024:1536]], yv[i[1024:1536]]))
    g1 = predict(model, x[:2000], yv[:2000], x[5000:])
    print(
        f"loss first/last {L[0]:.4f} {L[-1]:.4f} | fine-tuned rmse {np.sqrt(((g1 - yv[5000:]) ** 2).mean()):.2f} | "
        f"max mem {torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.2f} GiB | {time.time()-t0:.0f}s",
        flush=True,
    )
    sys.exit(0)
# ---------------- data: kumo_km_member.py's frame (same feature resolution, target offset, clean mask) + the timestamp for the airport-day VAL split
import polars as pl

HOLD = {"janjul": [1, 7], "months28": [2, 8], "months39": [3, 9], "months410": [4, 10]}
FEATS_L = "base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,adsb,adsb2,adsb3,xba,xe".split(",")
DROP = tuple(
    "te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy".split(
        ","
    )
)
SIDE = {
    "durp": "durp_features",
    "dreg": "dreg_features",
    "adsb": "adsb_features",
    "adsb2": "adsb2_features",
    "adsb3": "adsb3_features",
    "xba": "xba_features",
    "xe": "xe_queue_features",
}
_src = open(f"{ROOT}/scripts/train2.py").read()
ns = {}
exec(
    _src[_src.index("CAT_BASE = ") : _src.index("def dayshift_model")], ns
)  # CAT_BASE, G, cols_for exactly as train2.py / kumo_km_member.py
FV = f"{ROOT}/data/features_v9.parquet"
schema = pl.read_parquet_schema(FV)
META = [
    "MVT_ID_mvt",
    "y",
    "m",
    "is_rank",
    "ADEP_mvt",
    "fb",
    "fb_pru",
    "mvt_aobt3",
    "unmatched",
    "adsb_linked",
    "MVT_TIME_UTC_mvt",
]
side_sch = {
    g: {
        c: t
        for c, t in pl.read_parquet_schema(f"{ROOT}/data/external/{f}.parquet").items()
        if c != "MVT_ID_mvt" and c not in schema
    }
    for g, f in SIDE.items()
}
allsch = dict(schema)
[allsch.update(v) for v in side_sch.values()]
dummy = pl.DataFrame(schema={c: pl.Null for c in allsch})
feats = []
for g in FEATS_L:
    feats += ns["cols_for"](dummy, g)
feats = [f for f in dict.fromkeys(feats) if f not in ns["CAT_BASE"] and not f.startswith(DROP)]
num = [f for f in feats if allsch[f].is_numeric()]
assert len(num) == 412, len(num)  # kumo_member_screen arm M: 412 numeric columns
d = pl.read_parquet(FV, columns=[c for c in dict.fromkeys(META + num) if c in schema])
for g, f in SIDE.items():
    cols = [c for c in side_sch[g] if c in num or c == "adsb_linked"]
    if cols:
        d = d.join(
            pl.read_parquet(f"{ROOT}/data/external/{f}.parquet", columns=["MVT_ID_mvt"] + cols),
            on="MVT_ID_mvt",
            how="left",
        )
X = d.select(num).cast(pl.Float32).fill_nan(None).to_numpy().astype(np.float32)
y = d["y"].to_numpy().astype(float)
m = d["m"].to_numpy()
rank = d["is_rank"].to_numpy() == 1
apt = d["ADEP_mvt"].cast(pl.Utf8).to_numpy()
off = np.nan_to_num(d["mvt_aobt3"].to_numpy().astype(float), nan=0.0)
clean = d["fb_pru"].to_numpy() != 1
lk = d["adsb_linked"].fill_null(0).to_numpy() > 0
day = d["MVT_TIME_UTC_mvt"].dt.ordinal_day().fill_null(0).to_numpy()
ids = d["MVT_ID_mvt"].to_numpy().astype(float)
r_all = y - off
del d
fold = sys.argv[2]
hold = HOLD[fold]
val = ~rank & np.isin(m, hold)
trn = ~rank & ~np.isin(m, hold) & clean & np.isfinite(y)
print(
    f"{fold}: train rows {trn.sum():,} held rows {val.sum():,} features {len(num)} | {time.time()-t0:.0f}s", flush=True
)
ck = f"{ROOT}/artifacts/xtfm2/ft_{SIZE}_{fold}.pt"
model = sdm.models.KumoTabular(task="regression", size=SIZE, device=DEV)
if sys.argv[1] == "ft":
    rng = np.random.default_rng(SEED)
    key = np.char.add(apt.astype(str), day.astype(str))
    keys = np.unique(key[trn])
    vkeys = set(rng.choice(keys, int(VALFRAC * len(keys)), replace=False).tolist())
    isval = np.array([k in vkeys for k in key]) & trn
    fit = trn & ~isval
    apts = sorted(np.unique(apt[fit]))
    pa = np.array([(fit & (apt == a)).sum() for a in apts], float)
    pa /= pa.sum()
    rows_fit = {a: np.where(fit & (apt == a))[0] for a in apts}
    rows_val = {a: np.where(isval & (apt == a))[0] for a in apts}
    rec = train_recipe(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def val_rmse():
        se, n = 0.0, 0
        for a in apts:
            vi = rows_val[a]
            if len(vi) == 0:
                continue
            ci = rng.choice(rows_fit[a], min(CTX, len(rows_fit[a])), replace=False)
            g = predict(model, X[ci], r_all[ci], X[vi])
            se += ((g - r_all[vi]) ** 2).sum()
            n += len(vi)
        return float(np.sqrt(se / n))

    best = val_rmse()
    print(f"step 0 VAL rmse {best:.2f} (zero-shot) | {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), ck)
    hist = []
    for s in range(1, STEPS + 1):
        a = apts[rng.choice(len(apts), p=pa)]
        i = rng.permutation(rows_fit[a])[: CTX_FT + Q_FT]
        if len(i) < CTX_FT + Q_FT:
            continue
        hist.append(ft_step(model, rec, opt, X[i[:CTX_FT]], r_all[i[:CTX_FT]], X[i[CTX_FT:]], r_all[i[CTX_FT:]]))
        if s % EVAL == 0:
            v = val_rmse()
            print(
                f"step {s} train loss {np.mean(hist[-EVAL:]):.4f} VAL rmse {v:.2f} {'*' if v < best else ''} | {time.time()-t0:.0f}s",
                flush=True,
            )
            if v < best:
                best = v
                torch.save(model.state_dict(), ck)
    print(f"{fold} fine-tune done: best VAL rmse {best:.2f} -> {ck} | {time.time()-t0:.0f}s", flush=True)
elif (
    sys.argv[1] == "infer"
):  # = kumo_km_member.py's loop (stratified month x link context <= CTX, query = all fold rows of the airport)
    model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
    model.eval()
    taxi = np.full(len(y), np.nan)
    for a in sorted(np.unique(apt[val])):
        ti = np.where(trn & (apt == a))[0]
        qi = np.where(val & (apt == a))[0]
        t1 = time.time()
        rng = np.random.default_rng(SEED)
        frac = min(1.0, CTX / len(ti))
        cell = m[ti] * 2 + lk[ti]
        pick = []
        for c in np.unique(cell):
            ci = ti[cell == c]
            k = int(round(frac * len(ci)))
            pick.append(rng.choice(ci, min(k, len(ci)), replace=False))
        k = np.concatenate(pick)
        g = predict(model, X[k], r_all[k], X[qi])
        assert np.isfinite(g).all(), (fold, a)
        taxi[qi] = off[qi] + g
        ok = qi[clean[qi] & np.isfinite(y[qi])]
        e = taxi[ok] - y[ok]
        print(
            f"  {fold} {a}: ctx {len(k):,} query {len(qi):,} clean-row rmse {np.sqrt((e ** 2).mean()):.1f} bias {e.mean():+.1f} {time.time()-t1:.0f}s",
            flush=True,
        )
    vi = np.where(val)[0]
    np.save(f"{ROOT}/data/oof/{NAME}_{fold}.npy", np.vstack([ids[vi], taxi[vi]]))
    ok = vi[clean[vi] & np.isfinite(y[vi])]
    print(
        f"{fold} done rows {len(vi):,} clean rmse {np.sqrt(((taxi[ok] - y[ok]) ** 2).mean()):.1f} -> data/oof/{NAME}_{fold}.npy | {time.time()-t0:.0f}s",
        flush=True,
    )
print("done", f"{time.time()-t0:.0f}s")
