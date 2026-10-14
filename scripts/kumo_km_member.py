#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""kumo_member_screen arm M: NVIDIA Kumo Tabular as a MEMBER on raw features, fold OOF for the WIDELEAF/CAPACITY_XTD xtd harness.
Features = the L member's groups (FEATS_L of wideleaf_member.sh, same --drop-cols prefixes), resolved with train2.py's own G/cols_for; numeric
columns only (the CAT_BASE string categoricals are left out: their target encodings te*/te2*/ref* are in the groups). NaN kept (Kumo has a
learned missingness projection). Target = y - mvt_aobt3 (aobt NaN -> 0, the train2 resid_aobt3 offset); the context holds CLEAN training rows
only (not fb_pru: no artefact / > 2 h labels), i.e. the member is a genuine-taxi branch like row 3 of the tree OOF.
Per airport: context = stratified random subsample (proportional over month x adsb_linked cells) of <= CTX clean training-month rows;
query = ALL of the fold's rows of that airport (matched and unmatched). Prediction = aobt offset + mean of the 999 quantiles.
usage: kumo_km_member.py <fold>[,<fold>...]   fold in janjul|months28|months39|months410   ($KUMO_PY; DRY=1 = features only)
env: SIZE (medium), CTX (20000), NEST (4), SEED (0), CHUNK (4096), DEV (cuda), NAME (kumo_member_screen_km) -> data/oof/<NAME>_<fold>.npy = [id, taxi] (use @1)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, numpy as np, polars as pl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); t0 = time.time()
HOLD = {"janjul": [1, 7], "months28": [2, 8], "months39": [3, 9], "months410": [4, 10]}
FEATS_L = "base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,adsb,adsb2,adsb3,xba,xe".split(",")
DROP = tuple("te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy".split(","))
SIDE = {"durp": "durp_features", "dreg": "dreg_features", "adsb": "adsb_features", "adsb2": "adsb2_features", "adsb3": "adsb3_features",
        "xba": "xba_features", "xe": "xe_queue_features"}
src = open(f"{ROOT}/scripts/train2.py").read(); ns = {}
exec(src[src.index("CAT_BASE = "):src.index("def dayshift_model")], ns)   # CAT_BASE, G, cols_for exactly as train2.py
FV = f"{ROOT}/data/features_v9.parquet"; schema = pl.read_parquet_schema(FV)
META = ["MVT_ID_mvt", "y", "m", "is_rank", "ADEP_mvt", "fb", "fb_pru", "mvt_aobt3", "unmatched", "adsb_linked"]
side_sch = {g: {c: t for c, t in pl.read_parquet_schema(f"{ROOT}/data/external/{f}.parquet").items() if c != "MVT_ID_mvt" and c not in schema} for g, f in SIDE.items()}
allsch = dict(schema); [allsch.update(v) for v in side_sch.values()]
dummy = pl.DataFrame(schema={c: pl.Null for c in allsch})
feats = []
for g in FEATS_L: feats += ns["cols_for"](dummy, g)
feats = [f for f in dict.fromkeys(feats) if f not in ns["CAT_BASE"] and not f.startswith(DROP)]
num = [f for f in feats if allsch[f].is_numeric()]; nonnum = [f for f in feats if not allsch[f].is_numeric()]
print(f"features: {len(num)} numeric (+{len(nonnum)} non-numeric dropped: {nonnum[:8]}) | adsb_linked in schema {'adsb_linked' in allsch} | {time.time()-t0:.0f}s", flush=True)
if os.environ.get("DRY"): print(num); sys.exit(0)
d = pl.read_parquet(FV, columns=[c for c in dict.fromkeys(META + num) if c in schema])
for g, f in SIDE.items():
    cols = [c for c in side_sch[g] if c in num or c == "adsb_linked"]
    if cols: d = d.join(pl.read_parquet(f"{ROOT}/data/external/{f}.parquet", columns=["MVT_ID_mvt"] + cols), on="MVT_ID_mvt", how="left")
print(f"frame rows {d.height:,} cols {d.width} | {time.time()-t0:.0f}s", flush=True)
import torch, sdm
SIZE = os.environ.get("SIZE", "medium"); CTX = int(os.environ.get("CTX", "20000")); NEST = int(os.environ.get("NEST", "4")); SEED = int(os.environ.get("SEED", "0"))
CHUNK = int(os.environ.get("CHUNK", "4096")); DEV = os.environ.get("DEV", "cuda"); NAME = os.environ.get("NAME", "kumo_member_screen_km")
os.environ.setdefault("HF_HOME", f"{ROOT}/artifacts/xkumo/hf"); os.environ.setdefault("HF_HUB_OFFLINE", "1"); torch.set_num_threads(int(os.environ.get("PRC_THREADS", "2")))
model = sdm.models.KumoTabular(task="regression", size=SIZE, device=DEV)
print(f"torch {torch.__version__} sdm {getattr(sdm, '__version__', '?')} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} | size {SIZE} ctx {CTX} nest {NEST} seed {SEED}", flush=True)
X = d.select(num).cast(pl.Float32).fill_nan(None).to_numpy().astype(np.float32)   # nulls -> NaN
y = d["y"].to_numpy().astype(float); m = d["m"].to_numpy(); rank = d["is_rank"].to_numpy() == 1; apt = d["ADEP_mvt"].cast(pl.Utf8).to_numpy()
off = np.nan_to_num(d["mvt_aobt3"].to_numpy().astype(float), nan=0.0); clean = d["fb_pru"].to_numpy() != 1; lk = d["adsb_linked"].fill_null(0).to_numpy() > 0
ids = d["MVT_ID_mvt"].to_numpy().astype(float); del d
def kumo(xc, yc, xq):
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    model.fit(torch.as_tensor(xc, device=DEV), torch.as_tensor(yc.astype(np.float32)[:, None], device=DEV), num_estimators=NEST, generator=gen); out = []
    for i in range(0, len(xq), CHUNK):
        p = model.predict(torch.as_tensor(xq[i:i + CHUNK], device=DEV)); q = p.numerical if hasattr(p, "numerical") else p; assert q.shape[-1] == 999
        out.append(q.float().mean(-1).cpu().numpy())
    model.clear(); return np.concatenate(out)
for fold in sys.argv[1].split(","):
    hold = HOLD[fold]; val = ~rank & np.isin(m, hold); trn = ~rank & ~np.isin(m, hold) & clean & np.isfinite(y); taxi = np.full(len(y), np.nan)
    for a in sorted(np.unique(apt[val])):
        ti = np.where(trn & (apt == a))[0]; qi = np.where(val & (apt == a))[0]; t1 = time.time()
        rng = np.random.default_rng(SEED); frac = min(1.0, CTX / len(ti)); cell = m[ti] * 2 + lk[ti]; pick = []
        for c in np.unique(cell):   # proportional stratified subsample over month x link class
            ci = ti[cell == c]; k = int(round(frac * len(ci))); pick.append(rng.choice(ci, min(k, len(ci)), replace=False))
        k = np.concatenate(pick)
        g = kumo(X[k], y[k] - off[k], X[qi]); assert np.isfinite(g).all(), (fold, a); taxi[qi] = off[qi] + g
        ok = qi[clean[qi] & np.isfinite(y[qi])]; e = taxi[ok] - y[ok]
        print(f"  {fold} {a}: ctx {len(k):,} query {len(qi):,} clean-row rmse {np.sqrt((e ** 2).mean()):.1f} bias {e.mean():+.1f} {time.time()-t1:.0f}s", flush=True)
    vi = np.where(val)[0]; np.save(f"{ROOT}/data/oof/{NAME}_{fold}.npy", np.vstack([ids[vi], taxi[vi]]))
    ok = vi[clean[vi] & np.isfinite(y[vi])]; print(f"{fold} done rows {len(vi):,} clean rmse {np.sqrt(((taxi[ok] - y[ok]) ** 2).mean()):.1f} -> data/oof/{NAME}_{fold}.npy | {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s")
