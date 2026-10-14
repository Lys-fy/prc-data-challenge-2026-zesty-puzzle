#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Trains a day-level transformer member (one sequence = all departures of one airport x UTC day, full self-attention
within the day) on the residual target r = y - off (off = mvt_aobt3), extending the first dayset member with v2 options.
Inputs: --fold (held-out fold), --tag (output member name), data/features_v9.parquet. Writes
data/oof/big_<tag>_<fold>.npy = [MVT_ID, pred, nan]; --save-model also saves the fitted model and stats to a directory.

New vs the first dayset member: --tbias dir (directional time bias: separate slopes for earlier / later departures + a per-head sign constant),
--ctx-hours H (tokens of the previous day with tod >= 24-H and of the next day with tod < H as loss-free context with clock -24/+24),
--amp (bf16 autocast on CUDA). On CUDA the feature tensors live on the device (batch gather on-device; identical math).
usage: dayset_big_member.py --fold months28 --tag BIG --d 192 --layers 4 --heads 6 --epochs 16 [--tbias dir --ctx-hours 6 --amp --seed 0 --limit-days 0 --no-write]
"""

import argparse, os, sys, time, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
from train2 import CAT_BASE, cols_for

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--fold", required=True, help="held-out fold name (e.g. months28)")
ap.add_argument("--tag", required=True, help="output member name")
ap.add_argument("--epochs", type=int, default=12, help="training epochs")
ap.add_argument("--d", type=int, default=128, help="model hidden width")
ap.add_argument("--layers", type=int, default=3, help="number of transformer layers")
ap.add_argument("--heads", type=int, default=4, help="number of attention heads")
ap.add_argument("--bs", type=int, default=8, help="batch size in days")
ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
ap.add_argument("--seed", type=int, default=0, help="random seed")
ap.add_argument("--limit-days", type=int, default=0, help="cap on training days (0 = no limit)")
ap.add_argument("--tbias", default="abs", choices=["abs", "dir"], help="time-bias form: absolute or directional")
ap.add_argument("--ctx-hours", type=float, default=0.0, help="hours of adjacent-day loss-free context tokens")
ap.add_argument("--amp", action="store_true", help="use bf16 autocast on CUDA")
ap.add_argument("--feats", default="base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,day", help="comma list of feature groups")
ap.add_argument("--no-write", action="store_true", help="skip writing the output .npy file")
ap.add_argument(
    "--features",
    default="data/features_v9.parquet",
    help="SAVE_MODEL: features table (default = the hard-coded v9 path, unchanged)",
)
ap.add_argument(
    "--save-model",
    default="",
    help="SAVE_MODEL: after the fit save state_dict + standardisation stats (lo/hi/med/mu/sd, flag columns), vocabularies, target scale and a manifest to DIR",
)
ap.add_argument(
    "--load-model",
    default="",
    help="SAVE_MODEL: predict-only from a --save-model DIR (no training; every statistic and vocabulary is the saved one); writes data/oof/big_<tag>_<fold>.npy as usual",
)
a = ap.parse_args()
T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:7.0f}s]", *x, flush=True)


FOLDM = {
    "prod": [],
    "janjul": [1, 7],
    "months28": [2, 8],
    "months39": [3, 9],
    "months410": [4, 10],
    "months511": [5, 11],
    "months612": [6, 12],
}
HOLD = FOLDM[a.fold]
NT = int(os.environ.get("PRC_THREADS", "4"))
P = a.features
S = pl.read_parquet_schema(P)
dummy = pl.DataFrame(schema={c: pl.Null for c in S})
LEAK = {
    "y",
    "y_minus_ref",
    "fb",
    "fb_pru",
    "BLOCK_TIME_UTC_mvt",
    "TAXITIME_SEC_mvt",
    "t_block",
    "is_rank",
    "m",
    "MVT_ID_mvt",
}
num = []
for g in a.feats.split(","):
    num += cols_for(dummy, g)
NUMT = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt16, pl.UInt8, pl.Boolean)
num = [c for c in dict.fromkeys(num) if c in S and c not in LEAK and c not in CAT_BASE and S[c] in NUMT]
cats = [c for c in CAT_BASE if c in S]
meta = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "m", "is_rank", "fb", "y", "mvt_aobt3"]
d = pl.read_parquet(P, columns=list(dict.fromkeys(meta + cats + num)))
if a.fold != "prod":
    d = d.filter(pl.col("is_rank") == 0)  # prod: 2025 rows train, 2026 ranking airport-days predicted
d = d.with_columns(
    pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_day"),
    (
        pl.col("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64)
        + pl.col("MVT_TIME_UTC_mvt").dt.minute() / 60
        + pl.col("MVT_TIME_UTC_mvt").dt.second() / 3600
    ).alias("_tod"),
)
d = d.sort(["ADEP_mvt", "_day", "MVT_TIME_UTC_mvt"])
log(
    f"rows {d.height:,} num {len(num)} cats {len(cats)} fold {a.fold} hold {HOLD} tag {a.tag} tbias {a.tbias} ctx {a.ctx_hours} amp {a.amp}"
)
m = d["m"].to_numpy()
y = d["y"].to_numpy().astype(np.float64)
fb = d["fb"].to_numpy() == 1
off_raw = d["mvt_aobt3"].to_numpy().astype(np.float64)
offna = np.isnan(off_raw)
off = np.nan_to_num(off_raw, nan=0.0)
r = y - off
held = (d["is_rank"].to_numpy() == 1) if a.fold == "prod" else np.isin(m, HOLD)
trn = ~held
fit = trn & ~fb & (np.abs(r) < 7200) & np.isfinite(r)
ids = d["MVT_ID_mvt"].to_numpy()
tod = d["_tod"].to_numpy().astype(np.float32)
dk = (d["ADEP_mvt"] + "|" + d["_day"].cast(pl.Utf8)).to_numpy()
apt_all = d["ADEP_mvt"].cast(pl.Utf8).to_numpy()
date_all = d["_day"].to_numpy().astype("datetime64[D]")
# ---- numeric matrix: fit-row statistics only (identical to the first dayset member)
X = d.select([pl.col(c).cast(pl.Float32) for c in num]).to_numpy().astype(np.float32)
X[~np.isfinite(X)] = np.nan
rng = np.random.default_rng(a.seed)
if a.load_model:  # SAVE_MODEL: saved standardisation (the subsample of the saved run), nothing is re-estimated
    import json

    _ST = np.load(f"{a.load_model}/stats.npz")
    _MANI = json.load(open(f"{a.load_model}/manifest.json"))
    _bad = [
        k
        for k in ("fold", "d", "layers", "heads", "tbias", "ctx_hours", "feats", "seed", "epochs")
        if _MANI["args"].get(k) != vars(a).get(k)
    ]
    if _bad:
        raise SystemExit(
            "load-model: args differ from the saved run: "
            + ", ".join(f"{k} saved {_MANI['args'].get(k)!r} now {vars(a).get(k)!r}" for k in _bad)
        )
    assert (
        list(_ST["num"]) == num and list(_ST["cats"]) == cats
    ), "load-model: numeric / categorical column lists differ from the saved run"
    na = np.isnan(X)
    flag_cols = _ST["flag_cols"]
    lo, hi, med, mu, sd = (_ST[k] for k in ("lo", "hi", "med", "mu", "sd"))
    X = np.clip(X, lo, hi)
    X = np.where(na, med, X)
    X = (X - mu) / sd
else:
    sub = rng.choice(np.where(fit)[0], min(300000, fit.sum()), replace=False)
    na = np.isnan(X)
    narate = na[sub].mean(0)
    flag_cols = np.where(narate > 0.01)[0]
    lo = np.nanpercentile(X[sub], 0.5, axis=0)
    hi = np.nanpercentile(X[sub], 99.5, axis=0)
    lo = np.nan_to_num(lo)
    hi = np.nan_to_num(hi)
    X = np.clip(X, lo, hi)
    med = np.nan_to_num(np.nanmedian(X[sub], axis=0))
    X = np.where(na, med, X)
    mu = X[sub].mean(0)
    sd = X[sub].std(0)
    sd[sd < 1e-6] = 1.0
    X = (X - mu) / sd
X = np.concatenate([X, na[:, flag_cols].astype(np.float32), offna[:, None].astype(np.float32)], 1).astype(np.float32)
del na
C = np.zeros((d.height, len(cats)), np.int64)
VOC = []
KEEPS = []
for j, c in enumerate(cats):
    v = d[c].cast(pl.Utf8).fill_null("__na__").to_numpy()
    if a.load_model:
        keep = json.loads(str(_ST["keeps_json"]))[
            j
        ]  # SAVE_MODEL: saved vocabulary (values seen < 20 times or never -> 0, as in the fit)
    else:
        u, inv, cnt = np.unique(v[fit], return_inverse=True, return_counts=True)
        keep = {k: i + 1 for i, k in enumerate(u[cnt >= 20])}
    C[:, j] = np.array([keep.get(x, 0) for x in v])
    VOC.append(len(keep) + 1)
    KEEPS.append(keep)
del d
log(
    f"X {X.shape} flags {len(flag_cols)} vocab {VOC} fit {fit.sum():,} held {held.sum():,} held-with-y {(held & np.isfinite(y)).sum():,}"
)
rs = float(_ST["rs"]) if a.load_model else float(np.std(np.clip(r[fit], -3600, 3600)))
log(f"target scale {rs:.1f}")


def write(pred, tag):
    pred = np.maximum(pred, 60.0)
    ok = held & np.isfinite(y)
    if not ok.any():
        ok = held.copy()
        y_ = np.where(ok, pred, np.nan)
        log(
            f"PROD {tag}: {held.sum():,} rows, finite {np.isfinite(pred[held]).mean():.4f}, mean {np.nanmean(pred[held]):.1f}"
        )
    rm = np.sqrt(np.mean((pred[ok] - y[ok]) ** 2))
    rmf = np.sqrt(np.mean((pred[ok & ~fb] - y[ok & ~fb]) ** 2))
    log(f"MEMBER {tag} fold {a.fold} RMSE held {rm:.2f} (n {ok.sum():,}) | non-fb {rmf:.2f}")
    if not a.no_write:
        os.makedirs("data/oof", exist_ok=True)
        f = f"data/oof/big_{tag}_{a.fold}.npy"
        np.save(f, np.vstack([ids[held], pred[held], np.full(held.sum(), np.nan)]))
        log("wrote", f)


# ---------------- set transformer
import torch, torch.nn as nn, torch.nn.functional as Fn

torch.set_num_threads(NT)
torch.manual_seed(a.seed)
dev = torch.device(
    "cuda" if (torch.cuda.is_available() and os.environ.get("CAT_GPU") == "1") else "cpu"
)  # GPU only when the queue granted one
log("device", dev, "threads", NT)
days, start = np.unique(dk, return_index=True)
o = np.argsort(start)
start = start[o]
days = days[o]
end = np.append(start[1:], len(dk))
dheld = held[start]
assert all((held[s:e] == held[s]).all() for s, e in zip(start, end))
# context window (--ctx-hours): cs/ce extend each day's token range into the neighbouring days of the same airport (consecutive dates only)
cs = start.copy()
ce = end.copy()
H = a.ctx_hours
if H > 0:
    nd = len(start)
    apt_d = apt_all[start]
    date_d = date_all[start]
    for i in range(nd):
        if i > 0 and apt_d[i - 1] == apt_d[i] and (date_d[i] - date_d[i - 1]) == np.timedelta64(1, "D"):
            cs[i] = start[i - 1] + np.searchsorted(tod[start[i - 1] : end[i - 1]], 24.0 - H)
        if i + 1 < nd and apt_d[i + 1] == apt_d[i] and (date_d[i + 1] - date_d[i]) == np.timedelta64(1, "D"):
            ce[i] = start[i + 1] + np.searchsorted(tod[start[i + 1] : end[i + 1]], H)
    log(f"ctx-hours {H}: mean tokens/day {np.mean(end - start):.0f} -> {np.mean(ce - cs):.0f}, max {np.max(ce - cs)}")
tr_days = np.where(~dheld)[0]
ho_days = np.where(dheld)[0]
if a.limit_days:
    tr_days = rng.choice(tr_days, a.limit_days, replace=False)
    ho_days = ho_days[: max(20, a.limit_days // 4)]
mon_days = rng.choice(tr_days, max(10, len(tr_days) // 20), replace=False)
tr_fit_days = np.setdiff1d(tr_days, mon_days)  # 5 % monitor days: info only
Xt = torch.from_numpy(X).to(dev)
Ct = torch.from_numpy(C).to(dev)
tt = torch.from_numpy(tod).to(dev)
del X, C
tgt = torch.from_numpy((np.clip(np.nan_to_num(r), -3600, 3600) / rs).astype(np.float32)).to(dev)
wt = torch.from_numpy(fit.astype(np.float32)).to(dev)


def batch(ix):
    L = int(max(ce[i] - cs[i] for i in ix))
    B = len(ix)
    F = Xt.shape[1]
    xb = torch.zeros(B, L, F, device=dev)
    cb = torch.zeros(B, L, len(cats), dtype=torch.long, device=dev)
    tb = torch.zeros(B, L, device=dev)
    yb = torch.zeros(B, L, device=dev)
    wb = torch.zeros(B, L, device=dev)
    mk = torch.zeros(B, L, dtype=torch.bool, device=dev)
    for k, i in enumerate(ix):
        s0, e0 = cs[i], ce[i]
        n = e0 - s0
        xb[k, :n] = Xt[s0:e0]
        cb[k, :n] = Ct[s0:e0]
        t = tt[s0:e0].clone()
        if H > 0:
            np_, nn_ = start[i] - s0, e0 - end[i]
            if np_:
                t[:np_] -= 24.0
            if nn_:
                t[n - nn_ :] += 24.0
        tb[k, :n] = t
        yb[k, :n] = tgt[s0:e0]
        w = wt[s0:e0].clone()
        if H > 0:
            w[: start[i] - s0] = 0.0
            w[end[i] - s0 :] = 0.0  # context tokens: no loss
        wb[k, :n] = w
        mk[k, :n] = True
    return xb, cb, tb, yb, wb, mk


class Layer(nn.Module):
    def __init__(s, D, Hh, dirb):
        super().__init__()
        s.H = Hh
        s.dirb = dirb
        s.n1 = nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D)
        s.o = nn.Linear(D, D)
        s.n2 = nn.LayerNorm(D)
        s.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * D, D))
        s.dr = nn.Dropout(0.1)
        s.slope = nn.Parameter(torch.linspace(-2, 1, Hh))
        if dirb:
            s.slope_n = nn.Parameter(torch.linspace(-2, 1, Hh))
            s.csign = nn.Parameter(torch.zeros(Hh))

    def forward(s, x, tb, mk):
        B, L, D = x.shape
        h = s.n1(x)
        q, k, v = s.qkv(h).view(B, L, 3, s.H, D // s.H).permute(2, 0, 3, 1, 4)
        dt = tb[:, None, None, :] - tb[:, None, :, None]  # t_key - t_query
        if s.dirb:
            bias = (
                -Fn.softplus(s.slope).view(1, s.H, 1, 1) * dt.clamp(min=0)
                - Fn.softplus(s.slope_n).view(1, s.H, 1, 1) * (-dt).clamp(min=0)
                + s.csign.view(1, s.H, 1, 1) * torch.sign(dt)
            )
        else:
            bias = -Fn.softplus(s.slope).view(1, s.H, 1, 1) * dt.abs()
        allow = (mk[:, None, None, :] & mk[:, None, :, None]) | torch.eye(L, dtype=torch.bool, device=x.device)[
            None, None
        ]  # padded rows attend to themselves
        bias = bias.masked_fill(~allow, float("-inf")).to(q.dtype)
        att = Fn.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.1 if s.training else 0.0)
        x = x + s.dr(s.o(att.transpose(1, 2).reshape(B, L, D)))
        return x + s.dr(s.ff(s.n2(x)))


class Net(nn.Module):
    def __init__(s, F, D, Hh, NL, dirb):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(F, 2 * D), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * D, D))
        s.emb = nn.ModuleList([nn.Embedding(v, D) for v in VOC])
        s.tim = nn.Linear(5, D)
        s.layers = nn.ModuleList([Layer(D, Hh, dirb) for _ in range(NL)])
        s.nf = nn.LayerNorm(D)
        s.head = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 1))

    def forward(s, xb, cb, tb, mk):
        w = 2 * np.pi * tb / 24
        x0 = (
            s.inp(xb)
            + sum(e(cb[..., j]) for j, e in enumerate(s.emb))
            + s.tim(torch.stack([torch.sin(w), torch.cos(w), torch.sin(2 * w), torch.cos(2 * w), tb / 24 - 0.5], -1))
        )
        x = x0
        for l in s.layers:
            x = l(x, tb, mk)
        return s.head(torch.cat([x0, s.nf(x)], -1)).squeeze(-1)


net = Net(Xt.shape[1], a.d, a.heads, a.layers, a.tbias == "dir").to(dev)
log("params", sum(p.numel() for p in net.parameters()))
USE_AMP = bool(a.amp and dev.type == "cuda")


def fwd(xb, cb, tb, mk):
    if USE_AMP:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(xb, cb, tb, mk).float()
    return net(xb, cb, tb, mk)


nb = int(np.ceil(len(tr_fit_days) / a.bs))
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb, pct_start=0.1)


def predict(dlist):
    net.eval()
    out = np.full(len(y), np.nan)
    with torch.no_grad():
        for b in range(0, len(dlist), a.bs):
            ix = dlist[b : b + a.bs]
            xb, cb, tb, yb, wb, mk = batch(ix)
            p = fwd(xb, cb, tb, mk).cpu().numpy() * rs
            for k, i in enumerate(ix):
                o0 = start[i] - cs[i]
                out[start[i] : end[i]] = p[k, o0 : o0 + end[i] - start[i]]
    net.train()
    return out


if a.load_model:  # SAVE_MODEL: weights from the saved run, no training
    net.load_state_dict(torch.load(f"{a.load_model}/state_dict.pt", map_location=dev))
    log("loaded", f"{a.load_model}/state_dict.pt")
for ep in (range(0) if a.load_model else range(a.epochs)):
    perm = rng.permutation(tr_fit_days)
    tl = 0.0
    tn = 0.0
    te = time.time()
    for b in range(nb):
        xb, cb, tb, yb, wb, mk = batch(perm[b * a.bs : (b + 1) * a.bs])
        p = fwd(xb, cb, tb, mk)
        loss = (wb * (p - yb) ** 2).sum() / wb.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        tl += loss.item() * wb.sum().item()
        tn += wb.sum().item()
        if ep == 0 and b == 4:
            log(f"  5 steps {time.time() - te:.1f}s -> est epoch {(time.time() - te) / 5 * nb / 60:.1f} min")
    pm = predict(mon_days)
    mi = np.concatenate([np.arange(start[i], end[i]) for i in mon_days])
    mi = mi[fit[mi]]
    log(
        f"ep {ep} train mse(rs^2) {tl / tn:.4f} -> rmse {np.sqrt(tl / tn) * rs:.1f} | monitor-days rmse(r) {np.sqrt(np.mean((pm[mi] - np.clip(r[mi], -3600, 3600)) ** 2)):.1f} | {time.time() - te:.0f}s"
    )
ph = predict(ho_days)
pred = np.full(len(y), np.nan)
hm = np.isfinite(ph)
pred[hm] = off[hm] + ph[hm]
if a.limit_days:
    ok = hm & np.isfinite(y)
    log(f"SMOKE held subset rmse {np.sqrt(np.mean((np.maximum(pred[ok], 60) - y[ok]) ** 2)):.1f} n {ok.sum()}")
    sys.exit(0)
write(pred, a.tag)
if a.save_model or a.load_model:  # SAVE_MODEL
    import json, hashlib, platform, socket

    def _md5(p_):
        h = hashlib.md5()
        with open(p_, "rb") as fh:
            for b_ in iter(lambda: fh.read(1 << 22), b""):
                h.update(b_)
        return h.hexdigest()

    _pv = np.maximum(pred[held], 60.0)
    _pmd5 = hashlib.md5(np.ascontiguousarray(np.vstack([ids[held].astype(np.float64), _pv])).tobytes()).hexdigest()
    if a.save_model:
        os.makedirs(a.save_model, exist_ok=True)
        torch.save(net.state_dict(), f"{a.save_model}/state_dict.pt")
        np.savez(
            f"{a.save_model}/stats.npz",
            num=np.array(num),
            cats=np.array(cats),
            flag_cols=flag_cols,
            lo=lo,
            hi=hi,
            med=med,
            mu=mu,
            sd=sd,
            rs=rs,
            keeps_json=np.array(json.dumps(KEEPS)),
        )
        _man = dict(
            script=os.path.basename(__file__),
            script_md5=_md5(os.path.abspath(__file__)),
            argv=sys.argv,
            args=vars(a),
            host=socket.gethostname(),
            arch=platform.machine(),
            device=str(dev),
            torch=torch.__version__,
            t_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            vocab=VOC,
            n_num=len(num),
            n_flags=int(len(flag_cols)),
            features=dict(path=os.path.abspath(P), bytes=os.path.getsize(P), md5=_md5(P)),
            pred_md5=_pmd5,
            n_held=int(held.sum()),
            pred_mean=float(np.mean(_pv)),
            state_md5=_md5(f"{a.save_model}/state_dict.pt"),
            stats_md5=_md5(f"{a.save_model}/stats.npz"),
            secs=round(time.time() - T0, 1),
        )
        _man["save_bytes"] = int(sum(os.path.getsize(f"{a.save_model}/{f}") for f in os.listdir(a.save_model)))
        json.dump(_man, open(f"{a.save_model}/manifest.json", "w"), indent=1, default=str)
        log(f"SAVE_MODEL saved {a.save_model} ({_man['save_bytes'] / 1e6:.1f} MB) pred md5 {_pmd5}")
    else:
        log(
            f"SAVE_MODEL load done: pred md5 {_pmd5} ({'== saved run' if _pmd5 == _MANI.get('pred_md5') else 'differs from the saved run'}), {time.time() - T0:.0f}s"
        )
