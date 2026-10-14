#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Trains or applies the NBR_CLOCK LightGBM member (with or without neighbour ADS-B clock set-statistics), predicting
taxi time as mvt_aobt3 + a fitted offset.
Inputs: data/features_v9.parquet plus external ADS-B/durp/dreg/xba/xe feature files and queue_eb_ra3qa OOF files;
positional args ARM, UNITS; optional --save-model/--load-model/--pred-name.
Outputs: per-fold data/oof/nbr_clock_<ARM>_<fold>.npy, or (prod) data/preds/nbr_clock_<ARM>P_s0.parquet.

NBR_CLOCK: plain OUT-OF-MONTH member (the chain-member form) with / without neighbour ADS-B clock set-statistics.
usage: nbrclock_member.py ARM UNITS     ARM in MA (412 cols) | MB (412 + 10 clk_*) | MBm (MB with clk_* NaN at coverage-drift-masked airports, §4b)
       UNITS = comma list of folds (months39,months28,months410,janjul,months511,months612) and/or prod
Target clip(y - mvt_aobt3, -1800, 3600); LightGBM fixed recipe (lr 0.05, 255 leaves, min_data 200, ff 0.8, bag 0.8/1, l2 1, 900 rounds, seed 0).
Train = 2025 matched non-rank rows, finite y and mvt_aobt3, month not in the unit's hold (prod: all months), UNS days only (sealed never read).
Fold output data/oof/nbr_clock_<ARM>_<fold>.npy = [id, taxi, pred_offset] on is_rank==0 & m in hold rows (features_v9 order; the Kumo-member file format read by the chain variants);
prod output data/preds/nbr_clock_<ARM>P_s0.parquet (MVT_ID_mvt, taxi, g) on is_rank==1 rows. taxi = mvt_aobt3 + pred; rows without mvt_aobt3 take the
queue_eb_ra3qa taxi (folds) / the 2025 airport median y (prod), identical in every arm."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, time, numpy as np, polars as pl, lightgbm as lgb

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
THR = int(os.environ.get("PRC_THREADS", "4"))
OUT = f"{ROOT}/scratchpad/xnclk"
os.makedirs(OUT, exist_ok=True)
HOLD = {
    "months39": [3, 9],
    "months28": [2, 8],
    "months410": [4, 10],
    "janjul": [1, 7],
    "months612": [6, 12],
    "months511": [5, 11],
}
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
src = open(f"{ROOT}/scripts/train2.py").read()
ns = {}
exec(
    src[src.index("CAT_BASE = ") : src.index("def dayshift_model")], ns
)  # CAT_BASE, cols_for exactly as kumo_km_member.py
FV = f"{ROOT}/data/features_v9.parquet"
schema = pl.read_parquet_schema(FV)
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
assert len(num) == 412, len(num)
ARM = sys.argv[1]
UNITS = sys.argv[2].split(",")
assert ARM in ("MA", "MB", "MBm")


def _opt(k):  # SAVE_MODEL: optional trailing flags (absent = unchanged behaviour)
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else ""


opt_save, opt_load, opt_pred = _opt("--save-model"), _opt("--load-model"), _opt("--pred-name")
if (opt_save or opt_load) and UNITS != ["prod"]:
    raise SystemExit("SAVE_MODEL: --save-model / --load-model only with UNITS = prod")
if opt_load:
    import json, gzip

    opt_man = json.load(open(f"{opt_load}/manifest.json"))
    assert opt_man["arm"] == ARM, (opt_man["arm"], ARM)
PRM = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=THR,
    seed=0,
)
ROUNDS = 900


def log(*a):
    print(*a, flush=True)


# ---------- all-row frame (label-free) ----------
F = (
    pl.scan_parquet(FV)
    .select(
        "MVT_ID_mvt",
        pl.col("ADEP_mvt").cast(pl.Utf8),
        pl.col("RUNWAY_mvt").cast(pl.Utf8),
        "MVT_TIME_UTC_mvt",
        "mvt_aobt3",
        "ref_sr_med",
    )
    .collect()
    .with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
)
for f, c in [
    ("adsb_features", ["adsb_move_aobt", "adsb_linked"]),
    ("adsb2_features", ["adsb2_leave_aobt"]),
    ("adsb3_features", ["adsb3_mv_gs_aobt"]),
]:
    A = pl.read_parquet(f"{ROOT}/data/external/{f}.parquet", columns=["MVT_ID_mvt"] + c).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Float64)
    )
    F = F.join(A, on="MVT_ID_mvt", how="left")
F = F.filter(pl.col("MVT_TIME_UTC_mvt").is_not_null()).with_columns(
    t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)
)
F = F.sort(["ADEP_mvt", "t"])
log(f"# all rows {F.height:,}; leave clock finite {F['adsb2_leave_aobt'].is_not_null().mean():.3f}")


def wmean(grp, t, v, w):
    """mean of finite v over rows of the same group with |dt| <= w, self excluded; also count of finite and of all neighbours."""
    out = np.full(len(t), np.nan)
    nf = np.zeros(len(t))
    na = np.zeros(len(t))
    for g in np.unique(grp):
        ix = np.flatnonzero(grp == g)
        o = ix[np.argsort(t[ix], kind="stable")]
        tt = t[o]
        vv = v[o]
        fin = np.isfinite(vv)
        cs = np.concatenate([[0], np.cumsum(np.where(fin, vv, 0))])
        cn = np.concatenate([[0], np.cumsum(fin)])
        lo = np.searchsorted(tt, tt - w, "left")
        hi = np.searchsorted(tt, tt + w, "right")
        s = cs[hi] - cs[lo] - np.where(fin, vv, 0)
        n = cn[hi] - cn[lo] - fin
        a = hi - lo - 1
        out[o] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        nf[o] = n
        na[o] = a
    return out, nf, na


apt = F["ADEP_mvt"].to_numpy().astype(str)
rwy = (F["ADEP_mvt"] + "|" + F["RUNWAY_mvt"].fill_null("?")).to_numpy().astype(str)
tt = F["t"].to_numpy()
CL = {
    "lv": np.clip(F["adsb2_leave_aobt"].cast(pl.Float64).fill_null(np.nan).to_numpy(), -1800, 1800),
    "mv": np.clip(F["adsb_move_aobt"].cast(pl.Float64).fill_null(np.nan).to_numpy(), -1800, 1800),
    "gs": np.clip(F["adsb3_mv_gs_aobt"].cast(pl.Float64).fill_null(np.nan).to_numpy(), -1800, 1800),
}
FE = {}
for k, v in CL.items():
    FE[f"clk_a15_{k}"], nf, na = wmean(apt, tt, v, 900)
    if k == "lv":
        FE["clk_a15_n"] = np.where(na > 0, nf / np.maximum(na, 1), np.nan)
    FE[f"clk_a5_{k}"], _, _ = wmean(apt, tt, v, 300)
    FE[f"clk_r15_{k}"], _, _ = wmean(rwy, tt, v, 900)
CLK = list(FE)
ids_all = F["MVT_ID_mvt"].to_numpy()
NBX = np.column_stack([FE[c] for c in CLK]).astype(np.float32)
o_id = np.argsort(ids_all)
ids_sorted = ids_all[o_id]
log(f"clk features {CLK} | {time.time()-t0:.0f}s")
yr_all = F["MVT_TIME_UTC_mvt"].dt.year().to_numpy()
own = np.isfinite(CL["lv"])
MASK = []
for a in np.unique(apt):
    s5, s6 = (apt == a) & (yr_all == 2025), (apt == a) & (yr_all == 2026)
    dn = np.isfinite(FE["clk_a15_lv"][s6]).mean() - np.isfinite(FE["clk_a15_lv"][s5]).mean()
    do = own[s6].mean() - own[s5].mean()
    log(f"  coverage {a}: d_nbr {dn:+.3f} d_own {do:+.3f}{'  MASKED' if (abs(dn) > 0.25 or abs(do) > 0.25) else ''}")
    if abs(dn) > 0.25 or abs(do) > 0.25:
        MASK.append(a)
log(f"# §4b mask (label-free): {MASK}")
if (
    opt_load
):  # SAVE_MODEL: the model was fitted with the saved mask -> apply the saved one; a changed mask means the recipe would refit
    opt_mask_now = list(MASK)
    MASK = list(opt_man["mask"])
    if opt_mask_now != MASK:
        log(
            f"WARNING SAVE_MODEL: coverage mask on this table {opt_mask_now} != saved {MASK}; using the SAVED mask (the recipe would refit with the new one)"
        )
CLK = list(FE)
ids_all = F["MVT_ID_mvt"].to_numpy()
NBX = np.column_stack([FE[c] for c in CLK]).astype(np.float32)
o_id = np.argsort(ids_all)
ids_sorted = ids_all[o_id]
del F
SD = pl.read_parquet(f"{ROOT}/data/sealed_days.parquet").select(pl.col("ADEP_mvt").cast(pl.Utf8), "day", "sealed")
d = (
    pl.scan_parquet(FV)
    .select(
        [
            c
            for c in dict.fromkeys(
                [
                    "MVT_ID_mvt",
                    "ADEP_mvt",
                    "MVT_TIME_UTC_mvt",
                    "unmatched",
                    "is_rank",
                    "m",
                    "TAXITIME_SEC_mvt",
                    "mvt_aobt3",
                ]
                + num
            )
            if c in schema
        ]
    )
    .collect()
)
for g, f in SIDE.items():
    cols = [c for c in side_sch[g] if c in num]
    if cols:
        d = d.join(
            pl.read_parquet(f"{ROOT}/data/external/{f}.parquet", columns=["MVT_ID_mvt"] + cols),
            on="MVT_ID_mvt",
            how="left",
            maintain_order="left",
        )
d = d.with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Float64),
    apt=pl.col("ADEP_mvt").cast(pl.Utf8),
    day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
)
d = d.join(SD, left_on=["apt", "day"], right_on=["ADEP_mvt", "day"], how="left", maintain_order="left")
ids = d["MVT_ID_mvt"].to_numpy()
aptr = d["apt"].to_numpy().astype(str)
rk = d["is_rank"].to_numpy() == 1
mo = d["m"].to_numpy()
y = d["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
off = d["mvt_aobt3"].cast(pl.Float64).to_numpy()
sealed = d["sealed"].fill_null(True).to_numpy().astype(bool)
unm = d["unmatched"].to_numpy() == 1
X = d.select(num).cast(pl.Float32).fill_nan(None).to_numpy().astype(np.float32)
del d
if ARM != "MA":
    kn = np.clip(np.searchsorted(ids_sorted, ids), 0, len(ids_sorted) - 1)
    hit = ids_sorted[kn] == ids
    NBk = NBX[o_id[kn]].copy()
    NBk[~hit] = np.nan
    if ARM == "MBm":
        NBk[np.isin(aptr, MASK)] = np.nan
    X = np.hstack([X, NBk])
    log(f"clk cols added (hit {hit.mean():.4f}); MBm masked rows {np.isin(aptr, MASK).sum() if ARM == 'MBm' else 0:,}")
log(f"NBR_CLOCK member {ARM} host {os.uname().nodename} threads {THR} | X {X.shape} | {time.time()-t0:.0f}s")
tgt = np.clip(y - off, -1800, 3600)
base_tr = (~rk) & (~unm) & np.isfinite(y) & np.isfinite(off) & (~sealed)


def npy(p, t):
    c = glob.glob(f"{ROOT}/data/oof/{p}_{t}.npy") or glob.glob(f"{ROOT}/data/oof/{p}_{t}_*.npy")
    return np.load(sorted(c)[0])


for u in UNITS:
    t1 = time.time()
    if u == "prod":
        tr = base_tr
        te = rk
    else:
        tr = base_tr & ~np.isin(mo, HOLD[u])
        te = (~rk) & np.isin(mo, HOLD[u])
    if opt_load:
        with gzip.open(f"{opt_load}/lgb.txt.gz", "rt") as fh:
            b = lgb.Booster(model_str=fh.read())
        assert b.num_feature() == X.shape[1] == opt_man["n_feat"], (b.num_feature(), X.shape[1])
        log(f"SAVE_MODEL loaded {opt_load}/lgb.txt.gz ({b.num_trees()} trees)")
    else:
        b = lgb.train(PRM, lgb.Dataset(X[tr], tgt[tr], free_raw_data=True), ROUNDS)
    t_fit = time.time() - t1
    g = b.predict(X[te], num_threads=THR)
    taxi = off[te] + g
    miss = ~np.isfinite(taxi)
    if u == "prod":
        med = {a: np.nanmedian(y[base_tr & (aptr == a)]) for a in np.unique(aptr)}
        if opt_load:
            med = {k_: float(v_) for k_, v_ in opt_man["med"].items()}
        taxi[miss] = np.array([med[a] for a in aptr[te][miss]])
        _pf = f"{ROOT}/data/preds/{opt_pred or f'nbr_clock_{ARM}P_s0'}.parquet"
        pl.DataFrame({"MVT_ID_mvt": ids[te], "taxi": taxi, "g": g}).write_parquet(_pf)
        if opt_save or opt_load:
            import json, gzip, hashlib, platform

            def _md5(p_):
                h = hashlib.md5()
                with open(p_, "rb") as fh:
                    for b_ in iter(lambda: fh.read(1 << 22), b""):
                        h.update(b_)
                return h.hexdigest()

            _pm = _md5(_pf)
            if opt_save:
                os.makedirs(opt_save, exist_ok=True)
                t2 = time.time()
                with gzip.open(f"{opt_save}/lgb.txt.gz", "wt", compresslevel=6) as fh:
                    fh.write(b.model_to_string(num_iteration=-1))
                _rd = [FV, f"{ROOT}/data/sealed_days.parquet"] + [
                    f"{ROOT}/data/external/{f}.parquet"
                    for f in dict.fromkeys(list(SIDE.values()) + ["adsb_features", "adsb2_features", "adsb3_features"])
                ]
                man = dict(
                    script="nbrclock_member.py",
                    script_md5=_md5(os.path.abspath(__file__)),
                    argv=sys.argv,
                    arm=ARM,
                    host=os.uname().nodename,
                    arch=platform.machine(),
                    lightgbm=lgb.__version__,
                    t_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    prm={k_: v_ for k_, v_ in PRM.items() if k_ != "num_threads"},
                    rounds=ROUNDS,
                    num=num,
                    clk=CLK,
                    n_feat=int(X.shape[1]),
                    mask=list(MASK),
                    med={k_: float(v_) for k_, v_ in med.items()},
                    n_train=int(tr.sum()),
                    n_pred=int(te.sum()),
                    fit_s=round(t_fit, 1),
                    save_s=round(time.time() - t2, 1),
                    pred_file=_pf,
                    pred_md5=_pm,
                    model_md5=_md5(f"{opt_save}/lgb.txt.gz"),
                    reads=[dict(path=r_, bytes=os.path.getsize(r_), md5=_md5(r_)) for r_ in _rd if os.path.exists(r_)],
                )
                man["save_bytes"] = int(sum(os.path.getsize(f"{opt_save}/{f}") for f in os.listdir(opt_save)))
                json.dump(man, open(f"{opt_save}/manifest.json", "w"), indent=1)
                log(f"SAVE_MODEL saved {opt_save} ({man['save_bytes'] / 1e6:.1f} MB) pred md5 {_pm}")
            else:
                json.dump(
                    dict(
                        load_dir=opt_load,
                        pred_md5=_pm,
                        same_as_saved=_pm == opt_man["pred_md5"],
                        mask_now=opt_mask_now,
                        mask_saved=MASK,
                        mask_changed=opt_mask_now != MASK,
                        secs=round(time.time() - t0, 1),
                    ),
                    open(_pf + ".xsave_load.json", "w"),
                    indent=1,
                )
                log(
                    f"SAVE_MODEL load done: pred md5 {_pm} ({'== saved run' if _pm == opt_man['pred_md5'] else 'differs from the saved run'})"
                )
    else:
        Lz = npy("queue_eb_ra3qa", u)
        oL = np.argsort(Lz[0])
        k = np.clip(np.searchsorted(Lz[0][oL], ids[te]), 0, len(oL) - 1)
        hitL = Lz[0][oL][k] == ids[te]
        Lt = np.where(hitL, Lz[3][oL][k], np.nan)
        taxi[miss] = Lt[miss]
        assert np.isfinite(taxi).all(), (u, int((~np.isfinite(taxi)).sum()))
        np.save(f"{ROOT}/data/oof/nbr_clock_{ARM}_{u}.npy", np.vstack([ids[te], taxi, g]))
        sc = te & ~unm & np.isfinite(y) & ~sealed
        log(f"  {u}: rmse UNS matched {np.sqrt(np.nanmean((taxi[sc[te]] - y[sc]) ** 2)):.2f}")
    log(f"{ARM} {u}: train {tr.sum():,} pred {te.sum():,} fallback {miss.sum():,} | {time.time()-t1:.0f}s")
log(f"nbr_clock_member_done {ARM} {time.time()-t0:.0f}s")
