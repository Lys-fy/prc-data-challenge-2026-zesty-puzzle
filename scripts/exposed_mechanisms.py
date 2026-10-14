# SPDX-License-Identifier: GPL-3.0-only
"""Mechanism tests and pre-declared gates for the huge-error unmatched rows.
Part 1  M3 test: is BLOCK on OTHER-regime rows an arrival's in-block time at the same airport (same airline prefix)? placebo = BLOCK shifted +1 h.
Part 2  bias of the proxy / true frozen on unm_LIRF by MS band; 2026 unm_LIRF composition.
Part 3  PX / SNC / PXC on 2025 (proxy all months, TRUE frozen janjul) and the 2026 scenario ledger + per-row break-even (G1-G4).
Part 4  expected-SE map of the 412 frozen 2026 rows under the current prediction (regime mixture from 2025 cells).
Read-only on data/; writes scratchpad/xhr/ only."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(320)
pl.Config.set_tbl_rows(200)
pl.Config.set_fmt_str_lengths(16)
pl.Config.set_fmt_float("mixed")
RN = int(os.environ.get("P2_NRANK", "344841"))  # ranking rows (P2_NRANK from the period file; default: public file)
L = pl.read_parquet("scratchpad/xhr/unm2025b.parquet")
R = pl.read_parquet("scratchpad/xhr/unm2026.parquet")
E = pl.read_parquet("scratchpad/xhr/exposed_rows.parquet")
sec = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_seconds()

# ---------------- Part 1: M3 (BLOCK = an arrival's in-block) ----------------
RC = ["FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
arr = pl.concat(
    [pl.read_parquet(f, columns=RC) for f in sorted(glob.glob("data/raw/training_*.parquet"))], how="diagonal_relaxed"
).filter((pl.col("PHASE_mvt") == "ARR") & pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
arr = arr.with_columns(al=pl.col("FLIGHT_mvt").str.slice(0, 3))


def nearest(left, key_expr, by, prefix, rcol="BLOCK_TIME_UTC_mvt"):
    r = arr.select(
        by + [pl.col(rcol).alias("_k"), pl.col(rcol).alias(prefix + "t"), pl.col("FLIGHT_mvt").alias(prefix + "flt")]
    ).sort("_k")
    l = left.select("MVT_ID_mvt", *by, key_expr.alias("_k")).sort("_k")
    j = l.join_asof(r, on="_k", by=by, strategy="nearest", tolerance="6h")
    j = j.with_columns((pl.col("_k") - pl.col(prefix + "t")).dt.total_seconds().alias(prefix + "d")).drop("_k", *by)
    return left.join(j, on="MVT_ID_mvt", how="left")


L1 = L.filter(pl.col("BLOCK_TIME_UTC_mvt").is_not_null())
L1 = nearest(
    L1, pl.col("BLOCK_TIME_UTC_mvt"), ["ADEP_mvt", "al"], "m3_"
)  # same airline prefix, in-block nearest to BLOCK
L1 = nearest(L1, pl.col("BLOCK_TIME_UTC_mvt") + pl.duration(hours=1), ["ADEP_mvt", "al"], "pl_")  # placebo
L1 = nearest(L1, pl.col("BLOCK_TIME_UTC_mvt"), ["ADEP_mvt"], "m3any_")  # any airline
L1 = nearest(
    L1, pl.col("BLOCK_TIME_UTC_mvt"), ["ADEP_mvt", "al"], "m3land_", rcol="MVT_TIME_UTC_mvt"
)  # landing time instead of in-block
for tol in (30, 60, 120):
    print(
        f"\n=== M3: share of rows whose BLOCK is within {tol} s of a same-airline ARRIVAL in-block at the same airport (placebo: BLOCK+1h) | any airline | landing ==="
    )
    print(
        L1.group_by("cls", "regime")
        .agg(
            n=pl.len(),
            m3=(pl.col("m3_d").abs() <= tol).mean(),
            placebo=(pl.col("pl_d").abs() <= tol).mean(),
            any_al=(pl.col("m3any_d").abs() <= tol).mean(),
            landing=(pl.col("m3land_d").abs() <= tol).mean(),
        )
        .sort("cls", "regime")
    )
O = L1.filter((pl.col("regime") == "OTHER") & (pl.col("y") > 3600))
print("\n=== OTHER rows with y > 3600: LIRF + the largest non-LIRF, with the nearest same-airline arrival in-block ===")
print(
    O.sort("cls", pl.col("err").abs(), descending=[False, True])
    .select(
        "cls",
        "ADEP_mvt",
        "FLIGHT_mvt",
        "STAND_mvt",
        "SCHED_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt",
        "MVT_TIME_UTC_mvt",
        "y",
        "mvt_sched",
        "px",
        "err",
        "m3_flt",
        "m3_d",
        "m3any_flt",
        "m3any_d",
        "m3land_d",
    )
    .head(70)
)
print("\n=== M3 by airport for OTHER rows with y > 3600 (share within 60 s, same airline) ===")
print(
    O.group_by("ADEP_mvt")
    .agg(
        n=pl.len(),
        m3_60=(pl.col("m3_d").abs() <= 60).mean(),
        placebo=(pl.col("pl_d").abs() <= 60).mean(),
        y_med=pl.col("y").median(),
    )
    .sort("n", descending=True)
)
L1.select("MVT_ID_mvt", "m3_flt", "m3_d", "pl_d", "m3any_d", "m3land_d").write_parquet("scratchpad/xhr/m3.parquet")


# ---------------- Part 2: bias by MS band ----------------
def band(c):
    return (
        pl.when(pl.col(c) < 3600)
        .then(pl.lit("a<1h"))
        .when(pl.col(c) < 7200)
        .then(pl.lit("b1-2h"))
        .when(pl.col(c) < 14400)
        .then(pl.lit("c2-4h"))
        .when(pl.col(c) < 28800)
        .then(pl.lit("d4-8h"))
        .when(pl.col(c) < 43200)
        .then(pl.lit("e8-12h"))
        .when(pl.col(c) < 57600)
        .then(pl.lit("f12-16h"))
        .otherwise(pl.lit("g>16h"))
    )


L = L.with_columns(band=band("mvt_sched"))
R = R.with_columns(band=band("mvt_sched"), m=pl.col("MVT_TIME_UTC_mvt").dt.month())
LI = L.filter(pl.col("cls") == "unm_LIRF")
print("\n=== unm_LIRF 2025: proxy bias (mean pred - y) and MSE by MS band; regime shares ===")
print(
    LI.group_by("band")
    .agg(
        n=pl.len(),
        bias=pl.col("err").mean(),
        mse=(pl.col("err") ** 2).mean(),
        y_mean=pl.col("y").mean(),
        pred_mean=pl.col("px").mean(),
        SUB=(pl.col("regime") == "SUB").mean(),
        ORD=(pl.col("regime") == "ORD").mean(),
        OTH=(pl.col("regime") == "OTHER").mean(),
        y_oth_med=pl.col("y").filter(pl.col("regime") == "OTHER").median(),
    )
    .sort("band")
)
# true frozen janjul on unm_LIRF
V25 = [f"V25L_s{i}" for i in range(10)] + [f"V25Lb_s{i}" for i in range(10, 15)]
CAT9 = ["CATV9x_s0", "CATV9x_s1"]


def oof(names, ids, col=1):
    M = []
    for n in names:
        a = np.load(f"data/oof/{n}_janjul.npy")
        d = dict(zip(a[0].astype(np.int64), a[col]))
        M.append([d.get(i, np.nan) for i in ids])
    return np.nanmean(np.array(M, float), 0)


LJ = LI.filter(pl.col("m").is_in([1, 7])).sort("MVT_ID_mvt")
ids = LJ["MVT_ID_mvt"].to_numpy()
fz = np.maximum(0.5 * oof(V25, ids) + 0.5 * oof(CAT9, ids), 60.0)
fzp = 0.5 * oof(V25, ids, 2) + 0.5 * oof(CAT9, ids, 2)
LJ = LJ.with_columns(fz=pl.lit(fz), fz_p=pl.lit(fzp)).with_columns(fz_err=pl.col("fz") - pl.col("y"))
print("\n=== unm_LIRF janjul: TRUE frozen bias / MSE by MS band x month ===")
print(
    LJ.group_by("band", "m")
    .agg(
        n=pl.len(),
        bias=pl.col("fz_err").mean(),
        mse=(pl.col("fz_err") ** 2).mean(),
        proxy_mse=(pl.col("err") ** 2).mean(),
        SUB=(pl.col("regime") == "SUB").mean(),
        OTH=(pl.col("regime") == "OTHER").mean(),
    )
    .sort("band", "m")
)
print("\n=== 2026 unm_LIRF: rows by MS band x month, crossers, v62 median ===")
print(
    R.filter(pl.col("cls") == "unm_LIRF")
    .group_by("band", "m")
    .agg(n=pl.len(), cross=pl.col("cross").sum(), v62_med=pl.col("v62").median(), ms_med=pl.col("mvt_sched").median())
    .sort("band", "m")
)
LJ.select("MVT_ID_mvt", "fz", "fz_p", "fz_err").write_parquet("scratchpad/xhr/unm_lirf_janjul_frozen.parquet")

# ---- Part 2b: the -2 h class (TZ2) ----
A = LI.filter(pl.col("band") == "a<1h").with_columns(
    tz2=(pl.col("regime") == "OTHER") & (pl.col("y") >= 6000) & (pl.col("y") <= 10000)
)
print(
    "\n=== unm_LIRF MS<1h: y - 7200 for OTHER rows (should look like ordinary taxi) ===",
    sorted((A.filter(pl.col("regime") == "OTHER")["y"] - 7200).to_list()),
)
print("\n=== unm_LIRF MS<1h: airline x regime (2025) ===")
print(
    A.group_by("al")
    .agg(
        n=pl.len(),
        tz2=pl.col("tz2").sum(),
        SUB=(pl.col("regime") == "SUB").sum(),
        ORD=(pl.col("regime") == "ORD").sum(),
        OTH_other=((pl.col("regime") == "OTHER") & ~pl.col("tz2")).sum(),
        y_med=pl.col("y").median(),
        px_med=pl.col("px").median(),
    )
    .sort("n", descending=True)
)
print("\n=== unm_LIRF MS<1h: month x tz2 rate ===")
print(A.group_by("m").agg(n=pl.len(), tz2=pl.col("tz2").mean(), SUB=(pl.col("regime") == "SUB").mean()).sort("m"))
print("\n=== tz2-like rows (OTHER, 6000<=y<=10000, MS<1h) at other airports ===")
print(
    L.filter(
        (pl.col("band") == "a<1h") & (pl.col("regime") == "OTHER") & (pl.col("y") >= 6000) & (pl.col("y") <= 10000)
    )
    .group_by("ADEP_mvt")
    .agg(n=pl.len(), al=pl.col("al").unique().head(8))
    .sort("n", descending=True)
)
print("\n=== unm_LIRF MS in 1-2h: airline x regime for airlines in the MS<1h tz2 set ===")
S_all = (
    A.group_by("al")
    .agg(n=pl.len(), r=pl.col("tz2").mean())
    .filter((pl.col("n") >= 4) & (pl.col("r") >= 0.8))["al"]
    .to_list()
)
print("S (all months):", S_all)
print(
    LI.filter(pl.col("al").is_in(S_all))
    .group_by("band", "al")
    .agg(
        n=pl.len(),
        tz2=((pl.col("regime") == "OTHER") & (pl.col("y") >= 6000) & (pl.col("y") <= 10000)).mean(),
        SUB=(pl.col("regime") == "SUB").mean(),
        ORD=(pl.col("regime") == "ORD").mean(),
        y_med=pl.col("y").median(),
    )
    .sort("band", "al")
)
# T1: leave-one-month-out selection of S
hits = tot = 0
for m in range(1, 13):
    tr = A.filter(pl.col("m") != m)
    te = A.filter(pl.col("m") == m)
    S = (
        tr.group_by("al")
        .agg(n=pl.len(), r=pl.col("tz2").mean())
        .filter((pl.col("n") >= 4) & (pl.col("r") >= 0.8))["al"]
        .to_list()
    )
    k = te.filter(pl.col("al").is_in(S))
    hits += int(k["tz2"].sum())
    tot += k.height
print(f"T1 LOMO: held-out rows flagged {tot}, of which tz2 {hits} ({hits / max(tot, 1):.2f})")
taxi_ord = float((A.filter(pl.col("tz2") & pl.col("al").is_in(S_all))["y"] - 7200).median())
print("taxi_ord (median y-7200 on S rows):", taxi_ord)
# T3 proxy all months: rows in S, MS<1h
K = (
    A.filter(pl.col("al").is_in(S_all))
    .with_columns(tz=pl.lit(7200.0 + taxi_ord))
    .with_columns(d=(pl.col("tz") - pl.col("y")) ** 2 - (pl.col("px") - pl.col("y")) ** 2)
)
print(
    f"T3 proxy: n {K.height} | dSSE {K['d'].sum()/1e6:+.1f} M | improved {(K['d'] < 0).mean():.2f} | per-row mean gain {K['d'].mean()/1e6:+.2f} M"
)
print(K.select("FLIGHT_mvt", "m", "y", "mvt_sched", "px", "tz", "d", "regime"))
# T2 true frozen janjul
KJ = (
    LJ.filter((pl.col("band") == "a<1h") & pl.col("al").is_in(S_all))
    .with_columns(tz=pl.lit(7200.0 + taxi_ord))
    .with_columns(d=(pl.col("tz") - pl.col("y")) ** 2 - (pl.col("fz") - pl.col("y")) ** 2)
)
for m in (1, 7):
    k = KJ.filter(pl.col("m") == m)
    print(
        f"T2 frozen month {m}: n {k.height} | dSSE {k['d'].sum()/1e6:+.1f} M | improved {(k['d'] < 0).mean() if k.height else float('nan'):.2f}"
    )
print(KJ.select("FLIGHT_mvt", "m", "y", "mvt_sched", "fz", "fz_p", "tz", "d", "regime"))
# frozen janjul bias on a<1h (all airlines) for CB
for m in (1, 7):
    k = LJ.filter((pl.col("band") == "a<1h") & (pl.col("m") == m))
    print(
        f"CB frozen a<1h month {m}: n {k.height} bias {k['fz_err'].mean() if k.height else float('nan'):+.0f} mse {(k['fz_err']**2).mean() if k.height else float('nan'):,.0f}"
    )
# 2026 rows flagged
R26 = (
    R.filter((pl.col("cls") == "unm_LIRF") & (pl.col("band") == "a<1h"))
    .with_columns(al=pl.col("FLIGHT_mvt").str.slice(0, 3))
    .with_columns(inS=pl.col("al").is_in(S_all))
)
print(f"\n=== 2026 unm_LIRF MS<1h: {R26.height} rows, in S: {int(R26['inS'].sum())} ===")
print(
    R26.sort("inS", "al", descending=[True, False]).select(
        "FLIGHT_mvt", "al", "STAND_mvt", "m", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "mvt_sched", "v62", "inS"
    )
)
R26.write_parquet("scratchpad/xhr/tz2_2026.parquet")

# ---------------- Part 3: PX / SNC / PXC ----------------
X = L.filter((pl.col("cls") == "exposed") & (pl.col("ADEP_mvt") == "LIRF")).with_columns(cr=pl.col("cross") >= 1)
print("\n=== G1 census: 2025 LIRF exposed regimes by crossing ===")
print(X.group_by("cr", "regime").len().sort("cr", "regime"))
# proxy version: queue_eb_ra3qa already gates q on crossing, so PX_proxy = px_art (p -> 1)
X = X.with_columns(pxr=pl.when(pl.col("cr")).then(pl.col("px_art")).otherwise(pl.col("mvt_sched")).cast(pl.Float64))
X = X.with_columns(d_px=(pl.col("pxr") - pl.col("y")) ** 2 - (pl.col("px") - pl.col("y")) ** 2)


def summarise(df, dcol, tag):
    d = df[dcol].to_numpy().astype(float)
    o = np.argsort(-np.abs(d))
    print(
        f"[{tag}] n {len(d)} | improved {np.mean(d < 0):.2f} | dSSE {d.sum()/1e6:+.1f} M | w/o top-3 {np.delete(d, o[:3]).sum()/1e6:+.1f} M | median {np.median(d)/1e6:+.3f} M"
    )
    if "m" in df.columns:
        for m in sorted(df["m"].unique().to_list()):
            k = df["m"].to_numpy() == m
            if k.sum() >= 3:
                print(f"      month {m}: n {k.sum()} dSSE {d[k].sum()/1e6:+.1f} M improved {np.mean(d[k] < 0):.2f}")


print("\n=== G2: PX on the proxy (all months, 50 rows) — p->1 ===")
summarise(X, "d_px", "PX proxy all")
summarise(X.filter(~pl.col("cr")), "d_px", "SNC proxy")
summarise(X.filter(pl.col("cr")), "d_px", "PXC proxy")
print(
    X.sort("cr", "mvt_sched").select(
        "FLIGHT_mvt", "m", "cr", "regime", "mvt_sched", "y", "px", "px_p", "px_art", "pxr", "d_px"
    )
)
# TRUE frozen janjul (frozen_weight_u1 frame): fz_oof, fz_p, art_f (frozen's ungated art), q_ds_f
EJ = E.filter((pl.col("is_rank") == 0) & (pl.col("ADEP_mvt") == "LIRF") & pl.col("fz_oof").is_not_null()).with_columns(
    cr=pl.col("cross")
)
EJ = EJ.with_columns(pxf=pl.when(pl.col("cr")).then(pl.col("art_f")).otherwise(pl.col("mvt_sched")).cast(pl.Float64))
EJ = EJ.with_columns(d_fz=(pl.col("pxf") - pl.col("y")) ** 2 - (pl.col("fz_oof") - pl.col("y")) ** 2)
print("\n=== G3: PX on the TRUE frozen (janjul, 16 rows) ===")
summarise(EJ, "d_fz", "PX frozen janjul")
summarise(EJ.filter(~pl.col("cr")), "d_fz", "SNC frozen")
summarise(EJ.filter(pl.col("cr")), "d_fz", "PXC frozen")
print(
    EJ.sort("cr", "mvt_sched").select(
        "FLIGHT_mvt", "m", "cr", "regime", "mvt_sched", "y", "fz_oof", "fz_p", "q_ds_f", "art_f", "pxf", "d_fz"
    )
)
# ---- 2026 ledger ----
ER = (
    E.filter((pl.col("is_rank") == 1) & (pl.col("ADEP_mvt") == "LIRF"))
    .join(R.select("MVT_ID_mvt", "v62"), on="MVT_ID_mvt", how="left")
    .with_columns(cr=pl.col("cross"))
)
ER = ER.with_columns(
    px26=pl.when(pl.col("cr"))
    .then(pl.col("fzr_art"))
    .otherwise(pl.col("mvt_sched"))
    .cast(pl.Float64)
    .round()
    .clip(lower_bound=60)
)
ms = ER["mvt_sched"].to_numpy().astype(float)
v = ER["v62"].to_numpy().astype(float)
px = ER["px26"].to_numpy()
cr = ER["cr"].to_numpy()
q = ER["fzr_q"].to_numpy()
yS, yO, yD = (
    ms,
    np.full_like(ms, 1000.0),
    86400.0 + np.where(ms > 86400, ms - 86400 + 1000, 1000.0),
)  # DS label ~ 86400 + taxi (or +1 more day for MS > 24 h)
dS = (px - yS) ** 2 - (v - yS) ** 2
dO = (px - yO) ** 2 - (v - yO) ** 2
dD = (px - yD) ** 2 - (v - yD) ** 2
mv = np.abs(px - v) >= 1
# per-row break-even ORD probability: pi* = gS / (gS - dO) where gS = -dS (gain if SUB > 0); for crossers use the frozen-q mixture of SUB/DS as the 'non-ORD' outcome
gNon = -np.where(cr, (1 - q) * dS + q * dD, dS)
be = np.where(gNon > 0, gNon / np.maximum(gNon + dO, 1e-9), 0.0)
ub = np.where(cr, 0.072, 0.259)
g1 = mv & (be >= 2 * ub) & (gNon > 0)
T = ER.select("FLIGHT_mvt", "STAND_mvt", "mvt_sched", "cr", "v62", "fzr_p", "fzr_q", "fzr_art").with_columns(
    px=px, moved=mv, dSUB=dS / RN, dORD=dO / RN, dDS=dD / RN, breakeven_ORD=be, G1_pass=g1
)
print("\n=== G4 ledger: 2026 LIRF exposed rows (board s2 per row; DS scenario for crossers) ===")
print(T.sort("cr", "mvt_sched"))
for name, keep in (("PX all moved", mv), ("PX G1-passed", g1), ("SNC G1-passed", g1 & ~cr), ("PXC G1-passed", g1 & cr)):
    adverse = np.where(px > v, dO, dS)  # up-moved rows ORD, down-moved rows SUB
    exp_ = np.where(cr, (1 - q) * dS + q * dD, dS)
    print(
        f"[{name}] rows {keep.sum()} | all-SUB {dS[keep].sum()/RN:+.1f} | all-ORD {dO[keep].sum()/RN:+.1f} | crossers-DS(non-crossers SUB) {np.where(cr, dD, dS)[keep].sum()/RN:+.1f}"
        f" | adverse mix {adverse[keep].sum()/RN:+.1f} | frozen-q expectation {exp_[keep].sum()/RN:+.1f} board s2"
    )
T.with_columns(MVT_ID_mvt=ER["MVT_ID_mvt"]).write_parquet("scratchpad/xhr/px_ledger.parquet")


# ---------------- Part 4: expected-SE map of the 412 frozen rows ----------------
# regime probabilities and label pools from 2025, by (class, crossing, band); for exposed LIRF crossers DS prob by finer MS bins
def pools(df):
    return {r: df.filter(pl.col("regime") == r)["y"].to_numpy().astype(float) for r in ("ORD", "OTHER")}


rows = []
RF = R.filter(pl.col("cls").is_in(["unm_LIRF", "exposed"])).with_columns(cr=pl.col("cross") >= 1)
LX = L.with_columns(cr=pl.col("cross") >= 1)
for r in RF.iter_rows(named=True):
    cls, cr_, b, ms_, pred, apt = r["cls"], r["cr"], r["band"], float(r["mvt_sched"]), float(r["v62"]), r["ADEP_mvt"]
    if cls == "exposed" and apt != "LIRF":
        cell = LX.filter((pl.col("cls") == "exposed") & (pl.col("ADEP_mvt") != "LIRF"))
    elif cls == "exposed":
        cell = LX.filter((pl.col("cls") == "exposed") & (pl.col("ADEP_mvt") == "LIRF") & (pl.col("cr") == cr_))
        if cr_:  # DS share by MS bin ( bands)
            h = ms_ / 3600
            lo, hi = (
                (8, 12) if h < 12 else (12, 15) if h < 15 else (15, 18) if h < 18 else (18, 24) if h < 24 else (24, 99)
            )
            cell = cell.filter((pl.col("mvt_sched") >= lo * 3600) & (pl.col("mvt_sched") < hi * 3600))
            if cell.height < 4:
                cell = LX.filter((pl.col("cls") == "exposed") & (pl.col("ADEP_mvt") == "LIRF") & pl.col("cr"))
    else:
        cell = LX.filter((pl.col("cls") == "unm_LIRF") & (pl.col("band") == b) & (pl.col("cr") == cr_))
        if cell.height < 15:
            cell = LX.filter((pl.col("cls") == "unm_LIRF") & (pl.col("band") == b))
    n = cell.height
    P = {k: (cell["regime"] == k).sum() / n for k in ("SUB", "ORD", "DS", "OTHER")}
    pool = pools(cell)
    yv = {
        "SUB": np.array([ms_]),
        "DS": np.array([86400.0 + 1000.0]),
        "ORD": pool["ORD"] if len(pool["ORD"]) else np.array([1000.0]),
        "OTHER": pool["OTHER"] if len(pool["OTHER"]) else np.array([ms_]),
    }
    ese = sum(P[k] * np.mean((pred - yv[k]) ** 2) for k in P)
    mean_y = sum(P[k] * np.mean(yv[k]) for k in P)
    ese_opt = sum(P[k] * np.mean((mean_y - yv[k]) ** 2) for k in P)
    rows.append(
        {
            "MVT_ID_mvt": r["MVT_ID_mvt"],
            "cls": cls,
            "ADEP_mvt": apt,
            "FLIGHT_mvt": r["FLIGHT_mvt"],
            "cr": cr_,
            "band": b,
            "mvt_sched": ms_,
            "v62": pred,
            "cell_n": n,
            "P_SUB": P["SUB"],
            "P_ORD": P["ORD"],
            "P_DS": P["DS"],
            "P_OTH": P["OTHER"],
            "ESE": ese,
            "ESE_board": ese / RN,
            "opt_pred": mean_y,
            "ESE_opt_board": ese_opt / RN,
        }
    )
M = pl.DataFrame(rows)
print("\n=== expected-SE map (2025 cell mixture, current v62 prediction): totals by class, board s2 ===")
print(
    M.group_by("cls", "ADEP_mvt" if False else "cr")
    .agg(
        n=pl.len(),
        ESE_board=pl.col("ESE_board").sum(),
        ESE_if_cell_mean=pl.col("ESE_opt_board").sum(),
        top3=pl.col("ESE_board").sort(descending=True).head(3).sum(),
    )
    .sort("cls", "cr")
)
print(
    M.group_by("cls")
    .agg(n=pl.len(), ESE_board=pl.col("ESE_board").sum(), ESE_if_cell_mean=pl.col("ESE_opt_board").sum())
    .sort("cls")
)
print("\n=== the 40 rows with the largest expected SE ===")
print(M.sort("ESE_board", descending=True).head(40))
M.write_parquet("scratchpad/xhr/ese_map.parquet")
print("done")
