#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds the xe side table: runway departure-queue features not covered by the existing tl/nbr/day groups.
Reads data/features_v9.parquet (MVT_ID_mvt, ADEP_mvt, apt_rwy, MVT_TIME_UTC_mvt, mvt_lobt, mvt_sched); writes
data/external/xe_queue_features.parquet keyed by MVT_ID_mvt, for both training and ranking rows.

Existing groups already hold MVT-anchored counts (rwy_dep_/apt_dep_/apt_arr_/rwy_arr_ m5..m120, p5..p60), runway
gaps prev/next 1..10, queue runs (gap <= 90/120/180 s), heavy predecessors, SCHED/EOBT demand around MVT, hourly
runway share, day/hour totals. What is new here is anchoring on the flight's OWN PLANNED off-block (LOBT, or SCHED
when there is no flight plan) and on the planned ORDER of the other flights:

  xe_nr_Lt / xe_na_Lt      take-offs on my runway / at the airport between LOBT+u and my MVT (u = 300 s)
  xe_rate_Lt               xe_nr_Lt per minute of that window
  xe_arr_Lt                arrivals landing on my runway in the same window (mixed mode)
  xe_ahead_r / xe_ahead_a  flights planned (LOBT) BEFORE me that took off AFTER LOBT+u and BEFORE me (queue ahead)
  xe_passed_by_a/_r        flights planned AFTER me that took off before me (they overtook me)
  xe_i_passed_a            flights planned BEFORE me that took off after me (I overtook them)
  xe_slip_a                passed_by - i_passed (net places lost in the take-off sequence, +/-3 h)
  xe_vq_a / xe_vq_r        virtual queue at my take-off: others with LOBT <= MVT-u that are not yet airborne
  xe_coh_pos / _n / _first sequence position in my 15-min SCHED cohort (airport), cohort size, seconds after its first
  xe_sat_r30 / xe_sat_a60  30-min runway / 60-min airport take-off count over the p95 capacity (runway; airport x config)
  xe_busy_age / xe_busy_n  seconds since the start of my runway's current busy period (gaps < 300 s) and its count
  xe_same_flow60           share of the airport's take-offs in the last 60 min on a runway within 90 deg of mine
  xe_since_opp             seconds since the last airport take-off on an opposite-flow runway (cap 6 h)
  xe_cfg_chg60             1 if the set of departure runways in [t-60,t-30) differs from [t-30,t)
  xe_rwy_open_age          seconds since my runway came back into use after a > 30-min pause (cap 6 h)
Only observable fields are used: every departure's MVT / runway / LOBT / SCHED and every arrival's landing time /
runway. No BLOCK / TAXITIME of any departure is read. Training and ranking rows are built by the same code.
usage: build_xe_queue_features.py   (writes data/external/xe_queue_features.parquet)"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = 300
K = 400
WIN = 3 * 3600
F = pl.read_parquet(
    f"{ROOT}/data/features_v9.parquet",
    columns=["MVT_ID_mvt", "ADEP_mvt", "apt_rwy", "MVT_TIME_UTC_mvt", "mvt_lobt", "mvt_sched"],
)
F = F.with_columns(t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64)).with_columns(
    S=(pl.col("t") - pl.col("mvt_sched")).cast(pl.Int64), L=(pl.col("t") - pl.col("mvt_lobt")).cast(pl.Int64)
)
if os.environ.get("runway_queue_shuffle"):
    F = F.sample(fraction=1.0, shuffle=True, seed=int(os.environ["runway_queue_shuffle"]))  # tie-order audit
F = F.with_columns(Lp=pl.coalesce("L", "S"), rwy=pl.col("apt_rwy").str.split("_").list.get(1, null_on_oob=True))
F = F.with_columns(hdg=(pl.col("rwy").str.slice(0, 2).cast(pl.Int32, strict=False) * 10))
files = sorted(glob.glob(f"{ROOT}/data/raw/training_*.parquet")) + [f"{ROOT}/data/raw/ranking.parquet"]
R = pl.concat(
    [pl.read_parquet(f, columns=["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]) for f in files],
    how="diagonal_relaxed",
)
A = (
    R.filter(pl.col("PHASE_mvt") == "ARR")
    .select(
        apt_rwy=pl.col("ADES_mvt") + "_" + pl.col("RUNWAY_mvt"),
        t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64),
    )
    .drop_nulls()
)
print(f"departures {F.height:,}, arrivals {A.height:,}", flush=True)

out = {}


def put(name, idx, val):
    if name not in out:
        out[name] = np.full(F.height, np.nan)
    out[name][idx] = val


def seq_block(keycol, tag):
    """Shift-based pairwise counts inside groups sorted by take-off time (no O(n^2)): for k = 1..K compare row i with
    its k-th successor / predecessor in the same group, restricted to |dt| <= WIN."""
    o = np.lexsort((F["t"].to_numpy(), F[keycol].cast(pl.Categorical).to_physical().to_numpy()))
    g = F[keycol].cast(pl.Categorical).to_physical().to_numpy()[o]
    t = F["t"].to_numpy()[o]
    Lp = F["Lp"].to_numpy()[o].astype(float)
    n = len(t)
    ahead = np.zeros(n)
    passed_by = np.zeros(n)
    i_passed = np.zeros(n)
    vq = np.zeros(n)
    for k in range(1, K + 1):
        # predecessors: j = i - k (took off before me)
        sg = np.zeros(n, bool)
        sg[k:] = (g[k:] == g[:-k]) & (t[k:] - t[:-k] <= WIN)
        Lj = np.full(n, np.nan)
        Lj[k:] = Lp[:-k]
        tj = np.zeros(n)
        tj[k:] = t[:-k]
        ahead += sg & (Lj < Lp) & (tj > Lp + U)
        passed_by += sg & (Lj > Lp)
        # successors: j = i + k (take off after me)
        ss = np.zeros(n, bool)
        ss[:-k] = (g[:-k] == g[k:]) & (t[k:] - t[:-k] <= WIN)
        Ls = np.full(n, np.nan)
        Ls[:-k] = Lp[k:]
        i_passed += ss & (Ls < Lp)
        vq += ss & (Ls <= t - U)
    nanL = np.isnan(Lp)
    for nm, v in (("ahead", ahead), ("passed_by", passed_by), ("i_passed", i_passed), ("vq", vq)):
        v = v.astype(float)
        v[nanL] = np.nan
        put(f"xe_{nm}_{tag}", o, v)
    if tag == "a":
        put("xe_slip_a", o, np.where(nanL, np.nan, passed_by - i_passed))


print("sequence counts (airport)", flush=True)
seq_block("ADEP_mvt", "a")
print("sequence counts (runway)", flush=True)
seq_block("apt_rwy", "r")


def window_count(ev_key, ev_t, q_key, lo, hi):
    """count events of the same key with lo < t_ev <= hi (arrays aligned to F)."""
    res = np.full(len(q_key), np.nan)
    ek = pl.Series(ev_key).cast(pl.Utf8).to_numpy()
    qk = pl.Series(q_key).cast(pl.Utf8).to_numpy()
    o = np.argsort(ek, kind="stable")
    ek = ek[o]
    et = ev_t[o]
    keys, starts = np.unique(ek, return_index=True)
    ends = np.r_[starts[1:], len(ek)]
    for kk, s, e in zip(keys, starts, ends):
        tt = np.sort(et[s:e])
        m = qk == kk
        if not m.any():
            continue
        a = np.searchsorted(tt, lo[m], side="right")
        b = np.searchsorted(tt, hi[m], side="right")
        res[m] = np.maximum(b - a, 0)
    return res


t = F["t"].to_numpy()
Lp = F["Lp"].to_numpy().astype(float)
rk = F["apt_rwy"].fill_null("?").to_numpy()
ak = F["ADEP_mvt"].to_numpy()
lo = Lp + U
valid = ~np.isnan(Lp) & (t > lo)
print("LOBT-anchored windows", flush=True)
lo_ = np.where(valid, lo, t).astype(np.int64)
nr = window_count(rk, t, rk, lo_, t - 1)
na = window_count(ak, t, ak, lo_, t - 1)
arrk = A["apt_rwy"].to_numpy()
arrt = A["t"].to_numpy()
nar = window_count(arrk, arrt, rk, lo_, t)
dur_min = (t - lo) / 60.0
put("xe_nr_Lt", slice(None), np.where(valid, nr, np.nan))
put("xe_na_Lt", slice(None), np.where(valid, na, np.nan))
put("xe_arr_Lt", slice(None), np.where(valid, nar, np.nan))
put("xe_rate_Lt", slice(None), np.where(valid & (dur_min >= 2), nr / np.maximum(dur_min, 1e-9), np.nan))
put("xe_dur_Lt", slice(None), np.where(~np.isnan(Lp), t - lo, np.nan))

print("cohort / saturation / flow", flush=True)
G = F.select("MVT_ID_mvt", "ADEP_mvt", "apt_rwy", "t", "S", "hdg").with_row_index("i")
G = G.with_columns(sb=(pl.col("S") // 900)).with_columns(
    coh_pos=(pl.col("t").rank("ordinal").over(["ADEP_mvt", "sb"]) - 1) / pl.len().over(["ADEP_mvt", "sb"]).clip(1),
    coh_n=pl.len().over(["ADEP_mvt", "sb"]),
    coh_first=pl.col("t") - pl.col("t").min().over(["ADEP_mvt", "sb"]),
)
ii = G["i"].to_numpy()
put("xe_coh_pos", ii, G["coh_pos"].to_numpy())
put("xe_coh_n", ii, G["coh_n"].to_numpy())
put("xe_coh_first", ii, G["coh_first"].to_numpy())
# rolling counts on own runway (30 min) and airport (60 min), anchored at own take-off (closed both ends, own row included)
r30 = window_count(rk, t, rk, t - 1800, t)
a60 = window_count(ak, t, ak, t - 3600, t)
cap_r = pl.DataFrame({"k": rk, "c": r30}).group_by("k").agg(cap=pl.col("c").quantile(0.95))
capr = pl.DataFrame({"k": rk}).join(cap_r, on="k", how="left")["cap"].to_numpy()
put("xe_sat_r30", slice(None), r30 / np.maximum(capr, 1))
# configuration = sorted set of departure runways used at the airport in the last 60 min
H = G.sort("ADEP_mvt", "t").with_columns(ts=pl.from_epoch("t", time_unit="s"))
cfg = H.rolling(index_column="ts", period="60m", group_by="ADEP_mvt", closed="both").agg(
    cfg=pl.col("apt_rwy").unique().sort().str.join("+")
)
H = H.with_columns(cfg=cfg["cfg"])
cfg30a = H.rolling(index_column="ts", period="30m", group_by="ADEP_mvt", closed="both").agg(
    c30=pl.col("apt_rwy").unique().sort().str.join("+")
)
H = H.with_columns(c30=cfg30a["c30"]).with_columns(tm30=pl.col("t") - 1800)
# config of the preceding half hour: as-of the airport's last take-off at or before t-30 min
prev = H.select("ADEP_mvt", pl.col("t").alias("tp"), pl.col("c30").alias("c30p")).sort("tp")
H = H.sort("tm30").join_asof(prev, left_on="tm30", right_on="tp", by="ADEP_mvt", strategy="backward", tolerance=1800)
H = H.with_columns(
    cfg_chg=pl.when(pl.col("c30p").is_null()).then(None).otherwise((pl.col("c30p") != pl.col("c30")).cast(pl.Float64))
)
hi = H["i"].to_numpy()
put("xe_cfg_chg60", hi, H["cfg_chg"].to_numpy())
a60h = a60[hi]
Hc = pl.DataFrame({"a": H["ADEP_mvt"], "cfg": H["cfg"], "c": a60h})
cc = Hc.group_by("a", "cfg").agg(cap=pl.col("c").quantile(0.95), nn=pl.len())
capc = (
    Hc.join(cc, on=["a", "cfg"], how="left")
    .with_columns(cap=pl.when(pl.col("nn") >= 500).then(pl.col("cap")))["cap"]
    .to_numpy()
)
put("xe_sat_a60", hi, a60h / np.maximum(capc, 1))
put("xe_ncfg_rwy", hi, H["cfg"].str.count_matches(r"\+").to_numpy() + 1.0)
# flow: share of the airport's last-60-min take-offs on a runway within 90 deg of mine, and time since last opposite-flow take-off
hd = F["hdg"].to_numpy().astype(float)
same = np.zeros(F.height)
tot = np.zeros(F.height)
since_opp = np.full(F.height, np.nan)
o = np.lexsort((t, F["ADEP_mvt"].cast(pl.Categorical).to_physical().to_numpy()))
go = ak[o]
to = t[o]
ho = hd[o]
last_opp = np.full(len(o), np.nan)
for k in range(1, K + 1):
    sg = np.zeros(len(o), bool)
    sg[k:] = (go[k:] == go[:-k]) & (to[k:] - to[:-k] <= 3600)
    hj = np.full(len(o), np.nan)
    hj[k:] = ho[:-k]
    d = np.abs(((hj - ho) + 180) % 360 - 180)
    same[o] += sg & (d <= 90)
    tot[o] += sg & ~np.isnan(d)
put("xe_same_flow60", slice(None), np.where(tot > 0, same / np.maximum(tot, 1), np.nan))
# time since last opposite-flow take-off: per airport, running last time per flow bucket
flow = np.where(np.isnan(ho), -1, ((ho + 45) % 360 // 180).astype(int))  # two half-planes: [315,135) vs [135,315)
lastb = {}
for idx in range(len(o)):
    a_ = go[idx]
    f_ = flow[idx]
    if f_ >= 0:
        lo_t = lastb.get((a_, 1 - f_))
        last_opp[idx] = (to[idx] - lo_t) if lo_t is not None else np.nan
        lastb[(a_, f_)] = to[idx]
since_opp[o] = np.minimum(last_opp, 6 * 3600)
put("xe_since_opp", slice(None), since_opp)
# busy period and runway re-opening on my runway
o = np.lexsort((t, pl.Series(rk).cast(pl.Categorical).to_physical().to_numpy()))
gr = rk[o]
tr = t[o]
newg = np.r_[True, gr[1:] != gr[:-1]]
gap = np.r_[np.inf, np.diff(tr).astype(float)]
gap[newg] = np.inf
for thr, nm in ((300, "busy"), (1800, "open")):
    brk = gap >= thr
    start_idx = np.maximum.accumulate(np.where(brk, np.arange(len(tr)), 0))
    age = (tr - tr[start_idx]).astype(float)
    nn = (np.arange(len(tr)) - start_idx + 1).astype(float)
    if nm == "busy":
        put("xe_busy_age", o, age)
        put("xe_busy_n", o, nn)
    else:
        put("xe_rwy_open_age", o, np.minimum(age, 6 * 3600))

if os.environ.get("runway_queue_d4"):
    # D4: did the airport's departure-runway configuration change while I was (nominally) taxiing, i.e. in
    # (Lp + u, MVT)? Events = airport take-offs whose xe_cfg_chg60 == 1. OFF by default so the screened / production
    # xe table (26 columns) stays byte-identical; enable with runway_queue_d4=1 runway_queue_out=<new path>.
    chg = out["xe_cfg_chg60"] == 1
    n_chg = window_count(ak[chg], t[chg], ak, lo_, t - 1)
    put("xe_cfgchg_in_taxi", slice(None), np.where(valid, (n_chg > 0).astype(float), np.nan))
T = pl.DataFrame({"MVT_ID_mvt": F["MVT_ID_mvt"], **{k: v.astype(np.float32) for k, v in out.items()}})
os.makedirs(f"{ROOT}/data/external", exist_ok=True)
T.write_parquet(os.environ.get("runway_queue_out", f"{ROOT}/data/external/xe_queue_features.parquet"))
print(T.describe().transpose(include_header=True) if hasattr(T.describe(), "transpose") else T.describe())
print(f"wrote {T.height:,} rows x {T.width - 1} xe_ features")
