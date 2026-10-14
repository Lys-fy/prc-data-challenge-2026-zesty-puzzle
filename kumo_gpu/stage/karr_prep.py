#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Labelled ARRIVAL rows for the Kumo context (arm KARWPc100, knob KARR=1; not part of the final model).
For every airport APT: all ARR movements with ADES = APT from the organiser files (2025 training_*.parquet, 2026 ranking.parquet: Jan/Jul arrivals are
labelled there; submitting.parquet is DEP only, so no arrival is a scored row). Clean = taxi-in in [30, 3600] s.
Target r = taxi_in - median 2025 clean taxi-in at APT x RUNWAY_mvt (airport median if the runway has < 200 rows); offset 0 in the engine.
Inputs (built by the adapter): the 412 features of the NEAREST departure of the same payload frame by |MVT_TIME_dep - MVT_TIME_arr| (same UTC day
preferred, else overall nearest; features are label-free, held-month and 2026 departures included), per-flight columns NaN, + is_arr = 1.
Output: <PAYLOAD>/karr/arr_<APT>.parquet  [arr_id, year, m, day, taxi_in, rwy_med, y_res, dep_row, dt_min]  (+ karr_counts.tsv, label-free counts).
usage: python stage/karr_prep.py [APT,APT,...]   env PAYLOAD (default kumo_gpu/data/payload), RAW (default <project>/data/raw)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, time, numpy as np, polars as pl

B = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PAYLOAD = os.environ.get("PAYLOAD", f"{B}/data/payload")
RAW = os.environ.get("RAW", os.path.abspath(f"{B}/../data/raw"))
APTS = (sys.argv[1] if len(sys.argv) > 1 else "LTFM,EHAM,EGLL,LFPG,EDDF,LEMD,LEBL,EDDM,LIRF,LSZH").split(",")
HOLD = {
    "janjul": [1, 7],
    "months28": [2, 8],
    "months39": [3, 9],
    "months410": [4, 10],
    "months511": [5, 11],
    "months612": [6, 12],
    "prod": None,
}
TMIN, TMAX, MINRWY = 30, 3600, 200
t0 = time.time()
os.makedirs(f"{PAYLOAD}/karr", exist_ok=True)
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "TAXITIME_SEC_mvt"]
tr = pl.concat(
    [
        pl.read_parquet(f, columns=cols).with_columns(year=pl.lit(2025, pl.Int32))
        for f in sorted(glob.glob(f"{RAW}/training_2025-*.parquet"))
    ]
)
rk = pl.read_parquet(f"{RAW}/ranking.parquet", columns=cols).with_columns(year=pl.lit(2026, pl.Int32))
sub = pl.read_parquet(f"{RAW}/submitting.parquet", columns=["MVT_ID_mvt"])
ALL = pl.concat([tr, rk]).with_columns(
    t=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s"),
    day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
    m=pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int32),
)
arr_ids_all = ALL.filter(pl.col("PHASE_mvt") == "ARR")["MVT_ID_mvt"]
assert not sub["MVT_ID_mvt"].is_in(arr_ids_all).any(), "a submitting row is an arrival"
print(
    f"raw rows {len(ALL):,} (2025 {len(tr):,}, 2026 {len(rk):,}); submitting rows {len(sub):,} all DEP; 2026 ARR labelled: "
    f"{rk.filter(pl.col('PHASE_mvt') == 'ARR')['TAXITIME_SEC_mvt'].is_not_null().sum():,} | {time.time() - t0:.0f}s",
    flush=True,
)
CNT = []
for apt in APTS:
    d = pl.read_parquet(
        f"{PAYLOAD}/member/km_{apt}.parquet",
        columns=["meta__MVT_ID_mvt", "_row", "meta__is_rank", "meta__m", "meta__fb_pru", "meta__y"],
    )
    dep = d.join(
        ALL.filter(pl.col("PHASE_mvt") == "DEP").select(meta__MVT_ID_mvt="MVT_ID_mvt", t="t", dday="day"),
        on="meta__MVT_ID_mvt",
        how="left",
    )
    assert dep["t"].null_count() == 0, (apt, "departures without MVT time", dep["t"].null_count())
    dep = dep.sort("t")
    dt_ = dep["t"].to_numpy()
    dday = dep["dday"].to_physical().to_numpy()
    drow = dep["_row"].to_numpy()
    a = ALL.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == apt)).rename({"TAXITIME_SEC_mvt": "taxi_in"})
    n_all = len(a)
    a = a.filter(pl.col("taxi_in").is_not_null() & pl.col("taxi_in").is_between(TMIN, TMAX))
    a25 = a.filter(pl.col("year") == 2025)
    amed = float(a25["taxi_in"].median())
    rmed = (
        a25.group_by("RUNWAY_mvt")
        .agg(n=pl.len(), med=pl.col("taxi_in").median())
        .filter(pl.col("n") >= MINRWY)
        .select("RUNWAY_mvt", rwy_med=pl.col("med").cast(pl.Float64))
    )
    a = (
        a.join(rmed, on="RUNWAY_mvt", how="left")
        .with_columns(rwy_med=pl.col("rwy_med").fill_null(amed))
        .with_columns(y_res=pl.col("taxi_in").cast(pl.Float64) - pl.col("rwy_med"))
    )
    at = a["t"].to_numpy()
    aday = a["day"].to_physical().to_numpy()
    n = len(dt_)
    lo = np.searchsorted(dday, aday, "left")
    hi = np.searchsorted(dday, aday, "right")
    same = hi > lo
    lo = np.where(same, lo, 0)
    hi = np.where(same, hi, n)
    pos = np.searchsorted(dt_, at)
    c1 = np.clip(pos - 1, lo, hi - 1)
    c2 = np.clip(pos, lo, hi - 1)
    best = np.where(np.abs(dt_[c1] - at) <= np.abs(dt_[c2] - at), c1, c2)
    out = a.select(
        arr_id="MVT_ID_mvt",
        year="year",
        m="m",
        day="day",
        taxi_in=pl.col("taxi_in").cast(pl.Int32),
        rwy_med="rwy_med",
        y_res="y_res",
    ).with_columns(
        dep_row=pl.Series(drow[best]),
        dt_min=pl.Series(np.abs(dt_[best] - at) / 60.0).cast(pl.Float32),
        same_day=pl.Series(same),
    )
    out.write_parquet(f"{PAYLOAD}/karr/arr_{apt}.parquet")
    # label-free context counts per fold (engine split rule on departures; arrivals = 2025 all months for CV folds, 2025 + 2026 for prod)
    rank = d["meta__is_rank"].to_numpy() == 1
    m = d["meta__m"].to_numpy()
    clean = d["meta__fb_pru"].to_numpy() != 1
    fin = np.isfinite(d["meta__y"].to_numpy().astype(float))
    n25 = int((out["year"] == 2025).sum())
    n26 = int((out["year"] == 2026).sum())
    for f, h in HOLD.items():
        if h is None:
            ctx = int((~rank & clean & fin).sum())
            q = int(rank.sum())
            na = n25 + n26
        else:
            ctx = int((~rank & ~np.isin(m, h) & clean & fin).sum())
            q = int((~rank & np.isin(m, h)).sum())
            na = n25
        CNT.append((apt, f, ctx, na, q))
    print(
        f"{apt}: ARR {n_all:,} -> clean {len(out):,} (2025 {n25:,}, 2026 {n26:,}) | runways w/ own median {len(rmed)} | nearest dep: same day "
        f"{out['same_day'].mean():.3f}, |dt| median {out['dt_min'].median():.1f} min p95 {out['dt_min'].quantile(0.95):.1f} | {time.time() - t0:.0f}s",
        flush=True,
    )
with open(f"{PAYLOAD}/karr/karr_counts.tsv", "w") as fh:
    fh.write("apt\tfold\tdep_ctx\tarr_ctx\tquery\n")
    [fh.write("\t".join(map(str, r)) + "\n") for r in CNT]
C = pl.DataFrame(CNT, schema=["apt", "fold", "dep_ctx", "arr_ctx", "query"], orient="row")
S = C.with_columns(
    ctx_ratio=(pl.col("dep_ctx") + pl.col("arr_ctx")) / pl.col("dep_ctx"),
    w_q=pl.col("query") * pl.col("dep_ctx"),
    w_qa=pl.col("query") * (pl.col("dep_ctx") + pl.col("arr_ctx")),
)
print(
    S.group_by("fold")
    .agg(
        pl.col("dep_ctx").sum(),
        pl.col("arr_ctx").sum(),
        pl.col("query").sum(),
        max_ctx=(pl.col("dep_ctx") + pl.col("arr_ctx")).max(),
        cost_ratio=pl.col("w_qa").sum() / pl.col("w_q").sum(),
    )
    .sort("fold"),
    flush=True,
)
print(f"KARR_PREP_DONE {time.time() - t0:.0f}s")
