#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Computes the by-rule predictors for unmatched ranking rows (no flight-plan match) and writes them to
scratchpad/p2/U.parquet (all ranking rows; u = the by-rule value on unmatched rows, null on matched rows).
Inputs: data/features_v9.parquet and the member prediction parquets named by env vars FZ_L/FZ_C/C1_L/C1_C
(defaults in-code); env ROUND selects trunc|rint|floor. Run inside a workspace (cwd = workspace root).

phase2_ws step U: the by-rule predictors for the UNMATCHED ranking rows (no flight-plan match).
Run inside a workspace (cwd = workspace root).

Classes (label-free, = unm_resid_lib.cls_expr): exposed = unmatched & MVT-SCHED > 8 h; unm_LIRF = unmatched LIRF (<= 8 h); unm_other = the rest.
  unm_other  C1 = max(0.5 mean(V47L s0-9) + 0.5 mean(CATV17x s0-19), 60)                      (the unmatched-row baseline blend)
  unm_LIRF,  FZ = max(0.5 mean(V25L s0-9 + V25Lb s10-14) + 0.5 mean(CATV9x s0-1), 60)          (the v21/v28-era "frozen" blend)
  exposed    + its decomposition fzr_p / fzr_art / fzr_q (seed means of the members' p_raw / art / q_ds columns) for the px_safe PX rule
Rounding: int32 truncation after the 60 s floor (astype(np.int32))
'cross' = local-midnight crossing between SCHED and MVT (a crude local clock, = train2's cross_ definition).
Writes scratchpad/p2/U.parquet (all ranking rows; u = the by-rule value on unmatched rows, null on matched rows).
env: FZ_L, FZ_C, C1_L, C1_C = comma lists of member files (defaults above); ROUND=trunc|rint|floor."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
E = lambda k, d: os.environ.get(k, d)
FZ_L = E("FZ_L", ",".join([f"V25L_s{i}" for i in range(10)] + [f"V25Lb_s{i}" for i in range(10, 15)])).split(",")
FZ_C = E("FZ_C", "CATV9x_s0,CATV9x_s1").split(",")
C1_L = E("C1_L", ",".join(f"V47L_s{i}" for i in range(10))).split(",")
C1_C = E("C1_C", ",".join(f"CATV17x_s{i}" for i in range(20))).split(",")
RND = E("ROUND", "trunc")
TZ = {"LIRF": 1, "EHAM": 1, "EDDF": 1, "EDDM": 1, "LFPG": 1, "LSZH": 1, "LEBL": 1, "LEMD": 1, "EGLL": 0, "LTFM": 3}


def loc(c):
    return pl.col(c) + pl.duration(
        hours=pl.col("ADEP_mvt").replace_strict(TZ, default=1, return_dtype=pl.Int64)
        + pl.when((pl.col(c).dt.month() >= 4) & (pl.col(c).dt.month() <= 10) & (pl.col("ADEP_mvt") != "LTFM"))
        .then(1)
        .otherwise(0)
    )


F = (
    pl.read_parquet(
        "data/features_v9.parquet",
        columns=[
            "MVT_ID_mvt",
            "is_rank",
            "unmatched",
            "ADEP_mvt",
            "mvt_sched",
            "MVT_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt",
            "FLIGHT_mvt",
        ],
    )
    .filter(pl.col("is_rank") == 1)
    .with_columns(pl.col("ADEP_mvt").cast(pl.Utf8))
    .sort("MVT_ID_mvt")
)
ms0 = pl.col("mvt_sched").fill_null(0)
F = F.with_columns(
    cls=pl.when(pl.col("unmatched") == 0)
    .then(pl.lit("matched"))
    .when(ms0 > 28800)
    .then(pl.lit("exposed"))
    .when(pl.col("ADEP_mvt") == "LIRF")
    .then(pl.lit("unm_LIRF"))
    .otherwise(pl.lit("unm_other")),
    cross=((loc("MVT_TIME_UTC_mvt").dt.date() - loc("SCHED_TIME_UTC_mvt").dt.date()).dt.total_days() >= 1),
)
ids = F["MVT_ID_mvt"].to_numpy()


def load(names, cols):
    out = []
    for n in names:
        d = pl.read_parquet(f"data/preds/{n}.parquet").sort("MVT_ID_mvt")
        assert np.array_equal(d["MVT_ID_mvt"].to_numpy(), ids), f"{n}: ids differ from the period's ranking rows"
        out.append(
            np.vstack([d[c].to_numpy().astype(float) if c in d.columns else np.full(len(ids), np.nan) for c in cols])
        )
    return np.mean(out, axis=0)


C = ["TAXITIME_SEC_mvt", "p_raw", "art", "q_ds"]
fL, fC = load(FZ_L, C), load(FZ_C, C)
cL, cC = load(C1_L, C[:1]), load(C1_C, C[:1])
fz = np.maximum(0.5 * fL[0] + 0.5 * fC[0], 60.0)
c1 = np.maximum(0.5 * cL[0] + 0.5 * cC[0], 60.0)
R = {"trunc": lambda a: a.astype(np.int32).astype(float), "rint": np.rint, "floor": np.floor}[RND]
cls = F["cls"].to_numpy()
u = np.where(cls == "unm_other", R(c1), np.where(np.isin(cls, ["unm_LIRF", "exposed"]), R(fz), np.nan))
out = F.select("MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "mvt_sched", "unmatched", "cls", "cross").with_columns(
    fz=fz,
    c1=c1,
    fzr_p=0.5 * fL[1] + 0.5 * fC[1],
    fzr_art=0.5 * fL[2] + 0.5 * fC[2],
    fzr_q=0.5 * fL[3] + 0.5 * fC[3],
    u=u,
)
os.makedirs("scratchpad/p2", exist_ok=True)
out.write_parquet("scratchpad/p2/U.parquet")
print(
    "U: ranking rows",
    out.height,
    "| by class:",
    dict(out.group_by("cls").len().iter_rows()),
    "| rounding",
    RND,
    "| members FZ",
    len(FZ_L),
    "+",
    len(FZ_C),
    "C1",
    len(C1_L),
    "+",
    len(C1_C),
)
