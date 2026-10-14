#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Precomputes the C8 categorical codes per airport frame (same rule as kumo_kcat_member.py stage C: rank in the
airport frame's sorted value set, null = -1). Inputs: positional <FV parquet> <member dir> <OUT>. Writes
<OUT>/kcat_codes_<APT>.parquet (_row, MVT_ID_mvt, 8 int32 codes) for every airport found in <member dir>.

KCAT8 (docs/REPRODUCE.md B6b): folds  = kumo_kcat_codes.py data/features_v9.parquet <fold payload>/member scratchpad/xkumo3/codes/folds
                                prod   = kumo_kcat_codes.py phase2/work/final/data/features_v9.parquet <final K250 payload>/member scratchpad/xkumo3/codes/final
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, numpy as np, polars as pl

C8 = [
    "STAND_mvt",
    "RUNWAY_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "AIRCRAFT_TYPE_flt",
    "WK_TBL_CAT_flt",
    "ADES_flt",
    "MARKET_SEGMENT_flt",
    "FLIGHT_TYPE_flt",
]
kcat_d1_extras = [
    "FLIGHT_mvt",
    "prev_wtc",
]  # stage D1 extras read from features_v9; derived: SCHED_TOD15 (15-min bin of the scheduled time of day), STANDxSH (stand x scheduled hour)
EXT = os.environ.get("EXT") == "1"
fv, mdir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
v9 = pl.read_parquet(
    fv, columns=["ADEP_mvt", "MVT_ID_mvt"] + C8 + ((kcat_d1_extras + ["SCHED_TIME_UTC_mvt"]) if EXT else [])
).with_row_index("_row")
if EXT:
    t = pl.col("SCHED_TIME_UTC_mvt")
    v9 = v9.with_columns(
        (t.dt.hour().cast(pl.Int32) * 4 + t.dt.minute().cast(pl.Int32) // 15).cast(pl.Utf8).alias("SCHED_TOD15"),
        (pl.col("STAND_mvt") + "_" + t.dt.hour().cast(pl.Utf8)).alias("STANDxSH"),
    )
    C8 = C8 + kcat_d1_extras + ["SCHED_TOD15", "STANDxSH"]
for f in sorted(os.listdir(mdir)):
    if not (f.startswith("km_") and f.endswith(".parquet")):
        continue
    apt = f[3:7]
    k = pl.read_parquet(f"{mdir}/{f}", columns=["_row", "meta__MVT_ID_mvt"])
    sub = v9[k["_row"].to_list()]
    assert np.array_equal(
        sub["MVT_ID_mvt"].cast(pl.Float64).to_numpy(), k["meta__MVT_ID_mvt"].cast(pl.Float64).to_numpy()
    ), apt
    assert (sub["ADEP_mvt"] == apt).all(), apt
    cols = {}
    for c in C8:
        s = sub[c].cast(pl.Utf8)
        vals = np.sort(s.drop_nulls().unique().to_numpy())
        lut = {v: i for i, v in enumerate(vals)}
        cols[c] = np.array([lut[v] if v is not None else -1 for v in s.to_list()], dtype=np.int32)
    pl.DataFrame({"_row": k["_row"], "MVT_ID_mvt": k["meta__MVT_ID_mvt"], **cols}).write_parquet(
        f"{out}/kcat_codes_{apt}.parquet"
    )
    print(apt, k.height, {c: int(cols[c].max()) + 1 for c in C8}, flush=True)
