#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""phase2_ws: build a synthetic NEW PERIOD from labelled 2025 data, for an end-to-end dry run of the phase-2 pipeline.
The chosen month(s) of 2025 become the period's ranking set; their labels are removed from everything the pipeline can read.

Writes (workspace = phase2/work/<period>, the pipeline's only view of the data):
  data/features_dry.parquet   features_v9 with: the real 2026 ranking rows dropped; the period's rows (departures AND arrivals of the
                              chosen months, like the official ranking file) flagged is_rank = 1; every column that is all-null on the
                              real ranking rows (the label-derived ones: y, fb, BLOCK time, ...) set to null on the period's rows.
  data/oof/<file>             for every OOF file of the fold that holds the chosen months (name contains the fold tag): a local copy
                              without the period's rows, so no stage is ever fitted on a period label (the shared file is untouched).
  scratchpad/x4/X4_fold.parquet   the post-rule fit frame without the period's rows.
  scratchpad/xl/frame.parquet     minimal ranking-composition frame for the stage scripts' informational gates.
  data/raw/submitting.parquet     the period's submission template (ids of the period's departures).
Held-out labels go OUTSIDE the workspace (phase2/heldout/<period>_labels.parquet); the strict guard makes them unreadable to the pipeline.
usage (repo root, no guard): p2_mkdry.py <period> --year 2025 --months 2 --fold months28"""

import os, sys, glob, argparse, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
REPO = os.environ.get("P2_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
ap = argparse.ArgumentParser()
ap.add_argument("period")
ap.add_argument("--year", type=int, default=2025)
ap.add_argument("--months", default="2")
ap.add_argument("--fold", default="months28")
a = ap.parse_args()
MS = [int(x) for x in a.months.split(",")]
WS = f"{REPO}/phase2/work/{a.period}"
HO = f"{REPO}/phase2/heldout"
os.makedirs(HO, exist_ok=True)
FV = "data/features_v9.parquet"
L = pl.scan_parquet(FV)
nr = L.filter(pl.col("is_rank") == 1).select(pl.len()).collect().item()
nulls = L.filter(pl.col("is_rank") == 1).select(pl.all().null_count()).collect()
LAB = [c for c in nulls.columns if nulls[c][0] == nr and c not in ("MVT_ID_mvt",)]
print(f"label-derived columns (all-null on the {nr:,} real ranking rows): {len(LAB)}: {LAB}")
sel = (
    (pl.col("is_rank") == 0)
    & pl.col("MVT_TIME_UTC_mvt").dt.year().eq(a.year)
    & pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(MS)
)
P = L.filter(sel).select("MVT_ID_mvt", "is_dep", "unmatched", "ADEP_mvt", "y", "MVT_TIME_UTC_mvt").collect()
dep = pl.col("is_dep").cast(pl.Int8) == 1  # is_dep is Boolean in features_v9 (the table holds departures only)
print(f"period rows: {P.height:,} (departures {int(P['is_dep'].cast(pl.Int8).sum()):,})")
P.filter(dep).select("MVT_ID_mvt", "ADEP_mvt", "unmatched", "y").write_parquet(f"{HO}/{a.period}_labels.parquet")
out = (
    L.filter(pl.col("is_rank") == 0)
    .with_columns(_p=sel)
    .with_columns([pl.when(pl.col("_p")).then(None).otherwise(pl.col(c)).alias(c) for c in LAB])
    .with_columns(
        is_rank=pl.when(pl.col("_p")).then(1).otherwise(pl.col("is_rank")).cast(L.collect_schema()["is_rank"])
    )
    .drop("_p")
)
os.makedirs(f"{WS}/data", exist_ok=True)
fo = f"{WS}/data/features_dry.parquet"
out.sink_parquet(fo)
chk = pl.scan_parquet(fo).group_by("is_rank").len().collect().sort("is_rank")
print("features_dry:", chk.rows())
assert pl.read_parquet_schema(fo) == L.collect_schema(), "schema changed"
pid = set(P["MVT_ID_mvt"].cast(pl.Float64).to_list())
pida = np.array(sorted(pid))
# ---- OOF files of the fold that holds the period: local copies without the period's rows
n_f = n_r = 0
bad = []
os.makedirs(f"{WS}/data/oof", exist_ok=True)
for f in sorted(glob.glob(f"data/oof/*{a.fold}*.npy")):
    try:
        z = np.load(f, allow_pickle=False)
    except Exception as e:
        bad.append(f"{f}: {str(e)[:80]}")
        continue  # unreadable (truncated / being written) shared file: reported, never used by the chain
    if z.ndim != 2:
        continue
    k = np.isin(z[0], pida)
    if not k.any():
        continue
    t = f"{WS}/{f}"
    if os.path.islink(t):
        os.unlink(t)
    np.save(t, z[:, ~k])
    n_f += 1
    n_r += int(k.sum())
print(
    f"OOF files of {a.fold} with period rows removed: {n_f} (rows removed {n_r:,}); unreadable files skipped: {len(bad)}"
)
for b in bad:
    print("   UNREADABLE", b)
# ---- X4 fold frame without the period's rows
os.makedirs(f"{WS}/scratchpad/x4", exist_ok=True)
t = f"{WS}/scratchpad/x4/X4_fold.parquet"
if os.path.lexists(t):
    os.unlink(t)
X4 = pl.read_parquet("scratchpad/x4/X4_fold.parquet")
X4f = X4.filter(~pl.col("MVT_ID_mvt").cast(pl.Float64).is_in(pida))
X4f.write_parquet(t)
print(f"X4_fold: {X4.height:,} -> {X4f.height:,}")
# ---- minimal XL frame (ranking composition for the informational gates in combo_treed_apply / daystate_lib.gate)
os.makedirs(f"{WS}/scratchpad/xl", exist_ok=True)
t = f"{WS}/scratchpad/xl/frame.parquet"
if os.path.lexists(t):
    os.unlink(t)
A = pl.read_parquet("data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"])
(
    pl.scan_parquet(fo)
    .select(
        "MVT_ID_mvt", "is_rank", "is_dep", "ADEP_mvt", "unmatched", "mvt_lobt", "mvt_aobt3", "mvt_eobt1", "mvt_sched"
    )
    .collect()
    .join(A, on="MVT_ID_mvt", how="left")
    .write_parquet(t)
)
# ---- submission template
os.makedirs(f"{WS}/data/raw", exist_ok=True)
t = f"{WS}/data/raw/submitting.parquet"
if os.path.lexists(t):
    os.unlink(t)
S = (
    P.filter(dep)
    .select(pl.col("MVT_ID_mvt").cast(pl.Float64))
    .sort("MVT_ID_mvt")
    .with_columns(TAXITIME_SEC_mvt=pl.lit(None, dtype=pl.Int32))
)
S.write_parquet(t)
print(f"period ranking departures (NRANK): {S.height}")
open(f"{WS}/manifest_dry.txt", "w").write(f"NRANK={S.height}\nLABEL_COLS={','.join(LAB)}\nOOF_FILTERED={n_f}\n")
