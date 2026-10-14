#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""TW1' fit: the LFPG return-to-stand twin shift, without any MVT_ID ordering. Class: LFPG, non-null callsign, pair of exactly 2 DEP records
with |dMVT| <= 300 s, EXACTLY ONE record NM-matched -> that record. Matched+matched pairs: no shift.
Inputs: scratchpad/xbig2/twins.parquet (scripts/tw1_twins.py) and the matched fold line table scratchpad/xatlas/rows_matched.parquet
(scripts/error_atlas_line.sh). s' = n/(n+5)*mean(y-pred) over the 2025 class UNS rows (12 months); LOMO gate G1-G7, placebo 50 draws.
Writes scratchpad/xquick/tw1p_ids_2026.parquet (2026 class ids of the ranking file) and scratchpad/xquick/tw1p_shift.json
(s' and rint(s') = TW1S of phase2/steps/p2_v107.sh)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, glob, numpy as np, polars as pl

os.environ.setdefault("POLARS_MAX_THREADS", "3")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
os.chdir(os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + ""))
OUT = "scratchpad/xquick"
os.makedirs(OUT, exist_ok=True)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(300)
pl.Config.set_tbl_rows(60)
RN = 344841.0
T = pl.read_parquet("scratchpad/xbig2/twins.parquet")
T = T.filter(
    pl.col("FLIGHT_mvt").is_not_null()
    & (pl.col("pair_n") == 2)
    & ((pl.col("dprev").abs() <= 300) | (pl.col("dnext").abs() <= 300))
)
# ID-free selector: the matched record of a pair with exactly one matched record
T = T.with_columns(nm=pl.col("matched").cast(pl.Int32).sum().over("rank", "pair")).with_columns(
    sel=pl.col("matched") & (pl.col("nm") == 1), first=(pl.col("pos") == 1)
)
print(
    "pairs by rank x n_matched (LFPG):",
    T.filter(pl.col("ADEP_mvt") == "LFPG")
    .group_by("rank", "pair")
    .agg(nm=pl.col("nm").first())
    .group_by("rank", "nm")
    .len()
    .sort("rank", "nm")
    .rows(),
)
print(
    "TW1' rows where sel != (first & matched):",
    T.filter(pl.col("sel") != (pl.col("first") & pl.col("matched")))
    .select("rank", "pair", "pos", "MVT_ID_mvt", "ADEP_mvt", "FLIGHT_mvt", "matched", "nm")
    .sort("rank", "pair", "pos"),
)
M = pl.read_parquet(
    "scratchpad/xatlas/rows_matched.parquet",
    columns=["MVT_ID_mvt", "y", "ADEP_mvt", "MVT_TIME_UTC_mvt", "fold", "pred", "w", "uns", "drop"],
)
M = M.with_columns(
    pl.col("MVT_ID_mvt").cast(pl.Int64), mon=pl.col("MVT_TIME_UTC_mvt").dt.month(), e=pl.col("pred") - pl.col("y")
)
MT = M.join(
    T.filter(pl.col("rank") == 0).select("MVT_ID_mvt", "pair", "sel", "first", "nm"), on="MVT_ID_mvt", how="inner"
)
L = MT.filter((pl.col("ADEP_mvt") == "LFPG") & pl.col("sel") & (pl.col("uns") == 1) & (pl.col("drop") == 0))
L1 = MT.filter((pl.col("ADEP_mvt") == "LFPG") & pl.col("first") & (pl.col("uns") == 1) & (pl.col("drop") == 0))
print(
    f"\n2025 LFPG class UNS not-dropped rows: TW1' n={L.height} e mean {float(L['e'].mean()):.1f} | TW1 (first) n={L1.height} e mean {float(L1['e'].mean()):.1f}"
)
print(
    "rows in TW1 but not TW1' (matched+matched first records):",
    MT.filter((pl.col("ADEP_mvt") == "LFPG") & pl.col("first") & ~pl.col("sel")).select(
        "MVT_ID_mvt", "pair", "mon", "y", "pred", "e", "uns", "drop"
    ),
)
n_all = L.height
s_prime = n_all / (n_all + 5.0) * float((L["y"] - L["pred"]).mean())
print(f"TW1' production shift s' = {s_prime:.1f} s (TW1 was 1320)")
# ---- LOMO gate ----
rows = []
for m in range(1, 13):
    tr = L.filter(pl.col("mon") != m)
    te = L.filter(pl.col("mon") == m)
    if te.height == 0:
        continue
    n = tr.height
    s = n / (n + 5.0) * float((tr["y"] - tr["pred"]).mean())
    rows.append(te.with_columns(shift=pl.lit(s)))
LL = (
    pl.concat(rows)
    .with_columns(cand=pl.col("pred") + pl.col("shift"))
    .with_columns(
        d=(pl.col("w") * ((pl.col("cand") - pl.col("y")) ** 2 - pl.col("e") ** 2)) / RN,
        draw=((pl.col("cand") - pl.col("y")) ** 2 - pl.col("e") ** 2),
        imp=((pl.col("cand") - pl.col("y")).abs() < pl.col("e").abs()),
    )
)
FOLDS = {"m39": [3, 9], "m28": [2, 8], "m410": [4, 10], "janjul": [1, 7], "m511": [5, 11], "m612": [6, 12]}


def fold_tab(D):
    out = {}
    for f, ms in FOLDS.items():
        x = D.filter(pl.col("mon").is_in(ms))
        if x.height == 0:
            out[f] = (np.nan, np.nan, 0)
            continue
        raw = x["draw"].to_numpy()
        top3 = np.argsort(-np.abs(raw))[:3]
        mask = np.ones(len(raw), bool)
        mask[top3] = False
        out[f] = (float(x["d"].sum()), float(raw[mask].sum() / RN), x.height)
    return out


ft = fold_tab(LL)
gate = ["m39", "m28", "m410", "janjul"]
mw4 = float(np.mean([ft[f][0] for f in gate]))
print(
    "\nTW1' per fold (MW board s2, raw w/o top-3, n):",
    {k: (round(v[0], 1), round(v[1], 1), v[2]) for k, v in ft.items()},
)
mneg = int(sum(1 for m in range(1, 13) if float(LL.filter(pl.col("mon") == m)["d"].sum()) < 0))
print(
    "4-fold MW mean",
    round(mw4, 1),
    "| months < 0:",
    mneg,
    "of",
    LL["mon"].n_unique(),
    "| improved share",
    round(float(LL["imp"].mean()), 3),
    "| LOMO shift range",
    round(float(LL["shift"].min())),
    round(float(LL["shift"].max())),
)
rng = np.random.default_rng(0)
base = M.filter((pl.col("ADEP_mvt") == "LFPG") & (pl.col("uns") == 1) & (pl.col("drop") == 0))
plc = []
nm = {m: n for m, n in L.group_by("mon").len().sort("mon").rows()}
for k in range(50):
    samp = pl.concat(
        [
            base.filter(pl.col("mon") == m).sample(
                n=min(n, base.filter(pl.col("mon") == m).height), seed=int(rng.integers(1e9))
            )
            for m, n in nm.items()
        ]
    )
    rr = []
    for m in nm:
        tr = samp.filter(pl.col("mon") != m)
        te = samp.filter(pl.col("mon") == m)
        n = tr.height
        s = n / (n + 5.0) * float((tr["y"] - tr["pred"]).mean())
        rr.append(te.with_columns(shift=pl.lit(s)))
    P = (
        pl.concat(rr)
        .with_columns(cand=pl.col("pred") + pl.col("shift"))
        .with_columns(
            d=(pl.col("w") * ((pl.col("cand") - pl.col("y")) ** 2 - pl.col("e") ** 2)) / RN,
            draw=((pl.col("cand") - pl.col("y")) ** 2 - pl.col("e") ** 2),
        )
    )
    pf = fold_tab(P)
    plc.append(np.mean([pf[f][0] for f in gate]))
print("placebo 4-fold MW mean over 50 draws: mean", round(float(np.mean(plc)), 2), "sd", round(float(np.std(plc)), 2))
g1 = sum(1 for f in gate if ft[f][0] < 0) >= 3
g2 = ft["janjul"][0] <= 0
g3 = sum(1 for f in gate if ft[f][1] < 0) >= 3
g5 = mneg >= 7
g6 = mw4 <= float(np.mean(plc)) - 5
g7 = float(LL["imp"].mean()) >= 0.55
verdict = "PASS" if all([g1, g2, g3, g5, g6, g7]) else "FAIL"
print("GATES G1", g1, "G2", g2, "G3", g3, "G5", g5, "G6", g6, "G7", g7, "=>", verdict)
# ---- 2026 class (label-free) and the fitted constant ----
T26 = T.filter((pl.col("rank") == 1) & (pl.col("ADEP_mvt") == "LFPG") & pl.col("sel"))
ids = T26["MVT_ID_mvt"].to_list()
print("\n2026 TW1' rows:", len(ids))
old = T.filter((pl.col("rank") == 1) & (pl.col("ADEP_mvt") == "LFPG") & pl.col("matched") & pl.col("first"))[
    "MVT_ID_mvt"
].to_list()
print(
    "2026 TW1 (old) rows:", len(old), "| dropped:", sorted(set(old) - set(ids)), "| added:", sorted(set(ids) - set(old))
)
pl.DataFrame({"MVT_ID_mvt": pl.Series(ids, dtype=pl.Int64)}).write_parquet(f"{OUT}/tw1p_ids_2026.parquet")
sh = int(np.rint(s_prime))
json.dump(
    {
        "s_prime": s_prime,
        "TW1S": sh,
        "n": n_all,
        "gate": verdict,
        "lomo_4fold_mw": mw4,
        "placebo_mean": float(np.mean(plc)),
    },
    open(f"{OUT}/tw1p_shift.json", "w"),
    indent=1,
)
print("RESULT", verdict, "| s'", round(s_prime, 1), "| rint", sh, "->", f"{OUT}/tw1p_shift.json")
