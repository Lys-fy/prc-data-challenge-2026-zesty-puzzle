#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""NEWSTAND_DELTA: new-stand neighbour mapping.
sim U|N <out.parquet>  : janjul simulation file (hidden stands S; training-month S rows removed; stand encodings recomputed without S;
                         U = S val rows keep own stand (unseen), N = S val rows keyed by nearest non-S neighbour)
prod U|N <out.parquet> : production file (ranking rows on stands unseen in 2025; U = features_v9 recomputed as is, N = neighbour keys)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, numpy as np, polars as pl

mode, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
FV = "data/features_v9.parquet"
SCH = pl.read_parquet_schema(FV)
d = pl.read_parquet(
    FV,
    columns=[
        "MVT_ID_mvt",
        "ADEP_mvt",
        "ADES_mvt",
        "STAND_mvt",
        "RUNWAY_mvt",
        "hour",
        "dow",
        "m",
        "is_rank",
        "y",
        "mvt_aobt3",
        "apt_stand",
    ],
)
pos = (
    pl.read_parquet("data/external/adsblol/stand_pos.parquet")
    .filter(pl.col("n") >= 5)
    .rename({"apt": "ADEP_mvt", "stand": "STAND_mvt"})
)
tr25 = d.filter(pl.col("is_rank") == 0)
if mode == "sim":
    rng = np.random.default_rng(0)
    S = []
    val = tr25.filter(pl.col("m").is_in([1, 7]))
    for apt in sorted(val["ADEP_mvt"].unique().to_list()):
        v = val.filter(pl.col("ADEP_mvt") == apt)
        cnt = v.group_by("STAND_mvt").len()
        cand = (
            cnt.join(pos.filter(pl.col("ADEP_mvt") == apt), on="STAND_mvt")
            .filter(pl.col("len") >= 20)
            .sort("STAND_mvt")
        )
        order = rng.permutation(cand.height)
        cov = 0
        for i in order:
            S.append((apt, cand["STAND_mvt"][int(i)]))
            cov += int(cand["len"][int(i)])
            if cov >= 0.05 * v.height:
                break
    S = pl.DataFrame(S, schema=["ADEP_mvt", "STAND_mvt"], orient="row").with_columns(pl.lit(True).alias("inS"))
    d = d.join(S, on=["ADEP_mvt", "STAND_mvt"], how="left").with_columns(pl.col("inS").fill_null(False))
    n0 = d.height
    d = d.filter(~((pl.col("is_rank") == 0) & ~pl.col("m").is_in([1, 7]) & pl.col("inS")))
    print("hidden stands", S.height, "removed training rows", n0 - d.height, flush=True)
    target = (pl.col("is_rank") == 0) & pl.col("m").is_in([1, 7]) & pl.col("inS")
    pool_excl = pl.col("inS")
else:
    seen = tr25.select("ADEP_mvt", "STAND_mvt").unique().with_columns(pl.lit(False).alias("inS"))
    d = d.join(seen, on=["ADEP_mvt", "STAND_mvt"], how="left").with_columns(
        pl.col("inS").fill_null(True) & pl.col("STAND_mvt").is_not_null()
    )
    S = d.filter(pl.col("inS")).select("ADEP_mvt", "STAND_mvt").unique()
    target = (pl.col("is_rank") == 1) & pl.col("inS")
    pool_excl = pl.lit(False)
# nearest non-S neighbour with a position
P = pos.join(
    S.select("ADEP_mvt", "STAND_mvt").with_columns(pl.lit(1).alias("_s")), on=["ADEP_mvt", "STAND_mvt"], how="left"
)
seenpos = P.filter(pl.col("_s").is_null())
if mode == "prod":
    seenpos = seenpos.join(tr25.select("ADEP_mvt", "STAND_mvt").unique(), on=["ADEP_mvt", "STAND_mvt"])
nb = []
for r in P.filter(pl.col("_s").is_not_null()).iter_rows(named=True):
    c = seenpos.filter(pl.col("ADEP_mvt") == r["ADEP_mvt"])
    if c.height == 0:
        continue
    dist = ((c["lat"] - r["lat"]) ** 2 + ((c["lon"] - r["lon"]) * np.cos(np.radians(r["lat"]))) ** 2).sqrt() * 111000
    i = int(dist.arg_min())
    nb.append((r["ADEP_mvt"], r["STAND_mvt"], c["STAND_mvt"][i], float(dist[i])))
NB = pl.DataFrame(nb, schema=["ADEP_mvt", "STAND_mvt", "NB_STAND", "nb_dist_m"], orient="row")
print("mapped stands:", NB.height, "of", S.height)
print(NB.sort("ADEP_mvt", "STAND_mvt").head(60).rows())
d = d.join(NB, on=["ADEP_mvt", "STAND_mvt"], how="left")
use_nb = target & pl.col("NB_STAND").is_not_null() & pl.lit(arm == "N")
d = d.with_columns(pl.when(use_nb).then(pl.col("NB_STAND")).otherwise(pl.col("STAND_mvt")).alias("KS"))
# ---- recompute stand-keyed encodings (build_features.py logic) with key KS, pool = 2025 rows not in S
trn = d.filter((pl.col("is_rank") == 0) & ~pool_excl)


def lomo_quantile(keys, prefix):
    parts = []
    for mm in range(1, 13):
        sub = (
            trn.filter(pl.col("m") != mm)
            .group_by(keys)
            .agg(
                pl.col("y").quantile(0.1).alias(f"{prefix}_p10"),
                pl.col("y").median().alias(f"{prefix}_med"),
                pl.col("y").mean().alias(f"{prefix}_mean"),
                pl.len().alias(f"{prefix}_n"),
            )
        )
        parts.append(
            d.filter((pl.col("is_rank") == 0) & (pl.col("m") == mm))
            .select(["MVT_ID_mvt"] + keys)
            .join(sub, on=keys, how="left")
            .drop(keys)
        )
    full = trn.group_by(keys).agg(
        pl.col("y").quantile(0.1).alias(f"{prefix}_p10"),
        pl.col("y").median().alias(f"{prefix}_med"),
        pl.col("y").mean().alias(f"{prefix}_mean"),
        pl.len().alias(f"{prefix}_n"),
    )
    parts.append(
        d.filter(pl.col("is_rank") == 1).select(["MVT_ID_mvt"] + keys).join(full, on=keys, how="left").drop(keys)
    )
    return pl.concat(parts)


d = d.with_columns((pl.col("dow") * 24 + pl.col("hour")).alias("how"))
trn2 = trn.with_columns(pl.col("y").clip(0, 3600).alias("yc"))


def shrink_te(keys, parent, prefix, m_smooth=30):
    tot = trn2.group_by(keys).agg(pl.col("yc").sum().alias("s_"), pl.len().alias("n_"))
    per = trn2.group_by(keys + ["m"]).agg(pl.col("yc").sum().alias("sm_"), pl.len().alias("nm_"))
    ptot = trn2.group_by(parent).agg(pl.col("yc").sum().alias("ps_"), pl.len().alias("pn_"))
    pper = trn2.group_by(parent + ["m"]).agg(pl.col("yc").sum().alias("psm_"), pl.len().alias("pnm_"))
    cols = list(dict.fromkeys(keys + parent))
    j = (
        d.select(["MVT_ID_mvt", "is_rank", "m"] + cols)
        .join(tot, on=keys, how="left")
        .join(per, on=keys + ["m"], how="left")
        .join(ptot, on=parent, how="left")
        .join(pper, on=parent + ["m"], how="left")
    )
    j = j.with_columns([pl.col(c).fill_null(0) for c in ("s_", "n_", "sm_", "nm_", "ps_", "pn_", "psm_", "pnm_")])
    own = pl.col("is_rank") == 0
    j = j.with_columns(
        (pl.col("s_") - pl.when(own).then(pl.col("sm_")).otherwise(0)).alias("s_"),
        (pl.col("n_") - pl.when(own).then(pl.col("nm_")).otherwise(0)).alias("n_"),
        (pl.col("ps_") - pl.when(own).then(pl.col("psm_")).otherwise(0)).alias("ps_"),
        (pl.col("pn_") - pl.when(own).then(pl.col("pnm_")).otherwise(0)).alias("pn_"),
    )
    j = j.with_columns((pl.col("ps_") / pl.col("pn_").clip(1)).alias("prior_"))
    j = j.with_columns(
        ((pl.col("s_") + m_smooth * pl.col("prior_")) / (pl.col("n_") + m_smooth)).alias(f"{prefix}_te"),
        pl.col("n_").cast(pl.Float32).alias(f"{prefix}_n"),
    )
    return j.select("MVT_ID_mvt", f"{prefix}_te", f"{prefix}_n")


trn = trn.with_columns(pl.col("STAND_mvt").alias("KS"))
trn2 = trn2.with_columns(pl.col("STAND_mvt").alias("KS"))
new = []
for keys, prefix in [
    (["ADEP_mvt", "KS", "RUNWAY_mvt"], "ref_sr"),
    (["ADEP_mvt", "KS"], "ref_s"),
    (["ADEP_mvt", "KS", "hour"], "ref_sh"),
]:
    new.append(lomo_quantile(keys, prefix))
for keys, parent, prefix in [
    (["ADEP_mvt", "KS", "RUNWAY_mvt"], ["ADEP_mvt", "RUNWAY_mvt"], "te2_sr"),
    (["ADEP_mvt", "KS", "RUNWAY_mvt", "hour"], ["ADEP_mvt", "KS", "RUNWAY_mvt"], "te2_srh"),
    (["ADEP_mvt", "KS", "ADES_mvt"], ["ADEP_mvt", "KS"], "te2_sdes"),
]:
    new.append(shrink_te(keys, parent, prefix))
cols_new = [c for t in new for c in t.columns if c != "MVT_ID_mvt"]
d = d.drop([c for c in cols_new if c in d.columns])
for t in new:
    d = d.join(t, on="MVT_ID_mvt", how="left")
d = d.with_columns(
    (pl.col("mvt_aobt3") - pl.col("ref_sr_p10")).alias("aobt3_minus_ref"),
    (pl.col("y") - pl.col("ref_sr_p10")).alias("y_minus_ref"),
    pl.when(use_nb).then(pl.col("ADEP_mvt") + "_" + pl.col("KS")).otherwise(pl.col("apt_stand")).alias("apt_stand"),
)
ids = d.filter(target).select("MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "NB_STAND", "nb_dist_m")
repl = cols_new + ["aobt3_minus_ref", "y_minus_ref", "apt_stand"]
patch = d.select(["MVT_ID_mvt"] + [pl.col(c).cast(SCH[c]) for c in repl])
print(
    "target rows",
    ids.height,
    "mapped",
    int(ids["NB_STAND"].is_not_null().sum()),
    ids.group_by("ADEP_mvt").len().sort("ADEP_mvt").rows(),
)
print(
    "null ref_s_p10 on target rows:",
    patch.join(ids.select("MVT_ID_mvt"), on="MVT_ID_mvt")["ref_s_p10"].is_null().mean(),
)
del d, trn, trn2, tr25
keep = patch.select("MVT_ID_mvt")  # rows kept (sim: training-month S rows removed)
cols = list(SCH.keys())
lf = pl.scan_parquet(FV).drop(repl).join(patch.lazy(), on="MVT_ID_mvt", how="inner").select(cols)
lf.sink_parquet(out)
ids.write_parquet(out.replace(".parquet", "_ids.parquet"))
print("wrote", out, pl.scan_parquet(out).select(pl.len()).collect().item())
