#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""phase2_ws: score a dry-run period against its held-out labels (never readable by the pipeline itself).
Reports RMSE overall / by class / by airport, and the same rows under the historical v76-equivalent fold vectors (the (B)-stage pool
OOF pru_pxxb of the fold that held these months, + the unmatched-row OOF where available) as the accuracy the pipeline should reach.
usage (cwd = workspace, no guard): p2_score.py <period> [--fold months28]"""

import os, sys, json, argparse, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
REPO = os.environ.get("P2_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("period")
ap.add_argument("--fold", default="months28")
ap.add_argument("--file", default="submissions/final.parquet")
ap.add_argument("--tag", default="px")
a = ap.parse_args()
Y = pl.read_parquet(f"{REPO}/phase2/heldout/{a.period}_labels.parquet").with_columns(
    mid=pl.col("MVT_ID_mvt").cast(pl.Float64), ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8)
)
P = (
    pl.read_parquet(a.file)
    .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
    .select("mid", p=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
)
U = (
    pl.read_parquet("scratchpad/p2/U.parquet")
    .with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
    .select("mid", "cls")
)
z = np.load(f"{REPO}/scratchpad/xpr/pru_pxxb_{a.fold}.npy")
H = pl.DataFrame({"mid": z[0], "h": z[1]})
J = Y.join(P, on="mid", how="left").join(U, on="mid", how="left").join(H, on="mid", how="left")
assert J["p"].null_count() == 0, "prediction missing for some period rows"


def rm(d):
    return float(np.sqrt(np.mean(d**2))) if len(d) else float("nan")


e = (J["p"] - J["y"]).to_numpy()
out = {"rows": J.height, "rmse": rm(e), "by_class": {}, "by_airport": {}}
for c in ("matched", "unm_other", "unm_LIRF", "exposed"):
    k = (J["cls"] == c).to_numpy()
    out["by_class"][c] = {"n": int(k.sum()), "rmse": rm(e[k])}
M = J.filter(pl.col("cls") == "matched")
k = M["h"].is_not_null().to_numpy()
em = (M["p"] - M["y"]).to_numpy()
eh = (M["h"] - M["y"]).to_numpy()
out["matched_vs_fold"] = {
    "n_common": int(k.sum()),
    "pipeline_rmse": rm(em[k]),
    "fold_v76eq_rmse": rm(eh[k]),
    "rms_diff_pred": rm((M["p"] - M["h"]).to_numpy()[k]),
}
for ap_ in sorted(J["ADEP_mvt"].unique().to_list()):
    kk = (J["ADEP_mvt"] == ap_).to_numpy()
    out["by_airport"][ap_] = {"n": int(kk.sum()), "rmse": rm(e[kk])}
os.makedirs("report", exist_ok=True)
json.dump(out, open(f"report/score_{a.tag}.json", "w"), indent=1)
print(f"{a.period}: {out['rows']:,} departures, RMSE {out['rmse']:.2f} s")
for c, v in out["by_class"].items():
    print(f"   {c:9s} n {v['n']:7,} RMSE {v['rmse']:9.2f}")
m = out["matched_vs_fold"]
print(
    f"   matched rows with a {a.fold} fold vector: {m['n_common']:,} | pipeline RMSE {m['pipeline_rmse']:.2f} vs v76-equivalent fold OOF {m['fold_v76eq_rmse']:.2f} (RMS gap between the two {m['rms_diff_pred']:.1f} s)"
)
for ap_, v in out["by_airport"].items():
    print(f"   {ap_} n {v['n']:6,} RMSE {v['rmse']:8.2f}")
