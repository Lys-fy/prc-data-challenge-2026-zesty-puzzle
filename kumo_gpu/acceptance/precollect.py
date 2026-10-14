#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Kumo acceptance step 5b: assemble the arm-M fold files ON THE CLUSTER (no features_v9 there) from results/member/kumo_member_screen_km_<fold>_<APT>.npz:
sorted by `row` (= the features_v9 row index staged in the payload) -> results/oof/kumo_member_screen_km_<fold>.npy = [id, taxi] float64, exactly what collect.py
writes once its features_v9 row-set assertion passes on the build machine. Also writes results/MD5SUMS (every npz / npy) for the transfer back.
Checks: all 10 airports per fold, finite taxi, one consistent (size, nest, amp, seed) setting, no duplicate rows. Prints `PRECOLLECT {json}`; rc 1 on FAIL.
usage: python precollect.py <RUN> [NAME=kumo_member_screen_km] [folds=months39,months410]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, glob, hashlib, numpy as np

RUN = sys.argv[1]
NAME = sys.argv[2] if len(sys.argv) > 2 else "kumo_member_screen_km"
folds = (sys.argv[3] if len(sys.argv) > 3 else "months39,months410").split(",")
APTS = "EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH,LTFM".split(",")
out = {"name": NAME, "folds": {}, "fail": []}
os.makedirs(f"{RUN}/results/oof", exist_ok=True)
for fold in folds:
    fs = {a: f"{RUN}/results/member/{NAME}_{fold}_{a}.npz" for a in APTS}
    miss = [a for a, f in fs.items() if not os.path.exists(f)]
    if miss:
        out["fail"].append(f"{fold}: missing {miss}")
        out["folds"][fold] = "incomplete"
        continue
    Z = [np.load(fs[a]) for a in APTS]
    M = [json.loads(str(z["meta"])) for z in Z]
    sets = {
        json.dumps({k: m.get("profile", {}).get(k) for k in ("size", "nest", "amp", "cap")}, sort_keys=True)
        + f" seed {m.get('seed')}"
        for m in M
    }
    row = np.concatenate([z["row"] for z in Z])
    ids = np.concatenate([z["id"] for z in Z])
    taxi = np.concatenate([z["taxi"] for z in Z])
    o = np.argsort(row)
    row, ids, taxi = row[o], ids[o], taxi[o]
    if len(np.unique(row)) != len(row):
        out["fail"].append(f"{fold}: duplicate rows")
    if not np.isfinite(taxi).all():
        out["fail"].append(f"{fold}: non-finite taxi")
    if len(sets) > 1:
        out["fail"].append(f"{fold}: mixed settings {sets}")
    p = f"{RUN}/results/oof/{NAME}_{fold}.npy"
    np.save(p, np.vstack([ids, taxi]))
    out["folds"][fold] = dict(
        rows=int(len(ids)),
        taxi_mean=round(float(taxi.mean()), 1),
        taxi_sd=round(float(taxi.std()), 1),
        settings=sorted(sets)[0],
        secs=round(sum(m["secs"] for m in M)),
        peak_gib=round(max(m["peak_gib"] for m in M), 1),
        rows_s=round(sum(m["query"] for m in M) / max(1, sum(m["secs"] for m in M))),
    )
with open(f"{RUN}/results/MD5SUMS", "w") as fh:
    for f in sorted(glob.glob(f"{RUN}/results/**/*.np[zy]", recursive=True)):
        fh.write(f"{hashlib.md5(open(f, 'rb').read()).hexdigest()}  {os.path.relpath(f, RUN)}\n")
out["verdict"] = "PASS" if not out["fail"] else "FAIL"
print("PRECOLLECT " + json.dumps(out), flush=True)
sys.exit(0 if not out["fail"] else 1)
