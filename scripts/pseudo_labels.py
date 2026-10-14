#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds pseudo-label files for the pseudo-label member (fold analogue of transductive self-training): on held-out
matched rows with a low member OOF artefact probability and a chain-teacher prediction under 7,200 s, writes three
label columns (chain teacher, self member, oracle y).
Inputs: positional fold names (any of the six HOLD folds; default months39 months28 months410 janjul) or "prod"; data/features_v9.parquet,
data/oof/queue_eb_ra3qa, env pseudolabel_fold_teacher/pseudolabel_kind (fold mode) or
pseudolabel_prod_p/pseudolabel_teacher/pseudolabel_prod_out (prod mode).
Outputs: scratchpad/xsyn/pseudo_<fold>_<kind>.parquet per fold (kind: env pseudolabel_kind, default chainB), or the prod-mode
output file (env pseudolabel_prod_out, default scratchpad/xsyn/pseudo_prod_chain88.parquet).

Pseudo-label files of the pseudo-label member, fold analogue of transductive self-training.
For fold t the "unlabelled test set" = the held-out months' MATCHED rows. One row mask per fold (label-free):
  matched (unmatched == 0), is_rank == 0, m in HOLD[t], L member OOF p < 0.05 (queue_eb_ra3qa row 2, artefact probability, fitted on the other months),
  chain teacher exists and its prediction < 7,200 s (no tail rows: L6).
Three label columns on the SAME rows -> scratchpad/xsyn/pseudo_<fold>_<kind>.parquet (MVT_ID_mvt, yhat):
  chain  = the chain fold vector named by env pseudolabel_fold_teacher (default scratchpad/xau1/clockfold1_le3b, label column chainB;
           pseudo_member.sh PL3 reads kind chain, built by scripts/a_side.sh step pseudo from clockfold1_le3BA; chain OOF, never saw the held months' labels)   [arm PL]
  self   = the L member's own OOF prediction (queue_eb_ra3qa row 1)                                                             [placebo PLB: self-distillation, no chain knowledge]
  oracle = the true label y                                                                                                [ceiling ORC]
usage: pseudo_labels.py [fold ...]   (folds: janjul months28 months39 months410 months511 months612; default: months39 months28 months410 janjul)
       pseudo_labels.py prod        production file (docs/METHOD.md §9 item 3): matched 2026 ranking rows, mean production-member p_raw (env pseudolabel_prod_p, default
                                  data/preds/queue_eb_v47lqa_s*.parquet) < 0.05, teacher (env pseudolabel_teacher, default submissions/v99_iso_L.parquet = the period's first-pass line)
                                  < 7,200 s -> $pseudolabel_prod_out (default scratchpad/xsyn/pseudo_prod_chain88.parquet; yhat = the teacher's TAXITIME_SEC_mvt)
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "scratchpad/xsyn"
os.makedirs(OUT, exist_ok=True)
HOLD = {
    "janjul": [1, 7],
    "months28": [2, 8],
    "months39": [3, 9],
    "months410": [4, 10],
    "months511": [5, 11],
    "months612": [6, 12],
}
P_MAX, Y_MAX = 0.05, 7200.0
# PSEUDO_FIX: the fold teacher must be the (B)-only arm clockfold1_le3b (LE3BA = (B) + the monthly PRU add-on, a y_f aggregate). Env pseudolabel_fold_teacher
# (default clockfold1_le3b) -> label column "chainB" (kind suffix env pseudolabel_kind, default chainB); the original LE3BA files keep their name "chain".
FT = os.environ.get("pseudolabel_fold_teacher", "scratchpad/xau1/clockfold1_le3b")
KIND = os.environ.get("pseudolabel_kind", "chainB")


def npy(p, t):
    c = sorted(glob.glob(f"{p}_{t}.npy") or glob.glob(f"{p}_{t}_*.npy"))
    assert c, (p, t)
    return np.load(c[0])


def al(a, ids):
    o = np.argsort(a[0])
    k = np.clip(np.searchsorted(a[0][o], ids), 0, a.shape[1] - 1)
    hit = a[0][o][k] == ids
    return hit, a[1:, o][:, k]


F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "y", "m", "is_rank", "unmatched", "ADEP_mvt"])


def prod():  # the production analogue (teacher = the period's first-pass line, p = the production L member's mean p_raw over its seeds); label-free
    V = F.filter((pl.col("is_rank") == 1) & (pl.col("unmatched") == 0))
    ids = V["MVT_ID_mvt"].to_numpy().astype(float)
    apt = V["ADEP_mvt"].to_numpy()
    pf = sorted(glob.glob(os.environ.get("pseudolabel_prod_p", "data/preds/queue_eb_v47lqa_s*.parquet")))
    assert pf, "no production member files"
    P = None
    for f in pf:
        d = pl.read_parquet(f, columns=["MVT_ID_mvt", "p_raw"])
        a = np.vstack([d["MVT_ID_mvt"].to_numpy().astype(float), d["p_raw"].to_numpy().astype(float)])
        h, v = al(a, ids)
        assert h.all(), f
        P = v[0] if P is None else P + v[0]
    P = P / len(pf)
    tf = os.environ.get(
        "pseudolabel_teacher", "submissions/v99_iso_L.parquet"
    )  # the period's own first-pass line (phase2/run_pipeline.sh step xs_pseudo)
    d = pl.read_parquet(tf, columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
    hc, C = al(
        np.vstack([d["MVT_ID_mvt"].to_numpy().astype(float), d["TAXITIME_SEC_mvt"].to_numpy().astype(float)]), ids
    )
    assert hc.all(), tf
    sel = (P < P_MAX) & (C[0] < Y_MAX)
    po = os.environ.get("pseudolabel_prod_out", f"{OUT}/pseudo_prod_chain88.parquet")
    pl.DataFrame({"MVT_ID_mvt": ids[sel], "yhat": C[0][sel]}).write_parquet(po)
    print(
        f"prod -> {po}: matched ranking rows {len(ids):,} | p from {len(pf)} files {os.path.basename(pf[0])}.. | teacher {tf} | selected {sel.sum():,} ({100 * sel.mean():.1f} %)"
        f" | dropped p>={P_MAX} {int((P >= P_MAX).sum()):,}, teacher>={Y_MAX:.0f} {int((C[0] >= Y_MAX).sum()):,} | yhat mean {C[0][sel].mean():.0f}",
        flush=True,
    )
    for a_ in sorted(set(apt)):
        k = sel & (apt == a_)
        print(f"   {a_} selected {k.sum():,} of {(apt == a_).sum():,}", end="")
    print()


if sys.argv[1:] == ["prod"]:
    prod()
    sys.exit(0)
for t in sys.argv[1:] or ["months39", "months28", "months410", "janjul"]:
    V = F.filter((pl.col("is_rank") == 0) & (pl.col("unmatched") == 0) & pl.col("m").is_in(HOLD[t]))
    ids = V["MVT_ID_mvt"].to_numpy().astype(float)
    y = V["y"].to_numpy().astype(float)
    apt = V["ADEP_mvt"].to_numpy()
    hm, M = al(npy("data/oof/queue_eb_ra3qa", t), ids)
    hc, C = al(npy(FT, t), ids)
    sel = hm & hc & (M[1] < P_MAX) & (C[0] < Y_MAX) & np.isfinite(y)
    lab = {KIND: C[0], "self": M[0], "oracle": y}
    for kind, v in lab.items():
        pl.DataFrame({"MVT_ID_mvt": ids[sel], "yhat": v[sel]}).write_parquet(f"{OUT}/pseudo_{t}_{kind}.parquet")
    e_c = C[0][sel] - y[sel]
    e_m = M[0][sel] - y[sel]
    print(
        f"{t} [teacher {FT} -> {KIND}]: held matched rows {len(ids):,} | member hit {hm.sum():,} chain hit {hc.sum():,} | selected {sel.sum():,} ({100 * sel.mean():.1f} %)"
        f" | dropped p>={P_MAX} {int((hm & (M[1] >= P_MAX)).sum()):,}, chain>={Y_MAX:.0f} {int((hc & (C[0] >= Y_MAX)).sum()):,}"
        f" | on selected rows MSE chain {np.mean(e_c ** 2):,.0f} member {np.mean(e_m ** 2):,.0f} | y mean {y[sel].mean():.0f} chain mean {C[0][sel].mean():.0f}",
        flush=True,
    )
    for a_ in sorted(set(apt)):
        k = sel & (apt == a_)
        print(f"   {a_} selected {k.sum():,} of {(apt == a_).sum():,}", end="")
    print()
