#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""KCATLg: the log-target Kumo member with a tail guard. Per row, the KCAT8 member is used where the KCAT8 prediction is above T = 2,400 s,
and the KCATL16 bag elsewhere; all four member rows [id, taxi, spread, g_sd] come from the chosen source. The rule is label-free and the same
for every airport: a log target compresses the extreme tail (the mean of expm1 over log quantiles under-predicts very long taxi-outs), so the
raw-target member keeps the rows that it itself predicts long. T is one global value from the grid {1,800, 2,400, 3,600 s, KCAT8 q99}.
usage: kumo_kcat_logtarget_tailguard.py <fold> ...             fold side: data/oof/KCAT_<fold>.npy + KCATL16_<fold>.npy -> data/oof/KCATLg_<fold>.npy
       kumo_kcat_logtarget_tailguard.py prod <workspace>       period:   <ws>/data/oof/{KCAT,KCATL16}_prod.npy -> <ws>/data/oof/KCATLg_prod.npy
                                                                          and <ws>/data/preds/KCATLgP_s0.parquet (MVT_ID_mvt, taxi)
Root (fold side) = $PRC_ROOT, else the current directory."""

import os, sys, hashlib, numpy as np, polars as pl

T = 2400.0
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()


def guard(a, b):
    """a, b: member arrays [id, taxi, spread, g_sd] x rows of KCAT8 and KCATL16 (same ids, same order). Returns (out, use8): the
    column of a where a's taxi > T, else the column of b."""
    assert a.shape == b.shape and np.array_equal(a[0], b[0]), "KCAT8 and KCATL16 rows differ"
    use8 = a[1] > T
    return np.where(use8[None, :], a, b), use8


def main(argv):
    if argv[:1] in (["-h"], ["--help"]):
        sys.exit(print(__doc__))
    a = argv
    if a and a[0] == "prod":
        W = a[1]
        out, u = guard(np.load(f"{W}/data/oof/KCAT_prod.npy"), np.load(f"{W}/data/oof/KCATL16_prod.npy"))
        np.save(f"{W}/data/oof/KCATLg_prod.npy", out)
        pl.DataFrame({"MVT_ID_mvt": out[0], "taxi": out[1]}).write_parquet(f"{W}/data/preds/KCATLgP_s0.parquet")
        print(
            f"KCATLg prod: {out.shape[1]:,} rows, KCAT8 on {u.sum():,} ({u.mean():.2%}), taxi mean {out[1].mean():.1f} | "
            f"md5 {md5(f'{W}/data/oof/KCATLg_prod.npy')} / {md5(f'{W}/data/preds/KCATLgP_s0.parquet')}"
        )
    else:
        os.chdir(os.environ.get("PRC_ROOT", os.getcwd()))
        for f in a:
            out, u = guard(np.load(f"data/oof/KCAT_{f}.npy"), np.load(f"data/oof/KCATL16_{f}.npy"))
            p = f"data/oof/KCATLg_{f}.npy"
            np.save(p, out)
            print(
                f"KCATLg {f}: {out.shape[1]:,} rows, KCAT8 on {u.sum():,} ({u.mean():.2%}), taxi mean {out[1].mean():.1f} | md5 {md5(p)}",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
