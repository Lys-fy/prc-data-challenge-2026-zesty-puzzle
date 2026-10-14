#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""KSTK production pre-stage file: runs scripts/chainvar_assemble.py with one asserted extra patch of the production pre-stage source:
the line `gc = np.clip(meta.predict(X(d, COLS)), -CLIP_LO, CLIP_HI)` becomes the KSTK hook. Knobs, both off by default (then the result
equals chainvar_assemble.py's):
  KSTK_DUMP=<file.npz>  write the production stacker frame of the matched ranking rows (mid, apt, v41 = production v, lobt, X = the meta input
                        matrix as the meta reads it, cols, gM = the production meta output before clipping) and exit 0 (no pre file written).
  KSTK_GKP=<file.npy>   [id, taxi] of the Kumo run on the production frame (scripts/kstk_gk.py); gK = taxi - v on matched rows (100 % coverage
                        asserted), g = (1 - KSTK_W) * gM + KSTK_W * gK (KSTK_W default 0.3, as on the fold side), then the clip of +-150 s.
usage: as chainvar_assemble.py (env from scripts/kstk_bundle_prod.sh)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = "gc = np.clip(meta.predict(X(d, COLS)), -CLIP_LO, CLIP_HI)"
HOOK = """_gm = meta.predict(X(d, COLS))
if E("KSTK_DUMP", ""):   # KSTK_ASSEMBLE: production stacker frame (matched ranking rows) -> npz, then stop
    _xq = X(d, COLS); _Xm = _xq.to_numpy(dtype=np.float32)[m]
    np.savez_compressed(E("KSTK_DUMP", ""), mid=ids.astype(np.float64)[m], apt=apt[m].astype("U4"), v41=np.asarray(v, np.float64)[m], lobt=lo[m],
                        X=_Xm, cols=np.array(_xq.columns).astype("U32"), gM=np.asarray(_gm, np.float64)[m], y=np.full(int(m.sum()), np.nan))
    print(f"KSTK_DUMP prod: {int(m.sum()):,} matched ranking rows x {_Xm.shape[1]} cols, NaN share {np.isnan(_Xm).mean():.4f} -> {E('KSTK_DUMP', '')}", flush=True); sys.exit(0)
if E("KSTK_GKP", ""):   # KSTK_ASSEMBLE: KB = (1 - W) gM + W gK on matched ranking rows
    _a = np.load(E("KSTK_GKP", "")); _o = np.argsort(_a[0]); _i = ids.astype(np.float64); _k = np.clip(np.searchsorted(_a[0][_o], _i), 0, _a.shape[1] - 1)
    _h = _a[0][_o][_k] == _i; assert _h[m].all(), f"KSTK_ASSEMBLE: {E('KSTK_GKP', '')} covers {_h[m].mean():.5f} of the matched ranking rows"
    _gk = np.where(m, _a[1][_o][_k] - v, 0.0); assert np.isfinite(_gk[m]).all(); _w = float(E("KSTK_W", "0.3"))
    print(f"KSTK_GKP: W {_w} matched {int(m.sum()):,} corr(gK, gM) {np.corrcoef(_gk[m], _gm[m])[0, 1]:.3f} sd gK {_gk[m].std():.1f} gM {_gm[m].std():.1f}", flush=True)
    _gm = (1 - _w) * _gm + _w * _gk
gc = np.clip(_gm, -CLIP_LO, CLIP_HI)"""


def _kstk_patch(S):
    assert S.count(LINE) == 1, "KSTK_ASSEMBLE: production meta line not found exactly once"
    return S.replace(LINE, HOOK)


src = open(f"{ROOT}/scripts/chainvar_assemble.py").read()
A = 'g = {"__file__": os.path.abspath("scripts/audited_assemble.py")'
assert src.count(A) == 1
src = src.replace(A, "S = _kstk_patch(S)\n" + A)
exec(
    compile(src, "xkstk_assemble", "exec"),
    {"__file__": f"{ROOT}/scripts/chainvar_assemble.py", "__name__": "__main__", "_kstk_patch": _kstk_patch},
)
