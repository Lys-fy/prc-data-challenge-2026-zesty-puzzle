#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Applies the unm_resid model to unmatched 'unm_other' ranking rows by rule: value = max(rint(final), 60), where
'final' already keeps the baseline on LTFM rows and on rows whose candidate moves more than 7,200 s. unm_LIRF and
exposed rows are out of scope and keep step U's frozen-blend value.
Input: <S tag> (positional, selects the unm_resid stacker output scratchpad/xur/S_<tag>.npz) and
scratchpad/p2/U.parquet. Output: scratchpad/p2/U2.parquet (MVT_ID_mvt, u2 on unm_other rows).

phase2_ws: unmatched-row model unm_resid applied by rule.
Input: the unm_resid stacker output scratchpad/xur/S_<tag>.npz (unm_resid_d.py + unm_resid_s.py run in this workspace; its ranking 'base' is the C1
rule of step U through stages/unmatched_rule.parquet). Rule: unm_other rows get max(rint(final), 60); 'final' already keeps the
baseline on LTFM rows (declared scope rule APT_MEMBER=LTFM, round 2) and on rows whose candidate moves > 7,200 s.
unm_LIRF and exposed rows are not in scope (they keep the frozen-blend rule of step U).
Writes scratchpad/p2/U2.parquet (MVT_ID_mvt, u2 on unm_other rows).
usage: p2_unm_resid_apply.py <S tag>"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
T = sys.argv[1]
z = np.load(f"scratchpad/xur/S_{T}.npz", allow_pickle=True)
r = z["fold"] == "rank"
N = pl.DataFrame(
    {"mid": z["ids"][r].astype(float), "new": np.rint(z["final"][r]), "cls_s": z["cls"][r], "big": z["moved_big"][r]}
)
U = pl.read_parquet("scratchpad/p2/U.parquet").with_columns(mid=pl.col("MVT_ID_mvt").cast(pl.Float64))
J = U.filter(pl.col("cls") == "unm_other").join(N, on="mid", how="left", maintain_order="left")
assert (
    J["new"].null_count() == 0 and (J["cls_s"] == "unm_other").all()
), "unm_resid output does not cover every unm_other row"
u2 = np.maximum(J["new"].to_numpy(), 60).astype(np.int32)
d = u2.astype(float) - J["u"].to_numpy().astype(float)
apt = J["ADEP_mvt"].to_numpy()
assert (d[apt == "LTFM"] == 0).all() and (np.abs(d) <= 7200).all()
pl.DataFrame({"MVT_ID_mvt": J["MVT_ID_mvt"], "u2": u2}).write_parquet("scratchpad/p2/U2.parquet")
print(
    f"unm_resid on unm_other: rows {J.height}, changed {(d != 0).sum()} (LTFM kept {(apt == 'LTFM').sum()}), mean {d.mean():+.2f} s, RMS {np.sqrt((d**2).mean()):.1f}, max {np.abs(d).max():.0f}"
)
