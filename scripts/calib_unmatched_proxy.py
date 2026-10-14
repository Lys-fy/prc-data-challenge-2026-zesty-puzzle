#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds a 2025 unmatched-row proxy for pru_daily_calib --um25 (its (B) day sums include the unmatched predictions).
Reads scratchpad/xur/S_xLTFM_HUB_resb_c14400_k0_s0.5_g900.npz and scratchpad/xur2/ARM_A1.npz; writes the output npz
path given as the sole positional argument.

Replaces, in the unm_resid npz 'final' array, the leave-one-month-out value of the callsign-link arm A1
(scratchpad/xur2/ARM_A1.npz 'cand') on the 2025 unm_other rows where A1 changes its reference (cand != ref), as the
production line does on the ranking rows.
usage: calib_unmatched_proxy.py <out.npz>"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, numpy as np

z = dict(np.load("scratchpad/xur/S_xLTFM_HUB_resb_c14400_k0_s0.5_g900.npz", allow_pickle=True))
a = np.load("scratchpad/xur2/ARM_A1.npz", allow_pickle=True)
pos = {v: i for i, v in enumerate(z["ids"])}
r = (~a["isr"]) & a["uo"] & (a["cand"] != a["ref"])
ix = np.array([pos[v] for v in a["ids"][r]])
assert (z["fold"][ix] != "rank").all()
old = z["final"][ix].copy()
z["final"] = z["final"].copy()
z["final"][ix] = a["cand"][r]
d = z["final"][ix] - old
print(
    f"2025 rows replaced {len(ix):,}; change RMS {np.sqrt((d**2).mean()):.1f} mean {d.mean():+.1f}; y-MSE old {np.nanmean((old - z['y'][ix])**2):.0f} new {np.nanmean((z['final'][ix] - z['y'][ix])**2):.0f}"
)
np.savez(sys.argv[1], **z)
