#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Seed aliasing for a reduced-seed run of a new period. For every member family of manifest/members.tsv (and the
derived queue_decomp_ebdt), each v76 seed file that was NOT retrained in this workspace (missing, or still a link to the shared prediction that was
fitted for another ranking file) is replaced by a link to the family's retrained seed 0. Stage scripts with hard-coded seed counts then
average identical files (= seed 0). V25Lb (seeds 10-14, a sibling recipe of V25L) is aliased to V25L_s0.
Fails if a family has no retrained seed 0. Writes manifest/aliases.txt. Run with cwd = workspace."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, csv

P2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.getcwd()
fams = []
for r in csv.reader(open(f"{P2}/manifest/members.tsv"), delimiter="\t"):
    if not r or r[0].startswith("#") or len(r) < 6:
        continue
    lo, hi = r[1][1:].split("-")
    fams.append((r[0], int(lo), int(hi)))
fams += [("queue_decomp_ebdt", 0, 2), ("sixseed_chain_ebdt6", 0, 5), ("sixseed_chain_nnra3x6", 0, 5)]
local = lambda p: os.path.exists(p) and not os.path.islink(p)
out = []
for fam, lo, hi in fams:
    src = f"{ {'V25Lb': 'V25L', 'sixseed_chain_nnra3x6': 'meta_oof_realmlp_nnra3'}.get(fam, fam)}_s0.parquet"
    if fam in ("sixseed_chain_ebdt6", "sixseed_chain_nnra3x6") and not os.path.lexists(f"data/preds/{src}"):
        out.append(f"# {fam}: not built (v87 line only)")
        continue
    if fam.startswith(("wideleaf_", "pseudolabel_")) and not local(
        f"data/preds/{src}"
    ):  # optional v99-line families (flags WIDELEAF / PSEUDOLABEL; STAGE_COMPARE_V99)
        out.append(f"# {fam}: not retrained in this workspace (flag off, or its pass has not run yet)")
        continue
    assert local(f"data/preds/{src}") or (
        fam == "sixseed_chain_nnra3x6"
        and os.path.islink(f"data/preds/{src}")
        and local(os.path.join("data/preds", os.readlink(f"data/preds/{src}")))
    ), f"{fam}: retrained seed 0 missing ({src})"
    for s in range(lo, hi + 1):
        p = f"data/preds/{fam}_s{s}.parquet"
        if local(p) or (fam == "sixseed_chain_nnra3x6" and os.path.islink(p) and not os.path.isabs(os.readlink(p))):
            continue
        if os.path.lexists(p):
            os.unlink(p)
        os.symlink(src, p)
        out.append(f"{fam}_s{s} -> {src}")
os.makedirs("manifest", exist_ok=True)
open("manifest/aliases.txt", "w").write("\n".join(out) + "\n")
print(f"aliased {len(out)} seed files (manifest/aliases.txt)")
