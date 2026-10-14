#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Workspace preparation for FORM=v99 (run_pipeline.sh step v99prep; cwd = workspace, outside the guard).
1. Stage scripts: installs the 17 scripts of the stages that FORM=v99 adds to the v88 line (list LEVER) into the workspace's scripts/, from
   $P2_LEVER_SRC (default: the repository's scripts/). MODE=repro keeps any workspace copy that exists and installs only the missing ones;
   MODE=new installs all 17, so the workspace runs the versions whose base files are set by env. The copies are byte-identical to the
   source. List and md5 in manifest/v99_scripts.txt.
2. Links (never replacing a local file): data/sealed_days.parquet (the gate's sealed-day map, read by the Kumo scripts' gate code only),
   artifacts/xkumo (Kumo weights, HF_HOME), the 2025 context tables of F4 (scratchpad/xhr/unm2025b.parquet, unm_lirf_janjul_frozen.parquet:
   2025 rows only), and the data/oof + data/external files created after the workspace (2025 fold OOF such as wideleaf_retune_a2_<fold>, pseudolabel_pl3_<fold>).
   MODE=repro also links data/preds files created after the workspace (wideleaf_a2p_s*, pseudolabel_pl3p88_s*: fitted members of the public period).
3. FOLD_LINK=1 (default): links the fold side of the WIDELEAF / PSEUDOLABEL chains (scratchpad/xbase/P13_XB<C>x*, scratchpad/xsyn/P13_XSPL3<C>x*,
   scratchpad/xau1/clockfold1_xb<C>_*, clockfold1_xspl3<C>_*). The fold side is fitted on 2025 OOF only and does not depend on the period (phase2/README.md §2);
   FOLD_LINK=0 leaves them to the steps xb_fold_<C> / xs_fold_<C>, which rebuild them in the workspace.
usage: p2_v99_prep.py <workspace> <repo> <mode>"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, hashlib

WS, REPO, MODE = sys.argv[1:4]
P2 = f"{REPO}/phase2"
SRC = os.environ.get("P2_LEVER_SRC") or f"{REPO}/scripts"
LEVER = [
    "kumo_icl_unmatched.py",
    "kumo_icl_build.py",
    "kumo_icl_lirf.py",
    "rulebag_blend.py",
    "wideleaf_stk.sh",
    "wideleaf_stack.py",
    "wideleaf_assemble.py",
    "wideleaf_bundle_prod.sh",
    "wideleaf_member.sh",
    "train2_wideleaf.py",
    "pseudo_stk.sh",
    "pseudo_bundle_prod.sh",
    "pseudo_labels.py",
    "pseudo_member.sh",
    "pseudo_prod.sh",
    "train2_pseudo.py",
    "validate_submission.py",
]
log = []
for f in LEVER:
    s, d = f"{SRC}/{f}", f"{WS}/scripts/{f}"
    if os.path.exists(d) and MODE != "new":
        log.append(f"kept   {hashlib.md5(open(d, 'rb').read()).hexdigest()[:8]}  scripts/{f} (workspace copy)")
        continue
    if not os.path.exists(s):
        log.append(f"MISSING scripts/{f} (not in {SRC})")
        continue
    t = open(s).read()
    open(d, "w").write(t)
    os.chmod(d, 0o755)
    log.append(
        f"copied {hashlib.md5(open(d, 'rb').read()).hexdigest()[:8]}  scripts/{f} (from {SRC}, md5 src {hashlib.md5(open(s, 'rb').read()).hexdigest()[:8]})"
    )


def link(src, dst):
    if not os.path.exists(src) or os.path.lexists(dst):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.realpath(src), dst)
    return 1


n = {}
n["sealed"] = link(f"{REPO}/data/sealed_days.parquet", f"{WS}/data/sealed_days.parquet")
os.makedirs(f"{WS}/artifacts", exist_ok=True)
n["kumo"] = link(f"{REPO}/artifacts/xkumo", f"{WS}/artifacts/xkumo")
for f in ("unm2025b.parquet", "unm_lirf_janjul_frozen.parquet"):
    n[f] = link(f"{REPO}/scratchpad/xhr/{f}", f"{WS}/scratchpad/xhr/{f}")
for sub in ["oof", "external"] + (["preds"] if MODE == "repro" else []):
    k = 0
    for p in glob.glob(f"{REPO}/data/{sub}/*"):
        if os.path.isfile(p):
            k += link(p, f"{WS}/data/{sub}/{os.path.basename(p)}")
    n[f"data/{sub} new links"] = k
if os.environ.get("FOLD_LINK", "1") == "1":
    k = 0
    for C in ("Kq", "NNA3"):
        for g in (
            f"scratchpad/xbase/P13_XB{C}x*",
            f"scratchpad/xau1/clockfold1_xb{C}_*",
            f"scratchpad/xsyn/P13_XSPL3{C}x*",
            f"scratchpad/xau1/clockfold1_xspl3{C}_*",
        ):
            for p in glob.glob(f"{REPO}/{g}"):
                k += link(p, f"{WS}/{os.path.relpath(p, REPO)}")
    n["fold side XB/XS links"] = k
for d in ("xbase", "xsyn", "xtfm2", "xtfm2/frz", "xhr", "xhr2", "p2"):
    os.makedirs(f"{WS}/scratchpad/{d}", exist_ok=True)
os.makedirs(f"{WS}/manifest", exist_ok=True)
open(f"{WS}/manifest/v99_scripts.txt", "w").write("\n".join(log + [f"link {k}: {v}" for k, v in n.items()]) + "\n")
print("\n".join(log))
print("links", n)
