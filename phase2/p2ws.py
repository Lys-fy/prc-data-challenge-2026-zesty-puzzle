#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Creates a phase-2 workspace for one period: a self-contained project root under phase2/work/<period>/ with a
frozen copy of scripts/, symlink farms over the shared project data, and empty submissions/stages/scratchpad dirs.
Inputs: <period> (positional), optional --features/--raw-rank parquet overrides, --force (rebuild an existing
workspace's link farms); env P2_REPO/PRC_ROOT, P2_SCRIPTS_SRC. Output: the workspace directory
tree plus phase2/work/<period>/manifest/{scripts.md5,workspace.json}.

phase2_ws: create a phase-2 workspace for one period.

A workspace is a self-contained project root (phase2/work/<period>/) in which the ORIGINAL pipeline scripts run unchanged:
  scripts/        a frozen, byte-identical COPY of the project's scripts/ (md5 manifest in manifest/scripts.md5). The scripts read
                  their root from $PRC_ROOT (the drivers set it to the workspace).
  scratchpad/     empty per-stage output dirs + copies of the stage code that runs from scratchpad/xur (unm_resid).
  data/raw, data/external, data/oof, data/preds
                  symlink farms over the shared project data (read side). For a NEW period the period-dependent files
                  (ranking.parquet, features, per-row side tables, member predictions, q) are regenerated inside the
                  workspace; the copy-on-write guard (lib/guard/sitecustomize.py) replaces the link by a local file.
  submissions/    starts EMPTY; the line outputs of the workspace.
  stages/         starts EMPTY; the stage files the stage scripts read by fixed names (unmatched_rule, fallback_base, unmatched_resid,
                  unmatched_link_a1), built BY RULE by the pipeline (phase2/steps/p2_slot.py); nothing is copied from outside the workspace.
  .venv           symlink to the project venv.
usage: p2ws.py <period> [--features <parquet>] [--raw-rank <parquet>] [--force]"""

import os, sys, shutil, hashlib, argparse, json, time

REPO = (
    os.environ.get("P2_REPO")
    or os.environ.get("PRC_ROOT")
    or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
P2 = f"{REPO}/phase2"  # repo root = parent of phase2/ (or $P2_REPO)
SCRIPTS_SRC = os.environ.get("P2_SCRIPTS_SRC", f"{REPO}/scripts")  # the scripts copied into the workspace
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("period", help="period name; the workspace is created at phase2/work/<period>/")
ap.add_argument("--features", default="", help="features_v9 parquet to symlink in as data/features_v9.parquet")
ap.add_argument("--raw-rank", default="", help="ranking parquet to symlink in as data/raw/ranking.parquet")
ap.add_argument(
    "--force", action="store_true", help="rebuild the link farms of an existing workspace; outputs are kept"
)
a = ap.parse_args()
WS = f"{P2}/work/{a.period}"
if os.path.exists(WS):
    if not a.force:
        sys.exit(f"{WS} exists (use --force to rebuild the link farms; outputs are kept)")
os.makedirs(WS, exist_ok=True)
os.makedirs(f"{WS}/manifest", exist_ok=True)


def farm(src, dst, skip=()):
    """per-file symlinks src/* -> dst/* (recursing into sub-directories), never overwriting a local (non-link) file"""
    os.makedirs(dst, exist_ok=True)
    n = 0
    for e in os.scandir(src):
        if e.name in skip:
            continue
        t = f"{dst}/{e.name}"
        if e.is_dir(follow_symlinks=False):
            n += farm(e.path, t)
            continue
        if os.path.lexists(t):
            if os.path.islink(t):
                os.unlink(t)
            else:
                continue  # regenerated locally: keep
        os.symlink(os.path.realpath(e.path), t)
        n += 1
    return n


# ---- scripts: frozen copy
S = f"{WS}/scripts"
if os.path.exists(S):
    shutil.rmtree(S)
shutil.copytree(SCRIPTS_SRC, S, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), symlinks=False)
md5 = []
for root, _, files in os.walk(S):
    for f in sorted(files):
        p = f"{root}/{f}"
        md5.append(f"{hashlib.md5(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, WS)}")
open(f"{WS}/manifest/scripts.md5", "w").write("\n".join(md5) + "\n")
# stage code that runs from scratchpad/xur (unmatched-row residual model scripts)
os.makedirs(f"{WS}/scratchpad/xur", exist_ok=True)
for f in sorted(os.listdir(f"{REPO}/scripts")):
    if f.startswith("unm_resid_") and f.endswith(".py"):
        shutil.copyfile(f"{REPO}/scripts/{f}", f"{WS}/scratchpad/xur/{f}")
for d in (
    "xof",
    "xpr",
    "xpr/val",
    "xsx",
    "xsx/val",
    "xhr",
    "xfz",
    "p12",
    "xur",
    "x4",
    "xu",
    "xcr",
    "xz",
    "xlv",
    "xl",
    "p13",
    "xau1",
    "xau2",
    "xau3",
    "xur2",
    "xse2",
    "xpb",
    "p2",
):
    os.makedirs(f"{WS}/scratchpad/{d}", exist_ok=True)
# ---- data: link farms
n = {}
n["raw"] = farm(f"{REPO}/data/raw", f"{WS}/data/raw")
n["external"] = farm(f"{REPO}/data/external", f"{WS}/data/external")
n["oof"] = farm(f"{REPO}/data/oof", f"{WS}/data/oof")
n["preds"] = farm(f"{REPO}/data/preds", f"{WS}/data/preds")
fv = f"{WS}/data/features_v9.parquet"
if a.features:
    if os.path.lexists(fv):
        os.unlink(fv)
    os.symlink(os.path.abspath(a.features), fv)
elif not os.path.lexists(fv):
    os.symlink(os.path.realpath(f"{REPO}/data/features_v9.parquet"), fv)
if a.raw_rank:
    t = f"{WS}/data/raw/ranking.parquet"
    if os.path.lexists(t):
        os.unlink(t)
    os.symlink(os.path.abspath(a.raw_rank), t)
for d in ("submissions", "stages", "experiments", "logs"):
    os.makedirs(f"{WS}/{d}", exist_ok=True)
if not os.path.lexists(f"{WS}/.venv"):
    os.symlink(os.path.realpath(f"{REPO}/.venv"), f"{WS}/.venv")
for f in ("pyproject.toml", "uv.lock"):
    if os.path.exists(f"{REPO}/{f}") and not os.path.lexists(f"{WS}/{f}"):
        shutil.copy(f"{REPO}/{f}", f"{WS}/{f}")
json.dump(
    {
        "period": a.period,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "links": n,
        "features": os.path.realpath(fv),
        "ranking": os.path.realpath(f"{WS}/data/raw/ranking.parquet"),
    },
    open(f"{WS}/manifest/workspace.json", "w"),
    indent=1,
)
print(f"workspace {WS}: scripts copied ({len(md5)} files), links {n}")
