#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_CLUSTER_SETUP: progress per member profile NAME (and t2x) = units with a result file / units in the lists; measured rows/s and peak GPU memory.
B=$(cd "$(dirname "$0")/.." && pwd); R0=${RUN:-}; source "$B/jobs/config.env" 2>/dev/null || true; RUN=${R0:-${RUN:-$B/run}}   # an exported RUN (run_box.sh) wins
PYX=$B/.venv-kumo/bin/python; [ -x "${ENVDIR:-/nonexistent}/bin/python" ] && PYX=$ENVDIR/bin/python
"$PYX" - "$RUN" <<'PY'
import sys, os, glob, json, numpy as np, collections
R = sys.argv[1]; L = {n: open(f"{R}/lists/{n}.txt").read().split() for n in ("member", "t2x") if os.path.exists(f"{R}/lists/{n}.txt")}
names = sorted({os.path.basename(f).rsplit("_", 2)[0] for f in glob.glob(f"{R}/results/member/*.npz")}) or ["(none yet)"]
for n in names:
    fs = [f"{R}/results/member/{n}_{u.split(':')[1]}_{u.split(':')[2]}.npz" for u in L.get("member", [])]; d = [f for f in fs if os.path.exists(f)]
    M = [json.loads(str(np.load(f)["meta"])) for f in d]
    rate = np.median([(m["query"] + m.get("synth", 0)) / m["secs"] for m in M]) if M else float("nan"); peak = max([m.get("peak_gib", 0) for m in M], default=0)
    print(f"member {n}: {len(d)}/{len(fs)} units | median {rate:.0f} rows/s | max peak {peak:.1f} GiB | gpus {sorted({m['gpu'] for m in M})}")
if "t2x" in L:
    d = [u for u in L["t2x"] if os.path.exists(f"{R}/results/t2x/gK_{'_'.join(u.split(':')[1:])}.npz")]; print(f"t2x: {len(d)}/{len(L['t2x'])} units")
PY
