#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Kumo acceptance: build $RUN/acceptance/SUMMARY.md (relayable) from the step outputs. Pure stdlib (works with any python3)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, glob, re, time

RUN = sys.argv[1]
A = f"{RUN}/acceptance"
rd = lambda p: open(p).read() if os.path.exists(p) else ""
REF_SPEEDS = {
    ("medium", 20000, 412): (0.80, 6.2),
    ("medium", 20000, 100): (0.23, 2.8),
    ("medium", 50000, 100): (0.24, 4.4),
    ("medium", 50000, 412): (0.85, 12.3),
    ("large", 20000, 100): (0.26, 3.4),
    ("large", 20000, 412): (0.84, 6.8),
    ("large", 50000, 100): (0.30, 4.9),
    ("large", 50000, 412): (None, None),
}
L = [
    f"# KUMO_ENGINE cluster acceptance — {rd(f'{A}/00_header.txt').strip() or RUN}",
    f"_summary written {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_",
    "",
]
V = rd(f"{A}/verdicts.txt").strip().splitlines()
L += ["## Verdicts"] + [f"- {v}" for v in V] + [""]
steps = {int(m.group(1)): m.group(2) for v in V for m in [re.match(r"STEP (\d) (PASS|FAIL)", v)] if m}
L.append(
    f"**Overall: {'PASS — arm M stage 2 fits are on disk (results/oof)' if all(steps.get(i) == 'PASS' for i in range(1, 6)) else 'INCOMPLETE/FAIL — see verdicts'}**\n"
)
e = rd(f"{A}/01_envcheck.txt").strip()
if e:
    d = json.loads(e[9:])
    L += [
        "## 1. Environment",
        f"- host {d.get('host')} | torch {d.get('torch')} cuda {d.get('cuda')} | sdm {d.get('sdm')} {d.get('sdm_commit')}",
        f"- GPUs ({d.get('ngpu')}): {d.get('gpus')} mem GiB {d.get('mem_gib')} cap {d.get('cap')}",
        f"- fp16 matmul {d.get('fp16_matmul')} | SDPA backends {d.get('sdpa')}",
        f"- weights medium {d.get('weights_medium')} large {d.get('weights_large')} | payload frames {d.get('payload_frames')}/10 parity ref {d.get('parity_ref')}",
        f"- fail list: {d.get('fail')}",
        "",
    ]
p = rd(f"{A}/02_parity16.txt").strip()
if p:
    L += [
        "## 2. PARITY16 (fp16 vs the fp32 GPU workstation file, janjul x LSZH; build machine: RMS 0.39 s, corr 0.999999, 148 rows/s)",
        f"- {p}",
        "",
    ]
b = [json.loads(l[6:]) for l in rd(f"{A}/03_bench.txt").splitlines() if l.startswith("BENCH ")]
if b:
    L += [
        "## 3. Bench grid (one instance, fp16, QMAX query rows; GPU workstation fp16 pilot in brackets)",
        "| size | ctx | cols | fit s | predict ms/row | peak GiB | rmse clean (info) | vs GPU workstation ms/row / GiB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in b:
        g = REF_SPEEDS.get((d["size"], d["cap"], d["ncol"]), (None, None))
        sp = (
            f"{g[0]/d['ms_per_row']:.1f}x faster"
            if g[0] and d["ms_per_row"]
            else "n/a (never completed on the build machine)"
        )
        L.append(
            f"| {d['size']} | {d['cap']//1000}k | {d['ncol']} | {d['fit_s']} | {d['ms_per_row']} | {d['peak_gib']} | {d['rmse_clean']} | [{g[0]} / {g[1]}] {sp} |"
        )
    L.append("")
s = rd(f"{A}/04_scaling.txt").strip()
r = rd(f"{A}/04_resume.txt").strip()
if s or r:
    L += ["## 4. Scaling (4 units, one per GPU) and resume", f"- {s}", *[f"- {x}" for x in r.splitlines()], ""]
pc = rd(f"{A}/05_precollect.txt").strip()
if pc:
    d = json.loads(pc[pc.index("PRECOLLECT ") + 11 :].splitlines()[0])
    L += [
        "## 5. Arm M stage 2 (months39 + months410, shared20k construction in fp16) + precollect",
        f"- {rd(f'{A}/05_armM_wall.txt').strip()}",
    ]
    for f, v in d["folds"].items():
        L.append(f"- {f}: {v}")
    L += [
        f"- verdict {d['verdict']} {d['fail']}",
        f"- files: {sorted(os.path.basename(x) for x in glob.glob(f'{RUN}/results/oof/*.npy'))} ; MD5SUMS: results/MD5SUMS",
        "",
    ]
units = glob.glob(f"{RUN}/results/member/*.npz")
if units:
    secs = 0
    q = 0
    for f in units:
        try:
            import numpy as np

            m = json.loads(str(np.load(f)["meta"]))
            secs += m["secs"]
            q += m["query"]
        except Exception:
            pass
    L += [
        "## GPU time actually used",
        f"- member units {len(units)}: {secs/3600:.2f} GPU-h for {q:,} query rows ({q/max(secs,1):.0f} rows/s incl. context fits)",
        "",
    ]
L += [
    "## Bring back",
    "copy `$RUN/results` to the build machine and collect there with `kumo_gpu/collect/collect.py` (docs/kumo_gpu.md).",
    "",
]
print("\n".join(L))
