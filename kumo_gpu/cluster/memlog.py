#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Memory logger + guard for every Kumo GPU job. Writes one line per MEMLOG_SEC (60) seconds to MEMLOG (a file):
ts | unit | torch.cuda allocated / max_allocated / reserved GiB | cuda mem_get_info free/total GiB | host MemAvailable/MemTotal GiB.
On the GPU workstation the "GPU" memory IS the host memory (120 GB unified), so the host MemAvailable line is the one that matters for the crash question
. MEMGUARD_GIB (default 10): if host MemAvailable drops below it the process prints MEMGUARD and exits with rc 3
(the engine resumes from its part files, so an abort costs minutes; a hung box costs a day). Set MEMGUARD_GIB=0 to disable.
"""

import os, time, threading

STATE = {"unit": "-"}


def _host():
    mi = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        mi[k] = int(v.split()[0]) / 2**20
    return mi.get("MemAvailable", float("nan")), mi.get("MemTotal", float("nan"))


def line():
    import torch

    av, tot = _host()
    try:
        fr, ct = torch.cuda.mem_get_info()
        fr /= 2**30
        ct /= 2**30
    except Exception:
        fr = ct = float("nan")
    return (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {STATE['unit']} | cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} max {torch.cuda.max_memory_allocated()/2**30:.1f} "
        f"reserved {torch.cuda.memory_reserved()/2**30:.1f} GiB | mem_get_info free {fr:.1f}/{ct:.1f} | host avail {av:.1f}/{tot:.1f} GiB"
    ), av


_started = {"t": None}


def start(path=None, every=None, guard=None):
    if _started["t"] is not None:
        return _started["t"]  # idempotent: km_engine and km_task both call start()
    path = path or os.environ.get("MEMLOG")
    every = float(every or os.environ.get("MEMLOG_SEC", "60"))
    guard = float(os.environ.get("MEMGUARD_GIB", "10") if guard is None else guard)
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def loop():
        while True:
            try:
                s, av = line()
                with open(path, "a") as f:
                    f.write(s + "\n")
                if guard > 0 and av < guard:
                    with open(path, "a") as f:
                        f.write(f"MEMGUARD host MemAvailable {av:.1f} GiB < {guard} GiB -> exit 3 (resumable)\n")
                    print(f"MEMGUARD host MemAvailable {av:.1f} GiB < {guard} GiB -> exit 3", flush=True)
                    os._exit(3)
            except Exception as ex:
                with open(path, "a") as f:
                    f.write(f"memlog error {ex}\n")
            time.sleep(every)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    _started["t"] = t
    return t
