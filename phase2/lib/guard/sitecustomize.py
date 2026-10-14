# SPDX-License-Identifier: GPL-3.0-only
"""phase2_ws phase-2 workspace guard. Loaded automatically (PYTHONPATH) by every Python process that the phase-2 pipeline
starts inside a workspace. Two jobs:

1. Copy-on-write, fail-closed writes. A workspace is a symlink farm over the shared project data. Any write whose target lies
   inside the workspace but is a symlink pointing outside is turned into a fresh local file (the link is removed first, so the
   shared file is never touched). Any write outside the workspace (and outside /tmp, /dev) raises PermissionError.
   Covered: Python open()/os.open (numpy, zipfile, json, pickle, shutil), os.remove/rename/replace, and the polars / pandas
   parquet/csv writers (Rust / C++ I/O that bypasses open()).
2. Input ledger. Every file opened for reading (open(), numpy, polars/pandas readers) is appended to $P2_LOG (one JSON line
   per process at exit: reads, writes, argv) so the run records exactly which shared artefacts it consumed.

Env: P2_ROOT (workspace root, absolute; required to activate), P2_LOG (ledger file), P2_ALLOW (extra ':'-separated write prefixes).
"""

import os, sys, json, atexit

_ROOT = os.environ.get("P2_ROOT")
if _ROOT:
    _ROOT = os.path.abspath(_ROOT)
    _ALLOW = ["/tmp", "/dev", "/proc", os.path.expanduser("~/.cache"), os.path.expanduser("~/.config/matplotlib")] + [
        p for p in os.environ.get("P2_ALLOW", "").split(":") if p
    ]
    _READS, _WRITES = set(), set()

    def _inside(p, root=_ROOT):
        return p == root or p.startswith(root + os.sep)

    def _allowed_outside(p):
        return any(p == a or p.startswith(a.rstrip(os.sep) + os.sep) for a in _ALLOW)

    def _check_write(path, truncate=True):
        """Validate a write target; turn an outward symlink inside the workspace into a local file (copy-on-write)."""
        try:
            p = os.path.abspath(os.fsdecode(path))
        except Exception:
            return
        if _inside(p):
            parent = os.path.realpath(os.path.dirname(p))
            if not _inside(parent) and not _allowed_outside(parent):
                raise PermissionError(f"P2 guard: parent directory of {p} resolves outside the workspace ({parent})")
            if os.path.islink(p) and not _inside(os.path.realpath(p)):
                if not truncate:
                    raise PermissionError(
                        f"P2 guard: in-place update of a shared file through {p} (append / r+) is not allowed"
                    )
                os.unlink(p)  # copy-on-write: the new file lives in the workspace, the shared target is untouched
            _WRITES.add(p)
            return
        if _allowed_outside(os.path.realpath(p)):
            return
        raise PermissionError(f"P2 guard: write outside the workspace: {p}")

    _OUTSIDE = set()

    def _log_read(path):
        try:
            p = os.fsdecode(path)
            ab = os.path.abspath(p)
            _READS.add(os.path.realpath(p))
            if (
                not _inside(ab)
                and not ab.startswith(("/usr", "/proc", "/dev", "/sys", "/etc", "/tmp"))
                and "site-packages" not in ab
                and "/lib/python" not in ab
                and not ab.endswith((".py", ".pyc", ".so", ".pth"))
            ):
                _OUTSIDE.add(ab)  # opened by a path that does not go through the workspace (hard-coded root)
                if os.environ.get("P2_STRICT") == "1" and ab.startswith(
                    os.environ.get(
                        "P2_REPO",
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    )
                    + os.sep
                ):
                    raise PermissionError(f"P2 guard (strict): read of a project file outside the workspace: {ab}")
        except PermissionError:
            raise
        except Exception:
            pass

    def _hook(event, args):
        if event == "open":
            path, mode, flags = args
            if path is None or isinstance(path, int):
                return
            if mode is not None:
                w = any(c in mode for c in "wax+")
                trunc = ("w" in mode or "x" in mode) and "+" not in mode
            else:
                w = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                trunc = bool(flags & os.O_TRUNC) or bool(flags & os.O_EXCL)
                if w and not trunc and bool(flags & os.O_CREAT) and not os.path.exists(os.fsdecode(path)):
                    trunc = True
            if w:
                _check_write(path, trunc)
            else:
                _log_read(path)
        elif event in ("os.remove", "os.rmdir"):
            p = os.path.abspath(os.fsdecode(args[0]))
            if not (_inside(p) or _allowed_outside(p)):
                raise PermissionError(f"P2 guard: delete outside the workspace: {p}")
        elif event in ("os.rename", "os.replace", "shutil.move", "shutil.copyfile", "shutil.copytree"):
            dst = os.path.abspath(os.fsdecode(args[1]))
            if event in ("os.rename", "os.replace", "shutil.move"):
                src = os.path.abspath(os.fsdecode(args[0]))
                if not (_inside(src) or _allowed_outside(src)):
                    raise PermissionError(f"P2 guard: move of a file outside the workspace: {src}")
            _check_write(dst, True)

    sys.addaudithook(_hook)

    # ---- wrap writers/readers of polars and pandas once they are imported (their I/O bypasses open()) ----
    def _wrap_w(obj, name):
        f = getattr(obj, name, None)
        if f is None or getattr(f, "_p2", False):
            return

        def g(self, file=None, *a, **k):
            if isinstance(file, (str, bytes, os.PathLike)):
                _check_write(file, True)
            return f(self, file, *a, **k)

        g._p2 = True
        setattr(obj, name, g)

    def _wrap_r(mod, name):
        f = getattr(mod, name, None)
        if f is None or getattr(f, "_p2", False):
            return

        def g(source=None, *a, **k):
            if isinstance(source, (str, bytes, os.PathLike)):
                s = os.fsdecode(source)
                if any(ch in s for ch in "*?["):
                    import glob as _g

                    for x in _g.glob(s):
                        _log_read(x)
                else:
                    _log_read(s)
            return f(source, *a, **k)

        g._p2 = True
        setattr(mod, name, g)

    def _patch(name, mod):
        if name == "polars":
            for w in ("write_parquet", "write_csv", "write_ipc", "write_json", "write_ndjson"):
                _wrap_w(mod.DataFrame, w)
            _wrap_w(mod.LazyFrame, "sink_parquet")
            for r in ("read_parquet", "scan_parquet", "read_csv", "scan_csv", "read_ipc", "read_parquet_schema"):
                _wrap_r(mod, r)
        elif name == "pandas":
            for w in ("to_parquet", "to_csv", "to_pickle"):
                _wrap_w(mod.DataFrame, w)
            for r in ("read_parquet", "read_csv", "read_pickle"):
                _wrap_r(mod, r)

    import importlib.abc, importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in ("polars", "pandas"):
                return None
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = f.find_spec(fullname, path, target) if hasattr(f, "find_spec") else None
                if spec is not None and spec.loader is not None:
                    orig = spec.loader.exec_module

                    def exec_module(module, _o=orig, _n=fullname):
                        _o(module)
                        _patch(_n, module)

                    spec.loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())

    def _dump():
        log = os.environ.get("P2_LOG")
        if not log:
            return
        try:
            rec = {
                "pid": os.getpid(),
                "argv": sys.argv,
                "cwd": os.getcwd(),
                "reads": sorted(
                    r
                    for r in _READS
                    if not r.startswith(("/usr", "/proc", "/dev", "/sys", "/etc"))
                    and "site-packages" not in r
                    and "/lib/python" not in r
                    and not r.endswith((".py", ".pyc", ".so"))
                ),
                "writes": sorted(_WRITES),
                "outside_reads": sorted(_OUTSIDE),
            }
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    atexit.register(_dump)
