#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""trace_deps.py: static dependency trace of the final submission's build, plus an md5 manifest of every data
intermediate it names.

Starting from the build roots (the scripts that produced each file of the final build's lineage), the tool
  1. reads each script, resolves simple shell assignments (VAR=value, also several per line, and ${VAR:-default}),
  2. collects every token that looks like a project path under data/, scratchpad/, submissions/, artifacts/, phase2/work/ (data) or
     scripts/, jobs/ (code, followed recursively),
  3. resolves data tokens against PRC_ROOT: an exact path if it exists, otherwise a glob (shell $VAR / python {expr} / brace lists
     become '*'), and records every match.
It over-approximates on purpose (a glob over a family prefix lists every seed and fold of that family). Each row says how it was found
(exact / glob) and from which script. Raw open-data archives (adsb.lol traces, OPDI chunks) are listed by name and size only
(content md5 would mean hashing ~100 GB); everything else gets a content md5.

usage: PRC_ROOT=<project root> trace_deps.py <out.tsv> <root script> [<root script> ...] [--extra <file with extra paths>]
"""

import sys

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)

import glob, hashlib, os, re, sys

ROOT = os.environ.get("PRC_ROOT", os.getcwd())
DATA_PREFIX = r"(?:data|scratchpad|submissions|artifacts|phase2/work)/"
CODE_PREFIX = r"(?:scripts|jobs)/"
TOK = re.compile(r"(?<![\w/.-])((?:" + DATA_PREFIX + "|" + CODE_PREFIX + r")[^\s\"'`;|&<>()\[\],=]+)")
ASSIGN = re.compile(r"(?:^|[\s;])([A-Za-z_]\w*)=([^\s;]+)")
NAMEONLY = (
    "data/external/adsblol",
    "data/external/opdi",
    "data/external/adsblol_mlat",
    "data/external/adsblol_xgap",
    "data/raw/adsb",
)
MAXGLOB = 400


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def expand(tok, env):
    for _ in range(4):
        tok2 = re.sub(
            r"\$\{(\w+)(?::-([^}]*))?\}",
            lambda m: env.get(m.group(1), m.group(2) if m.group(2) is not None else "$" + m.group(1)),
            tok,
        )
        tok2 = re.sub(r"\$(\w+)", lambda m: env.get(m.group(1), "$" + m.group(1)), tok2)
        if tok2 == tok:
            break
        tok = tok2
    return tok


def to_glob(tok):
    g = re.sub(r"\$\{?\w+\}?", "*", tok)
    g = re.sub(r"\{[^}]*\}", "*", g)
    g = re.sub(r"%[\d.]*[sdf]", "*", g)
    g = g.rstrip(".:")
    return re.sub(r"\*+", "*", g)


def scan(path, seen_code, rows, queue):
    try:
        txt = open(os.path.join(ROOT, path), errors="replace").read()
    except OSError:
        return
    env = {}
    for line in txt.split("\n"):
        for m in ASSIGN.finditer(line):
            env[m.group(1)] = expand(m.group(2).strip("\"'"), env)
        for m in TOK.finditer(line):
            tok = expand(m.group(1), env).rstrip(".:,")
            for t in re.split(r"(?<=\.parquet)|(?<=\.npz)|(?<=\.npy)", tok):
                t = t.lstrip(",:")
                if not t:
                    continue
                if re.match(CODE_PREFIX, t) or t.endswith((".py", ".sh")):
                    if (
                        t.endswith((".py", ".sh"))
                        and "*" not in t
                        and "$" not in t
                        and "{" not in t
                        and t not in seen_code
                    ):
                        seen_code.add(t)
                        queue.append(t)
                    continue
                rows.setdefault(t, set()).add(path)


def main():
    a = sys.argv[1:]
    extra = []
    if "--extra" in a:
        i = a.index("--extra")
        extra = [l.strip() for l in open(a[i + 1]) if l.strip() and not l.startswith("#")]
        a = a[:i] + a[i + 2 :]
    out, roots = a[0], a[1:]
    seen_code, rows, queue = set(roots), {}, list(roots)
    while queue:
        scan(queue.pop(), seen_code, rows, queue)
    for e in extra:
        rows.setdefault(e, set()).add("lineage")
    res = {}
    for tok, srcs in rows.items():
        p = os.path.join(ROOT, tok)
        if os.path.isfile(p):
            res.setdefault(tok, ["exact", srcs])
            continue
        if os.path.isdir(p):
            g = tok.rstrip("/") + "/*"
        else:
            g = to_glob(tok)
            if g.count("*") and len(g.split("*")[0].rsplit("/", 1)[-1]) < 2 and g.split("*")[0].endswith("/"):
                pass  # bare directory globs are kept, they are bounded by MAXGLOB below
        hits = sorted(h for h in glob.glob(os.path.join(ROOT, g)) if os.path.isfile(h))
        if len(hits) > MAXGLOB:
            res.setdefault(tok, ["glob-too-wide(%d)" % len(hits), srcs])
            continue
        for h in hits:
            r = os.path.relpath(h, ROOT)
            if r not in res:
                res[r] = ["glob:" + g, srcs]
    with open(out, "w") as f:
        f.write("# path\tmd5\tbytes\tfound_by\treferenced_by\n")
        for r in sorted(res):
            how, srcs = res[r]
            p = os.path.join(ROOT, r)
            if not os.path.isfile(p):
                f.write(f"{r}\t-\t-\t{how}\t{','.join(sorted(srcs))[:300]}\n")
                continue
            size = os.path.getsize(p)
            h = "name-only" if r.startswith(NAMEONLY) else md5(p)
            f.write(f"{r}\t{h}\t{size}\t{how}\t{','.join(sorted(srcs))[:300]}\n")
    with open(out + ".code.txt", "w") as f:
        f.write("\n".join(sorted(seen_code)) + "\n")
    print(f"{len(res)} data paths, {len(seen_code)} code files -> {out}")


if __name__ == "__main__":
    main()
