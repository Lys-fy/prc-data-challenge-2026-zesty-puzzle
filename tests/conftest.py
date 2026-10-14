# SPDX-License-Identifier: GPL-3.0-only
"""Test setup: the production scripts are imported from scripts/ (their CLI code runs only under __main__)."""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def extract(path, names):
    """Return the source of every assignment (at any depth) whose target is one of `names`, and of top-level functions
    called one of `names`, in file order. Used for statements that live inside scripts that run on import."""
    src = open(os.path.join(ROOT, path)).read()
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            out.append((n.lineno, ast.get_source_segment(src, n)))
        if isinstance(n, ast.FunctionDef) and n.name in names:
            out.append((n.lineno, ast.get_source_segment(src, n)))
    return "\n".join(s for _, s in sorted(out))
