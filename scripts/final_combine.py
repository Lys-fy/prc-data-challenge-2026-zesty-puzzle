#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""final_combine.py (): compose pre-built submission files, with no board-derived parameters.

Mode `add` (isolate composition, isolate composition):
    out = BASE + sum_i sign_i * (ISO_i - ON_i)
Each component is `FILE@ONBASE`. Its delta is FILE - ONBASE, i.e. the isolate file minus the file it was built on.
`-c FILE@ON` adds a delta, `-r FILE@ON` removes one (a rollback). Components are applied in command-line order.
Example (one delta added, one removed):
    final_combine.py add --base line_b.parquet -r iso_x.parquet@line_a.parquet -c iso_y.parquet@line_a.parquet --out line_c.parquet --allow-stale

Mode `avg` ( P2, top-k equal weights): out = rint(mean of the k files). Weights are always equal; the tool has no weight option.

Row-level checks (both modes):
  * id set and order: every file must carry the same ids as the base (reordered by id if needed; any missing or extra id aborts).
  * OVERLAP conflict: a row touched by more than one delta. Aborts unless --allow-overlap. With --allow-overlap the deltas are
    summed and the rows are listed in the manifest.
  * STALE conflict: on the rows a delta touches, the base is not where the delta expects it: for an added delta BASE != ONBASE
    (something between ONBASE and BASE already moved those rows); for a removed delta BASE != FILE (the base no longer carries
    the isolate's values there). Either way the delta mixes two row histories. Aborts unless --allow-stale; the manifest records
    the count either way.
  * protected rows (--protect ids.parquet with MVT_ID_mvt, or --protect-from FILE: rows where FILE equals BASE): any change there
    aborts unless --allow-protected.
  * value checks: finite, no nulls, min >= 60 s (the submission floor); optional full validator via --validate-as N (runs
    scripts/validate_submission.py on a temp copy, incl. the LOBT law and the id set).
Output: parquet (same schema/dtype as the base, rint, base dtype) + <out>.manifest.json with every input's
md5, its role/sign, rows changed per delta, conflict counts, the output md5, the script md5, host and time.
"""
import argparse, hashlib, json, os, socket, subprocess, sys, tempfile, time, shutil
import numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR = 60


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p, ref_ids=None, key=None, col=None):
    d = pl.read_parquet(p)
    key = key or d.columns[0]; col = col or d.columns[1]
    assert d.columns[:2] == [key, col], f"{p}: columns {d.columns} != [{key}, {col}]"
    assert d[col].null_count() == 0, f"{p}: nulls in {col}"
    assert d[key].n_unique() == len(d), f"{p}: duplicate ids"
    if ref_ids is not None:
        assert len(d) == len(ref_ids), f"{p}: {len(d)} rows vs base {len(ref_ids)}"
        if not (d[key] == ref_ids).all():
            d = pl.DataFrame({key: ref_ids}).join(d, on=key, how="left")
            assert d[col].null_count() == 0, f"{p}: id set differs from the base"
            print(f"  note: {os.path.basename(p)} reordered to the base id order")
    return d, key, col


def vec(d, col):
    v = d[col].to_numpy().astype(np.float64)
    assert np.isfinite(v).all(), "non-finite values"
    return v


class Comp(argparse.Action):
    def __call__(self, parser, ns, value, opt=None):
        ns.comp.append(("-" if self.dest == "rm" else "+") + value)


def parse_comp(s):
    sign = -1 if s.startswith("-") else 1
    s = s[1:] if s[:1] in "+-" else s
    assert "@" in s, f"component '{s}' must be FILE@ONBASE"
    f, on = s.split("@", 1)
    return sign, f, on


def protected_mask(args, key, ids, base_v):
    m = np.zeros(len(ids), bool); src = []
    if args.protect:
        p = pl.read_parquet(args.protect)[key]
        m |= ids.is_in(p).to_numpy(); src.append({"ids_file": args.protect, "md5": md5(args.protect)})
    if args.protect_from:
        d, _, c = load(args.protect_from, ids, key)
        m |= vec(d, c) == base_v; src.append({"equal_rows_of": args.protect_from, "md5": md5(args.protect_from)})
    return m, src


def write_out(base, key, col, out, path):
    # write path (rint -> int64 -> base dtype, polars default parquet writer)
    res = base.with_columns(pl.Series(col, np.round(out).astype(np.int64)).cast(base[col].dtype))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    res.write_parquet(path)
    return md5(path)


def validate(path, n, ref):
    if n is None:
        return None
    tmp = tempfile.mkdtemp(); q = os.path.join(tmp, f"combine_check_{n}.parquet"); shutil.copyfile(path, q)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "validate_submission.py"), q] + ([ref] if ref else [])
    r = subprocess.run(cmd, capture_output=True, text=True); shutil.rmtree(tmp)
    print(r.stdout.strip()); print(r.stderr.strip()) if r.returncode else None
    assert r.returncode == 0, "validate_submission.py FAILED"
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["add", "avg"])
    ap.add_argument("--base", help="add: the file the deltas are applied to; avg: the reference for the diff summary (default: first file)")
    ap.add_argument("-c", "--comp", action=Comp, default=[], help="add: FILE@ONBASE, add its delta (repeatable)")
    ap.add_argument("-r", "--rm", action=Comp, help="add: FILE@ONBASE, remove its delta = rollback (repeatable)")
    ap.add_argument("files", nargs="*", help="avg: the k candidate files (equal weights)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-overlap", action="store_true")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--protect"); ap.add_argument("--protect-from"); ap.add_argument("--allow-protected", action="store_true")
    ap.add_argument("--validate-as", type=int, help="run validate_submission.py on the output (N = a free label)")
    ap.add_argument("--expect-md5", help="assert the output md5 starts with this prefix")
    ap.add_argument("--label", default="", help="free text for the manifest (rationale / slot)")
    a = ap.parse_args()

    man = {"tool": "scripts/final_combine.py", "tool_md5": md5(os.path.abspath(__file__)), "mode": a.mode, "label": a.label,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv[1:],
           "polars": pl.__version__, "numpy": np.__version__}

    if a.mode == "add":
        assert a.base and a.comp, "add needs --base and at least one -c"
        base, key, col = load(a.base); ids = base[key]; bv = vec(base, col)
        man["base"] = {"path": a.base, "md5": md5(a.base)}
        total = np.zeros_like(bv); touch = np.zeros(len(bv), np.int32); comps = []; stale_tot = 0
        for s in a.comp:
            sign, f, on = parse_comp(s)
            df, _, _ = load(f, ids, key, col); do, _, _ = load(on, ids, key, col)
            fv, ov = vec(df, col), vec(do, col); dlt = fv - ov; nz = dlt != 0
            stale = int((nz & (bv != (ov if sign > 0 else fv))).sum()); stale_tot += stale
            ov_rows = int((nz & (touch > 0)).sum())
            touch += nz; total += sign * dlt
            c = {"sign": "+" if sign > 0 else "-", "file": f, "file_md5": md5(f), "on": on, "on_md5": md5(on),
                 "rows_delta": int(nz.sum()), "delta_mean_s": float(dlt[nz].mean()) if nz.any() else 0.0,
                 "delta_rms_s": float(np.sqrt((dlt ** 2).mean())), "stale_rows": stale, "overlap_with_earlier": ov_rows}
            comps.append(c)
            print(f"  {c['sign']} {os.path.basename(f)} @ {os.path.basename(on)}: {c['rows_delta']:,} rows, "
                  f"RMS {c['delta_rms_s']:.2f} s, stale {stale:,}, overlap {ov_rows:,}")
        overlap = int((touch > 1).sum())
        man.update(components=comps, overlap_rows=overlap, stale_rows=stale_tot)
        if overlap:
            man["overlap_ids_head"] = ids.filter(pl.Series(touch > 1)).head(50).to_list()
            assert a.allow_overlap, f"OVERLAP: {overlap} rows touched by >1 delta (use --allow-overlap to sum them)"
        if stale_tot:
            assert a.allow_stale, f"STALE: {stale_tot} delta rows where BASE != ONBASE (use --allow-stale if intended)"
        out = bv + total; ref_v = bv; ref_path = a.base
    else:
        files = a.files; assert len(files) >= 2, "avg needs >= 2 files"
        refp = a.base or files[0]
        base, key, col = load(refp); ids = base[key]; ref_v = vec(base, col)
        V = np.vstack([vec(load(f, ids, key, col)[0], col) for f in files])
        out = V.mean(0)
        pair = {f"{i}-{j}": float(np.sqrt(((V[i] - V[j]) ** 2).mean())) for i in range(len(files)) for j in range(i + 1, len(files))}
        man.update(base={"path": refp, "md5": md5(refp)}, weights="equal",
                   components=[{"file": f, "file_md5": md5(f), "weight": 1 / len(files)} for f in files],
                   pairwise_rms_s=pair, rows_all_equal=int((V == V[0]).all(0).sum()))
        print(f"  equal-weight mean of {len(files)} files; pairwise RMS {pair}")
        ref_path = refp; bv = ref_v

    pm, psrc = protected_mask(a, key, ids, bv)
    ch = np.round(out) != bv
    if psrc:
        bad = int((ch & pm).sum()); man["protected"] = {"sources": psrc, "rows": int(pm.sum()), "changed": bad}
        assert bad == 0 or a.allow_protected, f"PROTECTED: {bad} protected rows change"
    o = np.round(out)
    assert np.isfinite(o).all() and o.min() >= FLOOR, f"value check: min {o.min()} < {FLOOR} s floor"
    d = o - ref_v
    man["vs_base"] = {"rows_changed": int((d != 0).sum()), "rms_s": float(np.sqrt((d ** 2).mean())), "mean_s": float(d.mean()),
                      "max_abs_s": float(np.abs(d).max()), "min_pred": float(o.min()), "mean_pred": float(o.mean())}
    print(f"  vs base: {man['vs_base']['rows_changed']:,} rows changed, RMS {man['vs_base']['rms_s']:.2f} s, "
          f"mean {man['vs_base']['mean_s']:+.3f} s, min pred {o.min():.0f}")
    h = write_out(base, key, col, out, a.out)
    man["out"] = {"path": a.out, "md5": h}
    man["validate"] = validate(a.out, a.validate_as, ref_path)
    with open(a.out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    print(f"md5 {h}\nmanifest {a.out}.manifest.json")
    if a.expect_md5:
        assert h.startswith(a.expect_md5), f"md5 {h} != expected {a.expect_md5}"
        print(f"md5 matches the expected prefix {a.expect_md5}")


if __name__ == "__main__":
    main()
