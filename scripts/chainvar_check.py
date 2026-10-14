#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Checks and placebo for the Kumo member input of the chain variants (xtd2). Reads 2025 labels only through the member files.
  km <NAME> [<PNAME>]       KM files: data/oof/<NAME>_<fold>.npy on the 6 folds (shape (>=2, n), ids == features_v9 rows is_rank==0 & month in hold, in
                            features_v9 order = collect.py / kumo_km_member.py's np.save; finite; every matched fold row covered) and the production file
                            data/preds/<PNAME>_s0.parquet (columns MVT_ID_mvt, taxi; ids == the is_rank rows; finite). md5 + shape -> scratchpad/xhook/km_<NAME>.tsv
  placebo <NAME> <PNAME> <SRC> <PSRC:n>   writes a PLACEBO KM in the exact KM format: fold taxi = row 3 of data/oof/<SRC>_<fold>* (a copy of an existing member),
                            production taxi = seed mean of data/preds/<PSRC>_s*.parquet "taxi". Refuses to overwrite and refuses NAME starting with kumo_member_screen_.
  inputs fold|prod|build <C|PFX>   every chain input of that stage exists (prints the missing list; exit 3 if any is missing)
  rank <PNAME>              unit test of the ranking-side block EXTRA2_BLOCK of chainvar_assemble.py (the exact code string) on the real ranking ids / matched mask
  manifest <tsv>            md5 of every KM-chain input listed in release/manifest/intermediates_v99.tsv (the v99 lineage, build machine) vs the local copy:
                            match / MISMATCH / ABSENT; writes scratchpad/xhook/manifest_check.tsv and the copy list scratchpad/xhook/rsync_from_build_host.txt
usage: .venv/bin/python scripts/chainvar_check.py <mode> ..."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, ast, glob, hashlib, numpy as np, polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
HOLD = {
    "months39": [3, 9],
    "months28": [2, 8],
    "months410": [4, 10],
    "janjul": [1, 7],
    "months612": [6, 12],
    "months511": [5, 11],
}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()


def frame():
    return pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "m", "unmatched"])


def fold_rows(F, f):
    msk = (F["is_rank"].to_numpy() == 0) & np.isin(F["m"].to_numpy(), HOLD[f])
    return F["MVT_ID_mvt"].to_numpy().astype(float)[msk], F["unmatched"].to_numpy()[msk] == 0


def npy(p, t):
    c = (
        glob.glob(f"data/oof/{p}_{t}.npy")
        or glob.glob(f"data/oof/{p}_{t}_*.npy")
        or (glob.glob(f"data/oof/{p}_months17.npy") if t == "janjul" else [])
    )
    return (np.load(sorted(c)[0]), sorted(c)[0]) if c else (None, None)


mode = sys.argv[1]
bad = []
if mode == "km":
    NAME = sys.argv[2]
    PNAME = sys.argv[3] if len(sys.argv) > 3 else None
    F = frame()
    os.makedirs("scratchpad/xhook", exist_ok=True)
    man = []
    for f in HOLD:
        p = f"data/oof/{NAME}_{f}.npy"
        if not os.path.exists(p):
            bad.append(f"missing {p}")
            print("MISSING", p)
            continue
        a = np.load(p)
        ids, mt = fold_rows(F, f)
        ok_shape = a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] == len(ids)
        ok_rows = ok_shape and np.array_equal(a[0], ids)
        ok_fin = ok_shape and bool(np.isfinite(a[1]).all())
        cov = float(np.isin(ids[mt], a[0]).mean()) if ok_shape else 0.0
        st = (
            f"taxi mean {np.nanmean(a[1]):.1f} sd {np.nanstd(a[1]):.1f} min {np.nanmin(a[1]):.0f} max {np.nanmax(a[1]):.0f}"
            if a.ndim == 2
            else ""
        )
        print(
            f"{f:10s} {p} md5 {md5(p)[:8]} shape {a.shape} expect (>=2, {len(ids):,}) | rows==features_v9 {ok_rows} finite {ok_fin} matched-row cover {cov:.5f} | {st}"
        )
        man.append((p, md5(p), "x".join(map(str, a.shape)), ok_rows, ok_fin))
        if not (ok_shape and ok_rows and ok_fin and cov == 1.0):
            bad.append(f"{p}: shape {ok_shape} rows {ok_rows} finite {ok_fin} cover {cov}")
    if PNAME:
        p = f"data/preds/{PNAME}_s0.parquet"
        if not os.path.exists(p):
            bad.append(f"missing {p}")
            print("MISSING", p)
        else:
            z = pl.read_parquet(p)
            R = F.filter(pl.col("is_rank") == 1)
            rid = np.sort(R["MVT_ID_mvt"].to_numpy().astype(float))
            zi = np.sort(z["MVT_ID_mvt"].to_numpy().astype(float)) if "MVT_ID_mvt" in z.columns else None
            ok_cols = {"MVT_ID_mvt", "taxi"} <= set(z.columns)
            ok_rows = zi is not None and np.array_equal(zi, rid)
            ok_fin = ok_cols and bool(np.isfinite(z["taxi"].to_numpy()).all())
            print(
                f"prod       {p} md5 {md5(p)[:8]} shape {z.shape} cols {z.columns} | ids == {len(rid):,} ranking rows {ok_rows} finite {ok_fin} | taxi mean {z['taxi'].mean():.1f}"
            )
            man.append((p, md5(p), "x".join(map(str, z.shape)), ok_rows, ok_fin))
            if not (ok_cols and ok_rows and ok_fin):
                bad.append(f"{p}: cols {ok_cols} rows {ok_rows} finite {ok_fin}")
    pl.DataFrame(man, schema=["path", "md5", "shape", "rows_ok", "finite"], orient="row").write_csv(
        f"scratchpad/xhook/km_{NAME}.tsv", separator="\t"
    )
elif mode == "placebo":
    NAME, PNAME, SRC, PSRC = sys.argv[2:6]
    assert not NAME.startswith("kumo_member_screen_") and not PNAME.startswith(
        "kumo_member_screen_"
    ), "placebo names must not shadow real KM files"
    F = frame()
    for f in HOLD:
        p = f"data/oof/{NAME}_{f}.npy"
        assert not os.path.exists(p), f"{p} exists"
        ids, _ = fold_rows(F, f)
        a, src = npy(SRC, f)
        o = np.argsort(a[0])
        k = np.clip(np.searchsorted(a[0][o], ids), 0, a.shape[1] - 1)
        hit = a[0][o][k] == ids
        tx = np.where(hit, a[3][o][k], np.nan)
        fill = ~hit | ~np.isfinite(tx)
        tx[fill] = np.nanmedian(tx)  # rows the source lacks (e.g. unmatched) -> fold median
        np.save(p, np.vstack([ids, tx]))
        print(
            f"placebo {p} <- {src} row 3 | rows {len(ids):,} filled with median {int(fill.sum()):,} | md5 {md5(p)[:8]}"
        )
    ps, n = PSRC.split(":")
    acc = None
    for i in range(int(n)):
        z = pl.read_parquet(f"data/preds/{ps}_s{i}.parquet", columns=["MVT_ID_mvt", "taxi"]).sort("MVT_ID_mvt")
        acc = z["taxi"].to_numpy() if acc is None else acc + z["taxi"].to_numpy()
    p = f"data/preds/{PNAME}_s0.parquet"
    assert not os.path.exists(p), f"{p} exists"
    pl.DataFrame({"MVT_ID_mvt": z["MVT_ID_mvt"].cast(pl.Float64), "taxi": acc / int(n)}).write_parquet(p)
    print(f"placebo {p} <- {ps} x{n} taxi | md5 {md5(p)[:8]}")
elif mode == "inputs":
    st, C = sys.argv[2], sys.argv[3]
    need = []
    if st == "fold":
        for f in HOLD:
            for m_ in (
                "queue_eb_ra3qa",
                "runway_queue_catq",
                "meta_oof_h_ctl",
                "NNA2",
                "meta_oof_realmlp_nna3",
                "fallback_regpl",
                "lgb_adsb_linked_a",
                "lgb_noclock_nc",
                "lgb_airport_ltfm",
                "lgb_airport_lfpg",
                "wideleaf_retune_a2",
            ):
                if npy(m_, f)[0] is None:
                    need.append(f"data/oof/{m_}_{f}*.npy")
        need += [
            f"data/external/{x}.parquet" for x in ("adsb_features", "adsb6_features", "adsbL_features", "xpr_pru_daily")
        ]
    elif st == "prod":
        S6 = {
            "queue_eb_v47lqa": 6,
            "wideleaf_a2p": 3,
            "lgb_adsb_linked_a": 3,
            "lgb_noclock_nc": 6,
            "lgb_airport_ltfm": 6,
            "lgb_airport_lfpg": 6,
            "runway_queue_v47lq": 6,
        }
        S6.update(
            {"runway_queue_catv17xq": 6, "NNRA2": 1} if C == "Kq" else {"CATV17x": 14, "sixseed_chain_nnra3x6": 1}
        )
        for pf, n in S6.items():
            need += [f"data/preds/{pf}_s{i}.parquet" for i in range(n)]
        need += [
            "data/oof/fallback_regpl_rank.npy",
            "data/external/xba_fallback_ids.parquet",
            "stages/fallback_base.parquet",
            "stages/unmatched_rule.parquet",
        ]
        PFX = os.environ.get("PFX", "kumo_member")
        need += [
            f"scratchpad/xau1/clockfold1_{PFX}{C}_HuJ.npz",
            f"scratchpad/xhook/P13_{PFX}{C}xSTK_janjul.npy",
            f"scratchpad/xhook/P13_{PFX}{C}xTD_janjul.npy",
        ]
        if os.environ.get("KMP"):
            need.append(f"data/preds/{os.environ['KMP']}_s0.parquet")
    elif st == "build":
        E = os.environ.get
        need += [
            E(k)
            for k in (
                "V62",
                "V62FK",
                "P12R",
                "V74X",
                "V72X",
                "unm_resid",
                "V84",
                "V85",
                "V86",
                "V88",
                "V99",
                "REF88",
                "UM",
                "C1",
            )
            if E(k)
        ]
        need += ["scratchpad/xau2/features_v9_prodN_ids.parquet"] + [
            f"scratchpad/xau1/clockfold1_ldtk_ha0l_{f}.npy" for f in HOLD
        ]
        need += [
            f"scratchpad/xau1/clockfold1_{C}{c}_HA0L_{f}.npy" for c in ("Kq", "NNA3") for f in HOLD if not E(f"FOLD{c}")
        ]
    for p in need:
        if not glob.glob(p):
            bad.append(p)
    print(
        f"CHAIN_HOOK inputs {st} {C}: {len(need)} checked, {len(bad)} missing"
        + ("".join(f"\n  MISSING {b}" for b in bad))
    )
    if bad:
        sys.exit(3)
elif mode == "manifest":
    import re

    PAT = [
        r"data/features_v9\.parquet",
        r"data/external/(adsb_features|adsb6_features|adsbL_features|xpr_pru_daily|xba_fallback_ids)\.parquet",
        r"data/oof/(queue_eb_ra3qa|runway_queue_catq|meta_oof_h_ctl|NNA2|meta_oof_realmlp_nna3|fallback_regpl|lgb_adsb_linked_a|lgb_noclock_nc|lgb_airport_ltfm|lgb_airport_lfpg|wideleaf_retune_a2)_(months\d+|janjul|rank)(_.*)?\.npy",
        r"data/preds/(queue_eb_v47lqa|runway_queue_catv17xq|CATV17x|NNRA2|sixseed_chain_nnra3x6|meta_oof_realmlp_nnra3|wideleaf_a2p|runway_queue_v47lq|lgb_adsb_linked_a|lgb_noclock_nc|lgb_airport_ltfm|lgb_airport_lfpg)_s\d+\.parquet",
        r"scratchpad/xau1/(PR_LdTK_AMXAU1_HA0L\.parquet|clockfold1_ldtk_ha0l_.*\.npy)",
        r"scratchpad/xau3/um25_A1\.npz",
        r"scratchpad/xau2/(features_v9_prodN_ids|unm_ids|newstand_delta_p[UN]_s[01])\.parquet",
        r"scratchpad/p12/P12_RdLN_HA0\.parquet",
        r"scratchpad/xbase/(P13_XB(Kq|NNA3)x(STK|TD).*\.npy|PR_XB(Kq|NNA3)x?_XB.*\.parquet)",
        r"submissions/zesty-puzzle_(v53|v58|v62|v62_fullK_xckHu6_xlvA0L_m12_cand|v74_xhr_pxsafe_cand|v72_xprBX_xsx_xpk_cand|v76_p13T2x_xur2A1_cand|v82b_xur2V3_cand|v82b_xur2V3E1s_cand|v84_xau2ns_cand|v87_v88combo_cand|v88_xbase_cand|v98_xtfm2F4_cand)\.parquet",
    ]
    rx = re.compile("^(" + "|".join(PAT) + ")$")
    rows = []
    cnt = {"match": 0, "MISMATCH": 0, "ABSENT": 0}
    for ln in open(sys.argv[2]):
        if ln.startswith("#"):
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) < 2 or not rx.match(c[0]) or not re.fullmatch(r"[0-9a-f]{32}", c[1]):
            continue
        st = "ABSENT" if not os.path.exists(c[0]) else ("match" if md5(c[0]) == c[1] else "MISMATCH")
        cnt[st] += 1
        rows.append((c[0], c[1], st))
    pl.DataFrame(rows, schema=["path", "md5_build_host", "status"], orient="row").write_csv(
        "scratchpad/xhook/manifest_check.tsv", separator="\t"
    )
    open("scratchpad/xhook/rsync_from_build_host.txt", "w").write("".join(r[0] + "\n" for r in rows if r[2] != "match"))
    print(
        f"CHAIN_HOOK manifest: {len(rows)} KM-chain input files in the v99 lineage manifest | "
        + " ".join(f"{k} {v}" for k, v in cnt.items())
    )
    for r in rows:
        if r[2] == "MISMATCH":
            print("  MISMATCH", r[0])
    grp = {}
    for r in rows:
        if r[2] == "ABSENT":
            k = re.sub(r"_(s\d+|months\d+|janjul)(_.*)?\.(npy|parquet)$", "_*", r[0])
            grp[k] = grp.get(k, 0) + 1
    for k, v in sorted(grp.items()):
        print(f"  ABSENT {v:3d} x {k}")
elif mode == "rank":
    PNAME = sys.argv[2]
    tree = ast.parse(open("scripts/chainvar_assemble.py").read())
    code = [
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "EXTRA2_BLOCK"
    ][0]
    R = (
        pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched"])
        .filter(pl.col("is_rank") == 1)
        .sort("MVT_ID_mvt")
    )
    ids = R["MVT_ID_mvt"].to_numpy()
    m = R["unmatched"].to_numpy() == 0
    n = len(ids)

    def avg(pref, k, cols):  # = audited_assemble.avg
        acc = None
        for i in range(k):
            d = pl.read_parquet(f"data/preds/{pref}_s{i}.parquet").sort("MVT_ID_mvt")
            a = np.vstack([d[c].to_numpy().astype(float) for c in cols])
            acc = a if acc is None else acc + a
            ii = d["MVT_ID_mvt"].to_numpy()
        return ii, acc / k

    os.environ["PXTD2"] = f"{PNAME}:1"
    g = {
        "np": np,
        "E": lambda k, d: os.environ.get(k, d),
        "avg": avg,
        "ids": ids,
        "m": m,
        "extra_inputs": ["xtd2"],
        "p": np.full(n, 0.5),
        "art": np.full(n, 600.0),
        "Lv": np.full(n, 900.0),
        "d": {},
    }
    exec(compile(code, "EXTRA2_BLOCK", "exec"), g)
    T2 = g["d"]["T2"]
    z = pl.read_parquet(f"data/preds/{PNAME}_s0.parquet").sort("MVT_ID_mvt")
    zz = dict(zip(z["MVT_ID_mvt"].to_numpy().astype(float), z["taxi"].to_numpy()))
    chk = np.array([0.5 * 600 + 0.5 * zz[x] for x in ids[m][:5000].astype(float)])
    ok = np.isfinite(T2[m]).all() and np.isnan(T2[~m]).all() and np.allclose(T2[m][:5000], chk)
    print(
        f"EXTRA2_BLOCK unit test on {n:,} ranking rows ({int(m.sum()):,} matched): T2 finite on all matched {np.isfinite(T2[m]).all()}, NaN on all unmatched {np.isnan(T2[~m]).all()}, id alignment (5k spot) {np.allclose(T2[m][:5000], chk)} -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        bad.append("rank")
if bad:
    print("chain_hook_check FAIL:", bad)
    sys.exit(1)
print(f"chain_hook_check {mode} PASS")
