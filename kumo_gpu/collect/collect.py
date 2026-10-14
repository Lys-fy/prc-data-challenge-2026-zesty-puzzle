#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Copies cluster run results into the build machine's layout so the existing screen scripts run unchanged. Run on
the build host after the cluster run/results/ (and, for the fine-tune, run/data/oof/ + run/artifacts/) are copied
back to $PRC_ROOT/scratchpad/xclu/run/. Positional arg: RUNDIR (default scratchpad/xclu/run); flags --replace, --dry.
Writes data/oof/, data/preds/, data/distill/ and scratchpad/xkt/t2x/ files, listed below.

  member  results/member/<NAME>_<fold>_<APT>.npz -> data/oof/<NAME>_<fold>.npy ([id, taxi], features_v9 row order = kumo_km_member.py's np.save), for every
          profile NAME found (kumo_member_screen_km = shared20k, kumo_member_screen_km8 = m20e8, kumo_member_screen_km50, kumo_member_screen_kl50); fold prod -> data/oof/<NAME>_prod.npy + data/preds/<NAME>P_s0.parquet.
          An existing data/oof/<NAME>_<fold>.npy (kumo_member_screen_km janjul/months28 from the GPU workstation) is NOT touched: the cluster copy goes to <NAME>c_<fold>.npy
          and a parity line is printed; --replace backs the old file up to scratchpad/xclu/backup/ and replaces it.
  distill data/distill/<NAME>_teacher.parquet: MVT_ID_mvt, is_rank, fold, m, teacher_taxi, teacher_g, q05..q95 for all 2025 rows (OOF by month:
          each row comes from the fold that holds its month out of the context) + the 2026 ranking rows; <NAME>_synth.parquet if SYNTH was used.
  t2x     results/t2x/gK_<chain>_<fold>_<APT>.npz  ->  scratchpad/xkt/t2x/gK_<chain>_<fold>.npy  (aligned to the dump npz rows, as kumo_t2x_screen2.py `kumo` writes)
  ft      data/oof/kumo_icl_lirf_ft_<fold>.npy (re-sorted to features_v9 order), artifacts/xtfm2/ft_<size>_<fold>.pt  (optional)
Every file is checked: complete airport coverage, exact row set and order vs features_v9 (member) / the dump npz (t2x), finite values, one
consistent (size, seed, nest, tf32) setting across units. Nothing is overwritten without --replace. --dry: check only, write nothing.
usage: .venv/bin/python kumo_gpu/collect/collect.py [RUNDIR=scratchpad/xclu/run] [--replace] [--dry]"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, glob, json, shutil, numpy as np, polars as pl

PRC = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()))
os.chdir(PRC)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
RUNDIR = os.path.abspath(args[0] if args else "scratchpad/xclu/run")
REPLACE = "--replace" in sys.argv
DRY = "--dry" in sys.argv
R = f"{RUNDIR}/results"
HOLD = {
    "janjul": [1, 7],
    "months28": [2, 8],
    "months39": [3, 9],
    "months410": [4, 10],
    "months511": [5, 11],
    "months612": [6, 12],
    "prod": None,
}
APTS = os.environ.get("kumo_cluster_setup_apts", "EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH,LTFM").split(",")


def put(path, arr=None, df=None):
    if DRY:
        print(f"    (dry) would write {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".xclu_tmp"
    if arr is not None:
        np.save(tmp + ".npy", arr)
        os.replace(tmp + ".npy", path)
    else:
        df.write_parquet(tmp)
        os.replace(tmp, path)
    print(f"    wrote {path}")


FV = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "m", "is_rank"])
fid = FV["MVT_ID_mvt"].to_numpy().astype(float)
fm = FV["m"].to_numpy()
frk = FV["is_rank"].to_numpy() == 1
summary = []
QN = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
names = sorted(
    {os.path.basename(f).rsplit("_", 2)[0] for f in glob.glob(f"{R}/member/*.npz")}
)  # kumo_member_screen_km (shared20k), kumo_member_screen_km8, kumo_member_screen_km50, kumo_member_screen_kl50 ...
# ---------------- member (per profile NAME) + distillation teacher table
for NAME in names:
    settings = set()
    teach = []
    synth_rows = []
    for fold, hold in HOLD.items():
        fs = {a: f"{R}/member/{NAME}_{fold}_{a}.npz" for a in APTS}
        have = [a for a, f in fs.items() if os.path.exists(f)]
        if not have:
            continue
        if len(have) < len(APTS):
            print(
                f"{NAME} {fold}: INCOMPLETE {len(have)}/10 airports (missing {sorted(set(APTS) - set(have))}) -> skipped"
            )
            summary.append((f"{NAME} {fold}", "incomplete"))
            continue
        Z = [np.load(fs[a]) for a in APTS]
        for z in Z:
            mt = json.loads(str(z["meta"]))
            settings.add(
                json.dumps(mt.get("profile", {k: mt.get(k) for k in ("size", "nest", "tf32")}), sort_keys=True)
                + f" seed {mt['seed']}"
            )
        row = np.concatenate([z["row"] for z in Z])
        ids = np.concatenate([z["id"] for z in Z])
        taxi = np.concatenate([z["taxi"] for z in Z])
        g = np.concatenate([z["g"] for z in Z])
        q = np.concatenate([z["q"] for z in Z]) if all("q" in z.files for z in Z) else None
        X2 = (
            np.vstack([np.concatenate([z[k] for z in Z]) for k in ("spread", "g_sd")])
            if all("spread" in z.files for z in Z)
            else None
        )  # engine extras
        o = np.argsort(row)
        row, ids, taxi, g = row[o], ids[o], taxi[o], g[o]
        q = q[o] if q is not None else None
        X2 = X2[:, o] if X2 is not None else None
        want = np.where(frk if hold is None else (~frk & np.isin(fm, hold)))[0]
        assert np.array_equal(row, want), f"{NAME} {fold}: row set differs from features_v9"
        assert np.array_equal(ids, fid[want]), f"{NAME} {fold}: ids differ"
        assert np.isfinite(taxi).all(), f"{NAME} {fold}: non-finite taxi"
        A = np.vstack([ids, taxi] + ([X2] if X2 is not None else []))
        print(
            f"{NAME} {fold}: {len(ids):,} rows OK | taxi mean {taxi.mean():.1f} sd {taxi.std():.1f}"
            + (
                f" | @2 spread med {np.median(X2[0]):.0f} s, @3 g_sd med {np.median(X2[1]):.1f} s"
                if X2 is not None
                else ""
            )
        )
        T = {
            "MVT_ID_mvt": ids,
            "is_rank": np.full(len(ids), hold is None),
            "fold": np.full(len(ids), fold),
            "m": fm[want],
            "teacher_taxi": taxi,
            "teacher_g": g,
        }
        if q is not None:
            T.update({n: q[:, i] for i, n in enumerate(QN)})
        if X2 is not None:
            T.update({"spread": X2[0], "g_sd": X2[1]})
        teach.append(pl.DataFrame(T))
        for a, z in zip(APTS, Z):
            if "synth_g" in z.files:
                synth_rows.append(
                    pl.DataFrame(
                        {
                            "apt": np.full(len(z["synth_g"]), a),
                            "fold": np.full(len(z["synth_g"]), fold),
                            "base_row": z["synth_base_row"],
                            "teacher_taxi": z["synth_off"] + z["synth_g"],
                            **{n: z["synth_q"][:, i] + z["synth_off"] for i, n in enumerate(QN)},
                        }
                    )
                )
        if hold is None:
            put(f"data/oof/{NAME}_prod.npy", A)
            put(f"data/preds/{NAME}P_s0.parquet", df=pl.DataFrame({"MVT_ID_mvt": ids, "taxi": taxi}))
            summary.append((f"{NAME} prod", "written"))
            continue
        dst = f"data/oof/{NAME}_{fold}.npy"
        if os.path.exists(dst):
            G = np.load(dst)
            assert np.array_equal(G[0], ids), f"{fold}: existing file has another row order"
            d = taxi - G[1]
            print(
                f"  parity vs existing {dst}: RMS diff {np.sqrt((d ** 2).mean()):.3f} s, max {np.abs(d).max():.2f} s, corr {np.corrcoef(taxi, G[1])[0, 1]:.6f}"
            )
            if REPLACE:
                if not DRY:
                    os.makedirs("scratchpad/xclu/backup", exist_ok=True)
                    shutil.copy2(dst, f"scratchpad/xclu/backup/{NAME}_{fold}.prev.npy")
                put(dst, A)
                summary.append((f"{NAME} {fold}", "replaced (previous backed up)"))
            else:
                put(f"data/oof/{NAME}c_{fold}.npy", A)
                summary.append((f"{NAME} {fold}", f"kept existing; cluster copy -> {NAME}c"))
        else:
            put(dst, A)
            summary.append((f"{NAME} {fold}", "written"))
    if len(settings) > 1:
        print(f"WARNING {NAME}: mixed unit settings: {settings}")
    elif settings:
        print(f"{NAME} settings: {settings.pop()}")
    if teach:  # distillation teacher table: 2025 rows OOF by month (6 two-month folds) + 2026 ranking rows
        Tt = pl.concat(teach).sort("MVT_ID_mvt")
        n25 = int((~Tt["is_rank"]).sum())
        n26 = int(Tt["is_rank"].sum())
        full = n25 == int((~frk).sum()) and n26 == int(frk.sum())
        put(f"data/distill/{NAME}_teacher.parquet", df=Tt)
        summary.append(
            (f"{NAME} teacher", f"2025 OOF rows {n25:,}, 2026 rows {n26:,} ({'complete' if full else 'PARTIAL'})")
        )
    if synth_rows:
        put(f"data/distill/{NAME}_synth.parquet", df=pl.concat(synth_rows))
        summary.append(
            (f"{NAME} synth", f"{sum(len(x) for x in synth_rows):,} rows (regenerate X with kumo_gpu/cluster/synth.py)")
        )
# ---------------- t2x (pre-declared T2x screen rule)
for ch in ("Kq", "NNA3"):
    for t in ("months39", "months28", "months410", "janjul"):
        fs = sorted(glob.glob(f"{R}/t2x/gK_{ch}_{t}_*.npz"))
        if not fs:
            continue
        z0 = np.load(f"scratchpad/xkt/t2x/{ch}_{t}.npz", allow_pickle=True)
        apt = z0["apt"].astype(str)
        gK = np.full(len(apt), np.nan)
        got = set()
        for f in fs:
            a = f[-8:-4]
            z = np.load(f)
            assert (apt[z["pos"]] == a).all() and (apt == a).sum() == len(z["pos"]), f
            gK[z["pos"]] = z["gK"]
            got.add(a)
        if not np.isfinite(gK).all():
            print(f"t2x {ch} {t}: INCOMPLETE ({len(got)} airports) -> skipped")
            continue
        dst = f"scratchpad/xkt/t2x/gK_{ch}_{t}.npy"
        if os.path.exists(dst) and not REPLACE:
            o = np.load(dst)
            print(
                f"t2x {ch} {t}: {dst} exists (max |diff| vs cluster {np.abs(o - gK).max():.3f}) -> kept; --replace to overwrite"
            )
            summary.append((f"t2x {ch} {t}", "kept existing"))
            continue
        put(dst, gK)
        summary.append((f"t2x {ch} {t}", "written"))
# ---------------- optional fine-tune (scripts/kumo_icl_finetune.py)
for f in sorted(glob.glob(f"{RUNDIR}/data/oof/kumo_icl_lirf_ft_*.npy")):
    fold = os.path.basename(f)[9:-4]
    A = np.load(f)
    hold = HOLD[fold]
    want = np.where(~frk & np.isin(fm, hold))[0]
    pos = {v: i for i, v in enumerate(fid[want])}
    o = np.argsort([pos[v] for v in A[0]])
    A = A[:, o]
    assert np.array_equal(A[0], fid[want]) and np.isfinite(A[1]).all(), f
    put(f"data/oof/kumo_icl_lirf_ft_{fold}.npy", A)
    summary.append((f"ft {fold}", "written"))
for f in sorted(glob.glob(f"{RUNDIR}/artifacts/xtfm2/ft_*.pt")):
    if not DRY:
        os.makedirs("artifacts/xtfm2", exist_ok=True)
        shutil.copy2(f, f"artifacts/xtfm2/{os.path.basename(f)}")
print("\nSUMMARY")
[print(f"  {k}: {v}") for k, v in summary]
print("""
NEXT (build machine, unchanged scripts):
  distillation          data/distill/<NAME>_teacher.parquet (2025 OOF-by-month + 2026) -> CPU LightGBM student (distillation test)
                        (needs data/oof/kumo_member_screen_km_{months39,months28,months410,janjul}.npy; writes scratchpad/xpla/kumo_member_screenb4_*, scratchpad/xkt/mscreen_final.txt)
  6-fold / production   data/oof/kumo_member_screen_km_months511/612.npy and data/oof/kumo_member_screen_km_prod.npy / data/preds/kumo_member_screen_kmp_s0.parquet are inputs for the 6-fold
                        protocol and a production build; the ranking-side member hook is an coordinator step.""")
