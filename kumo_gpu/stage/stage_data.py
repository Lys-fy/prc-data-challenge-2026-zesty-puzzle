#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Builds, on the CPU build machine, the minimal payload needed to run the Kumo cluster step elsewhere: per-airport
member parquet frames (features_v9 order, the 412 numeric feature columns), stripped t2x screen dumps, a parity
reference array, vendored sdm/HF model weights, and an MD5 manifest.
Inputs: scripts/kumo_km_member.py (run up to its torch import), kumo_gpu/data/km_features_412.txt,
scratchpad/xkt/t2x/*.npz, data/oof/kumo_member_screen_km_janjul.npy, the kumo venv's site-packages, the HF model
snapshot; positional OUT dir (default scratchpad/xclu/payload); env WITH_FT, PARITY_APT, VENDOR_SIZES, KUMO_PY.
Output: the OUT/ payload directory tree plus OUT/MANIFEST_staged.tsv.

KUMO_CLUSTER_SETUP STAGING, run CPU only (~25 GB RAM peak, ~5-10 min): builds the minimal payload the next step needs.
Nothing leaves the build machine here; the payload directory is what gets copied onward (docs/kumo_gpu.md).

  member/  km_<APT>.parquet  one file per airport: kumo_km_member.py's frame exactly (meta__<META> in original dtypes + the 412 numeric FEATS_L columns as Float32 with NaN -> null),
                             + `_row` (row index in features_v9 order) + `day` (ordinal day of MVT_TIME_UTC_mvt, for the optional fine-tune).
                             Rows keep features_v9 order within each airport (stable), so the per-airport context subsample of kumo_km_member.py
                             (np.where order -> rng.choice positions) is reproduced exactly.
  t2x/     t2x_<chain>_<fold>.npz  kumo_t2x_screen2.py dump files stripped to what the `kumo` step reads: apt (U4), y, v41, X (float32). No ids, no gM/gS.
  ref/     parity_XKT_KM_janjul_LSZH.npy  the GPU workstation member output for janjul x LSZH ([id, taxi], 22,346 rows): the cluster parity reference.
  vendor/  sdm_ee6acd4_site.tgz   structured-data-models at ee6acd4 (pure Python, from the kumo env's site-packages; Apache-2.0) = offline fallback
           hf/                    nvidia/Kumo-Tabular v1.0.0 (commit bd7fa122) medium + large regressor.pt + LICENSE (OpenMDW-1.1), HF cache layout
  MANIFEST_staged.tsv            path, bytes, md5 of every payload file
usage (build machine):  .venv/bin/python kumo_gpu/stage/stage_data.py [OUT=scratchpad/xclu/payload]
env: WITH_FT=1 also keeps `day` (default on; 1 column), PARITY_APT (LSZH)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, time, hashlib, tarfile, shutil, glob, numpy as np, polars as pl

t0 = time.time()
PRC = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()))
os.chdir(PRC)
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/xclu/payload")
for s in ("member", "t2x", "ref", "vendor"):
    os.makedirs(f"{OUT}/{s}", exist_ok=True)

# ---- 1. member frame: kumo_km_member.py's own feature resolution + frame build, executed unchanged up to the torch import
src = open("scripts/kumo_km_member.py").read()
head = src.split("import torch, sdm")[0]
assert 'if os.environ.get("DRY")' in head and "d = pl.read_parquet(FV" in head
os.environ.pop("DRY", None)
ns = {"__file__": f"{PRC}/scripts/kumo_km_member.py", "__name__": "xclu_stage"}
exec(head, ns)
d, num = ns["d"], ns["num"]
assert len(num) == 412, len(num)
ref_list = (
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "km_features_412.txt")).read().split()
)
assert (
    num == ref_list
), "feature list differs from kumo_gpu/data/km_features_412.txt (kumo_km_member.py or a side file changed)"
day = pl.read_parquet(ns["FV"], columns=["MVT_TIME_UTC_mvt"])["MVT_TIME_UTC_mvt"].dt.ordinal_day().fill_null(0)
assert len(day) == d.height
META = ["MVT_ID_mvt", "y", "m", "is_rank", "ADEP_mvt", "fb_pru", "mvt_aobt3", "adsb_linked"]
# identical value semantics to kumo_km_member.py: X = d.select(num).cast(Float32).fill_nan(None); y/off as float; adsb_linked null -> 0
# META columns keep their original dtype under a "meta__" prefix (mvt_aobt3 is ALSO a feature: the feature copy is Float32, the offset stays Float64)
F = d.select([pl.col(c).alias(f"meta__{c}") for c in META]).with_columns(pl.col("meta__ADEP_mvt").cast(pl.Utf8))
F = F.with_columns(pl.Series("_row", np.arange(d.height, dtype=np.int64)), pl.Series("day", day.cast(pl.Int32)))
X = d.select(num).cast(pl.Float32).fill_nan(None)
F = pl.concat([F, X], how="horizontal_extend")
del X, d
print(f"frame {F.height:,} x {F.width} | {time.time()-t0:.0f}s", flush=True)
for a in sorted(F["meta__ADEP_mvt"].unique().to_list()):
    Fa = F.filter(pl.col("meta__ADEP_mvt") == a)  # filter keeps order
    assert (np.diff(Fa["_row"].to_numpy()) > 0).all()
    Fa.write_parquet(f"{OUT}/member/km_{a}.parquet", compression="zstd", compression_level=9, statistics=False)
    print(
        f"  member {a}: {Fa.height:,} rows -> {os.path.getsize(f'{OUT}/member/km_{a}.parquet')/2**20:.0f} MiB",
        flush=True,
    )
del F

# ---- 2. t2x dumps (kumo_t2x_screen2.py dump outputs) stripped to the kumo-step inputs
for f in sorted(glob.glob("scratchpad/xkt/t2x/*_*.npz")):
    b = os.path.basename(f)[:-4]
    if b.startswith(("gK_", "mix_")):
        continue
    z = np.load(f, allow_pickle=True)
    np.savez_compressed(
        f"{OUT}/t2x/t2x_{b}.npz",
        apt=z["apt"].astype("U4"),
        y=z["y"].astype(np.float64),
        v41=z["v41"].astype(np.float64),
        X=z["X"].astype(np.float32),
        cols=np.array(z["cols"]).astype("U32"),
    )
    print(f"  t2x {b}: {len(z['y']):,} x {z['X'].shape[1]}", flush=True)

# ---- 3. parity reference: GPU workstation member output for one (fold, airport) cell
PA = os.environ.get("PARITY_APT", "LSZH")
K = np.load("data/oof/kumo_member_screen_km_janjul.npy")
ids_a = (
    pl.read_parquet(f"{OUT}/member/km_{PA}.parquet", columns=["meta__MVT_ID_mvt"])["meta__MVT_ID_mvt"]
    .to_numpy()
    .astype(float)
)
sel = np.isin(K[0], ids_a)
np.save(f"{OUT}/ref/parity_XKT_KM_janjul_{PA}.npy", K[:, sel])
print(f"  parity ref janjul {PA}: {sel.sum():,} rows", flush=True)

# ---- 4. vendor: sdm site-packages tarball + HF medium weights (public; offline fallback for boxes without GitHub/HF access)
sp = glob.glob(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(os.environ.get("KUMO_PY", ".venv-kumo/bin/python")))),
        "lib",
        "python3*",
        "site-packages",
    )
)[
    0
]  # the kumo env
with tarfile.open(f"{OUT}/vendor/sdm_ee6acd4_site.tgz", "w:gz") as tf:
    tf.add(f"{sp}/sdm", arcname="sdm", filter=lambda ti: None if "__pycache__" in ti.name else ti)
    for di in glob.glob(f"{sp}/structured_data_models-*.dist-info"):
        tf.add(di, arcname=os.path.basename(di))
HF = "artifacts/xkumo/hf/hub/models--nvidia--Kumo-Tabular"
SNAP = "bd7fa122b516c7355583873ffcf38f5e79403ebd"
dst = f"{OUT}/vendor/hf/hub/models--nvidia--Kumo-Tabular"
for rel in [f"{z}/regressor.pt" for z in os.environ.get("VENDOR_SIZES", "medium,large").split(",")] + [
    "LICENSE"
]:  # large = profile l50e16 (863 MB)
    os.makedirs(os.path.dirname(f"{dst}/snapshots/{SNAP}/{rel}"), exist_ok=True)
    shutil.copyfile(
        os.path.realpath(f"{HF}/snapshots/{SNAP}/{rel}"), f"{dst}/snapshots/{SNAP}/{rel}"
    )  # plain files (no symlinks/blobs)
os.makedirs(f"{dst}/refs", exist_ok=True)
open(f"{dst}/refs/v1.0.0", "w").write(SNAP)

# ---- 5. manifest
rows = []
for p in sorted(glob.glob(f"{OUT}/**/*", recursive=True)):
    if os.path.isfile(p) and not p.endswith("MANIFEST_staged.tsv"):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        rows.append(f"{os.path.relpath(p, OUT)}\t{os.path.getsize(p)}\t{h.hexdigest()}")
open(f"{OUT}/MANIFEST_staged.tsv", "w").write("path\tbytes\tmd5\n" + "\n".join(rows) + "\n")
tot = sum(int(r.split("\t")[1]) for r in rows)
print(f"payload {OUT}: {len(rows)} files, {tot/2**30:.2f} GiB | {time.time()-t0:.0f}s\nSTAGE_DONE", flush=True)
