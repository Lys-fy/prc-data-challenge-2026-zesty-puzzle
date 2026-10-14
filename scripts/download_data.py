#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Download the PRC Data Challenge 2026 competition data from the OpenSky MinIO/S3.

Credentials (team access key / secret key, received after team approval):
  1. env vars  PRC_ACCESS_KEY / PRC_SECRET_KEY, or
  2. ~/.config/pyopensky/secret.conf  ([default] access_key = ..., secret_key = ...)

Files with a known md5 (docs/REPRODUCE.md §8) are skipped only when their md5 matches, and the script exits
with an error when a downloaded file does not match. Other files are skipped when their size matches.

Usage:
  uv run scripts/download_data.py            # list buckets + download competition data to data/raw/, final_* objects to data/final/
  uv run scripts/download_data.py --list     # only list what is visible
  uv run scripts/download_data.py --bucket competition-data-2026   # override bucket
"""

import argparse, configparser, hashlib, os, sys
from pathlib import Path

ENDPOINT = "s3.opensky-network.org"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
FINAL = ROOT / "data" / "final"
KNOWN_MD5 = {  # docs/REPRODUCE.md §8 (final_ranking.parquet: the 8 Oct 2026 11:04 UTC re-issue)
    "final_ranking.parquet": "1ce0c6ee9ea15b8c3d0cd282591d4e0a",
    "final_submitting.parquet": "2c1f407296e217fd7d242c9d51ba27ad",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def dest(name):
    """Local path of a bucket object: final_* files to data/final/<basename>, the rest to data/raw/<name>."""
    base = name.rsplit("/", 1)[-1]
    return FINAL / base if base.startswith("final_") else OUT / name


def up_to_date(dst, size):
    """True when dst need not be downloaded: md5 match for files of KNOWN_MD5, size match otherwise."""
    if not dst.exists():
        return False
    want = KNOWN_MD5.get(dst.name)
    return md5(dst) == want if want else dst.stat().st_size == size


def verify(dst):
    want = KNOWN_MD5.get(dst.name)
    if want and md5(dst) != want:
        sys.exit(f"md5 mismatch for {dst}: got {md5(dst)}, expected {want} (docs/REPRODUCE.md §8)")


def creds():
    ak, sk = os.getenv("PRC_ACCESS_KEY"), os.getenv("PRC_SECRET_KEY")
    if ak and sk:
        return ak, sk
    cfg = configparser.ConfigParser()
    cfg.read(Path.home() / ".config/pyopensky/secret.conf")
    if cfg.has_section("default"):
        ak = cfg["default"].get("access_key")
        sk = cfg["default"].get("secret_key")
        if ak and sk:
            return ak, sk
    sys.exit("No credentials. Set PRC_ACCESS_KEY/PRC_SECRET_KEY or fill ~/.config/pyopensky/secret.conf.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--bucket", default=None, help="bucket holding the 2026 data (auto-detect if omitted)")
    a = ap.parse_args()
    from minio import Minio

    ak, sk = creds()
    c = Minio(ENDPOINT, access_key=ak, secret_key=sk, secure=True)
    buckets = [b.name for b in c.list_buckets()]
    print("visible buckets:", buckets)
    if a.list:
        for b in buckets:
            for o in c.list_objects(b, recursive=True):
                print(f"  {b}/{o.object_name}  {o.size / 1e6:.1f} MB")
        return
    bucket = a.bucket
    if bucket is None:
        cands = [b for b in buckets if "2026" in b or "competition" in b]
        # the 2024 'competition-data' bucket is public and huge (158 GB) -> avoid it
        cands = [b for b in cands if b != "competition-data"] or cands
        if not cands:
            sys.exit("Could not guess the bucket, pass --bucket. Visible: %s" % buckets)
        bucket = cands[0]
    OUT.mkdir(parents=True, exist_ok=True)
    objs = [o for o in c.list_objects(bucket, recursive=True)]
    print(f"downloading {len(objs)} objects from {bucket} -> {OUT}, {FINAL}")
    for o in objs:
        if not (o.object_name.endswith(".parquet") or o.object_name.endswith(".csv")):
            continue
        dst = dest(o.object_name)
        if up_to_date(dst, o.size):
            print("  skip", o.object_name)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {o.object_name} ({o.size / 1e6:.1f} MB) -> {dst.relative_to(ROOT)}")
        c.fget_object(bucket, o.object_name, str(dst))
        verify(dst)
    print("done")


if __name__ == "__main__":
    main()
