#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/upload_wrds.py
PURPOSE: Uploads the WRDS pipeline's processed/final/ folder to S3: the Parquet files with their relative paths,
         then prices_daily/_manifest.json last. Dry run unless --upload. Never deletes anything.

Usage (from the repo root; the AWS keys come from .streamlit/secrets.toml, which is local only):
    uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed/final"            # dry run
    uv run python scripts/upload_wrds.py "<final>" --offline                               # no keys, no network
    uv run python scripts/upload_wrds.py "<final>" --upload
    uv run python scripts/upload_wrds.py "<final>" --check

The target is s3://<bucket>/factor-lake-data/wrds_YYYY_MM/ from final/manifest.json (--prefix to change it). The
script refuses any prefix outside factor-lake-data/wrds_* or inside the live app's files. Objects whose size and
ETag already match are skipped; an object that differs stops the run unless --overwrite. Any AWS error stops it with
the error and the file. --check compares Parquet files by size and ETag and JSON files by content (created_at
ignored), and downloads the daily manifest and the smallest Parquet file back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

try:
    import tomllib                                   # Python 3.11+
except ImportError:
    import tomli as tomllib                          # Python 3.10: installed with the repo's dev group

APP_PREFIX = "factor-lake-data"
APP = (f"{APP_PREFIX}/full_precision_test.parquet", f"{APP_PREFIX}/last_price_mapping.parquet",
       f"{APP_PREFIX}/daily_last_prices.parquet", f"{APP_PREFIX}/daily_prices/", f"{APP_PREFIX}/backtest-results/",
       f"{APP_PREFIX}/archive/", f"{APP_PREFIX}/_deploy/")   # what the live app and its scripts read or write
CHUNK = 64 * 1024 * 1024                             # multipart part size; also predicts the S3 ETag
TYPES = {".parquet": "application/vnd.apache.parquet", ".json": "application/json"}
DAILY_MANIFEST = "prices_daily/_manifest.json"


def s3_client(secrets: Path):
    """boto3 S3 client from the Streamlit secrets (top-level AWS_* keys, or a section such as [aws]). The values are
    never printed."""
    import boto3
    with open(secrets, "rb") as f:
        s = tomllib.load(f)
    found = {k: v for k, v in s.items() if isinstance(v, str)}
    for table in (v for v in s.values() if isinstance(v, dict)):
        found.update({k.upper(): v for k, v in table.items() if isinstance(v, str)})
    missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not found.get(k)]
    if missing:
        sys.exit(f"Not in {secrets}: {', '.join(missing)}")
    return boto3.client("s3", aws_access_key_id=found["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=found["AWS_SECRET_ACCESS_KEY"],
                        region_name=found.get("AWS_DEFAULT_REGION") or "us-east-1")


def etag(path: Path) -> str:
    """The ETag S3 gives this file when uploaded with CHUNK-sized parts (plain MD5 below one part)."""
    parts = []
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            parts.append(hashlib.md5(block).digest())
    if path.stat().st_size < CHUNK:
        return parts[0].hex() if parts else hashlib.md5(b"").hexdigest()
    return f"{hashlib.md5(b''.join(parts)).hexdigest()}-{len(parts)}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def without_created_at(obj):
    if isinstance(obj, dict):
        return {k: without_created_at(v) for k, v in obj.items() if k != "created_at"}
    return obj


def check_prefix(prefix: str) -> None:
    if not prefix.startswith(f"{APP_PREFIX}/wrds_") or any(prefix.startswith(a.rstrip("/")) for a in APP):
        sys.exit(f"Refusing prefix {prefix!r}: it must be a new {APP_PREFIX}/wrds_* folder, apart from the app's files")


def local_files(final: Path, prefix: str) -> dict[str, Path]:
    """S3 key -> local file: every Parquet file, then the daily manifest last."""
    local = {f"{prefix}/{p.relative_to(final).as_posix()}": p for p in sorted(final.rglob("*.parquet"))}
    manifest = final / DAILY_MANIFEST
    if not manifest.is_file():
        sys.exit(f"{manifest} is missing; rerun s08")
    local[f"{prefix}/{DAILY_MANIFEST}"] = manifest
    return local


def remote_objects(s3, bucket: str, prefix: str) -> dict[str, tuple[int, str]]:
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        out.update({o["Key"]: (o["Size"], o["ETag"].strip('"')) for o in page.get("Contents", [])})
    return out


def same_json(s3, bucket: str, key: str, path: Path) -> bool:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return without_created_at(json.loads(body)) == without_created_at(json.loads(path.read_text(encoding="utf-8")))


def check(s3, bucket: str, local: dict[str, Path], remote: dict) -> bool:
    """Parquet by size and ETag, JSON by content without created_at; the daily manifest and the smallest Parquet
    file downloaded back (sha256 for Parquet). Prints one line per key."""
    ok_keys = set()
    for key, path in local.items():
        if key not in remote:
            status = "MISSING"
        elif path.suffix == ".json":
            status = "ok" if same_json(s3, bucket, key, path) else "DIFFERS"
        else:
            status = "ok" if remote[key] == (path.stat().st_size, etag(path)) else "DIFFERS"
        ok_keys |= {key} if status == "ok" else set()
        print(f"{status:8} s3://{bucket}/{key} ({path.stat().st_size / 1e6:,.1f} MB)")
    extra = sorted(set(remote) - set(local))
    for key in extra:
        print(f"{'EXTRA':8} s3://{bucket}/{key}")
    parquet = [k for k in ok_keys if k.endswith(".parquet")]
    good = len(ok_keys) == len(local) and not extra and bool(parquet)
    if parquet:
        small = min(parquet, key=lambda k: local[k].stat().st_size)
        with tempfile.TemporaryDirectory() as tmp:
            back = Path(tmp) / "back.parquet"
            s3.download_file(bucket, small, str(back))
            sha_ok = sha256(back) == sha256(local[small])
        good &= sha_ok
        print(f"downloaded {small} back: sha256 {'matches' if sha_ok else 'DIFFERS'}")
    manifest = next(k for k in local if k.endswith(DAILY_MANIFEST))
    if manifest in remote:
        ok = same_json(s3, bucket, manifest, local[manifest])
        good &= ok
        print(f"downloaded {manifest} back: content {'matches' if ok else 'DIFFERS'} (created_at ignored)")
    mb = sum(remote[k][0] for k in ok_keys) / 1e6
    print(f"{len(ok_keys)} of {len(local)} files match (Parquet: size and ETag; JSON: content), {mb:,.0f} MB; "
          f"{len(extra)} objects not local")
    return good


def upload(s3, bucket: str, local: dict[str, Path], remote: dict, really: bool, overwrite: bool) -> None:
    """Every Parquet file, then the daily manifest (local's last item) once all of them are verified."""
    from boto3.exceptions import S3UploadFailedError
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import BotoCoreError, ClientError

    def matches(key, path):
        if key not in remote:
            return False
        if path.suffix == ".json":
            return same_json(s3, bucket, key, path)
        return remote[key] == (path.stat().st_size, etag(path))
    same = {k for k, p in local.items() if matches(k, p)}
    differ = sorted(k for k in local if k in remote and k not in same)
    if differ and not overwrite:
        sys.exit(f"{len(differ)} objects in the prefix differ from the local files (e.g. {differ[0]}); "
                 "rerun with --overwrite to replace them")
    config = TransferConfig(multipart_threshold=CHUNK, multipart_chunksize=CHUNK)
    for key, path in local.items():
        size = path.stat().st_size
        if key in same:
            print(f"skip      s3://{bucket}/{key} (already the same)")
        elif not really:
            print(f"would upload s3://{bucket}/{key} ({size / 1e6:,.1f} MB)")
        else:
            try:
                s3.upload_file(str(path), bucket, key, Config=config, ExtraArgs={"ContentType": TYPES[path.suffix]})
                head = s3.head_object(Bucket=bucket, Key=key)
            except (ClientError, S3UploadFailedError, BotoCoreError) as e:   # stop: no retry another way
                sys.exit(f"FAILED on {path.name} -> s3://{bucket}/{key}: {type(e).__name__}: {e}")
            if head["ContentLength"] != size or head["ETag"].strip('"') != etag(path):
                sys.exit(f"Size or ETag mismatch after uploading {key}: S3 {head['ContentLength']} bytes, "
                         f"ETag {head['ETag']}; local {size} bytes, ETag {etag(path)}")
            print(f"uploaded  s3://{bucket}/{key} ({size / 1e6:,.1f} MB)")


def main(argv: list[str] | None = None, s3=None) -> int:
    parser = argparse.ArgumentParser(description="Upload processed/final/ to S3 (dry run unless --upload).")
    parser.add_argument("final_dir", type=Path, help="the processed/final folder")
    parser.add_argument("--prefix", help="target prefix (default: from final/manifest.json, factor-lake-data/wrds_YYYY_MM)")
    parser.add_argument("--secrets", type=Path, default=Path(__file__).resolve().parents[1] / ".streamlit" /
                        "secrets.toml", help="Streamlit secrets with the AWS keys (default: the repo's)")
    parser.add_argument("--upload", action="store_true", help="really upload (default: dry run)")
    parser.add_argument("--check", action="store_true", help="compare the S3 folder with the local files")
    parser.add_argument("--overwrite", action="store_true", help="replace objects in the prefix that differ")
    parser.add_argument("--offline", action="store_true", help="dry run without credentials or network: list the keys")
    args = parser.parse_args(argv)
    if args.offline and (args.upload or args.check):
        sys.exit("--offline is a dry run only")
    final = args.final_dir.resolve()
    if not (final / "manifest.json").is_file():
        sys.exit(f"{final} has no manifest.json; pass the processed/final folder")
    target = json.loads((final / "manifest.json").read_text(encoding="utf-8"))["s3_target"]
    bucket, default_prefix = target.removeprefix("s3://").split("/", 1)
    prefix = (args.prefix or default_prefix).strip("/")
    check_prefix(prefix)
    local = local_files(final, prefix)

    if args.offline:
        for key, path in local.items():
            print(f"would upload s3://{bucket}/{key} ({path.stat().st_size / 1e6:,.1f} MB)")
        print(f"{len(local)} files, {sum(p.stat().st_size for p in local.values()) / 1e6:,.0f} MB (offline dry run)")
        return 0
    s3 = s3 or s3_client(args.secrets)
    remote = remote_objects(s3, bucket, prefix)
    if args.check:
        return 0 if check(s3, bucket, local, remote) else 1
    upload(s3, bucket, local, remote, args.upload, args.overwrite)
    print(f"{len(local)} files, {sum(p.stat().st_size for p in local.values()) / 1e6:,.0f} MB"
          + ("" if args.upload else " (dry run: nothing uploaded)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
