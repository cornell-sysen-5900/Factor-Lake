#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/build_dollar_volume.py
PURPOSE: Builds a WRDS folder's dollar_volume_monthly.parquet (the liquidity limit's volume) from its S3 daily prices
         and uploads it to that folder. Dry run unless --upload. Only adds that file; never changes another one.

Usage (from the repo root; the AWS keys come from .streamlit/secrets.toml, which is local only):
    uv run python scripts/build_dollar_volume.py                          # the code's folder, dry run
    uv run python scripts/build_dollar_volume.py --upload
    uv run python scripts/build_dollar_volume.py --data-dir wrds_2026_11 --upload

The file is (date, permno, dollar_vol_3m), built by src/wrds_liquidity.read_dollar_volume from
s3://<bucket>/factor-lake-data/<data dir>/prices_daily/: each stock's mean daily price x share volume over its last
63 trading days (80% needed), at each month end. The backtest Lambda reads it for jobs with a liquidity limit
(src/wrds_service.py). If the folder already has the file, the script stops without building or uploading. After
the upload it reads the object back and checks its size and ETag (the MD5 of the body, for a single-part upload).

This is a stopgap for wrds_2026_11: at the next data refresh s08_export should write the file with the other
outputs, and this script can be retired (pipeline/PLAN.md, pipeline/DATA_UPDATE_GUIDE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.upload_wrds import s3_client  # noqa: E402
from src.s3_client import DEFAULT_BUCKET, DEFAULT_PREFIX, S3DataManager  # noqa: E402
from src.wrds_data import DOLLAR_VOLUME_FILE, wrds_dir  # noqa: E402
from src.wrds_liquidity import daily_years, read_dollar_volume  # noqa: E402

FOLDER = re.compile(r'^wrds_\d{4}_\d{2}$')
MISSING = ('404', 'NoSuchKey', 'NotFound')


def exists(s3, bucket: str, key: str) -> bool:
    """True when the object exists, False when S3 says it does not; other errors are raised."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        if str(getattr(e, 'response', {}).get('Error', {}).get('Code', '')) in MISSING:
            return False
        raise


def build(manager: S3DataManager, data_dir: str) -> bytes:
    """The Parquet bytes of the folder's dollar volume table."""
    read_bytes = lambda filename: manager.read_bytes(f'{data_dir}/{filename}')  # noqa: E731
    table = read_dollar_volume(read_bytes, daily_years(read_bytes))
    body = io.BytesIO()
    table.to_parquet(body, index=False, compression='zstd')
    print(f"  {len(table):,} rows, {table['date'].min().date()} to {table['date'].max().date()}, "
          f"{table['permno'].nunique():,} permnos")
    return body.getvalue()


def main(argv: list[str] | None = None, s3=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data-dir', default=None, help="WRDS folder under the prefix (default: the code's, "
                                                         "src/wrds_data.wrds_dir())")
    parser.add_argument('--bucket', default=DEFAULT_BUCKET)
    parser.add_argument('--upload', action='store_true', help="upload the file (default: build it, upload nothing)")
    parser.add_argument('--secrets', type=Path, default=PROJECT_ROOT / '.streamlit' / 'secrets.toml')
    args = parser.parse_args(argv)
    data_dir = (args.data_dir or wrds_dir()).strip('/')
    if not FOLDER.match(data_dir):
        sys.exit(f"Refusing {data_dir!r}: only a WRDS folder (wrds_YYYY_MM) gets this file.")
    if s3 is None:
        s3 = s3_client(args.secrets)
    manager = S3DataManager(bucket=args.bucket, prefix=DEFAULT_PREFIX, client=s3)
    key = manager.object_key(f'{data_dir}/{DOLLAR_VOLUME_FILE}')

    if exists(s3, args.bucket, key):
        print(f"s3://{args.bucket}/{key} already exists; nothing built or uploaded (this script never replaces it).")
        return 1
    print(f"Building {DOLLAR_VOLUME_FILE} from s3://{args.bucket}/{DEFAULT_PREFIX}/{data_dir}/prices_daily/")
    started = time.perf_counter()
    body = build(manager, data_dir)
    md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()
    print(f"  {len(body):,} bytes, MD5 {md5}, built in {time.perf_counter() - started:.0f} s")
    if not args.upload:
        print("Dry run: nothing uploaded (add --upload).")
        return 0

    s3.put_object(Bucket=args.bucket, Key=key, Body=body, ContentType='application/vnd.apache.parquet')
    head = s3.head_object(Bucket=args.bucket, Key=key)
    size, etag = int(head['ContentLength']), head['ETag'].strip('"')
    print(f"Uploaded s3://{args.bucket}/{key}: {size:,} bytes, ETag {etag}")
    if size != len(body):
        print(f"FAIL: S3 has {size:,} bytes, the file has {len(body):,}.")
        return 2
    if etag != md5:          # e.g. KMS encryption: the ETag is not the MD5, so compare the content
        if s3.get_object(Bucket=args.bucket, Key=key)['Body'].read() != body:
            print("FAIL: the object read back differs from the file.")
            return 2
        print("Check: size matches; the ETag is not the MD5 (encryption), and the content read back matches.")
    else:
        print("Check: size and ETag (MD5) match.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
