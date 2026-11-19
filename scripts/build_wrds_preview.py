"""Build a versioned WRDS preview index. Dry run unless --upload is supplied.

Only the preview index is written. Raw data, saved analyses and existing indexes
are never overwritten. This does NOT enable sample display in Lambda or the app.
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.s3_client import S3DataManager  # noqa: E402
from src.wrds_data import PANEL_FILE, wrds_dir  # noqa: E402
from src.wrds_preview import PREVIEW_COLUMNS, build_index, encode_index, index_filename  # noqa: E402


def build(manager, data_dir):
    key = manager.object_key(f'{data_dir}/{PANEL_FILE}')
    head = manager.client.head_object(Bucket=manager.bucket, Key=key)
    etag = head['ETag'].strip('"')
    # Stream the source to disk, then read projected Arrow batches. It never resides
    # in app memory; this separate preparation command can run before deployment.
    response = manager.client.get_object(Bucket=manager.bucket, Key=key, IfMatch=head['ETag'])
    stream = response['Body']
    try:
        with tempfile.TemporaryFile() as tmp:
            while chunk := stream.read(1024 * 1024):
                tmp.write(chunk)
            tmp.seek(0)
            parquet = pq.ParquetFile(tmp)
            index = build_index(parquet.iter_batches(batch_size=16384, columns=list(PREVIEW_COLUMNS)))
    finally:
        stream.close()
    current = manager.client.head_object(Bucket=manager.bucket, Key=key)['ETag'].strip('"')
    if current != etag:
        raise ValueError('The panel changed during preparation; rebuild the preview.')
    return index_filename(data_dir, etag), encode_index(index, etag)


def main(argv=None, manager=None):
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=wrds_dir())
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'wrds_\d{4}_\d{2}', args.data_dir):
        parser.error('Use a versioned wrds_YYYY_MM folder.')
    manager = manager or S3DataManager()
    filename, body = build(manager, args.data_dir)
    if args.upload:
        manager.client.put_object(Bucket=manager.bucket, Key=manager.object_key(filename), Body=body,
                                  ContentType='application/vnd.apache.parquet', IfNoneMatch='*')
        check = manager.read_bytes(filename)
        if check != body:
            raise RuntimeError('The uploaded preview index failed verification.')
        print('Preview index uploaded and verified. Sample display remains controlled by the approval gate.')
    else:
        print('Preview index built. Dry run: nothing uploaded and sample display was not enabled.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
