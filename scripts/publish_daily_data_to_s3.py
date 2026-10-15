#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/publish_daily_data_to_s3.py
PURPOSE: Cleans the daily CRSP stock file (russell2000_cleaned.csv) and publishes it to S3 as yearly Parquet files.

Usage (from the project root, with AWS keys in .env or the environment):
    uv run python scripts/publish_daily_data_to_s3.py ~/Desktop/Cornell/Factor-Lake/data/archive/russell2000_cleaned.csv

    # Clean and check everything, keeping the Parquet files locally, without uploading
    uv run python scripts/publish_daily_data_to_s3.py russell2000_cleaned.csv --dry-run --output-dir daily_parquet

Output in s3://<bucket>/<prefix>/ (default s3://sysen-5900-factor-lake/factor-lake-data/):
    daily_prices/year=YYYY/data.parquet   one file per year, sorted by security and date
    daily_last_prices.parquet             each security's last trading day and price (delistings)
    daily_prices/_manifest.json           years, row counts, columns and cleaning summary (uploaded last)

Cleaning, matching the PR #21 daily loader and the yearly data:
    - Column names are matched case-insensitively (date, permno, ticker, prc, cfacpr, shrout, ret, vol).
    - Text placeholders ('--', 'N/A', ...) and CRSP letter codes in numeric columns become missing.
    - Rows without a date, a price, or an identifier (permno or ticker) are dropped. A price of 0
      (CRSP's "no price available") counts as missing.
    - prc = |prc| (CRSP marks bid/ask midpoints with a negative price).
    - prc_adj = prc / cfacpr (split-adjusted price; prc when cfacpr is missing or 0).
    - mkt_cap = prc * shrout (CRSP shrout is in thousands of shares, so mkt_cap is in $ thousands).
    - Duplicate security/date rows keep the first occurrence.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Allow `python scripts/publish_daily_data_to_s3.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.s3_client import (  # noqa: E402
    DAILY_LAST_PRICES_FILE,
    DAILY_MANIFEST_FILE,
    S3DataManager,
    daily_year_file,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CSV header (lower-cased) -> output column
COLUMN_ALIASES = {
    'date': 'date', 'permno': 'permno', 'ticker': 'ticker', 'tic': 'ticker',
    'prc': 'prc', 'cfacpr': 'cfacpr', 'shrout': 'shrout', 'ret': 'ret', 'vol': 'vol',
}
NUMERIC_COLUMNS = ['prc', 'cfacpr', 'shrout', 'ret', 'vol']
SENTINELS = ['--', 'N/A', '#N/A', 'NULL', 'null', 'nan', 'NaN', '']
OUTPUT_COLUMNS = ['permno', 'ticker', 'date', 'prc', 'cfacpr', 'prc_adj', 'shrout', 'mkt_cap', 'ret', 'vol']
SCHEMA = pa.schema([
    ('permno', pa.int64()), ('ticker', pa.string()), ('date', pa.timestamp('ns')),
    ('prc', pa.float64()), ('cfacpr', pa.float64()), ('prc_adj', pa.float64()),
    ('shrout', pa.float64()), ('mkt_cap', pa.float64()), ('ret', pa.float64()), ('vol', pa.float64()),
])
PARQUET_TYPE = 'application/vnd.apache.parquet'


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parses YYYY-MM-DD quickly, then YYYYMMDD (WRDS exports), then any other format."""
    dates = pd.to_datetime(raw, format='%Y-%m-%d', errors='coerce')
    for fmt in ('%Y%m%d', 'mixed'):
        retry = dates.isna() & raw.notna()
        if not retry.any():
            break
        dates.loc[retry] = pd.to_datetime(raw[retry], format=fmt, errors='coerce')
    return dates


def clean_chunk(chunk: pd.DataFrame, counts: Dict[str, int]) -> pd.DataFrame:
    """
    Cleans one block of CSV rows (all read as text) into the output columns.
    Adds the number of rows dropped for each reason to `counts`.
    """
    renamed = {c: COLUMN_ALIASES[c.strip().lower()] for c in chunk.columns if c.strip().lower() in COLUMN_ALIASES}
    df = chunk[list(renamed)].rename(columns=renamed)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    missing = [c for c in ('date', 'prc') if c not in df.columns]
    if missing or not ({'permno', 'ticker'} & set(df.columns)):
        raise ValueError(f"The CSV needs date, prc and permno or ticker columns; missing {missing or ['permno/ticker']}. "
                         f"Found: {list(chunk.columns)}")

    df = df.replace(SENTINELS, np.nan)
    out = pd.DataFrame(index=df.index)
    out['date'] = _parse_dates(df['date'])
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else np.nan
    out['permno'] = pd.to_numeric(df['permno'], errors='coerce').astype('Int64') if 'permno' in df.columns \
        else pd.Series(pd.NA, index=df.index, dtype='Int64')
    out['ticker'] = (df['ticker'].astype('string').str.strip().str.upper().replace('', pd.NA)
                     if 'ticker' in df.columns else pd.Series(pd.NA, index=df.index, dtype='string'))
    out.loc[out['prc'] == 0, 'prc'] = np.nan

    total = len(out)
    out = out[out['date'].notna()]
    counts['dropped_missing_date'] += total - len(out)
    before = len(out)
    out = out[out['prc'].notna()]
    counts['dropped_missing_price'] += before - len(out)
    before = len(out)
    key = 'permno' if 'permno' in df.columns else 'ticker'
    out = out[out[key].notna()]
    counts['dropped_missing_id'] += before - len(out)

    out['prc'] = out['prc'].abs()
    usable_factor = out['cfacpr'].where(out['cfacpr'] > 0)
    out['prc_adj'] = (out['prc'] / usable_factor).fillna(out['prc'])
    out['mkt_cap'] = out['prc'] * out['shrout']
    return out[OUTPUT_COLUMNS]


def build_daily_dataset(csv_path: Path, output_dir: Path, chunk_rows: int = 500_000) -> Dict[str, Any]:
    """
    Cleans the CSV in chunks into output_dir/daily_prices/year=YYYY/data.parquet plus
    output_dir/daily_last_prices.parquet, and returns the manifest describing them.
    """
    counts = {'rows_read': 0, 'dropped_missing_date': 0, 'dropped_missing_price': 0,
              'dropped_missing_id': 0, 'dropped_duplicates': 0}
    staging = output_dir / '_staging'
    staging.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(csv_path, nrows=0).columns
    ignored = [c for c in header if c.strip().lower() not in COLUMN_ALIASES]
    key = 'permno' if any(c.strip().lower() == 'permno' for c in header) else 'ticker'

    # Pass 1: clean each chunk and stage its rows by year
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_rows, dtype=str, keep_default_na=False)):
        counts['rows_read'] += len(chunk)
        cleaned = clean_chunk(chunk, counts)
        for year, rows in cleaned.groupby(cleaned['date'].dt.year):
            part_dir = staging / f'year={int(year)}'
            part_dir.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pandas(rows, schema=SCHEMA, preserve_index=False),
                           part_dir / f'part-{i:05d}.parquet')
        logger.info(f"Read {counts['rows_read']:,} rows")

    # Pass 2: one sorted, de-duplicated file per year
    years: Dict[str, Dict[str, int]] = {}
    last_rows: List[pd.DataFrame] = []
    for part_dir in sorted(staging.glob('year=*'), key=lambda p: int(p.name.split('=')[1])):
        year = int(part_dir.name.split('=')[1])
        df = pq.read_table(part_dir, schema=SCHEMA).to_pandas()
        before = len(df)
        df = df.drop_duplicates(subset=[key, 'date'], keep='first').sort_values([key, 'date'], kind='stable')
        counts['dropped_duplicates'] += before - len(df)
        target = output_dir / daily_year_file(year)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), schema=SCHEMA, preserve_index=False),
                       target, compression='zstd', row_group_size=250_000)
        if pq.read_table(target).num_rows != len(df):   # the written file must load back intact
            raise RuntimeError(f"Verification failed for {target}")
        years[str(year)] = {'rows': len(df), 'bytes': target.stat().st_size, 'securities': int(df[key].nunique())}
        last_rows.append(df.groupby(key, sort=False).tail(1))
        shutil.rmtree(part_dir)
        logger.info(f"{year}: {len(df):,} rows, {years[str(year)]['securities']:,} securities")
    shutil.rmtree(staging)
    if not years:
        raise ValueError("No usable rows: every row was missing a date, price or identifier.")

    # Each security's last trading day, like last_price_mapping for the yearly data
    last = pd.concat(last_rows).sort_values('date').groupby(key, sort=False).tail(1)
    last_date = last['date'].max()
    last_prices = pd.DataFrame({
        'permno': last['permno'], 'ticker': last['ticker'], 'last_date': last['date'],
        'last_prc': last['prc'], 'last_prc_adj': last['prc_adj'],
        # Stopped trading more than a week before the data ends
        'delisted': last['date'] < last_date - pd.Timedelta(days=7),
    }).sort_values([key]).reset_index(drop=True)
    last_prices.to_parquet(output_dir / DAILY_LAST_PRICES_FILE, index=False)

    first_year, final_year = min(years, key=int), max(years, key=int)
    first_day = pq.read_table(output_dir / daily_year_file(int(first_year)), columns=['date']).column('date')
    final_day = pq.read_table(output_dir / daily_year_file(int(final_year)), columns=['date']).column('date')
    return {
        'created_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'source_file': csv_path.name,
        'source_sha256': _sha256(csv_path),
        'security_key': key,
        'date_range': [str(pc.min(first_day).as_py().date()), str(pc.max(final_day).as_py().date())],
        'rows_written': sum(y['rows'] for y in years.values()),
        'securities': int(last_prices[key].nunique()),
        'delisted_securities': int(last_prices['delisted'].sum()),
        'years': years,
        'columns': {f.name: str(f.type) for f in SCHEMA},
        'ignored_csv_columns': ignored,
        'cleaning': counts,
        'units': {'mkt_cap': '$ thousands (prc x CRSP shrout, which is in thousands of shares)',
                  'ret': 'daily simple return (decimal)'},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def publish(output_dir: Path, manifest: Dict[str, Any], manager: S3DataManager) -> None:
    """
    Uploads the yearly files and the last-price file, checks each upload's size, writes the
    manifest last, then reads the smallest year back from S3 to prove the data loads.
    """
    files = [daily_year_file(int(y)) for y in manifest['years']] + [DAILY_LAST_PRICES_FILE]
    for filename in files:
        local = output_dir / filename
        manager.upload_file(str(local), filename, PARQUET_TYPE)
        if manager.object_size(filename) != local.stat().st_size:
            raise RuntimeError(f"Upload size mismatch for {manager.s3_uri(filename)}")
        logger.info(f"Uploaded {manager.s3_uri(filename)} ({local.stat().st_size / 1e6:.1f} MB)")

    manifest_path = output_dir / DAILY_MANIFEST_FILE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manager.upload_file(str(manifest_path), DAILY_MANIFEST_FILE, 'application/json')

    check_year = min(manifest['years'], key=lambda y: manifest['years'][y]['rows'])
    rows_back = len(manager.read_daily_prices(years=[int(check_year)]))
    if rows_back != manifest['years'][check_year]['rows']:
        raise RuntimeError(f"Read-back check failed for {check_year}: {rows_back} rows")
    logger.info(f"Read {check_year} back from S3 ({rows_back:,} rows): the data loads.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clean the daily CRSP CSV and publish it to S3 as yearly Parquet.")
    parser.add_argument('csv', type=Path, help="Path to russell2000_cleaned.csv")
    parser.add_argument('--output-dir', type=Path, help="Keep the Parquet files here (default: a temporary folder)")
    parser.add_argument('--dry-run', action='store_true', help="Clean and check without uploading")
    parser.add_argument('--bucket', help="S3 bucket (default: FACTOR_LAKE_S3_BUCKET or sysen-5900-factor-lake)")
    parser.add_argument('--prefix', help="Key prefix (default: FACTOR_LAKE_S3_PREFIX or factor-lake-data)")
    parser.add_argument('--chunk-rows', type=int, default=500_000, help="CSV rows per chunk (default 500,000)")
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        parser.error(f"CSV not found: {args.csv}")

    with tempfile.TemporaryDirectory(prefix='factor_lake_daily_') as tmp:
        output_dir = args.output_dir or Path(tmp)
        manifest = build_daily_dataset(args.csv, output_dir, args.chunk_rows)
        c = manifest['cleaning']
        logger.info(f"Cleaned {c['rows_read']:,} rows into {manifest['rows_written']:,} "
                    f"({manifest['securities']:,} securities, {manifest['date_range'][0]} to {manifest['date_range'][1]}). "
                    f"Dropped: {c['dropped_missing_date']:,} no date, {c['dropped_missing_price']:,} no price, "
                    f"{c['dropped_missing_id']:,} no identifier, {c['dropped_duplicates']:,} duplicates.")
        if manifest['ignored_csv_columns']:
            logger.info(f"Columns not published: {manifest['ignored_csv_columns']}")
        if args.dry_run:
            (output_dir / DAILY_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
            logger.info(f"Dry run: nothing uploaded. Files are in {output_dir}"
                        + ("" if args.output_dir else " (deleted on exit; use --output-dir to keep them)"))
            return 0
        publish(output_dir, manifest, S3DataManager(bucket=args.bucket, prefix=args.prefix))
    logger.info("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
