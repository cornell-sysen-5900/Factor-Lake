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
    daily_last_prices.parquet             each security's last trading day and price, CRSP delisting
                                          code/return, delisted (code >= 200) and ended_early (rows stop
                                          more than 7 days before the data ends)
    daily_prices/_manifest.json           years, row counts, columns and cleaning summary (uploaded last)

Columns kept (CSV names are matched case-insensitively; anything else is listed and skipped):
    identifiers  permno, permco, ticker, cusip, comnam
    codes        shrcd, exchcd, siccd, naics, dlstcd
    prices       prc, cfacpr, openprc, askhi, bidlo, bid, ask
    other        shrout, vol, ret, dlret, shares_held

Cleaning, matching the PR #21 daily loader and the yearly data:
    - Text placeholders ('--', 'N/A', ...) and CRSP letter codes in numeric columns become missing.
    - CRSP's numeric missing codes count as missing too: returns or delisting returns below -100%
      (-55, -66, -77, -88, -99), negative volume (-99) and non-positive shares outstanding.
    - Rows without a date, a price, or an identifier (permno or ticker) are dropped. A price of 0
      (CRSP's "no price available") counts as missing; so does 0 for openprc/askhi/bidlo/bid/ask.
    - prc = |prc| (CRSP marks bid/ask midpoints with a negative price). askhi and bidlo are made
      positive the same way (negative there means a bid/ask quote, not a traded high or low).
    - prc_adj = prc / cfacpr (split-adjusted price; prc when cfacpr is missing or 0).
    - mkt_cap = prc * shrout (CRSP shrout is in thousands of shares, so mkt_cap is in $ thousands).
    - ret_total = ret combined with the delisting return on each security's last trading day:
      (1 + ret) * (1 + dlret) - 1, or dlret alone when ret is missing that day. The delisting
      return is taken from the security's latest row that has one, even if that row has no price,
      and is applied once (CRSP exports may repeat dlret on every row of a security). A delisting
      code or return dated before the security's last price is ignored (an earlier company that
      used the same ticker), and so is one for a ticker several securities share on its last day.
    - Duplicate permno/date rows keep the first occurrence. Without a permno column only exact
      duplicate rows are dropped, because share classes can share a ticker; the number of
      ticker/date pairs that still have several rows is reported.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

FLOAT_COLUMNS = ['prc', 'cfacpr', 'openprc', 'askhi', 'bidlo', 'bid', 'ask',
                 'shrout', 'vol', 'ret', 'dlret', 'shares_held']
CODE_COLUMNS = ['permno', 'permco', 'shrcd', 'exchcd', 'siccd', 'dlstcd']   # whole-number CRSP codes
TEXT_COLUMNS = ['ticker', 'cusip', 'comnam', 'naics']                        # kept as text (leading zeros)
QUOTE_COLUMNS = ['openprc', 'askhi', 'bidlo', 'bid', 'ask']                  # CRSP uses 0 for missing
# CSV header (lower-cased) -> output column
COLUMN_ALIASES = {c: c for c in ['date'] + FLOAT_COLUMNS + CODE_COLUMNS + TEXT_COLUMNS}
COLUMN_ALIASES['tic'] = 'ticker'
SENTINELS = ['--', 'N/A', '#N/A', 'NULL', 'null', 'nan', 'NaN', '']
OUTPUT_COLUMNS = ['permno', 'permco', 'ticker', 'cusip', 'comnam', 'date', 'shrcd', 'exchcd', 'siccd', 'naics',
                  'prc', 'cfacpr', 'prc_adj', 'openprc', 'askhi', 'bidlo', 'bid', 'ask', 'shrout', 'mkt_cap',
                  'vol', 'ret', 'dlret', 'dlstcd', 'ret_total', 'shares_held']
SCHEMA = pa.schema(
    [(c, pa.int64()) if c in CODE_COLUMNS else
     (c, pa.string()) if c in TEXT_COLUMNS else
     (c, pa.timestamp('ns')) if c == 'date' else
     (c, pa.float64())
     for c in OUTPUT_COLUMNS])
PARQUET_TYPE = 'application/vnd.apache.parquet'
DELISTED_FROM_CODE = 200   # CRSP DLSTCD: 100 = still active, 200 and above = delisted


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parses YYYY-MM-DD quickly, then YYYYMMDD (WRDS exports), then any other format."""
    dates = pd.to_datetime(raw, format='%Y-%m-%d', errors='coerce')
    for fmt in ('%Y%m%d', 'mixed'):
        retry = dates.isna() & raw.notna()
        if not retry.any():
            break
        dates.loc[retry] = pd.to_datetime(raw[retry], format=fmt, errors='coerce')
    return dates


def _check_columns(columns, found) -> str:
    """Returns the security key ('permno' or 'ticker'), or explains which required columns are missing."""
    names = {COLUMN_ALIASES.get(str(c).strip().lower()) for c in columns}
    missing = [c for c in ('date', 'prc') if c not in names]
    if missing or not ({'permno', 'ticker'} & names):
        raise ValueError(f"The CSV needs date, prc and permno or ticker columns; missing {missing or ['permno/ticker']}. "
                         f"Found: {list(found)}")
    return 'permno' if 'permno' in names else 'ticker'


def clean_chunk(chunk: pd.DataFrame, counts: Dict[str, int],
                start: Optional[date] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans one block of CSV rows (all read as text) into the output columns.
    Adds the number of rows dropped for each reason to `counts`.

    Returns (cleaned rows, delisting rows). The delisting rows are every dated, identified row
    that carries a delisting code or return, including rows later dropped for having no price.
    """
    key = _check_columns(chunk.columns, chunk.columns)
    renamed = {c: COLUMN_ALIASES[c.strip().lower()] for c in chunk.columns if c.strip().lower() in COLUMN_ALIASES}
    df = chunk[list(renamed)].rename(columns=renamed)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df = df.mask(df.isin(SENTINELS))   # placeholders -> missing (all columns are text here)

    out = pd.DataFrame(index=df.index)
    out['date'] = _parse_dates(df['date'])
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else np.nan
    for col in CODE_COLUMNS:
        values = pd.to_numeric(df[col], errors='coerce') if col in df.columns else pd.Series(np.nan, index=df.index)
        out[col] = values.where(values == values.round()).astype('Int64')   # 'Z' or 3.5 -> missing
    for col in TEXT_COLUMNS:
        text = df[col].astype('string').str.strip() if col in df.columns \
            else pd.Series(pd.NA, index=df.index, dtype='string')
        out[col] = text.replace('', pd.NA)
    out['ticker'] = out['ticker'].str.upper()

    out.loc[out['prc'] == 0, 'prc'] = np.nan
    for col in QUOTE_COLUMNS:
        out.loc[out[col] == 0, col] = np.nan
    # CRSP numeric missing codes: a simple return cannot be below -100%, volume cannot be negative
    for col in ('ret', 'dlret'):
        out.loc[out[col] < -1, col] = np.nan
    out.loc[out['vol'] < 0, 'vol'] = np.nan
    out.loc[out['shrout'] <= 0, 'shrout'] = np.nan

    total = len(out)
    out = out[out['date'].notna()]
    counts['dropped_missing_date'] += total - len(out)
    if start is not None:
        before = len(out)
        out = out[out['date'] >= pd.Timestamp(start)]
        counts['dropped_before_start'] += before - len(out)
    has_delisting = out[key].notna() & (out['dlret'].notna() | out['dlstcd'].notna())
    delisting = out.loc[has_delisting, [key, 'date', 'dlstcd', 'dlret']]

    before = len(out)
    out = out[out['prc'].notna()]
    counts['dropped_missing_price'] += before - len(out)
    before = len(out)
    out = out[out[key].notna()]
    counts['dropped_missing_id'] += before - len(out)

    out['prc'] = out['prc'].abs()
    for col in ('askhi', 'bidlo'):
        out[col] = out[col].abs()
    usable_factor = out['cfacpr'].where(out['cfacpr'] > 0)
    out['prc_adj'] = (out['prc'] / usable_factor).fillna(out['prc'])
    out['mkt_cap'] = out['prc'] * out['shrout']
    out['ret_total'] = out['ret']   # the delisting return is added in pass 2, once the last day is known
    return out[OUTPUT_COLUMNS], delisting


def build_daily_dataset(csv_path: Path, output_dir: Path, chunk_rows: int = 200_000,
                        start: Optional[date] = None) -> Dict[str, Any]:
    """
    Cleans the CSV in chunks into output_dir/daily_prices/year=YYYY/data.parquet plus
    output_dir/daily_last_prices.parquet, and returns the manifest describing them.
    """
    counts = {'rows_read': 0, 'dropped_missing_date': 0, 'dropped_before_start': 0, 'dropped_missing_price': 0,
              'dropped_missing_id': 0, 'dropped_duplicates': 0, 'ticker_date_conflicts': 0,
              'delisting_returns_applied': 0, 'delistings_ignored': 0}
    header = pd.read_csv(csv_path, nrows=0).columns
    key = _check_columns(header, header)
    ignored = [c for c in header if c.strip().lower() not in COLUMN_ALIASES]
    staging = output_dir / '_staging'
    shutil.rmtree(staging, ignore_errors=True)   # never mix in parts left by an interrupted run
    staging.mkdir(parents=True)
    try:
        years, last_rows, delist = _stage_and_write_years(csv_path, output_dir, staging, key, chunk_rows, counts, start)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not years:
        raise ValueError("No usable rows: every row was missing a date, price or identifier.")
    manifest = _finish(csv_path, output_dir, key, years, last_rows, delist, ignored, counts)
    manifest['start_date'] = start.isoformat() if start else None
    return manifest


def _latest_per_security(frames: List[Optional[pd.DataFrame]], key: str, col: str) -> pd.DataFrame:
    """
    Each security's latest row (by date) with a value in `col`. Frames are in file order; for
    rows on the same date the first one in the file wins, as for duplicate price rows.
    """
    rows = pd.concat([f for f in frames if f is not None])
    rows = rows.loc[rows[col].notna(), [key, 'date', col]].drop_duplicates(subset=[key, 'date'], keep='first')
    return rows.sort_values('date', kind='stable').groupby(key).tail(1)


def _stage_and_write_years(csv_path: Path, output_dir: Path, staging: Path, key: str, chunk_rows: int,
                           counts: Dict[str, int], start: Optional[date] = None):
    """Pass 1 stages cleaned rows by year; pass 2 writes one sorted, de-duplicated file per year."""
    # Pass 1: clean each chunk, stage its rows by year, and track each security's last day and delisting
    last_dates: Optional[pd.Series] = None
    latest = {col: None for col in ('dlstcd', 'dlret')}   # each security's latest row with that value
    reader = pd.read_csv(csv_path, chunksize=chunk_rows, dtype=str, keep_default_na=False,
                         usecols=lambda c: c.strip().lower() in COLUMN_ALIASES)
    for i, chunk in enumerate(reader):
        counts['rows_read'] += len(chunk)
        cleaned, delisting = clean_chunk(chunk, counts, start)
        for col in latest:
            latest[col] = _latest_per_security([latest[col], delisting], key, col)
        chunk_last = cleaned.groupby(key)['date'].max()
        last_dates = chunk_last if last_dates is None else pd.concat([last_dates, chunk_last]).groupby(level=0).max()
        for year, rows in cleaned.groupby(cleaned['date'].dt.year):
            part_dir = staging / f'year={int(year)}'
            part_dir.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pandas(rows, schema=SCHEMA, preserve_index=False),
                           part_dir / f'part-{i:05d}.parquet')
        logger.info(f"Read {counts['rows_read']:,} rows")

    if last_dates is None:
        return {}, [], pd.DataFrame(columns=['dlstcd', 'dlret'])
    # Each security's latest delisting code and return (from any row, priced or not). One dated
    # before the security's last price cannot be its delisting (e.g. an earlier company that used
    # the same ticker), so it is ignored and counted.
    delist = pd.DataFrame(index=last_dates.index)
    ignored_keys = set()
    for col, rows in latest.items():
        rows = rows.set_index(key)
        last_price_day = last_dates.reindex(rows.index)   # missing for a security that never had a price
        delist[col] = rows.loc[rows['date'] >= last_price_day, col]
        ignored_keys.update(rows.index[rows['date'] < last_price_day])
    counts['delistings_ignored'] += len(ignored_keys)

    # Pass 2: one sorted, de-duplicated file per year
    years: Dict[str, Dict[str, int]] = {}
    last_rows: List[pd.DataFrame] = []
    for part_dir in sorted(staging.glob('year=*'), key=lambda p: int(p.name.split('=')[1])):
        year = int(part_dir.name.split('=')[1])
        df = pq.read_table(part_dir, schema=SCHEMA).to_pandas()
        # Whole-number codes come back as float wherever a value is missing; keep them nullable
        # integers so every year's file reads back with the same column types
        df[CODE_COLUMNS] = df[CODE_COLUMNS].astype('Int64')
        before = len(df)
        if key == 'permno':
            df = df.drop_duplicates(subset=['permno', 'date'], keep='first')
        else:
            # Share classes can share a ticker: drop only exact copies and report what is left
            df = df.drop_duplicates(keep='first')
            counts['ticker_date_conflicts'] += int(df.duplicated(subset=['ticker', 'date']).sum())
        df = df.sort_values([key, 'date'], kind='stable').reset_index(drop=True)
        counts['dropped_duplicates'] += before - len(df)

        last_day = df['date'] == df[key].map(last_dates)
        if key == 'ticker':
            # Several securities share the ticker on its last day: the delisting can't be placed
            shared = df.loc[last_day & df.duplicated(subset=['ticker', 'date'], keep=False), 'ticker'].unique()
            known = delist.index.intersection(shared)
            known = known[delist.loc[known].notna().any(axis=1).to_numpy()]
            counts['delistings_ignored'] += len(known)
            delist = delist.drop(known)

        # Delisting return, once, on each security's last trading day
        dlret = df[key].map(delist['dlret']).astype('float64')
        apply = last_day & dlret.notna()
        df.loc[apply, 'ret_total'] = (1 + df.loc[apply, 'ret'].fillna(0)) * (1 + dlret[apply]) - 1
        counts['delisting_returns_applied'] += int(apply.sum())

        target = output_dir / daily_year_file(year)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False),
                       target, compression='zstd', row_group_size=250_000)
        if pq.read_table(target).num_rows != len(df):   # the written file must load back intact
            raise RuntimeError(f"Verification failed for {target}")
        years[str(year)] = {'rows': len(df), 'bytes': target.stat().st_size, 'securities': int(df[key].nunique())}
        last_rows.append(df.groupby(key, sort=False).tail(1))
        shutil.rmtree(part_dir)
        logger.info(f"{year}: {len(df):,} rows, {years[str(year)]['securities']:,} securities")
    return years, last_rows, delist


def _finish(csv_path: Path, output_dir: Path, key: str, years: Dict[str, Dict[str, int]],
            last_rows: List[pd.DataFrame], delist: pd.DataFrame, ignored: List[str],
            counts: Dict[str, int]) -> Dict[str, Any]:
    """Writes the last-price file and returns the manifest."""
    # Each security's last trading day, like last_price_mapping for the yearly data
    last = pd.concat(last_rows).sort_values('date').groupby(key, sort=False).tail(1)
    last_date = last['date'].max()
    delist_code = last[key].map(delist['dlstcd']).astype('Int64')
    last_prices = pd.DataFrame({
        'permno': last['permno'], 'ticker': last['ticker'], 'comnam': last['comnam'], 'last_date': last['date'],
        'last_prc': last['prc'], 'last_prc_adj': last['prc_adj'],
        'delist_code': delist_code,
        'delist_ret': last[key].map(delist['dlret']).astype('float64'),
        # From the CRSP code; missing when the file has no code for the security
        'delisted': (delist_code >= DELISTED_FROM_CODE).astype('boolean'),
        # Rows stop more than a week before the data ends: a delisting, or leaving the
        # universe if the file only covers index membership periods
        'ended_early': last['date'] < last_date - pd.Timedelta(days=7),
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
        'ended_early_securities': int(last_prices['ended_early'].sum()),
        'years': years,
        'columns': {f.name: str(f.type) for f in SCHEMA},
        'ignored_csv_columns': ignored,
        'cleaning': counts,
        'units': {
            'mkt_cap': '$ thousands (prc x CRSP shrout, which is in thousands of shares)',
            'ret': 'daily simple return (decimal)',
            'ret_total': "ret combined with the delisting return (dlret) on each security's last trading day",
            'askhi/bidlo': 'absolute values; CRSP marks a bid/ask quote (not a traded high/low) with a negative value',
            'dlstcd': 'CRSP delisting code: 100 = active, 200 and above = delisted',
        },
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
    parser.add_argument('--chunk-rows', type=int, default=200_000, help="CSV rows per chunk (default 200,000)")
    parser.add_argument('--start-date', type=date.fromisoformat,
                        help="Drop rows before this date, e.g. 2010-01-01 (default: keep everything)")
    args = parser.parse_args(argv)
    # Expand ~ even when the path was quoted (the shell only expands it unquoted)
    args.csv = args.csv.expanduser()
    if args.output_dir:
        args.output_dir = args.output_dir.expanduser()
    if not args.csv.is_file():
        parser.error(f"CSV not found: {args.csv}")

    manager = None
    if not args.dry_run:
        # Fail in seconds on bad keys or permissions, not after cleaning the whole file
        manager = S3DataManager(bucket=args.bucket, prefix=args.prefix)
        manager.check_write_access()
        logger.info(f"AWS access OK for s3://{manager.bucket}/{manager.prefix}/")

    with tempfile.TemporaryDirectory(prefix='factor_lake_daily_') as tmp:
        output_dir = args.output_dir or Path(tmp)
        manifest = build_daily_dataset(args.csv, output_dir, args.chunk_rows, args.start_date)
        c = manifest['cleaning']
        logger.info(f"Cleaned {c['rows_read']:,} rows into {manifest['rows_written']:,} "
                    f"({manifest['securities']:,} securities, {manifest['date_range'][0]} to {manifest['date_range'][1]}). "
                    f"Dropped: {c['dropped_missing_date']:,} no date, {c['dropped_before_start']:,} before the start date, "
                    f"{c['dropped_missing_price']:,} no price, "
                    f"{c['dropped_missing_id']:,} no identifier, {c['dropped_duplicates']:,} duplicates. "
                    f"Delisted: {manifest['delisted_securities']:,} securities; "
                    f"{c['delisting_returns_applied']:,} delisting returns added to ret_total.")
        if c['ticker_date_conflicts']:
            logger.warning(f"{c['ticker_date_conflicts']:,} ticker/date pairs still have more than one row "
                           "(e.g. share classes sharing a ticker). Add a permno column to tell them apart.")
        if c['delistings_ignored']:
            logger.warning(f"{c['delistings_ignored']:,} securities have a delisting code or return that could not be "
                           "placed (dated before their last price, or on a ticker shared that day); it was ignored.")
        if manifest['ignored_csv_columns']:
            logger.info(f"Columns not published: {manifest['ignored_csv_columns']}")
        if args.dry_run:
            (output_dir / DAILY_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
            logger.info(f"Dry run: nothing uploaded. Files are in {output_dir}"
                        + ("" if args.output_dir else " (deleted on exit; use --output-dir to keep them)"))
            return 0
        publish(output_dir, manifest, manager)
    logger.info("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
