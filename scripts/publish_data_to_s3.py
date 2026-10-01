#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/publish_data_to_s3.py
PURPOSE: Publishes the app's data tables to S3 as Parquet (from Supabase or from local files).

Usage:
    # One-time migration: copy the current Supabase tables to S3
    python scripts/publish_data_to_s3.py --from-supabase

    # Data refresh: publish new files with the same columns as the Supabase tables
    python scripts/publish_data_to_s3.py --universe full_precision.xlsx --delisting last_price_mapping.csv

    # Check everything without uploading
    python scripts/publish_data_to_s3.py --from-supabase --dry-run

Needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and SUPABASE_URL / SUPABASE_KEY for
--from-supabase) in the environment or a .env file. Every publish also keeps a copy under
<prefix>/archive/<UTC timestamp>/ so a bad upload can be rolled back.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd

# Allow `python scripts/publish_data_to_s3.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_standardization import standardize_universe, validate_universe  # noqa: E402
from src.s3_client import DELISTING_FILE, UNIVERSE_FILE, S3DataManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_TABLES = {UNIVERSE_FILE: 'Full Precision Test', DELISTING_FILE: 'last_price_mapping'}


def read_local_table(path: str) -> pd.DataFrame:
    """Reads a .csv, .xlsx/.xls or .parquet file."""
    suffix = Path(path).suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}' for {path} (use .csv, .xlsx, .xls or .parquet)")


def load_tables(from_supabase: bool, universe_path: Optional[str],
                delisting_path: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Returns {S3 file name: raw table} for everything that should be published."""
    if from_supabase:
        from src.supabase_client import SupabaseManager
        manager = SupabaseManager()
        return {filename: manager.fetch_table(table) for filename, table in SUPABASE_TABLES.items()}

    tables = {UNIVERSE_FILE: read_local_table(universe_path)}
    if delisting_path:
        tables[DELISTING_FILE] = read_local_table(delisting_path)
    return tables


def check_tables(tables: Dict[str, pd.DataFrame]) -> None:
    """Refuses to publish data the app could not run on."""
    validate_universe(standardize_universe(tables[UNIVERSE_FILE].copy()), 'universe file')
    if DELISTING_FILE in tables:
        missing = {'ticker', 'last_date', 'last_price'} - set(tables[DELISTING_FILE].columns)
        if missing:
            raise ValueError(f"The delisting file is missing required columns: {sorted(missing)}")


def publish(tables: Dict[str, pd.DataFrame], manager: S3DataManager,
            timestamp: str, dry_run: bool = False) -> Dict[str, str]:
    """
    Uploads each table to its live key plus an archive copy, then reads the
    live file back to confirm the row count. Returns {file name: live S3 URI}.
    """
    check_tables(tables)
    published = {}
    for filename, df in tables.items():
        live_uri = manager.s3_uri(filename)
        archive_key = manager.object_key(f"archive/{timestamp}/{filename}")
        if dry_run:
            logger.info(f"[dry run] would upload {len(df):,} rows to {live_uri} (archive: {archive_key})")
            published[filename] = live_uri
            continue
        manager.upload_parquet(df, filename, key=archive_key)
        manager.upload_parquet(df, filename)
        rows_back = len(manager.read_parquet(filename))
        if rows_back != len(df):
            raise RuntimeError(f"Verification failed for {live_uri}: wrote {len(df)} rows, read back {rows_back}")
        logger.info(f"Published {len(df):,} rows to {live_uri} (archive: s3://{manager.bucket}/{archive_key})")
        published[filename] = live_uri
    return published


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Publish Factor-Lake data tables to S3 as Parquet.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-supabase', action='store_true', help="Copy the current Supabase tables")
    source.add_argument('--universe', help="Local 'Full Precision Test' file (.csv/.xlsx/.xls/.parquet)")
    parser.add_argument('--delisting', help="Local last_price_mapping file (optional with --universe)")
    parser.add_argument('--bucket', help="S3 bucket (default: FACTOR_LAKE_S3_BUCKET or sysen-5900-factor-lake)")
    parser.add_argument('--prefix', help="Key prefix (default: FACTOR_LAKE_S3_PREFIX or factor-lake-data)")
    parser.add_argument('--dry-run', action='store_true', help="Validate and report without uploading")
    args = parser.parse_args(argv)
    if args.delisting and args.from_supabase:
        parser.error("--delisting can only be used with --universe")

    tables = load_tables(args.from_supabase, args.universe, args.delisting)
    manager = S3DataManager(bucket=args.bucket, prefix=args.prefix)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    publish(tables, manager, timestamp, dry_run=args.dry_run)
    if not args.dry_run:
        logger.info("Done. The app picks up new data within 6 hours, or immediately after rebooting it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
