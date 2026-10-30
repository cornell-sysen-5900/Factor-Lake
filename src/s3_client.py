"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/s3_client.py
PURPOSE: Loads the investment universe from Parquet files in AWS S3 (replaces the Supabase database).
VERSION: 1.1.0
"""

import io
import json
import logging
import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .data_standardization import (
    attach_delisting_info,
    prepare_last_price_mapping,
    standardize_universe,
)

logger = logging.getLogger(__name__)

# Location of the data files; override with the FACTOR_LAKE_S3_* environment variables / Streamlit secrets
DEFAULT_BUCKET = 'sysen-5900-factor-lake'
DEFAULT_PREFIX = 'factor-lake-data'
DEFAULT_REGION = 'us-east-1'

# One Parquet file per former Supabase table, stored exactly as exported (raw columns)
UNIVERSE_FILE = 'full_precision_test.parquet'
DELISTING_FILE = 'last_price_mapping.parquet'

# Daily stock data: one Parquet file per year plus a manifest that lists the published years.
# The manifest is uploaded last, so readers only ever see a complete set of years.
DAILY_DIR = 'daily_prices'
DAILY_MANIFEST_FILE = f'{DAILY_DIR}/_manifest.json'
DAILY_LAST_PRICES_FILE = 'daily_last_prices.parquet'


def daily_year_file(year: int) -> str:
    """Path (under the prefix) of one year of daily data."""
    return f'{DAILY_DIR}/year={int(year)}/data.parquet'


class S3DataError(RuntimeError):
    """Raised when the data files cannot be read from S3."""


class S3DataManager:
    """
    Reads the Factor-Lake data files from an S3 bucket.

    S3 is always on (unlike a free-tier database, it never pauses), and the
    whole universe downloads as one compressed file instead of dozens of
    paginated database queries.
    """

    def __init__(self, bucket: Optional[str] = None, prefix: Optional[str] = None,
                 region: Optional[str] = None, client: Optional[Any] = None):
        self.bucket = bucket or os.environ.get('FACTOR_LAKE_S3_BUCKET') or DEFAULT_BUCKET
        self.prefix = (prefix if prefix is not None
                       else os.environ.get('FACTOR_LAKE_S3_PREFIX', DEFAULT_PREFIX)).strip('/')
        self.region = region or os.environ.get('AWS_DEFAULT_REGION') or DEFAULT_REGION
        if client is None:
            import boto3
            # Credentials come from the standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY variables
            client = boto3.client('s3', region_name=self.region)
        self.client = client

    def object_key(self, filename: str) -> str:
        """Returns the S3 key for a data file under the configured prefix."""
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def s3_uri(self, filename: str) -> str:
        return f"s3://{self.bucket}/{self.object_key(filename)}"

    def fetch_all_data(self) -> pd.DataFrame:
        """
        Downloads the universe and delisting data and standardizes them exactly
        like SupabaseManager.fetch_all_data. Raises S3DataError on failure.
        """
        df = self.read_parquet(UNIVERSE_FILE)
        df = standardize_universe(df)

        try:
            lpm = self.read_parquet(DELISTING_FILE)
        except S3DataError as e:
            # Same behaviour as Supabase: without delisting data the engine falls back to its defaults
            logger.warning(f"Delisting data unavailable, continuing without it: {e}")
            lpm = pd.DataFrame()
        return attach_delisting_info(df, prepare_last_price_mapping(lpm))

    def read_parquet(self, filename: str) -> pd.DataFrame:
        """Downloads one Parquet data file into a DataFrame."""
        return pd.read_parquet(io.BytesIO(self._read_bytes(filename)))

    def read_bytes(self, filename: str) -> bytes:
        """Downloads one object under the prefix. Raises S3DataError on failure."""
        return self._read_bytes(filename)

    def read_daily_manifest(self) -> Dict[str, Any]:
        """Returns the manifest of the published daily data (years, row counts, columns)."""
        return json.loads(self._read_bytes(DAILY_MANIFEST_FILE))

    def read_daily_prices(self, years: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Downloads the daily stock data for the given years (all published years by default).
        Only years listed in the manifest are read. Raises S3DataError for unpublished years.
        """
        published = sorted(int(y) for y in self.read_daily_manifest()['years'])
        wanted = published if years is None else sorted({int(y) for y in years})
        missing = [y for y in wanted if y not in published]
        if missing:
            raise S3DataError(f"No daily data published for {missing}. Published years: "
                              f"{published[0]}-{published[-1]}." if published else
                              "No daily data has been published.")
        frames = [self.read_parquet(daily_year_file(y)) for y in wanted]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _read_bytes(self, filename: str) -> bytes:
        """Downloads one object under the prefix, turning AWS errors into readable S3DataErrors."""
        uri = self.s3_uri(filename)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.object_key(filename))
            return response['Body'].read()
        except Exception as e:
            raise _readable_error(e, 'read', uri) from e

    def upload_parquet(self, df: pd.DataFrame, filename: str, key: Optional[str] = None) -> str:
        """Writes a DataFrame as Parquet to S3 and returns the key it was written to."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine='pyarrow', index=False)
        key = key or self.object_key(filename)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue(),
                               ContentType='application/vnd.apache.parquet')
        return key

    def upload_file(self, local_path: str, filename: str, content_type: str) -> str:
        """Uploads a local file (multipart for large files) and returns its S3 key."""
        key = self.object_key(filename)
        try:
            self.client.upload_file(str(local_path), self.bucket, key,
                                    ExtraArgs={'ContentType': content_type})
        except Exception as e:
            raise _readable_error(e, 'write', self.s3_uri(filename)) from e
        return key

    def object_size(self, filename: str) -> int:
        """Size in bytes of an object under the prefix (used to verify uploads)."""
        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=self.object_key(filename))['ContentLength'])
        except Exception as e:
            raise _readable_error(e, 'check', self.s3_uri(filename)) from e

    def check_write_access(self) -> None:
        """Writes and deletes a tiny object so bad keys or permissions fail fast, before long work."""
        filename = '_write_check'
        try:
            self.client.put_object(Bucket=self.bucket, Key=self.object_key(filename), Body=b'ok')
            self.client.delete_object(Bucket=self.bucket, Key=self.object_key(filename))
        except Exception as e:
            raise _readable_error(e, 'write', self.s3_uri(filename)) from e


def _readable_error(error: Exception, action: str, uri: str) -> Exception:
    """Turns AWS errors into a short S3DataError; anything else is returned unchanged."""
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    if isinstance(error, NoCredentialsError):
        return S3DataError("AWS credentials are not configured. Set AWS_ACCESS_KEY_ID and "
                           "AWS_SECRET_ACCESS_KEY (Streamlit secrets or .env).")
    if isinstance(error, ClientError):
        code = error.response.get('Error', {}).get('Code', 'Unknown')
        hint = {
            'InvalidAccessKeyId': ' AWS_ACCESS_KEY_ID is not a valid key.',
            'SignatureDoesNotMatch': ' AWS_SECRET_ACCESS_KEY does not match AWS_ACCESS_KEY_ID; re-copy it exactly.',
            'AccessDenied': ' This key is not allowed to do that.',
        }.get(code, '')
        return S3DataError(f"Could not {action} {uri} ({code}).{hint}")
    if isinstance(error, BotoCoreError):
        return S3DataError(f"Could not reach S3 for {uri}: {error}")
    # s3transfer wraps client errors raised during multipart uploads
    cause = getattr(error, '__cause__', None) or getattr(error, '__context__', None)
    if cause is not None and cause is not error:
        mapped = _readable_error(cause, action, uri)
        if isinstance(mapped, S3DataError):
            return mapped
    return error
