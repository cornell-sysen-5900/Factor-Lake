"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/s3_client.py
PURPOSE: Loads the investment universe from Parquet files in AWS S3 (replaces the Supabase database).
VERSION: 1.0.0
"""

import io
import logging
import os
from typing import Any, Optional

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
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

        uri = self.s3_uri(filename)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.object_key(filename))
            body = response['Body'].read()
        except NoCredentialsError as e:
            raise S3DataError(
                "AWS credentials are not configured. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY (Streamlit secrets or .env)."
            ) from e
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code', 'Unknown')
            raise S3DataError(f"Could not read {uri} ({code}).") from e
        except BotoCoreError as e:
            raise S3DataError(f"Could not reach S3 for {uri}: {e}") from e
        return pd.read_parquet(io.BytesIO(body))

    def upload_parquet(self, df: pd.DataFrame, filename: str, key: Optional[str] = None) -> str:
        """Writes a DataFrame as Parquet to S3 and returns the key it was written to."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine='pyarrow', index=False)
        key = key or self.object_key(filename)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue(),
                               ContentType='application/vnd.apache.parquet')
        return key
