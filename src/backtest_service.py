"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/backtest_service.py
PURPOSE: Runs backtest jobs inside AWS Lambda, with results cached in S3.
VERSION: 1.0.0

For each job (see src/backtest_jobs.py) the service:
1. Hashes the job's inputs with the data version (the S3 ETags of the market data
   files) and the backtest code version.
2. Returns the saved result for that hash when one exists, without rerunning.
3. Otherwise reads the market data from S3, runs the backtest and saves the result
   under backtest-results/<hash>.json (an S3 lifecycle rule deletes it after 90 days).
4. Returns the result in the response, or a temporary S3 link when it is too big
   for a Lambda response (6 MB).
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .backtest_jobs import code_version, decode_saved, describe_job, encode_saved, job_hash, run_job, validate_job
from .data_standardization import validate_universe
from .s3_client import DELISTING_FILE, UNIVERSE_FILE, S3DataManager

logger = logging.getLogger(__name__)

RESULTS_DIR = 'backtest-results'
# Lambda responses are limited to 6 MB; leave room for the rest of the response
INLINE_LIMIT_BYTES = 5_000_000
# How long the S3 link to a large result stays valid
LINK_EXPIRY_SECONDS = 3600
# Saved results are deleted after this many days (S3 lifecycle rule, set by scripts/deploy_backtest_lambda.py)
RESULT_RETENTION_DAYS = 90


def result_file(key: str) -> str:
    """Path (under the data prefix) of a saved result."""
    return f'{RESULTS_DIR}/{key}.json'


def summarize_job(job: Dict[str, Any]) -> str:
    """
    A short description of a job for listings, e.g. "ROE using 9/30 Data (higher), ROA % (higher)
    | Mixing | 2002-2024 | $1,000". Stored as S3 metadata, so it is kept ASCII and short.
    """
    factors = ', '.join(
        f"{job['factor_labels'].get(f, f)} ({'higher' if job['factor_directions'][f] == 'top' else 'lower'})"
        for f in job['factors'])
    text = (f"{factors} | {job['factor_combination'].capitalize()} | "
            f"{job['start_year']}-{job['end_year']} | ${job['initial_aum']:,.0f}")
    return text.encode('ascii', 'replace').decode()[:500]


class BacktestService:
    """
    Runs and caches backtests. One instance lives for the life of a Lambda container,
    so warm invocations reuse the market data already in memory.
    """

    def __init__(self, manager: Optional[S3DataManager] = None,
                 inline_limit: int = INLINE_LIMIT_BYTES):
        self.manager = manager or S3DataManager()
        self.inline_limit = inline_limit
        self.code_version = code_version()
        self._universe: Optional[Tuple[str, pd.DataFrame]] = None   # (data version, data)
        self._wrds = None   # WrdsBacktestService, created on the first WRDS job

    @property
    def wrds(self):
        """The WRDS backtest service (src/wrds_service.py), sharing this service's S3 and cache."""
        if self._wrds is None:
            from .wrds_service import WrdsBacktestService
            self._wrds = WrdsBacktestService(self)
        return self._wrds

    def handle_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lambda entry point. Returns {'status': 'ok', ...} (see run) or
        {'status': 'error', 'error': message} instead of raising, so the app can
        show the reason.
        """
        try:
            if event.get('action') == 'ping':
                return {'status': 'ok', 'code_version': self.code_version}
            return self.run(event.get('job', event))
        except ValueError as e:
            return {'status': 'error', 'error_type': 'invalid_request', 'error': str(e)}
        except Exception as e:   # reported to the app instead of a bare Lambda failure
            logger.exception("Backtest failed")
            return {'status': 'error', 'error_type': type(e).__name__, 'error': str(e)}

    def run(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the job's result, from the cache when it was computed before.

        The response has the hash, whether it was cached, when and how fast it was
        computed, its size, and either 'result' (the result itself) or 'result_url'
        (a temporary S3 link to it, for results over the inline limit).

        WRDS jobs (data_mode 'wrds') go to the WRDS service; every other job is Legacy.
        """
        if isinstance(job, dict) and job.get('data_mode') == 'wrds':
            return self.wrds.run(job)
        validate_job(job)
        data_version = self.data_version()
        key = job_hash(job, data_version, self.code_version)
        filename = result_file(key)

        cached = self._read_cached(filename)
        if cached is not None:
            body, info = cached
            logger.info(f"Cache hit {key[:12]} for {describe_job(job)}")
        else:
            started = time.perf_counter()
            result = run_job(self._load_universe(data_version), job)
            info = {
                'computed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'compute_seconds': f"{time.perf_counter() - started:.3f}",
                # Read by the list of saved analyses without downloading each result
                'kind': job['kind'],
                'summary': summarize_job(job),
            }
            body = encode_saved(job, result)   # with its job, so the hash alone can reopen it
            self.manager.client.put_object(
                Bucket=self.manager.bucket, Key=self.manager.object_key(filename), Body=body,
                ContentType='application/json', Metadata=info,
            )
            logger.info(f"Computed {key[:12]} in {info['compute_seconds']} s for {describe_job(job)}")
        return self._respond(key, filename, body, info, cached is not None, data_version, self.code_version)

    def _respond(self, key: str, filename: str, body: bytes, info: Dict[str, str], cached: bool,
                 data_version: str, code: str) -> Dict[str, Any]:
        """The Lambda response for a saved result: inline, or a temporary S3 link when too big."""
        response = {
            'status': 'ok',
            'hash': key,
            'cached': cached,
            'data_version': data_version,
            'code_version': code,
            'computed_at': info.get('computed_at'),
            'compute_seconds': float(info['compute_seconds']) if info.get('compute_seconds') else None,
            'size_bytes': len(body),
        }
        if len(body) <= self.inline_limit:
            response['result'] = decode_saved(body)[1]
        else:   # the link is to the saved analysis; the client takes the result out of it
            response['result_url'] = self.manager.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.manager.bucket, 'Key': self.manager.object_key(filename)},
                ExpiresIn=LINK_EXPIRY_SECONDS,
            )
        return response

    def data_version(self) -> str:
        """
        Identifies the market data the backtest reads: the S3 ETags of the universe and
        delisting files. Republishing either file changes it, so old results are not reused.
        """
        parts = []
        for filename in (UNIVERSE_FILE, DELISTING_FILE):
            try:
                head = self.manager.client.head_object(Bucket=self.manager.bucket,
                                                       Key=self.manager.object_key(filename))
                parts.append(f"{filename}:{head['ETag'].strip(chr(34))}")
            except Exception as e:
                if filename == DELISTING_FILE and _is_missing(e):
                    parts.append(f"{filename}:missing")   # fetch_all_data runs without it too
                else:
                    raise
        return '|'.join(parts)

    def _load_universe(self, data_version: str) -> pd.DataFrame:
        """The full market data, standardized like the app's; reused while the data version is unchanged."""
        if self._universe is None or self._universe[0] != data_version:
            self._universe = None   # free the old copy before downloading the new one
            df = self.manager.fetch_all_data()
            validate_universe(df, 's3')
            self._universe = (data_version, df)
        return self._universe[1]

    def _read_cached(self, filename: str) -> Optional[Tuple[bytes, Dict[str, str]]]:
        """
        (body, metadata) of a saved result, or None when there is none.

        Without s3:ListBucket (e.g. the class's shared Lambda role), S3 answers AccessDenied
        instead of NoSuchKey for a result that is not saved yet, so that counts as "not
        saved" too. A real permission problem still surfaces when the result is saved.
        """
        try:
            response = self.manager.client.get_object(Bucket=self.manager.bucket,
                                                      Key=self.manager.object_key(filename))
            return response['Body'].read(), response.get('Metadata') or {}
        except Exception as e:
            if _is_missing(e) or _error_code(e) in ('AccessDenied', '403'):
                return None
            raise


def _error_code(error: Exception) -> str:
    """The AWS error code of a botocore ClientError ('' for other errors)."""
    return str(getattr(error, 'response', {}).get('Error', {}).get('Code', ''))


def _is_missing(error: Exception) -> bool:
    """True when an S3 error means the object does not exist."""
    return _error_code(error) in ('NoSuchKey', '404', 'NotFound')
