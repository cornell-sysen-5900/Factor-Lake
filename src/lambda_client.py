"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/lambda_client.py
PURPOSE: Sends backtest jobs from the Streamlit app to the backtest Lambda.
VERSION: 1.0.0
"""

import json
import os
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .backtest_jobs import decode_result

DEFAULT_FUNCTION_NAME = 'factor-lake-backtest'
# A cold Lambda downloads the market data before it runs; allow up to the Lambda's own limit
INVOKE_TIMEOUT_SECONDS = 900


class BacktestLambdaError(RuntimeError):
    """Raised when the backtest Lambda cannot be reached or reports an error."""


class BacktestLambdaClient:
    """Runs backtest jobs in AWS Lambda and returns their results and how they were produced."""

    def __init__(self, function_name: Optional[str] = None, region: Optional[str] = None,
                 client: Optional[Any] = None):
        self.function_name = function_name or os.environ.get('FACTOR_LAKE_BACKTEST_LAMBDA') or DEFAULT_FUNCTION_NAME
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client(
                'lambda',
                region_name=region or os.environ.get('AWS_DEFAULT_REGION') or 'us-east-1',
                # One attempt: a retry after a read timeout would start a second backtest
                config=Config(read_timeout=INVOKE_TIMEOUT_SECONDS, connect_timeout=10,
                              retries={'total_max_attempts': 1}),
            )
        self.client = client

    def run(self, job: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs a job (src/backtest_jobs.build_job) in Lambda.

        Returns:
            (result, info): the result, as run_job would return it locally, and how it
            was produced: hash, cached, computed_at, compute_seconds, size_bytes,
            delivery ('inline' or 's3_link') and round_trip_seconds.
        """
        started = time.perf_counter()
        try:
            response = self.client.invoke(FunctionName=self.function_name,
                                          InvocationType='RequestResponse',
                                          Payload=json.dumps(job).encode())
        except Exception as e:
            raise BacktestLambdaError(f"Could not call the backtest Lambda '{self.function_name}': {e}") from e

        payload = json.loads(response['Payload'].read() or b'{}')
        if response.get('FunctionError'):
            raise BacktestLambdaError(f"The backtest Lambda failed: {payload.get('errorMessage', payload)}")
        if payload.get('status') != 'ok':
            raise BacktestLambdaError(f"The backtest Lambda could not run this backtest: {payload.get('error', payload)}")

        if 'result' in payload:
            result, delivery = payload['result'], 'inline'
        else:
            result, delivery = _download(payload['result_url']), 's3_link'

        info = {k: payload.get(k) for k in ('hash', 'cached', 'computed_at', 'compute_seconds', 'size_bytes')}
        info.update(function_name=self.function_name, delivery=delivery,
                    round_trip_seconds=round(time.perf_counter() - started, 3))
        return decode_result(job['kind'], result), info


def _download(url: str) -> bytes:
    """Downloads a result from its temporary S3 link."""
    if not url.startswith('https://'):
        raise BacktestLambdaError(f"Unexpected result link from the backtest Lambda: {url[:60]}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:   # nosec B310 - https only, checked above
            return response.read()
    except Exception as e:
        raise BacktestLambdaError(f"Could not download the backtest result from S3: {e}") from e
