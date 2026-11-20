"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/lambda_client.py
PURPOSE: Sends backtest jobs from the Streamlit app to the backtest Lambda.
VERSION: 1.1.0
"""

import json
import os
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.config import Config

from .backtest_jobs import decode_result, decode_saved
from .wrds_preview import (
    DISABLED_MESSAGE,
    MAX_RESPONSE_BYTES,
    NOT_READY_MESSAGE,
    PREVIEW_COLUMNS,
    PREVIEW_ROWS,
    validate_request,
)

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
            client = boto3.client(
                'lambda',
                region_name=region or os.environ.get('AWS_DEFAULT_REGION') or 'us-east-1',
                # One attempt: a retry after a read timeout would start a second backtest
                config=Config(read_timeout=INVOKE_TIMEOUT_SECONDS, connect_timeout=10,
                              retries={'total_max_attempts': 1}),
            )
        self.client = client

    def preview(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch bounded sample rows without entering the backtest/result-download path."""
        validate_request(request)
        try:
            response = self.client.invoke(FunctionName=self.function_name, InvocationType='RequestResponse',
                                          Payload=json.dumps({'action': 'wrds_preview', 'request': request}).encode())
            stream = response['Payload']
            try:
                body = stream.read(MAX_RESPONSE_BYTES + 1)
            finally:
                stream.close()
            if len(body) > MAX_RESPONSE_BYTES or response.get('FunctionError'):
                raise ValueError('Invalid preview response')
            payload = json.loads(body)
            if payload.get('status') in ('disabled', 'unavailable'):
                return {'status': payload['status'], 'message': DISABLED_MESSAGE
                        if payload['status'] == 'disabled' else NOT_READY_MESSAGE}
            if payload.get('status') != 'ok':
                raise ValueError('Preview unavailable')
            preview = payload['preview']
            if preview['columns'] != list(PREVIEW_COLUMNS) or preview['row_limit'] != PREVIEW_ROWS:
                raise ValueError('Invalid preview columns or limit')
            rows = preview['rows']
            if len(rows) > PREVIEW_ROWS:
                raise ValueError('Invalid preview row count')
            if any(set(row) != set(PREVIEW_COLUMNS) for row in rows):
                raise ValueError('Invalid preview fields')
            if len({row['permno'] for row in rows}) != len(rows):
                raise ValueError('Preview rows must be different companies')
            return payload
        except Exception as exc:
            raise BacktestLambdaError('The WRDS preview could not be loaded. Check the development Lambda '
                                      'deployment and try again.') from exc

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
            raise BacktestLambdaError("The backtest Lambda could not run this backtest: "
                                      f"{payload.get('error', payload)}")

        if 'result' in payload:
            result, delivery = payload['result'], 'inline'
        else:
            result, delivery = decode_saved(_download(payload['result_url']))[1], 's3_link'

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
