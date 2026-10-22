"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/backtest_jobs.py
PURPOSE: Describes a backtest as plain JSON inputs, runs it, and hashes it for the result cache.
VERSION: 1.0.0

A job holds every input that changes a backtest's results, so the Streamlit app
can send it to AWS Lambda, and run_job gives exactly what the app computes
locally for the same inputs. Used by the app (lambda_client), the Lambda
(backtest_service) and the tests.
"""

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest_engine import FACTOR_COMBINATIONS, rebalance_portfolio, run_cohort_comparison
from .universe_selection import select_universe

JOB_KINDS = ('backtest', 'cohort')
# Version of the saved-analysis format (see encode_saved)
SAVED_FORMAT = 2
DELISTING_STRATEGIES = ('zero_return', 'hold_cash', 'reinvest')
# Every key of a job (see build_job)
JOB_FIELDS = ('kind', 'factors', 'factor_directions', 'factor_labels', 'start_year', 'end_year',
              'initial_aum', 'benchmark_index', 'top_pct', 'use_market_cap_weight', 'delisting_strategy',
              'factor_combination', 'restrict_fossil_fuels', 'selected_sectors', 'cohort_pct')

# Source files whose code decides a backtest's results. Their contents are part of the
# cache key, so deploying changed backtest code never returns results from the old code.
ENGINE_SOURCES = (
    'backtest_engine.py', 'backtest_jobs.py', 'benchmarks.py', 'data_standardization.py',
    'factor_registry.py', 'factor_utils.py', 'performance_metrics.py', 'portfolio.py',
    'portfolio_filters.py', 's3_client.py', 'universe_selection.py',
)


def build_job(kind: str, factors: List[str], factor_directions: Dict[str, str],
              settings: Dict[str, Any], factor_labels: Optional[Dict[str, str]] = None,
              cohort_pct: Optional[float] = None) -> Dict[str, Any]:
    """
    Builds a job from the app's run inputs (factor columns, their directions and
    display labels, and the sidebar settings).

    Values are normalized (ints, floats, sorted sectors) so equal inputs always give
    an identical job and therefore the same cache key. Factor order is kept: it sets
    the column order of the holdings table.
    """
    if kind not in JOB_KINDS:
        raise ValueError(f"kind must be one of {JOB_KINDS}, not '{kind}'.")
    factors = list(factors)
    if not factors:
        raise ValueError("A backtest needs at least one factor.")
    labels = factor_labels or {}
    job = {
        'kind': kind,
        'factors': factors,
        'factor_directions': {f: factor_directions.get(f, 'top') for f in factors},
        'factor_labels': {f: labels.get(f, f) for f in factors},
        'start_year': int(settings['start_year']),
        'end_year': int(settings['end_year']),
        'initial_aum': float(settings['initial_aum']),
        'benchmark_index': int(settings.get('benchmark_index', 1)),
        'top_pct': float(settings.get('top_pct', 10.0)),
        'use_market_cap_weight': bool(settings.get('use_market_cap_weight', False)),
        'delisting_strategy': settings.get('delisting_strategy', 'zero_return'),
        'factor_combination': settings.get('factor_combination', 'mixing'),
        'restrict_fossil_fuels': bool(settings.get('restrict_fossil_fuels', False)),
        'selected_sectors': sorted(settings.get('selected_sectors') or []),
        'cohort_pct': float(cohort_pct) if kind == 'cohort' else None,
    }
    validate_job(job)
    return job


def validate_job(job: Dict[str, Any]) -> None:
    """Raises ValueError with a readable message when a job cannot be run."""
    if missing := [k for k in JOB_FIELDS if k not in job]:
        raise ValueError(f"The backtest request is missing {missing}.")
    if job.get('kind') not in JOB_KINDS:
        raise ValueError(f"kind must be one of {JOB_KINDS}, not '{job.get('kind')}'.")
    if not job.get('factors'):
        raise ValueError("A backtest needs at least one factor.")
    if job.get('factor_combination') not in FACTOR_COMBINATIONS:
        raise ValueError(f"factor_combination must be one of {FACTOR_COMBINATIONS}.")
    if job.get('delisting_strategy') not in DELISTING_STRATEGIES:
        raise ValueError(f"delisting_strategy must be one of {DELISTING_STRATEGIES}.")
    if bad := [d for d in job['factor_directions'].values() if d not in ('top', 'bottom')]:
        raise ValueError(f"Factor directions must be 'top' or 'bottom', not {bad}.")
    if job['initial_aum'] <= 0:
        raise ValueError("Initial AUM must be greater than $0.")
    if job['kind'] == 'cohort' and not job.get('cohort_pct'):
        raise ValueError("A cohort comparison needs cohort_pct.")


def run_job(raw: pd.DataFrame, job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a job on the full market data, the way the app runs it locally: filter the
    universe like Load Market Data does, then call the backtest engine.

    Returns:
        backtest: the rebalance_portfolio results.
        cohort: {'top': [...], 'bottom': [...]} portfolio values, like run_cohort_comparison.
    """
    validate_job(job)
    data = select_universe(raw, job['start_year'], job['end_year'],
                           selected_sectors=job['selected_sectors'],
                           restrict_fossil_fuels=job['restrict_fossil_fuels'])
    if job['kind'] == 'cohort':
        top, bottom = run_cohort_comparison(
            data=data,
            selected_factors=job['factors'],
            factor_directions=job['factor_directions'],
            cohort_pct=job['cohort_pct'],
            user_settings=job,   # has the start/end year, AUM, weighting, delisting and combination keys
        )
        return {'top': list(top), 'bottom': list(bottom)}
    return rebalance_portfolio(
        data,
        job['factors'],
        factor_directions=job['factor_directions'],
        start_year=job['start_year'],
        end_year=job['end_year'],
        initial_aum=job['initial_aum'],
        benchmark_index=job['benchmark_index'],
        top_pct=job['top_pct'],
        use_market_cap_weight=job['use_market_cap_weight'],
        delisting_strategy=job['delisting_strategy'],
        factor_combination=job['factor_combination'],
        factor_labels=job['factor_labels'],
    )


def code_version(source_dir: Optional[Path] = None) -> str:
    """Hash of the backtest source files (ENGINE_SOURCES)."""
    source_dir = source_dir or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ENGINE_SOURCES:
        digest.update(name.encode())
        digest.update((source_dir / name).read_bytes())
    return digest.hexdigest()[:16]


def job_hash(job: Dict[str, Any], data_version: str, code: Optional[str] = None) -> str:
    """
    The cache key of a job's result: a hash of its inputs, the version of the data it
    reads and the version of the backtest code. Any change to one of them changes it.
    """
    payload = json.dumps({'job': job, 'data_version': data_version, 'code_version': code or code_version()},
                         sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def _to_json_value(value: Any) -> Any:
    """json.dumps fallback for the numpy / pandas values the engine returns."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"Cannot convert {type(value).__name__} to JSON")


def encode_result(result: Dict[str, Any]) -> bytes:
    """
    Serializes a job result to JSON bytes. Floats are written exactly (Python's
    shortest round-trip repr), and NaN is kept as NaN, so decode_result gives back
    the same numbers.
    """
    return json.dumps(result, default=_to_json_value, allow_nan=True, separators=(',', ':')).encode()


def encode_saved(job: Dict[str, Any], result: Dict[str, Any]) -> bytes:
    """
    A saved analysis, as stored in S3 under its hash: the job and its result, so the
    analysis can be shown again (with its factors and settings) from the hash alone.
    """
    return json.dumps({'format': SAVED_FORMAT, 'job': job, 'result': result},
                      default=_to_json_value, allow_nan=True, separators=(',', ':')).encode()


def decode_saved(payload: Any) -> Tuple[Optional[Dict[str, Any]], Any]:
    """
    (job, result JSON) of a saved analysis. Results saved before jobs were stored
    with them come back as (None, result JSON). Pass the result to decode_result.
    """
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    if isinstance(payload, dict) and payload.get('format') == SAVED_FORMAT and 'result' in payload:
        return payload.get('job'), payload['result']
    return None, payload


def decode_result(kind: str, payload: Any) -> Dict[str, Any]:
    """
    Turns a JSON result back into what run_job returned. JSON object keys are always
    strings, so the year keys of holdings_by_year become ints again.
    """
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    if kind == 'backtest' and isinstance(payload.get('holdings_by_year'), dict):
        payload['holdings_by_year'] = {int(year): rows for year, rows in payload['holdings_by_year'].items()}
    return payload


def results_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True when two results hold the same values (numpy vs Python types and NaN compare equal)."""
    return _canonical(a) == _canonical(b)


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return 'NaN'
    if isinstance(value, (np.generic, pd.Timestamp, datetime, date)) or value is pd.NA:
        return _canonical(_to_json_value(value))
    return value


def describe_job(job: Dict[str, Any]) -> str:
    """One-line summary of a job, for logs."""
    factors = ', '.join(f"{job['factor_labels'].get(f, f)} ({job['factor_directions'][f]})" for f in job['factors'])
    return (f"{job['kind']} {job['start_year']}-{job['end_year']} {job['factor_combination']}: {factors}"
            + (f", cohort {job['cohort_pct']:g}%" if job['kind'] == 'cohort' else ''))

