"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_jobs.py
PURPOSE: Describes a WRDS backtest as plain JSON inputs, validates it, and versions its code.
VERSION: 1.0.0

A WRDS job is built by the app and run by the Lambda (src/wrds_service.py); see
DOCS/WRDS_MODE_DESIGN.md, section 5. It always has data_mode 'wrds', which Legacy
jobs (src/backtest_jobs.py) never have, so the two can never share a cache key.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .backtest_jobs import code_version
from .wrds_calendar import FREQUENCIES, period_boundaries
from .wrds_engine import FACTOR_COMBINATIONS
from .wrds_factors import WRDS_FACTOR_COLUMNS, WRDS_SECTORS

WRDS_JOB_KINDS = ('backtest', 'cohort', 'ranked')
WRDS_JOB_VERSION = 1
# Every key of a WRDS job (see build_wrds_job)
WRDS_JOB_FIELDS = ('kind', 'data_mode', 'wrds_job_version', 'factors', 'factor_directions', 'factor_labels',
                   'frequency', 'start_year', 'end_year', 'initial_aum', 'top_pct', 'use_market_cap_weight',
                   'factor_combination', 'restrict_fossil_fuels', 'selected_sectors', 'cohort_pct',
                   'rebalance_date')
# Top % of every WRDS backtest (fixed, as in Legacy)
WRDS_TOP_PCT = 10.0

# Source files whose code decides a WRDS result: part of its cache key. Legacy's
# ENGINE_SOURCES are kept separate, so changing WRDS code never recomputes Legacy results.
WRDS_ENGINE_SOURCES = (
    'backtest_jobs.py', 'performance_metrics.py', 's3_client.py', 'wrds_calendar.py', 'wrds_data.py',
    'wrds_engine.py', 'wrds_factors.py', 'wrds_holdings.py', 'wrds_jobs.py', 'wrds_service.py',
)

_MONTH = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')


def wrds_code_version() -> str:
    """Hash of the WRDS engine's source files (WRDS_ENGINE_SOURCES)."""
    return code_version(sources=WRDS_ENGINE_SOURCES)


def is_wrds_job(job: Dict[str, Any]) -> bool:
    return isinstance(job, dict) and job.get('data_mode') == 'wrds'


def build_wrds_job(kind: str, factors: List[str], factor_directions: Dict[str, str], settings: Dict[str, Any],
                   factor_labels: Optional[Dict[str, str]] = None, cohort_pct: Optional[float] = None,
                   rebalance_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds a WRDS job from the app's run inputs (factor columns, their directions and
    display labels, and the WRDS sidebar settings).

    Values are normalized (ints, floats, sorted sectors) so equal inputs always give an
    identical job and therefore the same cache key. Factor order is kept.
    """
    labels = factor_labels or {}
    factors = list(factors)
    job = {
        'kind': kind,
        'data_mode': 'wrds',
        'wrds_job_version': WRDS_JOB_VERSION,
        'factors': factors,
        'factor_directions': {f: factor_directions.get(f, 'top') for f in factors},
        'factor_labels': {f: labels.get(f, f) for f in factors},
        'frequency': settings.get('frequency'),
        'start_year': int(settings['start_year']),
        'end_year': int(settings['end_year']),
        'initial_aum': float(settings['initial_aum']),
        'top_pct': WRDS_TOP_PCT,
        'use_market_cap_weight': bool(settings.get('use_market_cap_weight', False)),
        'factor_combination': settings.get('factor_combination', 'mixing'),
        'restrict_fossil_fuels': bool(settings.get('restrict_fossil_fuels', False)),
        'selected_sectors': sorted(settings.get('selected_sectors') or []),
        'cohort_pct': float(cohort_pct) if kind == 'cohort' and cohort_pct is not None else None,
        'rebalance_date': rebalance_date if kind == 'ranked' else None,
    }
    validate_wrds_job(job)
    return job


def validate_wrds_job(job: Dict[str, Any]) -> None:
    """Raises ValueError with a readable message when a WRDS job cannot be run."""
    if not isinstance(job, dict):
        raise ValueError("A WRDS backtest request must be a JSON object.")
    if missing := [k for k in WRDS_JOB_FIELDS if k not in job]:
        raise ValueError(f"The WRDS backtest request is missing {missing}.")
    if unknown := [k for k in job if k not in WRDS_JOB_FIELDS]:
        raise ValueError(f"The WRDS backtest request has unknown fields {unknown}.")
    if job['data_mode'] != 'wrds':
        raise ValueError("A WRDS backtest request needs data_mode 'wrds'.")
    if job['wrds_job_version'] != WRDS_JOB_VERSION:
        raise ValueError(f"wrds_job_version {job['wrds_job_version']} is not supported (expected {WRDS_JOB_VERSION}).")
    if job['kind'] not in WRDS_JOB_KINDS:
        raise ValueError(f"kind must be one of {WRDS_JOB_KINDS}, not '{job['kind']}'.")

    factors = job['factors']
    if not isinstance(factors, list) or not factors:
        raise ValueError("A backtest needs at least one factor.")
    if len(set(factors)) != len(factors):
        raise ValueError("Each factor can be chosen only once.")
    if bad := [f for f in factors if f not in WRDS_FACTOR_COLUMNS]:
        raise ValueError(f"Not WRDS factors: {bad}.")
    if set(job['factor_directions']) != set(factors) or set(job['factor_labels']) != set(factors):
        raise ValueError("factor_directions and factor_labels must have exactly the job's factors.")
    if bad := [d for d in job['factor_directions'].values() if d not in ('top', 'bottom')]:
        raise ValueError(f"Factor directions must be 'top' or 'bottom', not {bad}.")
    if bad := [v for v in job['factor_labels'].values() if not isinstance(v, str) or not v]:
        raise ValueError(f"Factor labels must be text, not {bad}.")

    if job['frequency'] not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}, not '{job['frequency']}'.")
    for key in ('start_year', 'end_year'):
        if not isinstance(job[key], int) or isinstance(job[key], bool):
            raise ValueError(f"{key} must be a whole number.")
    months = period_boundaries(job['frequency'], job['start_year'], job['end_year'])   # readable ValueError
    if not _is_number(job['initial_aum']) or job['initial_aum'] <= 0:
        raise ValueError("Initial AUM must be greater than $0.")
    if not _is_number(job['top_pct']) or not 0 < job['top_pct'] <= 100:
        raise ValueError("top_pct must be above 0 and at most 100.")
    for key in ('use_market_cap_weight', 'restrict_fossil_fuels'):
        if not isinstance(job[key], bool):
            raise ValueError(f"{key} must be true or false.")
    if job['factor_combination'] not in FACTOR_COMBINATIONS:
        raise ValueError(f"factor_combination must be one of {FACTOR_COMBINATIONS}, not '{job['factor_combination']}'.")
    sectors = job['selected_sectors']
    if not isinstance(sectors, list) or sectors != sorted(set(sectors)):
        raise ValueError("selected_sectors must be a sorted list without repeats.")
    if bad := [s for s in sectors if s not in WRDS_SECTORS]:
        raise ValueError(f"Unknown GICS sectors: {bad}.")

    if job['kind'] == 'cohort':
        if not _is_number(job['cohort_pct']) or not 0 < job['cohort_pct'] <= 50:
            raise ValueError("A cohort comparison needs cohort_pct above 0 and at most 50.")
    elif job['cohort_pct'] is not None:
        raise ValueError("cohort_pct is only for cohort comparisons.")
    if job['kind'] == 'ranked':
        date = job['rebalance_date']
        if not isinstance(date, str) or not _MONTH.match(date):
            raise ValueError("A ranked list needs rebalance_date as 'YYYY-MM'.")
        if date not in months[:-1]:
            raise ValueError(f"{date} is not a rebalance month of this backtest "
                             f"({months[0]} to {months[-2]}, {job['frequency']}).")
    elif job['rebalance_date'] is not None:
        raise ValueError("rebalance_date is only for ranked lists.")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def engine_arguments(job: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Dict[str, Any]]:
    """(factors, directions, keyword arguments) of run_wrds_backtest for a validated job."""
    options = {k: job[k] for k in ('frequency', 'start_year', 'end_year', 'initial_aum', 'top_pct',
                                   'use_market_cap_weight', 'factor_combination', 'restrict_fossil_fuels',
                                   'selected_sectors', 'factor_labels')}
    return list(job['factors']), dict(job['factor_directions']), options


def describe_wrds_job(job: Dict[str, Any]) -> str:
    """One-line summary of a WRDS job, for logs."""
    factors = ', '.join(f"{job['factor_labels'].get(f, f)} ({job['factor_directions'][f]})" for f in job['factors'])
    extra = (f", cohort {job['cohort_pct']:g}%" if job['kind'] == 'cohort'
             else f", ranked {job['rebalance_date']}" if job['kind'] == 'ranked' else '')
    return (f"wrds {job['kind']} {job['frequency']} {job['start_year']}-{job['end_year']} "
            f"{job['factor_combination']}: {factors}{extra}")
