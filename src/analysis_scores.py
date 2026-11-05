"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/analysis_scores.py
PURPOSE: An analysis's headline metrics ("scores"), saved with it in S3 so the scoreboard
         can rank every saved analysis without downloading any of them.
VERSION: 1.0.0

When the backtest Lambda saves an analysis (src/backtest_service.py, src/wrds_service.py),
it adds score_metadata(job, result) to the result's S3 metadata: one small JSON value with
the analysis's settings (to filter on) and metrics (to rank by). The metrics use the
Results tab's formulas (and the Compare tab's, app/compare.py), so a scoreboard row shows
exactly what opening the analysis shows.

This module is not part of the backtest code version (ENGINE_SOURCES), so changing it never
recomputes saved results.
"""

import json
import math
from typing import Any, Dict, Optional

import numpy as np

from .trading_costs import job_trading_cost
from .wrds_calendar import FREQUENCY_LABELS

# The S3 metadata field, and the version of its JSON (bump it when the layout changes)
SCORES_FIELD = 'scores'
SCORES_VERSION = 1
# S3 allows 2 KB of metadata in all, shared with the summary (up to 500 characters)
MAX_SCORES_LENGTH = 1200

# The metrics saved, by name. Each is a number, or None when the result does not have it.
METRICS = ('final_value', 'total_return', 'cagr', 'outperformance', 'sharpe', 'max_drawdown',
           'volatility', 'win_rate')


def _number(value: Any) -> Optional[float]:
    """A float, or None for a missing, NaN or infinite value (JSON has no NaN)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def score_result(job: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    An analysis's headline metrics (METRICS), as the Results tab shows them:
    CAGR counts years as periods / periods per year, cumulative outperformance is against
    the analysis's own benchmark (Russell 2000, or IWM for WRDS), and Sharpe, max drawdown,
    raw volatility and win rate are the result's own.
    """
    initial = float(job['initial_aum'])
    final = _number(result.get('final_value'))
    n_years = len(result.get('yearly_returns') or []) / (result.get('periods_per_year') or 1)
    cagr = outperformance = total_return = None
    if final is not None and initial > 0:
        total_return = final / initial - 1
        cagr = (final / initial) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        bench = result.get('benchmark_returns')
        if bench is not None:
            outperformance = final / (initial * float(np.prod([1 + r / 100.0 for r in bench]))) - 1
    return {
        'final_value': final,
        'total_return': _number(total_return),
        'cagr': _number(cagr),
        'outperformance': _number(outperformance),
        'sharpe': _number(result.get('sharpe_portfolio')),
        'max_drawdown': _number(result.get('max_drawdown_portfolio')),
        'volatility': _number(result.get('vol_raw_portfolio')),
        'win_rate': _number(result.get('win_rate')),
    }


def analysis_scores(job: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    What the scoreboard keeps of an analysis: its settings to filter on ('mode', 'frequency',
    'combination', 'start_year', 'end_year', 'initial_aum', 'trading_cost_bps', 'factors'
    as display labels) and its metrics ('metrics', see score_result).
    """
    wrds = job.get('data_mode') == 'wrds'
    labels = job.get('factor_labels') or {}
    return {
        'v': SCORES_VERSION,
        'mode': 'wrds' if wrds else 'legacy',
        'frequency': FREQUENCY_LABELS.get(job.get('frequency'), job.get('frequency')) if wrds else 'Annual',
        'combination': job.get('factor_combination') or 'mixing',
        'start_year': job.get('start_year'),
        'end_year': job.get('end_year'),
        'initial_aum': float(job['initial_aum']),
        'trading_cost_bps': job_trading_cost(job),
        'factors': [labels.get(f, f) for f in job.get('factors', [])],
        'metrics': score_result(job, result),
    }


def encode_scores(scores: Dict[str, Any]) -> str:
    """
    The scores as an S3 metadata value: compact ASCII JSON of at most MAX_SCORES_LENGTH
    characters. Long factor labels are replaced by their count, never cut mid-label.
    """
    text = json.dumps(scores, separators=(',', ':'), ensure_ascii=True)
    if len(text) > MAX_SCORES_LENGTH:
        text = json.dumps({**scores, 'factors': None}, separators=(',', ':'), ensure_ascii=True)
    return text


def score_metadata(job: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, str]:
    """
    The S3 metadata to add when saving a result: {SCORES_FIELD: JSON} for an analysis
    (kind 'backtest'), nothing for cohort comparisons and ranked lists.
    """
    if job.get('kind') != 'backtest':
        return {}
    return {SCORES_FIELD: encode_scores(analysis_scores(job, result))}


def decode_scores(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The scores saved in S3 metadata, or None when there are none or they cannot be read."""
    if not text:
        return None
    try:
        scores = json.loads(text)
    except ValueError:
        return None
    if not isinstance(scores, dict) or scores.get('v') != SCORES_VERSION or not isinstance(scores.get('metrics'), dict):
        return None
    return scores
