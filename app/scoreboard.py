"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/scoreboard.py
PURPOSE: Pure helpers for the Scoreboard tab: everyone's saved analyses ranked by a metric,
         with filters by data source, factor combination, factors and period.
         No Streamlit calls, so they can be unit tested directly.
VERSION: 1.0.0

The entries come from src/saved_analyses.list_scored_analyses: each analysis's scores
(src/analysis_scores.py) were saved with it, so nothing is downloaded or recomputed.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# How many rows the scoreboard shows until "Show all" is on
TOP_ROWS = 20
COMBINATION_LABELS = {'mixing': 'Mixing', 'integrating': 'Integrating'}
NOT_SCORED = 'Not scored'
NO_SUMMARY = 'Details not recorded'


@dataclass(frozen=True)
class Metric:
    """
    A metric the scoreboard ranks by: its key in the scores, its column name, which side is
    better ('higher' or 'lower'), whether it is a fraction shown as a percent, and the
    decimals shown (values that look the same tie).
    """
    key: str
    name: str
    best: str = 'higher'
    percent: bool = True
    decimals: int = 4


METRICS: Tuple[Metric, ...] = (
    Metric('sharpe', 'Sharpe', percent=False),
    Metric('cagr', 'CAGR'),
    Metric('outperformance', 'Outperformance'),
    # Drawdowns are negative: the best is the smallest loss, the highest value
    Metric('max_drawdown', 'Max Drawdown'),
    Metric('volatility', 'Volatility', best='lower'),
    Metric('win_rate', 'Win Rate'),
    Metric('total_return', 'Total Return'),
    Metric('final_value', 'Final Value', percent=False, decimals=2),
)
METRICS_BY_NAME = {m.name: m for m in METRICS}
DEFAULT_RANK = 'Sharpe'


def source_label(scores: Dict[str, Any]) -> str:
    """The data source and frequency, e.g. "Legacy, Annual" or "WRDS, Quarterly"."""
    return f"{'WRDS' if scores.get('mode') == 'wrds' else 'Legacy'}, {scores.get('frequency') or 'Annual'}"


def strategy_key(entry: Dict[str, Any]) -> str:
    """
    What makes two scored analyses the same row: the same settings except the initial amount,
    and the same rates as shown (every metric except final value, which scales with the amount).
    """
    scores = entry['scores']
    settings = {k: v for k, v in scores.items() if k not in ('initial_aum', 'metrics', 'factors')}
    rates = {m.key: metric_value(entry, m) for m in METRICS if m.key != 'final_value'}
    return json.dumps([settings, sorted(scores.get('factors') or []), rates], sort_keys=True)


def distinct_entries(entries: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Each strategy once: scored analyses with the same strategy_key (the same job run again
    after an update, or with another initial amount, e.g. by the Lambda check script) are
    one row, the newest. Returns (the entries kept, in order, and how many were folded).
    """
    newest: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        key = strategy_key(entry)
        if key not in newest or entry['saved_at'] > newest[key]['saved_at']:
            newest[key] = entry
    kept_ids = {e['analysis_id'] for e in newest.values()}
    kept = [e for e in entries if e['analysis_id'] in kept_ids]
    return kept, len(entries) - len(kept)


def filter_options(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The choices of each filter among the scored entries: 'sources', 'combinations' and
    'factors' (sorted), and 'years' (first and last year of any period, None without scores).
    """
    scored = [e['scores'] for e in entries if e.get('scores')]
    years = [y for s in scored for y in (s.get('start_year'), s.get('end_year')) if isinstance(y, int)]
    return {
        'sources': sorted({source_label(s) for s in scored}),
        'combinations': [label for key, label in COMBINATION_LABELS.items()
                         if any(s.get('combination') == key for s in scored)],
        'factors': sorted({f for s in scored for f in (s.get('factors') or [])}, key=str.lower),
        'years': (min(years), max(years)) if years else None,
    }


def filter_entries(entries: Sequence[Dict[str, Any]], sources: Sequence[str] = (),
                   combinations: Sequence[str] = (), factors: Sequence[str] = (),
                   years: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    """
    The entries that pass every filter set: a source and a combination among those chosen,
    every chosen factor, and a whole period within `years`. An empty filter passes everything.
    Unscored entries have nothing to filter on, so they are only kept while no filter is set.
    """
    active = bool(sources or combinations or factors or years)
    kept = []
    for entry in entries:
        scores = entry.get('scores')
        if not scores:
            if not active:
                kept.append(entry)
            continue
        if sources and source_label(scores) not in sources:
            continue
        if combinations and COMBINATION_LABELS.get(scores.get('combination')) not in combinations:
            continue
        if factors and not set(factors) <= set(scores.get('factors') or []):
            continue
        if years and not (years[0] <= (scores.get('start_year') or years[0])
                          and (scores.get('end_year') or years[1]) <= years[1]):
            continue
        kept.append(entry)
    return kept


def metric_value(entry: Dict[str, Any], metric: Metric) -> Optional[float]:
    """An entry's value of a metric, rounded as shown; None when it has none."""
    value = ((entry.get('scores') or {}).get('metrics') or {}).get(metric.key)
    if value is None:
        return None
    # Percents show 2 decimals of a percent: 4 of the fraction
    return round(float(value), metric.decimals)


def rank_entries(entries: Sequence[Dict[str, Any]], rank_by: str = DEFAULT_RANK) -> List[Dict[str, Any]]:
    """
    The entries ranked by a metric, best first, each a copy with 'rank': entries whose
    values tie (as shown) share a rank (1, 1, 3), and a tie is listed newest first. Entries
    without a value (n/a, or not scored) come last with rank None, scored before not scored.
    """
    metric = METRICS_BY_NAME[rank_by]
    sign = -1 if metric.best == 'higher' else 1

    def order(entry: Dict[str, Any]):
        value = metric_value(entry, metric)
        return (value is None, not entry.get('scores'), sign * value if value is not None else 0.0,
                -entry['saved_at'].timestamp())
    ranked, previous, rank = [], None, 0
    for position, entry in enumerate(sorted(entries, key=order), start=1):
        value = metric_value(entry, metric)
        if value is not None and value != previous:
            rank, previous = position, value
        ranked.append({**entry, 'rank': rank if value is not None else None})
    return ranked


def _period(scores: Dict[str, Any]) -> str:
    return f"{scores.get('start_year')}-{scores.get('end_year')}"


def row_label(entry: Dict[str, Any]) -> str:
    """An entry as the analysis picker shows it, e.g. "#1 · ROE (higher) | Mixing | 2002-2024 | $1,000"."""
    place = f"#{entry['rank']}" if entry.get('rank') else (NOT_SCORED if not entry.get('scores') else 'n/a')
    return f"{place} · {entry.get('summary') or NO_SUMMARY} · {entry['analysis_id'][:8]}"


def scoreboard_table(ranked: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """
    The table shown: one row per entry in ranked order, with its rank, summary, source,
    period, every metric (percents in percent, e.g. 12.34 for 12.34%; empty for n/a or not
    scored), when it was saved and its short ID.
    """
    rows = []
    for entry in ranked:
        scores = entry.get('scores') or {}
        metrics = scores.get('metrics') or {}
        row = {
            'Rank': entry.get('rank'),
            'Analysis': entry.get('summary') or NO_SUMMARY,
            'Source': source_label(scores) if scores else NOT_SCORED,
            'Period': _period(scores) if scores else '',
        }
        for metric in METRICS:
            value = metrics.get(metric.key)
            row[metric.name] = None if value is None else float(value) * (100 if metric.percent else 1)
        row['Saved (UTC)'] = entry['saved_at'].strftime('%Y-%m-%d %H:%M')
        row['ID'] = entry['analysis_id'][:8]
        rows.append(row)
    columns = ['Rank', 'Analysis', 'Source', 'Period', *[m.name for m in METRICS], 'Saved (UTC)', 'ID']
    table = pd.DataFrame(rows, columns=columns)
    table['Rank'] = table['Rank'].astype('Int64')
    return table
