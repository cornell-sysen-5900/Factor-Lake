"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/compare.py
PURPOSE: Pure helpers for the Compare tab: saved analyses (Legacy or WRDS) side by side,
         as rows of settings and metrics with the best value of each row, the rows that
         differ, the sort order and each analysis's growth series on a date axis.
         No Streamlit calls, so they can be unit tested directly.
VERSION: 1.0.0

Everything comes from the saved results (src/saved_analyses.load_saved_analysis): nothing
is recomputed, and the metrics use the same formulas, names and formats as the Results tab.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.saved_runs import trading_cost_suffix
from app.wrds_tables import FREQUENCY_LABELS, period_label, short_benchmark
from src.trading_costs import has_trading_costs, job_trading_cost

MIN_COMPARED, MAX_COMPARED = 2, 4
NA = 'n/a'
GROUPS = ('Settings', 'Performance', 'Risk', 'Costs')
# Added to the name of a row whose values differ
DIFFERS_MARK = ' ≠'

# The result keys the Compare tab uses. The rest (holdings, the win/loss ledger, ...) is
# dropped when an analysis is loaded, so comparing keeps little in memory.
RESULT_KEYS = (
    'final_value', 'portfolio_values', 'yearly_returns', 'benchmark_returns', 'years', 'frequency',
    'periods_per_year', 'rebalance_dates', 'period_end_dates', 'benchmark_labels',
    'vol_raw_portfolio', 'sharpe_portfolio', 'max_drawdown_portfolio', 'portfolio_beta', 'information_ratio',
    'trading_cost_bps', 'cagr_before_costs', 'average_turnover_pct', 'total_trading_cost', 'cost_drag',
)
COMBINATION_LABELS = {'mixing': 'Mixing', 'integrating': 'Integrating'}
LEGACY_DELISTING = {'zero_return': 'Zero Return', 'hold_cash': 'Hold Cash (risk-free rate)',
                    'reinvest': 'Reinvest (pro-rata)'}
WRDS_DELISTING = 'CRSP delisting returns, then the risk-free rate'
LEGACY_BENCHMARK = 'Russell 2000'
# Legacy rebalances once a year on the 9/30 data, so its yearly values go on September 30
LEGACY_DATE = '{year}-09-30'


def compact_analysis(saved: Dict[str, Any]) -> Dict[str, Any]:
    """
    A loaded saved analysis ({'id', 'job', 'result', ...}) as the Compare tab keeps it:
    its ID, job, the result keys it uses (RESULT_KEYS), mode, frequency and label.
    """
    job, result = saved['job'], saved['result']
    wrds = job.get('data_mode') == 'wrds'
    frequency = FREQUENCY_LABELS.get(job['frequency'], job['frequency']) if wrds else 'Annual'
    labels = [job['factor_labels'].get(f, f) for f in job['factors']]
    factors = labels[0] + (f" +{len(labels) - 1}" if len(labels) > 1 else '') if labels else 'No factors'
    return {
        'id': saved['id'],
        'job': job,
        'result': {k: result[k] for k in RESULT_KEYS if k in result},
        'mode': 'wrds' if wrds else 'legacy',
        'frequency': frequency,
        'label': f"{'WRDS' if wrds else 'Legacy'}, {frequency}: {factors}{trading_cost_suffix(job_trading_cost(job))}",
    }


def short_id(analysis_id: str) -> str:
    """The first 8 characters of an analysis ID."""
    return analysis_id[:8]


def column_titles(analyses: Sequence[Dict[str, Any]]) -> List[str]:
    """Each analysis's column header: its label and short ID, e.g. "WRDS, Quarterly: ROE +1 · 1a2b3c4d"."""
    return [f"{a['label']} · {short_id(a['id'])}" for a in analyses]


# ---- Metrics, as the Results tab computes them ---------------------------------------

def _initial(a: Dict[str, Any]) -> float:
    return float(a['job']['initial_aum'])


def _years(a: Dict[str, Any]) -> float:
    r = a['result']
    return len(r.get('yearly_returns') or []) / (r.get('periods_per_year') or 1)


def final_value(a: Dict[str, Any]) -> Optional[float]:
    return a['result'].get('final_value')


def total_return(a: Dict[str, Any]) -> Optional[float]:
    final = final_value(a)
    return None if final is None else final / _initial(a) - 1


def cagr(a: Dict[str, Any]) -> Optional[float]:
    """The Performance Summary's CAGR: years = periods / periods per year."""
    final, n_years = final_value(a), _years(a)
    if final is None:
        return None
    return (final / _initial(a)) ** (1 / n_years) - 1 if n_years > 0 else 0.0


def outperformance(a: Dict[str, Any]) -> Optional[float]:
    """Cumulative outperformance against the analysis's own benchmark (Russell 2000 or IWM)."""
    final, bench = final_value(a), a['result'].get('benchmark_returns')
    if final is None or bench is None:
        return None
    return final / (_initial(a) * float(np.prod([1 + r / 100.0 for r in bench]))) - 1


def _period_names(a: Dict[str, Any]) -> List[str]:
    """Each period's name: the year for Legacy (as its Year-by-Year table), start to end for WRDS."""
    r = a['result']
    n = len(r.get('yearly_returns') or [])
    if a['mode'] == 'wrds':
        return [period_label(s, e) for s, e in zip(r['rebalance_dates'][:n], r['period_end_dates'][:n])]
    return [str(int(y)) for y in r.get('years', [])[:n]]


def _extreme_period(a: Dict[str, Any], pick: Callable) -> Optional[Tuple[float, str]]:
    returns = a['result'].get('yearly_returns') or []
    if not returns:
        return None
    i = int(pick(np.asarray(returns, dtype=float)))
    return returns[i] / 100.0, _period_names(a)[i]


def best_period(a: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    return _extreme_period(a, np.argmax)


def worst_period(a: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    return _extreme_period(a, np.argmin)


def _result(key: str) -> Callable[[Dict[str, Any]], Optional[float]]:
    return lambda a: a['result'].get(key)


def cagr_before_costs(a: Dict[str, Any]) -> Optional[float]:
    """With trading costs, the result's CAGR before costs; without, the CAGR (nothing was charged)."""
    return a['result']['cagr_before_costs'] if has_trading_costs(a['result']) else cagr(a)


def average_turnover(a: Dict[str, Any]) -> Optional[float]:
    """Average turnover per rebalance in percent, or None (n/a) when the result does not have it."""
    return a['result'].get('average_turnover_pct')


def total_cost(a: Dict[str, Any]) -> float:
    return a['result'].get('total_trading_cost', 0.0) if has_trading_costs(a['result']) else 0.0


def cost_drag(a: Dict[str, Any]) -> float:
    return a['result'].get('cost_drag', 0.0) if has_trading_costs(a['result']) else 0.0


# ---- Settings, as the Results tab's captions describe them ---------------------------

def _factors(a: Dict[str, Any]) -> str:
    job = a['job']
    return ', '.join(f"{job['factor_labels'].get(f, f)} ({'higher' if job['factor_directions'][f] == 'top' else 'lower'})"
                     for f in job['factors'])


def _period(a: Dict[str, Any]) -> str:
    job, years = a['job'], a['result'].get('years') or []
    text = f"{job['start_year']}-{job['end_year']}"
    if a['mode'] == 'wrds' and years:
        text += f" ({years[0]} to {years[-1]})"
    return text


def _sectors(a: Dict[str, Any]) -> str:
    sectors = a['job'].get('selected_sectors') or []
    return ', '.join(sectors) if sectors else 'All sectors'


def _benchmark(a: Dict[str, Any]) -> str:
    return short_benchmark(a['result'], 'benchmark') if a['mode'] == 'wrds' else LEGACY_BENCHMARK


def _delisting(a: Dict[str, Any]) -> str:
    if a['mode'] == 'wrds':
        return WRDS_DELISTING
    return LEGACY_DELISTING.get(a['job'].get('delisting_strategy'), 'Zero Return')


# ---- Rows -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    """
    One row of the comparison. `value` gives a number (None for n/a) or, for settings,
    text. `best` is 'higher', 'lower' or None (settings, and metrics with no better side
    such as beta); `decimals` is the precision shown, so values that look the same tie.
    `per_period` rows (a period's return, turnover per rebalance) only have a best value
    among analyses of the same frequency: a best year and a best month do not compare.
    `in_dollars` rows (final value, cost paid) only have a best value when every compared
    analysis has the same initial AUM: dollars from $1,000 and from $1,000,000 do not
    compare. Rates (returns, CAGR, ratios, cost drag) are always compared.
    """
    group: str
    name: str
    value: Callable[[Dict[str, Any]], Any]
    show: Callable[[Any], str] = str
    best: Optional[str] = None
    decimals: int = 4
    per_period: bool = False
    in_dollars: bool = False


def _pct(v: float) -> str:
    return f"{v:.2%}"


def _period_pct(v: Tuple[float, str]) -> str:
    return f"{v[0]:.2%} ({v[1]})"


ROWS: Tuple[Row, ...] = (
    Row('Settings', 'Data source', lambda a: 'WRDS' if a['mode'] == 'wrds' else 'Legacy'),
    Row('Settings', 'Frequency', lambda a: a['frequency']),
    Row('Settings', 'Period', _period),
    Row('Settings', 'Initial AUM', lambda a: f"${_initial(a):,.0f}"),
    Row('Settings', 'Factors', _factors),
    Row('Settings', 'Factor Combination',
        lambda a: COMBINATION_LABELS.get(a['job'].get('factor_combination'), 'Mixing')),
    Row('Settings', 'Weighting',
        lambda a: 'Market Cap Weighted' if a['job'].get('use_market_cap_weight') else 'Equal Weighted'),
    Row('Settings', 'Sectors', _sectors),
    Row('Settings', 'Fossil fuels', lambda a: 'Excluded' if a['job'].get('restrict_fossil_fuels') else 'Included'),
    Row('Settings', 'Delisting Strategy', _delisting),
    Row('Settings', 'Benchmark', _benchmark),
    Row('Settings', 'Trading Cost', lambda a: f"{job_trading_cost(a['job']):g} bps"),
    Row('Performance', 'Final Portfolio Value', final_value, lambda v: f"${v:,.2f}", 'higher', 2,
        in_dollars=True),
    Row('Performance', 'Total Return', total_return, _pct, 'higher'),
    Row('Performance', 'CAGR', cagr, _pct, 'higher'),
    Row('Performance', 'Cumulative Outperformance', outperformance, _pct, 'higher'),
    Row('Performance', 'Best Period', best_period, _period_pct, 'higher', per_period=True),
    Row('Performance', 'Worst Period', worst_period, _period_pct, 'higher', per_period=True),
    Row('Risk', 'Volatility (Raw)', _result('vol_raw_portfolio'), _pct, 'lower'),
    Row('Risk', 'Sharpe', _result('sharpe_portfolio'), lambda v: f"{v:.4f}", 'higher'),
    # Drawdowns are negative: the best is the smallest loss, the highest value
    Row('Risk', 'Max Drawdown', _result('max_drawdown_portfolio'), _pct, 'higher'),
    Row('Risk', 'Beta vs Benchmark', _result('portfolio_beta'), lambda v: f"{v:.3f}", None, 3),
    Row('Risk', 'Information Ratio', _result('information_ratio'), lambda v: f"{v:.4f}", 'higher'),
    Row('Costs', 'CAGR Before Costs', cagr_before_costs, _pct, 'higher'),
    Row('Costs', 'Avg Turnover per Rebalance', average_turnover, lambda v: f"{v:.2f}%", 'lower', 2,
        per_period=True),
    Row('Costs', 'Total Trading Cost Paid', total_cost, lambda v: f"${v:,.2f}", 'lower', 2, in_dollars=True),
    Row('Costs', 'Annualized Cost Drag', cost_drag, _pct, 'lower'),
)
ROWS_BY_NAME = {row.name: row for row in ROWS}

# The Sort by menu: label -> the row it sorts by (None keeps the order entered)
SORT_OPTIONS = {
    'Entered order': None,
    'CAGR': 'CAGR',
    'Total return': 'Total Return',
    'Sharpe': 'Sharpe',
    'Max drawdown': 'Max Drawdown',
    'Volatility': 'Volatility (Raw)',
    'Information ratio': 'Information Ratio',
    'Cost drag': 'Annualized Cost Drag',
}


def _number(value: Any) -> Optional[float]:
    """A row value as the number it compares by (a period's return for best/worst period)."""
    if value is None:
        return None
    if isinstance(value, tuple):
        value = value[0]
    value = float(value)
    return None if np.isnan(value) else value


def _key(row: Row, value: Any) -> Optional[float]:
    """The value rounded as shown, so values that look the same compare equal; None for n/a."""
    number = _number(value)
    if number is None:
        return None
    # Percent formats show 2 decimals of a percent: 4 of the fraction
    return round(number, row.decimals)


def best_positions(row: Row, values: Sequence[Any], frequencies: Optional[Sequence[str]] = None,
                   initial_aums: Optional[Sequence[float]] = None) -> List[int]:
    """
    The positions of the row's best value: every analysis that ties for best (as shown).
    None are best (an empty list) for settings and rows with no better side, when fewer
    than 2 values are numbers (n/a is never best), when every number ties, for per-period
    rows when the analyses with a value differ in frequency, and for dollar rows when the
    compared analyses differ in initial AUM.
    """
    if row.best is None:
        return []
    keys = [(i, _key(row, v)) for i, v in enumerate(values)]
    keys = [(i, k) for i, k in keys if k is not None]
    if len(keys) < 2:
        return []
    if row.per_period and frequencies is not None and len({frequencies[i] for i, _ in keys}) > 1:
        return []
    if row.in_dollars and initial_aums is not None and not same_initial_aum(initial_aums):
        return []
    target = (max if row.best == 'higher' else min)(k for _, k in keys)
    winners = [i for i, k in keys if k == target]
    return winners if len(winners) < len(keys) else []


def same_initial_aum(initial_aums: Sequence[float]) -> bool:
    """True when every analysis starts with the same initial AUM (to the cent)."""
    return len({round(float(v), 2) for v in initial_aums}) <= 1


def display(row: Row, value: Any) -> str:
    """A row value as shown: formatted, or n/a."""
    if row.group == 'Settings':
        return str(value)
    return NA if _number(value) is None else row.show(value)


def comparison_rows(analyses: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Every row for these analyses (in this order): {'group', 'name', 'values' (shown text,
    one per analysis), 'best' (positions of the best value, empty for none), 'differs'}.
    """
    rows = []
    for row in ROWS:
        values = [row.value(a) for a in analyses]
        shown = [display(row, v) for v in values]
        rows.append({'group': row.group, 'name': row.name, 'values': shown,
                     'best': best_positions(row, values, [a['frequency'] for a in analyses],
                                        [_initial(a) for a in analyses]),
                     'differs': len(set(shown)) > 1})
    return rows


def sort_order(analyses: Sequence[Dict[str, Any]], option: str) -> List[int]:
    """
    The analyses' positions sorted by a Sort by option: best first by the same rule as the
    highlight, n/a last, ties (as shown) in the order entered. 'Entered order' keeps it.
    """
    name = SORT_OPTIONS.get(option)
    if name is None:
        return list(range(len(analyses)))
    row = ROWS_BY_NAME[name]
    keys = [_key(row, row.value(a)) for a in analyses]
    sign = -1 if row.best == 'higher' else 1

    def order(i: int):
        return (keys[i] is None, sign * keys[i] if keys[i] is not None else 0.0, i)
    return sorted(range(len(analyses)), key=order)


# ---- Growth chart ---------------------------------------------------------------------

def wealth_series(returns: Optional[Sequence[float]], initial: float) -> Optional[List[float]]:
    """Percent returns as a value series starting at `initial` (as the Results tab builds its benchmarks)."""
    if not returns:
        return None
    values = [initial]
    for r in returns:
        values.append(values[-1] * (1 + r / 100.0))
    return values


def growth_series(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    An analysis's growth chart data: 'dates' (ISO dates; Legacy years on September 30),
    'portfolio' (after costs), 'benchmark' (its own, from the same initial AUM) and
    'benchmark_name'.
    """
    r = a['result']
    years = r.get('years') or []
    dates = list(years) if a['mode'] == 'wrds' else [LEGACY_DATE.format(year=int(y)) for y in years]
    return {'dates': dates, 'portfolio': list(r.get('portfolio_values') or []),
            'benchmark': wealth_series(r.get('benchmark_returns'), _initial(a)),
            'benchmark_name': _benchmark(a)}


# Lines of analyses with different initial AUMs are drawn as the growth of this amount
GROWTH_BASE = 1000.0


def chart_series(analyses: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Every analysis's growth_series for the chart, and whether they were scaled: when the
    initial AUMs differ, each portfolio and its benchmark are scaled to start at
    GROWTH_BASE ("Growth of $1,000"), so the lines compare; when they match, the dollar
    values are kept.
    """
    series = [growth_series(a) for a in analyses]
    if same_initial_aum([_initial(a) for a in analyses]):
        return series, False
    for s, a in zip(series, analyses):
        scale = GROWTH_BASE / _initial(a)
        s['portfolio'] = [v * scale for v in s['portfolio']]
        if s['benchmark']:
            s['benchmark'] = [v * scale for v in s['benchmark']]
    return series, True


# ---- The table ------------------------------------------------------------------------

def comparison_table(analyses: List[Dict[str, Any]], order: List[int],
                     differences_only: bool) -> Optional[pd.DataFrame]:
    """
    The table shown: one row per setting or metric under its group's row, one column per
    analysis in the sorted order, a ≠ after the name of rows that differ. With
    differences_only, the rows that are the same (and empty groups) are left out.
    Each row's best columns are in df.attrs['best'] ({row name: [columns]}). None if no row is left.
    """
    titles = column_titles(analyses)
    columns = [titles[i] for i in order]
    index, data, best, groups = [], [], {}, set()
    rows = comparison_rows(analyses)
    for group in GROUPS:
        kept = [r for r in rows if r['group'] == group and (r['differs'] or not differences_only)]
        if not kept:
            continue
        index.append(group.upper())
        data.append([''] * len(columns))
        groups.add(group.upper())
        for r in kept:
            name = r['name'] + (DIFFERS_MARK if r['differs'] else '')
            index.append(name)
            data.append([r['values'][i] for i in order])
            if r['best']:
                best[name] = [titles[i] for i in r['best']]
    if not index:
        return None
    table = pd.DataFrame(data, index=index, columns=columns)
    table.attrs.update(best=best, groups=groups)
    return table
