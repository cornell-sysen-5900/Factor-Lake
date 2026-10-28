"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/wrds_tables.py
PURPOSE: Pure helpers for the Results tab of WRDS runs: holdings, ranked stocks, period and
         metric tables in Legacy's units and formats, with WRDS display names.
         No Streamlit calls, so they can be unit tested directly.
VERSION: 1.0.0
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.holdings_table import holdings_number_formats
from app.streamlit_config import WRDS_MILLIONS_FACTORS, WRDS_PERCENT_FACTORS

FREQUENCY_LABELS = {'semiannual': 'Semi-annual', 'quarterly': 'Quarterly', 'monthly': 'Monthly'}
# The three benchmark roles of a WRDS result, its result keys, and the short role name
BENCHMARKS = [('benchmark', 'benchmark_returns', ''), ('value', 'value_benchmark_returns', 'Value'),
              ('growth', 'growth_benchmark_returns', 'Growth')]


def benchmark_label(results: Dict[str, Any], role: str) -> str:
    """'IWM (ETF, total return)', 'Value: IWN (ETF, total return)', 'Growth: IWO (ETF, total return)'."""
    label = (results.get('benchmark_labels') or {}).get(role, role)
    short = dict((r, s) for r, _, s in BENCHMARKS)[role]
    return f"{short}: {label}" if short else label


def short_benchmark(results: Dict[str, Any], role: str) -> str:
    """'IWM', 'IWN (Value)', 'IWO (Growth)'."""
    ticker = (results.get('benchmark_labels') or {}).get(role, role).split(' ')[0]
    short = dict((r, s) for r, _, s in BENCHMARKS)[role]
    return f"{ticker} ({short})" if short else ticker


def factor_header(column: str, label: str) -> str:
    """A factor's column header: its display name, with ' %' or ' ($M)' for its unit."""
    if column in WRDS_PERCENT_FACTORS:
        return f"{label} %"
    if column in WRDS_MILLIONS_FACTORS:
        return f"{label} ($M)"
    return label


def _factor_values(values: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(values, errors='coerce')
    return values * 100.0 if column in WRDS_PERCENT_FACTORS else values


def holdings_display_table(holdings: pd.DataFrame, factors: List[str], labels: Dict[str, str]) -> pd.DataFrame:
    """
    One rebalance date's holdings (as stored in the holdings Parquet) as Legacy's holdings
    table: the same columns and units (weights, returns and contributions in percent), plus
    PERMNO, factor values under their display names, and months without a return.
    """
    table = pd.DataFrame({'Rank': holdings['rank'].astype(int), 'Ticker': holdings['ticker'].astype(str),
                          'PERMNO': holdings['permno'].astype(int), 'Name': holdings['name'].astype(str),
                          'Sector': holdings['sector'].astype(str)})
    if 'picked_by' in holdings:
        table['Picked By'] = holdings['picked_by'].astype(str)
    table['Score'] = pd.to_numeric(holdings['score'])
    for f in factors:
        table[factor_header(f, labels.get(f, f))] = _factor_values(holdings[f], f).to_numpy()
    table['Weight %'] = holdings['weight'] * 100.0
    table['Shares'] = holdings['shares']
    table['Buy Price'] = holdings['buy_price']
    table['Dollar Value'] = holdings['dollar_value']
    table['Period Return %'] = holdings['period_return'] * 100.0
    table['Contribution %'] = holdings['contribution'] * 100.0
    table['Delisted'] = holdings['delisted'].astype(bool)
    table['Months Without Return'] = holdings['missing_return_months'].astype(int)
    return table.reset_index(drop=True)


def ranked_display_table(rows: pd.DataFrame, factors: List[str], labels: Dict[str, str],
                         factor_combination: str) -> pd.DataFrame:
    """
    Every stock ranked on one rebalance date (the ranked job's rows) with the holdings
    table's names and units: rank by the composite score, Held and Weight %, each factor's
    value and z-score, and for Mixing each factor's own rank.
    """
    table = pd.DataFrame({'Rank': rows['rank'], 'Ticker': rows['ticker'].astype(str),
                          'PERMNO': rows['permno'].astype(int), 'Name': rows['name'].astype(str),
                          'Sector': rows['sector'].astype(str), 'Held': rows['held'].astype(bool),
                          'Weight %': pd.to_numeric(rows['weight']) * 100.0, 'Score': pd.to_numeric(rows['score'])})
    for f in factors:
        label = labels.get(f, f)
        table[factor_header(f, label)] = _factor_values(rows[f], f).to_numpy()
        table[f"{label} Z-Score"] = pd.to_numeric(rows[f'z_{f}']).to_numpy()
        if factor_combination == 'mixing':
            table[f"Rank by {label}"] = rows[f'rank_{f}'].to_numpy()
    table['Price'] = pd.to_numeric(rows['price']).to_numpy()
    return table.reset_index(drop=True)


def table_number_formats(table: pd.DataFrame, factors: List[str], labels: Dict[str, str]) -> Dict[str, str]:
    """On-screen number formats: Legacy's holdings formats, plus the WRDS-only columns."""
    headers = [factor_header(f, labels.get(f, f)) for f in factors]
    formats = holdings_number_formats(headers)
    formats.update({'PERMNO': '%d', 'Months Without Return': '%d', 'Price': 'dollar'})
    formats.update({c: '%.4f' for c in table.columns if c.endswith(' Z-Score')})
    formats.update({c: '%d' for c in table.columns if c.startswith('Rank by ')})
    return {c: fmt for c, fmt in formats.items() if c in table.columns}


def holdings_period_caption(results: Dict[str, Any], date: str) -> Optional[str]:
    """'(2026-03-31 to 2026-06-30): 3.21%' for the period starting at date, or None."""
    dates = results.get('rebalance_dates') or []
    if date not in dates:
        return None
    i = dates.index(date)
    return f"({date} to {results['period_end_dates'][i]}): {results['yearly_returns'][i]:.2f}%"


def period_table(results: Dict[str, Any]) -> pd.DataFrame:
    """One row per holding period: dates, portfolio value and return, benchmarks, risk-free, wins."""
    n = len(results['yearly_returns'])
    table = pd.DataFrame({
        'Period Start': results['rebalance_dates'][:n],
        'Period End': results['period_end_dates'][:n],
        'Portfolio Value': [f"${v:,.2f}" for v in results['portfolio_values'][1:n + 1]],
        'Portfolio Return': [f"{r:.2f}%" for r in results['yearly_returns']],
    })
    for role, key, _ in BENCHMARKS:
        table[f"{short_benchmark(results, role)} Return"] = [f"{r:.2f}%" for r in results[key][:n]]
    table['Risk-Free'] = [f"{r:.2f}%" for r in results['rf_returns'][:n]]
    for role, key, _ in BENCHMARKS:
        table[f"Beat {short_benchmark(results, role)}"] = [
            '✅' if p > b else '❌' for p, b in zip(results['yearly_returns'], results[key][:n])]
    return table


def calendar_year_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Returns compounded by the calendar year each period ends in (the first and last years may be partial)."""
    n = len(results['yearly_returns'])
    years = [int(d[:4]) for d in results['period_end_dates'][:n]]
    series = {'Portfolio': results['yearly_returns']}
    series.update({short_benchmark(results, role): results[key][:n] for role, key, _ in BENCHMARKS})
    frame = pd.DataFrame({name: np.array(values, dtype=float) / 100.0 for name, values in series.items()})
    frame['Year'] = years
    compounded = frame.groupby('Year').agg(lambda r: np.prod(1.0 + r) - 1.0)
    periods = frame.groupby('Year').size()
    table = pd.DataFrame({'Year': compounded.index, 'Periods': periods.to_numpy()})
    for name in series:
        table[f"{name} Return"] = [f"{r * 100:.2f}%" for r in compounded[name]]
    return table


def _total_and_cagr(returns_pct: List[float], periods_per_year: int):
    growth = float(np.prod([1.0 + r / 100.0 for r in returns_pct]))
    return growth - 1.0, growth ** (periods_per_year / len(returns_pct)) - 1.0 if returns_pct else 0.0


def metrics_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    The advanced statistics as one table: the portfolio and each benchmark side by side,
    returns, volatility and Sharpe annualized by the run's frequency, then the portfolio
    against each benchmark.
    """
    k = int(results.get('periods_per_year') or 1)
    columns = ['Portfolio'] + [short_benchmark(results, role) for role, _, _ in BENCHMARKS]
    suffixes = {'Portfolio': 'portfolio', 'benchmark': 'benchmark', 'value': 'value', 'growth': 'growth'}
    keys = ['Portfolio'] + [role for role, _, _ in BENCHMARKS]
    returns = {'Portfolio': results['yearly_returns']}
    returns.update({role: results[key] for role, key, _ in BENCHMARKS})
    totals = {name: _total_and_cagr(returns[name], k) for name in keys}

    def row(metric, values):
        return dict(zip(['Metric'] + columns, [metric] + values))

    rows = [
        row('Total Return', [f"{totals[n][0]:.2%}" for n in keys]),
        row('CAGR', [f"{totals[n][1]:.2%}" for n in keys]),
        row('Volatility (annualized)', [f"{results[f'vol_raw_{suffixes[n]}']:.2%}" for n in keys]),
        row('Excess Volatility (annualized)', [f"{results[f'vol_excess_{suffixes[n]}']:.2%}" for n in keys]),
        row('Sharpe Ratio (annualized)', [f"{results[f'sharpe_{suffixes[n]}']:.4f}" for n in keys]),
        row('Max Drawdown', [f"{results[f'max_drawdown_{suffixes[n]}']:.2%}" for n in keys]),
        row('Portfolio Beta vs', ['', f"{results['portfolio_beta']:.3f}", f"{results['portfolio_beta_value']:.3f}",
                                  f"{results['portfolio_beta_growth']:.3f}"]),
        row('Information Ratio vs (annualized)', ['', f"{results['information_ratio']:.4f}",
                                                  f"{results['information_ratio_value']:.4f}",
                                                  f"{results['information_ratio_growth']:.4f}"]),
        row('Period Win Rate vs', ['', f"{results['win_rate']:.2%}", f"{results['win_rate_value']:.2%}",
                                   f"{results['win_rate_growth']:.2%}"]),
    ]
    return pd.DataFrame(rows)


def frequency_caption(results: Dict[str, Any]) -> str:
    """How the metrics were annualized, e.g. 'Quarterly returns (4 per year): ... annualized by √4.'"""
    k = int(results.get('periods_per_year') or 1)
    label = FREQUENCY_LABELS.get(results.get('frequency'), str(results.get('frequency')))
    return (f"{label} returns ({k} per year). CAGR uses calendar years ({len(results['yearly_returns'])} periods "
            f"= {len(results['yearly_returns']) / k:.1f} years); volatility, Sharpe and information ratios are "
            f"annualized by √{k}. Risk-free: {results.get('risk_free_rate_source', 'Fama-French')}.")


def result_notes(results: Dict[str, Any]) -> List[str]:
    """Short notes for the counts the engine reports, only those that are not zero."""
    notes = []
    if results.get('total_delisted_positions'):
        notes.append(f"Delisted positions encountered: {results['total_delisted_positions']} (CRSP delisting "
                     f"returns are included; the money then earned the risk-free rate until the next rebalance).")
    if results.get('missing_return_stock_months'):
        notes.append(f"Stock-months without a return (counted as 0%): {results['missing_return_stock_months']}.")
    if results.get('rf_filled_months'):
        months = results['rf_filled_months']
        notes.append(f"Risk-free months filled from the 3-month T-bill / 12: {', '.join(months)}.")
    if results.get('empty_rebalance_dates'):
        dates = results['empty_rebalance_dates']
        notes.append(f"Rebalance dates with nothing to buy (the money earned the risk-free rate): {', '.join(dates)}.")
    return notes


def wealth_series(returns_pct: Optional[List[float]], initial: float) -> Optional[List[float]]:
    """A return series (percent per period) as values starting at initial."""
    if not returns_pct:
        return None
    values = [initial]
    for r in returns_pct:
        values.append(values[-1] * (1.0 + r / 100.0))
    return values


def drawdowns(values: List[float]) -> List[float]:
    """Drawdown from the running peak at each point, as a decimal (0 at a new high)."""
    values = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(values)
    return list(values / peak - 1.0)
