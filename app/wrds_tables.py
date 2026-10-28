"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/wrds_tables.py
PURPOSE: Pure helpers for the Results tab of WRDS runs: holdings and ranked stocks in Legacy's
         units and formats with WRDS display names, and WRDS results in the shapes Legacy's
         render functions (app/components/results_tab.py) expect.
         No Streamlit calls, so they can be unit tested directly.
VERSION: 2.0.0
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.holdings_table import holdings_number_formats
from app.streamlit_config import WRDS_MILLIONS_FACTORS, WRDS_PERCENT_FACTORS

FREQUENCY_LABELS = {'semiannual': 'Semi-annual', 'quarterly': 'Quarterly', 'monthly': 'Monthly'}
# The three benchmark roles of a WRDS result, its result keys, and the short role name
BENCHMARKS = [('benchmark', 'benchmark_returns', ''), ('value', 'value_benchmark_returns', 'Value'),
              ('growth', 'growth_benchmark_returns', 'Growth')]
# Legacy's tables list the benchmark, then growth, then value
TABLE_ORDER = ('benchmark', 'growth', 'value')


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


def period_label(start: str, end: str) -> str:
    """A holding period as the tables show it: '2026-03-31 to 2026-06-30'."""
    return f"{start} to {end}"


def period_by_period_columns(results: Dict[str, Any], first: str = 'Period',
                             period_return: str = 'Period Return') -> List[str]:
    """Legacy's Year-by-Year headers with the period (or year) and the ETFs in place of Legacy's names."""
    return [first, 'Portfolio Value', period_return] + [f"{short_benchmark(results, r)} Returns" for r in TABLE_ORDER]


def win_loss_columns(results: Dict[str, Any]) -> List[str]:
    """Legacy's Win/Loss headers with the period and the ETFs in place of Legacy's names."""
    columns = ['Period', 'Portfolio Return']
    for role in TABLE_ORDER:
        name = short_benchmark(results, role)
        columns += [f"{name} Return", f"Outperformed {name}"]
    return columns


def calendar_year_results(results: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
    """
    The periods compounded by the calendar year they end in, as (years, a result in
    Legacy's shape) for Legacy's Year-by-Year table: the portfolio value at each year's
    last period end, and each year's portfolio and benchmark returns in percent.
    The first and last years can be partial.
    """
    n = len(results['yearly_returns'])
    ends = np.array([int(d[:4]) for d in results['period_end_dates'][:n]])
    years = sorted(set(ends.tolist()))

    def compound(returns: List[float]) -> List[float]:
        growth = 1.0 + np.asarray(returns[:n], dtype=float) / 100.0
        return [float((np.prod(growth[ends == year]) - 1.0) * 100.0) for year in years]

    last = [int(np.flatnonzero(ends == year)[-1]) for year in years]
    values = results['portfolio_values']
    by_year = {'portfolio_values': [values[0]] + [values[i + 1] for i in last],
               'yearly_returns': compound(results['yearly_returns'])}
    by_year.update({key: compound(results[key]) for _, key, _ in BENCHMARKS})
    return years, by_year


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
    table: the same columns, order and units (weights, returns and contributions in percent),
    factor values under their display names, then the WRDS-only PERMNO and months without
    a return.
    """
    table = pd.DataFrame({'Rank': holdings['rank'].astype(int), 'Ticker': holdings['ticker'].astype(str),
                          'Name': holdings['name'].astype(str), 'Sector': holdings['sector'].astype(str)})
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
    table['PERMNO'] = holdings['permno'].astype(int)
    table['Months Without Return'] = holdings['missing_return_months'].astype(int)
    return table.reset_index(drop=True)


def ranked_display_table(rows: pd.DataFrame, factors: List[str], labels: Dict[str, str],
                         factor_combination: str) -> pd.DataFrame:
    """
    Every stock ranked on one rebalance date (the ranked job's rows) in the holdings
    table's order, names and units: rank by the composite score, Held where the holdings
    table has Picked By, each factor's value and z-score (and, for Mixing, its own rank),
    Weight %, the price and PERMNO.
    """
    table = pd.DataFrame({'Rank': rows['rank'], 'Ticker': rows['ticker'].astype(str),
                          'Name': rows['name'].astype(str), 'Sector': rows['sector'].astype(str),
                          'Held': rows['held'].astype(bool), 'Score': pd.to_numeric(rows['score'])})
    for f in factors:
        label = labels.get(f, f)
        table[factor_header(f, label)] = _factor_values(rows[f], f).to_numpy()
        table[f"{label} Z-Score"] = pd.to_numeric(rows[f'z_{f}']).to_numpy()
        if factor_combination == 'mixing':
            table[f"Rank by {label}"] = rows[f'rank_{f}'].to_numpy()
    table['Weight %'] = pd.to_numeric(rows['weight']).to_numpy() * 100.0
    table['Price'] = pd.to_numeric(rows['price']).to_numpy()
    table['PERMNO'] = rows['permno'].astype(int).to_numpy()
    return table.reset_index(drop=True)


def table_number_formats(table: pd.DataFrame, factors: List[str], labels: Dict[str, str]) -> Dict[str, str]:
    """On-screen number formats: Legacy's holdings formats, plus the WRDS-only columns."""
    headers = [factor_header(f, labels.get(f, f)) for f in factors]
    formats = holdings_number_formats(headers)
    formats.update({'PERMNO': '%d', 'Months Without Return': '%d', 'Price': 'dollar'})
    formats.update({c: '%.4f' for c in table.columns if c.endswith(' Z-Score')})
    formats.update({c: '%d' for c in table.columns if c.startswith('Rank by ')})
    return {c: fmt for c, fmt in formats.items() if c in table.columns}


def frequency_caption(results: Dict[str, Any]) -> str:
    """How the statistics were annualized and which ETF each benchmark name stands for."""
    k = int(results.get('periods_per_year') or 1)
    label = FREQUENCY_LABELS.get(results.get('frequency'), str(results.get('frequency')))
    n = len(results['yearly_returns'])
    names = {role: short_benchmark(results, role).split(' ')[0] for role in TABLE_ORDER}
    return (f"{label} returns ({k} per year, {n} periods = {n / k:.1f} years): volatility, Sharpe and "
            f"information ratios are annualized by √{k}, and win rates count {label.lower()} periods. "
            f"Benchmark = {names['benchmark']}, Growth = {names['growth']}, Value = {names['value']} "
            f"(ETF total returns).")


def result_notes(results: Dict[str, Any]) -> List[str]:
    """
    Captions for the WRDS counts that are not zero, in the style of Legacy's
    "Delisted positions encountered" caption (which WRDS runs show as Legacy does).
    """
    notes = []
    if results.get('missing_return_stock_months'):
        notes.append(f"**Stock-months without a return (counted as 0%):** {results['missing_return_stock_months']}")
    if results.get('rf_filled_months'):
        notes.append(f"**Risk-free months filled from the 3-month T-bill / 12:** "
                     f"{', '.join(results['rf_filled_months'])}")
    if results.get('empty_rebalance_dates'):
        notes.append(f"**Rebalance dates with nothing to buy (the money earned the risk-free rate):** "
                     f"{', '.join(results['empty_rebalance_dates'])}")
    return notes
