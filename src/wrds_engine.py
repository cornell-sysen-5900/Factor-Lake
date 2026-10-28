"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_engine.py
PURPOSE: Vectorized WRDS backtest: point-in-time universe, factor scores, buy-and-hold periods.
VERSION: 1.0.0

See DOCS/WRDS_MODE_DESIGN.md, sections 3, 4 and 7. At each rebalance date t:

1. Universe: the panel rows at t (Russell 2000 members known at t) with a price and a
   market cap, after the sector and fossil fuel filters. Nothing dated after t is read.
2. Scores: each factor's z-score per date, exactly as Legacy's normalize_series
   (winsorize at 1% / 99%, negate for 'bottom', z-score with ddof=1).
   Integrating ranks the mean z-score of stocks that have every factor; Mixing ranks
   each factor in its own sleeve with an equal share of the money.
3. Selection: the top max(1, floor(n * top_pct / 100)) by score, ties broken by permno,
   equal or market cap (mkt_cap_musd at t) weighted.
4. Holding period (t, t']: buy and hold, so weights drift with each stock's compounded
   monthly returns (delisting returns included). After a stock's delisting month its
   money earns the risk-free rate until t'. A month without a return, before any
   delisting, counts as 0% and is counted in missing_return_stock_months.

Stocks are keyed on permno (two securities can share a ticker on one date).
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .performance_metrics import compute_comprehensive_metrics
from .wrds_calendar import FIRST_MONTH, LAST_MONTH, PERIODS_PER_YEAR, period_boundaries
from .wrds_data import WrdsData
from .wrds_factors import WRDS_FACTOR_COLUMNS, WRDS_SECTORS

FACTOR_COMBINATIONS = ('mixing', 'integrating')
# Tail share capped at both ends before z-scoring (Legacy's normalize_series default)
WINSORIZE_PCT = 0.01
# Fossil fuel companies, matching Legacy's five FactSet industries (design 2.3):
# GICS sub-industries, and SIC codes for stocks without GICS
FOSSIL_GSUBIND = (10101010, 10101020, 10102010, 10102020, 10102030, 10102050)
FOSSIL_SIC = (1221, 1222, 1231, 1311, 1381, 1382, 1389, 2911)
BENCHMARK_COLUMNS = {'benchmark': 'iwm_total_ret', 'growth': 'iwo_total_ret', 'value': 'iwn_total_ret'}
BENCHMARK_LABELS = {'benchmark': 'IWM (ETF, total return)', 'growth': 'IWO (ETF, total return)',
                    'value': 'IWN (ETF, total return)'}
RISK_FREE_SOURCE = "Fama-French one-month T-bill (a missing month: 3-month T-bill / 12)"


def validate_inputs(factors: List[str], factor_directions: Dict[str, str], factor_combination: str,
                    top_pct: float, selected_sectors: Optional[Iterable[str]]) -> None:
    """Raises ValueError for inputs a WRDS backtest cannot run."""
    if not factors:
        raise ValueError("A backtest needs at least one factor.")
    if bad := [f for f in factors if f not in WRDS_FACTOR_COLUMNS]:
        raise ValueError(f"Not WRDS factors: {bad}. Factors must be columns known at the rebalance date.")
    if bad := [d for d in (factor_directions.get(f, 'top') for f in factors) if d not in ('top', 'bottom')]:
        raise ValueError(f"Factor directions must be 'top' or 'bottom', not {bad}.")
    if factor_combination not in FACTOR_COMBINATIONS:
        raise ValueError(f"factor_combination must be one of {FACTOR_COMBINATIONS}, not '{factor_combination}'.")
    if not 0 < top_pct <= 100:
        raise ValueError("top_pct must be above 0 and at most 100.")
    if bad := [s for s in (selected_sectors or []) if s not in WRDS_SECTORS]:
        raise ValueError(f"Unknown GICS sectors: {bad}.")


def eligible_universe(panel: pd.DataFrame, restrict_fossil_fuels: bool = False,
                      selected_sectors: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    The panel rows a backtest can buy: a positive price and market cap at t, in the
    selected GICS sectors (Unclassified stocks are left out while a sector filter is
    on) and, with restrict_fossil_fuels, not a fossil fuel company.
    """
    keep = (panel['prc'] > 0) & (panel['mkt_cap_musd'] > 0)
    sectors = list(selected_sectors or [])
    if sectors:
        keep &= panel['sector_name'].isin(sectors)
    if restrict_fossil_fuels:
        gsubind = panel['gsubind']
        fossil = (gsubind.isin(FOSSIL_GSUBIND).to_numpy(dtype=bool, na_value=False)
                  | (gsubind.isna().to_numpy() & panel['siccd'].isin(FOSSIL_SIC).to_numpy()))
        keep &= ~fossil
    return panel[keep.to_numpy()]


def factor_zscores(universe: pd.DataFrame, column: str, higher_is_better: bool) -> pd.Series:
    """
    One factor's z-scores within each date, as Legacy's normalize_series computes them
    for one year: non-numbers and infinities become NaN, values are capped at the date's
    1% and 99% quantiles, negated when lower is better, then z-scored (ddof=1). A date
    whose values do not vary is only centered.
    """
    s = pd.to_numeric(universe[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    by_date = s.groupby(universe['date'])
    s = s.clip(by_date.transform('quantile', WINSORIZE_PCT), by_date.transform('quantile', 1 - WINSORIZE_PCT))
    if not higher_is_better:
        s = -s
    by_date = s.groupby(universe['date'])
    centered = s - by_date.transform('mean')
    sigma = by_date.transform('std')
    return pd.Series(np.where(sigma > 1e-12, centered / sigma, centered), index=s.index)


def _top_ranked(universe: pd.DataFrame, scores: pd.Series, top_pct: float,
                use_market_cap_weight: bool) -> pd.DataFrame:
    """
    The top top_pct% by score on each date (at least one), best first with ties broken
    by permno, and each one's weight within the selection.
    """
    ranked = pd.DataFrame({'date': universe['date'], 'permno': universe['permno'], 'score': scores,
                           'cap': universe['mkt_cap_musd']})
    ranked = ranked[ranked['score'].notna()].sort_values(
        ['date', 'score', 'permno'], ascending=[True, False, True], kind='mergesort')
    by_date = ranked.groupby('date', sort=False)
    ranked['rank'] = by_date.cumcount() + 1
    n = by_date['permno'].transform('size')
    picked = ranked[ranked['rank'] <= np.maximum(1, np.floor(n * (top_pct / 100.0)))].copy()
    raw = picked['cap'] if use_market_cap_weight else pd.Series(1.0, index=picked.index)
    picked['weight'] = raw / raw.groupby(picked['date']).transform('sum')
    return picked.drop(columns='cap')


def select_positions(universe: pd.DataFrame, factors: List[str], factor_directions: Dict[str, str],
                     factor_combination: str = 'mixing', top_pct: float = 10.0,
                     use_market_cap_weight: bool = False) -> pd.DataFrame:
    """
    The positions bought at every date in universe: date, sleeve, permno, rank (within
    the sleeve), score, and weight (share of the whole portfolio; each date sums to 1).

    Integrating is one sleeve ranked by the mean z-score of the stocks that have every
    factor. Mixing is one sleeve per factor, each with 1 / len(factors) of the money.
    """
    zscores = {f: factor_zscores(universe, f, factor_directions.get(f, 'top') == 'top') for f in factors}
    if factor_combination == 'integrating':
        composite = pd.concat(zscores.values(), axis=1)
        composite = composite.mean(axis=1).where(composite.notna().all(axis=1))
        sleeves = [_top_ranked(universe, composite, top_pct, use_market_cap_weight).assign(sleeve=0)]
    else:
        sleeves = []
        for i, f in enumerate(factors):
            sleeve = _top_ranked(universe, zscores[f], top_pct, use_market_cap_weight)
            sleeves.append(sleeve.assign(sleeve=i, weight=sleeve['weight'] / len(factors)))
    positions = pd.concat(sleeves, ignore_index=True)
    return positions[['date', 'sleeve', 'permno', 'rank', 'score', 'weight']]


def _risk_free(data: WrdsData) -> Tuple[np.ndarray, np.ndarray]:
    """Monthly risk-free rate (Fama-French rf; a missing month is tbill_3m / 12) and which months were filled."""
    rf = data.market['rf'].to_numpy(dtype=float)
    filled = np.isnan(rf)
    return np.where(filled, data.market['tbill_3m'].to_numpy(dtype=float) / 12.0, rf), filled


def _month_range(data: WrdsData, start: pd.Timestamp, end: pd.Timestamp) -> slice:
    """The months of the holding period (start, end] as a slice of data.months."""
    return slice(data.months.get_loc(start) + 1, data.months.get_loc(end) + 1)


def holding_period_growth(data: WrdsData, permnos: np.ndarray, start: pd.Timestamp,
                          end: pd.Timestamp, rf_monthly: np.ndarray) -> pd.DataFrame:
    """
    Buy-and-hold growth of each stock over (start, end]: the product of (1 + monthly
    return), and after its delisting month (1 + risk-free) instead. A month without a
    return before any delisting counts as 0%.

    Returns one row per permno: growth, delisted (in the period), missing_return_months.
    """
    months = _month_range(data, start, end)
    cols = data.permnos.get_indexer(permnos)
    known = cols >= 0                                      # stocks never in returns_monthly have no returns
    returns = np.where(known, data.returns[months][:, cols], np.nan)
    delisted = np.where(known, data.delisted[months][:, cols], False)
    after_delisting = (np.cumsum(delisted, axis=0) - delisted) > 0
    missing = np.isnan(returns) & ~after_delisting
    monthly = np.where(after_delisting, 1.0 + rf_monthly[months][:, None],
                       np.where(missing, 1.0, 1.0 + np.nan_to_num(returns)))
    return pd.DataFrame({'permno': permnos, 'growth': monthly.prod(axis=0),
                         'delisted': delisted.any(axis=0), 'missing_return_months': missing.sum(axis=0)})


def run_periods(data: WrdsData, dates: List[pd.Timestamp], factors: List[str],
                factor_directions: Dict[str, str], initial_aum: float, periods_per_year: int,
                frequency: Optional[str] = None, top_pct: float = 10.0,
                use_market_cap_weight: bool = False, factor_combination: str = 'mixing',
                restrict_fossil_fuels: bool = False, selected_sectors: Optional[Iterable[str]] = None,
                factor_labels: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Backtests the holding periods between consecutive dates (rebalancing at every date
    but the last). See run_wrds_backtest for the arguments and the results.
    """
    validate_inputs(factors, factor_directions, factor_combination, top_pct, selected_sectors)
    if len(dates) < 2:
        raise ValueError("A backtest needs at least one holding period.")
    factor_labels = factor_labels or {}
    rebalance_dates, period_ends = list(dates[:-1]), list(dates[1:])
    in_panel = pd.DatetimeIndex(rebalance_dates).isin(data.panel_dates())
    if missing := [d.date().isoformat() for d, ok in zip(rebalance_dates, in_panel) if not ok]:
        raise ValueError(f"The WRDS panel has no rows on the rebalance dates {missing}.")

    # Only the columns this run needs, and only the rebalance dates' rows, are copied
    columns = list(dict.fromkeys(['date', 'permno', 'ticker', 'security_name', 'sector_name', 'gsubind',
                                  'siccd', 'prc', 'mkt_cap_musd'] + list(factors)))
    panel = data.panel.loc[data.panel['date'].isin(rebalance_dates).to_numpy(), columns]
    universe = eligible_universe(panel, restrict_fossil_fuels, selected_sectors)
    positions = select_positions(universe, factors, factor_directions, factor_combination,
                                 top_pct, use_market_cap_weight)

    rf_monthly, rf_filled = _risk_free(data)
    by_date = dict(tuple(positions.groupby('date')))
    period_returns, rf_period, growth_rows = [], [], []
    benchmarks = {role: [] for role in BENCHMARK_COLUMNS}
    used_months = np.zeros(len(data.months), dtype=bool)
    for start, end in zip(rebalance_dates, period_ends):
        months = _month_range(data, start, end)
        used_months[months] = True
        rf_period.append(float(np.prod(1.0 + rf_monthly[months]) - 1.0))
        for role, column in BENCHMARK_COLUMNS.items():
            monthly = data.market[column].to_numpy(dtype=float)[months]
            if np.isnan(monthly).any():
                raise ValueError(f"market_monthly has no {column} for a month of {start.date()} to {end.date()}.")
            benchmarks[role].append(float(np.prod(1.0 + monthly) - 1.0))

        held = by_date.get(start)
        if held is None:          # nothing to buy (e.g. a narrow sector filter): the money earns rf
            period_returns.append(rf_period[-1])
            continue
        growth = holding_period_growth(data, held['permno'].unique(), start, end, rf_monthly)
        growth_rows.append(growth.assign(date=start))
        g = held['permno'].map(growth.set_index('permno')['growth'])
        period_returns.append(float((held['weight'] * g).sum() - 1.0))

    values = [float(initial_aum)]
    for r in period_returns:
        values.append(values[-1] * (1.0 + r))

    p_ret = np.array(period_returns)
    results = compute_comprehensive_metrics(
        p_ret, np.array(benchmarks['benchmark']), np.array(benchmarks['growth']),
        np.array(benchmarks['value']), np.array(rf_period), values, 0, len(p_ret),
        periods_per_year=periods_per_year)

    labels = [d.strftime('%Y-%m-%d') for d in dates]
    for i, comparison in enumerate(results['yearly_comparisons']):
        comparison.update(year=period_ends[i].year, period_start=labels[i], period_end=labels[i + 1])
    growth = (pd.concat(growth_rows, ignore_index=True) if growth_rows
              else pd.DataFrame(columns=['date', 'permno', 'growth', 'delisted', 'missing_return_months']))
    results.update(
        data_mode='wrds',
        frequency=frequency,
        periods_per_year=periods_per_year,
        years=labels,                       # start of the first period, then each period's end
        rebalance_dates=labels[:-1],
        period_end_dates=labels[1:],
        cagr=(values[-1] / values[0]) ** (periods_per_year / len(p_ret)) - 1.0,
        rf_returns=(np.array(rf_period) * 100).tolist(),
        benchmark_labels=dict(BENCHMARK_LABELS),
        risk_free_rate_source=RISK_FREE_SOURCE,
        rf_filled_months=[m.strftime('%Y-%m') for m in data.months[rf_filled & used_months]],
        factor_combination=factor_combination,
        total_delisted_positions=int(growth['delisted'].sum()),
        missing_return_stock_months=int(growth['missing_return_months'].sum()),
        empty_rebalance_dates=[d.strftime('%Y-%m-%d') for d in rebalance_dates if d not in by_date],
    )
    holdings = _holdings_table(positions, growth, panel, values, rebalance_dates, factors,
                               factor_combination, factor_labels)
    return results, holdings


def _holdings_table(positions: pd.DataFrame, growth: pd.DataFrame, panel: pd.DataFrame,
                    values: List[float], rebalance_dates: List[pd.Timestamp], factors: List[str],
                    factor_combination: str, factor_labels: Dict[str, str]) -> pd.DataFrame:
    """
    One row per stock held per rebalance date, merged over Mixing sleeves the way
    Legacy's holdings table merges them: weights summed, rank and score from the sleeve
    where it ranked best, picked_by listing every sleeve that bought it.

    Weights, returns and contributions are decimals (Legacy's table shows percent).
    """
    best = positions.sort_values(['date', 'permno', 'rank', 'sleeve']).drop_duplicates(['date', 'permno'])
    table = best[['date', 'permno', 'rank', 'score']].merge(
        positions.groupby(['date', 'permno'], as_index=False)['weight'].sum(), on=['date', 'permno'])
    if factor_combination == 'mixing':
        # Each stock's sleeves as a bitmask (a sleeve buys a stock at most once; at most 40
        # factors exist, so it fits an int64), then one label per distinct mask
        names = [factor_labels.get(f, f) for f in factors]
        masks = positions.assign(mask=np.left_shift(1, positions['sleeve'].to_numpy(dtype=np.int64)))
        picked = masks.groupby(['date', 'permno'], as_index=False)['mask'].sum()
        labels = {m: ', '.join(n for i, n in enumerate(names) if m >> i & 1) for m in picked['mask'].unique()}
        picked['picked_by'] = picked.pop('mask').map(labels)
        table = table.merge(picked, on=['date', 'permno'])
    attributes = panel[['date', 'permno', 'ticker', 'security_name', 'sector_name', 'prc']
                       + list(dict.fromkeys(factors))]
    table = table.merge(attributes, on=['date', 'permno'], how='left').merge(growth, on=['date', 'permno'], how='left')

    start_value = pd.Series(values[:-1], index=rebalance_dates)
    table['dollar_value'] = table['weight'] * table['date'].map(start_value)
    table['shares'] = table['dollar_value'] / table['prc']
    table['period_return'] = table['growth'] - 1.0
    table['contribution'] = table['weight'] * table['period_return']
    table = table.rename(columns={'date': 'rebalance_date', 'security_name': 'name', 'sector_name': 'sector',
                                  'prc': 'buy_price'})
    table['_order'] = -table['score'].round(10)
    table = table.sort_values(['rebalance_date', 'rank', '_order', 'permno']).drop(columns=['_order', 'growth'])
    columns = (['rebalance_date', 'rank', 'permno', 'ticker', 'name', 'sector']
               + (['picked_by'] if factor_combination == 'mixing' else []) + ['score']
               + list(dict.fromkeys(factors))
               + ['weight', 'shares', 'buy_price', 'dollar_value', 'period_return', 'contribution',
                  'delisted', 'missing_return_months'])
    for col in ('ticker', 'name', 'sector'):     # only the values still used
        table[col] = table[col].cat.remove_unused_categories()
    return table[columns].reset_index(drop=True)


def run_wrds_backtest(data: WrdsData, factors: List[str], factor_directions: Dict[str, str],
                      frequency: str, start_year: int, end_year: int, initial_aum: float,
                      top_pct: float = 10.0, use_market_cap_weight: bool = False,
                      factor_combination: str = 'mixing', restrict_fossil_fuels: bool = False,
                      selected_sectors: Optional[Iterable[str]] = None,
                      factor_labels: Optional[Dict[str, str]] = None,
                      first_month: str = FIRST_MONTH,
                      last_month: str = LAST_MONTH) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Backtests factor portfolios on the WRDS data, rebalanced at the frequency's month
    ends over the calendar years start_year to end_year (src/wrds_calendar.py).

    Args:
        factors: allowlisted panel columns (src/wrds_factors.WRDS_FACTOR_COLUMNS).
        factor_directions: column -> 'top' (higher is better) or 'bottom'.
        frequency: 'semiannual', 'quarterly' or 'monthly'.
        factor_labels: column -> display name, for picked_by in the holdings.
        first_month, last_month: the data's bounds (tests use shorter data).

    Returns:
        (results, holdings). results has the keys of compute_comprehensive_metrics, with
        per-period returns (in percent, under the Legacy key names such as yearly_returns),
        metrics annualized by frequency, 'years' as the period boundary dates, plus
        rebalance_dates, period_end_dates, cagr, benchmark_labels, rf_returns,
        rf_filled_months, total_delisted_positions, missing_return_stock_months and
        empty_rebalance_dates. holdings is one row per stock held per rebalance date
        (see _holdings_table).
    """
    months = period_boundaries(frequency, start_year, end_year, first_month, last_month)
    dates = [data.month_date(m) for m in months]
    return run_periods(data, dates, factors, factor_directions, initial_aum, PERIODS_PER_YEAR[frequency],
                       frequency=frequency, top_pct=top_pct, use_market_cap_weight=use_market_cap_weight,
                       factor_combination=factor_combination, restrict_fossil_fuels=restrict_fossil_fuels,
                       selected_sectors=selected_sectors, factor_labels=factor_labels)
