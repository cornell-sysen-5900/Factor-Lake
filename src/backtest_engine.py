"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/backtest_engine.py
PURPOSE: Backtesting engine with integrated on-the-fly filtering.
VERSION: 2.5.0
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Any, Tuple, Optional  # Added Tuple and Optional

from .factor_utils import normalize_series
from .benchmarks import get_benchmark_list
from .portfolio import Portfolio
from src.factor_registry import get_factor_column
from .portfolio_filters import filter_universe
from .performance_metrics import compute_comprehensive_metrics


# How rebalance_portfolio combines several factors (see its docstring)
FACTOR_COMBINATIONS = ('mixing', 'integrating')


def _dedupe_year_slice_by_ticker(df_year: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures each ticker appears once in a given rebalance year slice.

    Some source pulls can contain duplicate (Year, Ticker) rows. Downstream
    `.loc[ticker, col]` access expects a scalar value, so duplicates must be
    removed to avoid ambiguous Series truth-value errors.
    """
    if df_year.empty:
        return df_year
    if df_year.index.has_duplicates:
        return df_year[~df_year.index.duplicated(keep='first')]
    return df_year


def _resolve_factor_column(df_year: pd.DataFrame, factor_key: str) -> str:
    """
    Maps an internal factor key to the column that holds it in df_year.
    """
    # 1. Resolve internal key to database column name
    factor_col = get_factor_column(factor_key)

    # 2. Safety Check & Emergency Path Cleaning
    # If the registry mapping fails, we attempt to resolve common formatting mismatches
    if factor_col not in df_year.columns:
        # Convert "6-Mo Momentum %" -> "6-Mo_Momentum"
        cleaned_fallback = factor_key.replace(" ", "_").replace("%", "").strip("_")
        if cleaned_fallback in df_year.columns:
            factor_col = cleaned_fallback
        else:
            raise KeyError(
                f"Factor column '{factor_col}' not found in data. "
                f"Available columns: {df_year.columns.tolist()}"
            )
    return factor_col


def factor_scores(df_year: pd.DataFrame, factor_key: str, higher_is_better: bool = True) -> pd.Series:
    """
    One factor's z-scores for a rebalance year (see normalize_series).

    higher_is_better=True ranks high values at the top (Long)
    higher_is_better=False ranks low values at the top (Short/Bottom)
    """
    factor_col = _resolve_factor_column(df_year, factor_key)
    return normalize_series(df_year[factor_col], higher_is_better=higher_is_better)


def composite_scores(df_year: pd.DataFrame, factor_keys: List[str],
                     factor_directions: Dict[str, str]) -> pd.Series:
    """
    Mean of the selected factors' z-scores, best first.

    Only stocks that have every selected factor get a score: a stock missing
    any factor is left out rather than scored on the factors it has.
    """
    scores = pd.concat(
        [factor_scores(df_year, f, factor_directions.get(f) == 'top') for f in factor_keys],
        axis=1
    )
    return scores.dropna().mean(axis=1).sort_values(ascending=False)


def _select_top(ranked_scores: pd.Series, top_pct: float) -> pd.Series:
    """The top top_pct% of ranked_scores (at least one), best first, one entry per ticker."""
    n_select = max(1, math.floor(len(ranked_scores) * (top_pct / 100.0)))
    selected = ranked_scores.head(n_select)
    # Preserve order while removing duplicate tickers that can appear in some year slices.
    return selected[~selected.index.duplicated(keep='first')]


def _allocate_positions(df_year: pd.DataFrame, selected_scores: pd.Series, aum: float,
                        use_market_cap_weight: bool, name: str) -> Portfolio:
    """
    Buys the selected stocks with aum, equal or market cap weighted.

    Stocks without a valid positive Ending_Price are not bought. The selection
    scores are kept on the portfolio (Portfolio.scores) for the holdings table.
    """
    selected_tickers = selected_scores.index.tolist()
    holdings_data = df_year.loc[selected_tickers]

    portfolio = Portfolio(name=name)
    portfolio.scores = selected_scores

    # Calculate Weights (Market Cap vs. Equal Weight)
    if use_market_cap_weight and 'Market_Capitalization' in holdings_data.columns:
        caps = pd.to_numeric(holdings_data['Market_Capitalization'], errors='coerce').fillna(0)
        caps_by_ticker = caps.groupby(level=0).sum()
        if caps_by_ticker.sum() > 0:
            weights = caps_by_ticker / caps_by_ticker.sum()
        else:
            weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
    else:
        weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)

    # Allocate Positions
    # We verify required columns exist to prevent runtime errors during the loop
    has_price = 'Ending_Price' in holdings_data.columns

    for ticker in selected_tickers:
        if has_price:
            price = holdings_data.loc[ticker, 'Ending_Price']
            if isinstance(price, pd.Series):
                price_candidates = pd.to_numeric(price, errors='coerce').dropna()
                if price_candidates.empty:
                    continue
                price = float(price_candidates.iloc[0])
            else:
                price = pd.to_numeric(pd.Series([price]), errors='coerce').iloc[0]
                if pd.isna(price):
                    continue
                price = float(price)

            weight = weights.get(ticker, 0.0)
            if isinstance(weight, pd.Series):
                weight = pd.to_numeric(weight, errors='coerce').fillna(0).sum()
            weight = float(weight)
            # Only add investment if price is valid and positive
            if pd.notnull(price) and price > 0:
                shares = (weight * aum) / price
                portfolio.add_investment(ticker, shares)

    return portfolio


def calculate_holdings(
    df_year: pd.DataFrame,
    factor_key: str,
    aum: float,
    higher_is_better: bool = True,
    top_pct: float = 10.0,
    use_market_cap_weight: bool = False
) -> Portfolio:
    """
    Constructs one factor's sleeve for a specific period (Mixing).

    Buys the top top_pct% of stocks by this factor's z-score, equal or market
    cap weighted within the sleeve.
    """
    scores = factor_scores(df_year, factor_key, higher_is_better=higher_is_better)
    valid_scores = scores.dropna().sort_values(ascending=False)

    if valid_scores.empty:
        return Portfolio(name=f"Empty_{factor_key}")

    return _allocate_positions(
        df_year, _select_top(valid_scores, top_pct), aum,
        use_market_cap_weight, name=f"Portfolio_{factor_key}"
    )


def calculate_integrated_holdings(
    df_year: pd.DataFrame,
    factor_keys: List[str],
    factor_directions: Dict[str, str],
    aum: float,
    top_pct: float = 10.0,
    use_market_cap_weight: bool = False
) -> Portfolio:
    """
    Constructs one portfolio from the composite score of all factors (Integrating).

    Buys the top top_pct% of stocks by composite_scores (stocks missing any
    selected factor are left out), one position per stock, equal or market cap
    weighted across the whole list.
    """
    ranked = composite_scores(df_year, factor_keys, factor_directions)

    if ranked.empty:
        return Portfolio(name="Empty_Integrated")

    return _allocate_positions(
        df_year, _select_top(ranked, top_pct), aum,
        use_market_cap_weight, name="Portfolio_Integrated"
    )


def _build_holdings_rows(portfolios: List[Portfolio], positions: List[Dict[str, Any]],
                         df_year: pd.DataFrame, factors: List[str], factor_combination: str,
                         factor_labels: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    One row per stock the backtest held this year, built from the positions it valued.

    portfolios are the year's sleeves (one per factor for Mixing, a single one
    for Integrating) and positions come from _value_positions. A stock bought by
    several Mixing sleeves is one row: its shares and value are summed, Rank and
    Score come from the sleeve where it ranked best, and Picked By lists every
    sleeve that bought it.
    """
    if not positions:
        return []

    # (rank, score) of every stock each sleeve selected
    sleeve_ranks = [
        {t: (i + 1, float(s)) for i, (t, s) in enumerate(p.scores.items())}
        if p.scores is not None else {}
        for p in portfolios
    ]

    by_ticker: Dict[str, Dict[str, Any]] = {}
    for pos in positions:
        rank, score = sleeve_ranks[pos['sleeve']].get(pos['ticker'], (np.nan, np.nan))
        held = by_ticker.get(pos['ticker'])
        if held is None:
            held = by_ticker[pos['ticker']] = {
                'rank': rank, 'score': score, 'sleeves': [], 'shares': 0.0, 'value': 0.0,
                'price': pos['price'], 'return': pos['return'], 'delisted': pos['delisted'],
            }
        elif rank < held['rank']:
            held['rank'], held['score'] = rank, score
        held['sleeves'].append(pos['sleeve'])
        held['shares'] += pos['shares']
        held['value'] += pos['value']

    total_value = sum(pos['value'] for pos in positions)
    factor_cols = {f: _resolve_factor_column(df_year, f) for f in factors}

    rows = []
    # Scores are compared to 10 decimals: the top stocks of different Mixing sleeves often tie
    # exactly in theory, and float rounding (which differs between machines, e.g. a Mac and
    # AWS Lambda) must not decide their order. True ties fall through to the ticker.
    for ticker, held in sorted(by_ticker.items(),
                               key=lambda kv: (kv[1]['rank'], -round(kv[1]['score'], 10), kv[0])):
        row: Dict[str, Any] = {'Rank': held['rank'], 'Ticker': ticker}
        if 'Security_Name' in df_year.columns:
            row['Name'] = df_year.loc[ticker, 'Security_Name']
        if 'Scotts_Sector_5' in df_year.columns:
            row['Sector'] = df_year.loc[ticker, 'Scotts_Sector_5']
        if factor_combination == 'mixing':
            row['Picked By'] = ", ".join(
                factor_labels.get(factors[i], factors[i]) for i in dict.fromkeys(held['sleeves'])
            )
        row['Score'] = held['score']
        for f in factors:
            row[factor_labels.get(f, f)] = float(pd.to_numeric(df_year.loc[ticker, factor_cols[f]], errors='coerce'))
        weight = held['value'] / total_value
        row['Weight %'] = weight * 100.0
        row['Shares'] = held['shares']
        row['Buy Price'] = held['price']
        row['Dollar Value'] = held['value']
        row['Period Return %'] = held['return'] * 100.0
        row['Contribution %'] = weight * held['return'] * 100.0
        row['Delisted'] = held['delisted']
        rows.append(row)
    return rows

def rebalance_portfolio(data: pd.DataFrame, factors: List[str], factor_directions: Dict[str, str],
                        start_year: int, end_year: int, initial_aum: float,
                        benchmark_index: int = 1, top_pct: float = 10.0,
                        use_market_cap_weight: bool = False,
                        delisting_strategy: str = 'zero_return',
                        factor_combination: str = 'mixing',
                        factor_labels: Optional[Dict[str, str]] = None,
                        exclude_fossil: bool = False,
                        selected_sectors: Optional[List[str]] = None) -> Dict[str, Any]:
    """Executes simulation using a filtered working copy of the master data.

    Takes no Streamlit state, so the same code runs in the app and in AWS Lambda.

    Args:
        delisting_strategy: 'zero_return', 'hold_cash', or 'reinvest'.
        factor_combination: how several factors are combined each year.
            'mixing': one sleeve per factor (top top_pct% by that factor),
            AUM split equally across sleeves; a stock picked by two sleeves is
            held in both. 'integrating': one portfolio of the top top_pct% by
            composite_scores.
        factor_labels: factor column -> display label, used for the factor
            columns and Picked By in the holdings table (default: column names).
        exclude_fossil, selected_sectors: extra on-the-fly filters. The app and the
            Lambda filter the data first (src/universe_selection.py), so they leave
            these at their defaults.

    Returns:
        The metrics from compute_comprehensive_metrics, plus 'holdings_by_year':
        {rebalance year: holdings rows (see _build_holdings_rows)} for every
        year whose return is in the results.
    """
    if factor_combination not in FACTOR_COMBINATIONS:
        raise ValueError(f"factor_combination must be one of {FACTOR_COMBINATIONS}, not '{factor_combination}'.")
    factor_labels = factor_labels or {}

    # ON-THE-FLY FILTERING: Apply constraints to a copy
    working_data = filter_universe(data, exclude_fossil, selected_sectors or [])

    aum = initial_aum
    portfolio_values = [aum]
    portfolio_returns = []
    total_delisted = 0
    # Rebalance year of each entry in portfolio_returns, and what was held that year
    return_years = []
    holdings_by_year = {}

    # Pre-fetch risk-free rates for hold_cash strategy
    rf_by_year = {}
    if delisting_strategy == 'hold_cash':
        rf_full = get_benchmark_list(4, start_year, end_year)
        for i, yr in enumerate(range(start_year, end_year)):
            rf_by_year[yr] = rf_full[i] if i < len(rf_full) else 0.0

    # Simulation Loop
    for year in range(start_year, end_year):
        df_year = _dedupe_year_slice_by_ticker(
            working_data[working_data['Year'] == year].set_index('Ticker')
        )
        if df_year.empty: continue

        if factor_combination == 'integrating':
            yearly_components = [
                calculate_integrated_holdings(df_year, factors, factor_directions, aum, top_pct, use_market_cap_weight)
            ]
        else:
            factor_aum = aum / len(factors)
            yearly_components = [
                calculate_holdings(df_year, f, factor_aum, (factor_directions.get(f) == 'top'), top_pct, use_market_cap_weight)
                for f in factors
            ]

        year_ret, delisted_count, positions = _value_positions(
            yearly_components, df_year,
            delisting_strategy=delisting_strategy,
            risk_free_rate=rf_by_year.get(year, 0.0),
            rebalance_year=year
        )
        holdings_by_year[year] = _build_holdings_rows(
            yearly_components, positions, df_year, factors, factor_combination, factor_labels
        )
        return_years.append(year)
        total_delisted += delisted_count
        portfolio_returns.append(year_ret)
        aum *= (1 + year_ret)
        portfolio_values.append(aum)

    # Benchmarks
    rf_rates = np.array(get_benchmark_list(4, start_year, end_year))
    bench_rets = np.array(get_benchmark_list(benchmark_index, start_year + 1, end_year + 1)) / 100.0
    growth_rets = np.array(get_benchmark_list(2, start_year + 1, end_year + 1)) / 100.0
    value_rets = np.array(get_benchmark_list(3, start_year + 1, end_year + 1)) / 100.0

    min_len = min(len(portfolio_returns), len(bench_rets), len(growth_rets), len(value_rets))

    results = compute_comprehensive_metrics(
        np.array(portfolio_returns[:min_len]), bench_rets[:min_len],
        growth_rets[:min_len], value_rets[:min_len], rf_rates[:min_len],
        portfolio_values[:min_len + 1], start_year, start_year + min_len
    )
    results['delisting_strategy'] = delisting_strategy
    results['total_delisted_positions'] = total_delisted
    results['factor_combination'] = factor_combination
    results['holdings_by_year'] = {yr: holdings_by_year[yr] for yr in return_years[:min_len]}
    return results

def _calculate_annual_return(portfolios: List[Portfolio], df_year: pd.DataFrame,
                             delisting_strategy: str = 'zero_return',
                             risk_free_rate: float = 0.0,
                             rebalance_year: int = 0) -> Tuple[float, int]:
    """Realized return logic with time-adjusted delisting handling.

    Args:
        delisting_strategy: 'zero_return', 'hold_cash', or 'reinvest'.
        risk_free_rate: annual RF rate as decimal (e.g. 0.02).
        rebalance_year: formation year; holding period is rebalance_year+1.

    Returns:
        (annual_return, delisted_count)
    """
    annual_return, delisted_count, _ = _value_positions(
        portfolios, df_year, delisting_strategy, risk_free_rate, rebalance_year
    )
    return annual_return, delisted_count


def _value_positions(portfolios: List[Portfolio], df_year: pd.DataFrame,
                     delisting_strategy: str = 'zero_return',
                     risk_free_rate: float = 0.0,
                     rebalance_year: int = 0) -> Tuple[float, int, List[Dict[str, Any]]]:
    """The year's return (see _calculate_annual_return) and every position it valued.

    Positions without a valid price, or whose ticker is not in df_year, are
    skipped and not returned. Each returned position is a dict with sleeve
    (index into portfolios), ticker, shares, price, value (at formation),
    return (the effective period return as a decimal, after delisting
    handling) and delisted. Summed over positions, value * return / total value
    equals annual_return.

    Returns:
        (annual_return, delisted_count, positions)
    """
    import datetime

    holding_end = datetime.date(rebalance_year + 1, 12, 31)
    has_delist_date = 'Delist_Date' in df_year.columns
    has_delist_price = 'Delist_Price' in df_year.columns

    live_positions = []       # (position_value, return)
    delisted_pre_profit = 0.0 # profit earned before delisting
    delisted_post_capital = [] # (capital_at_delist, fraction) for post-delist strategies
    delisted_value = 0.0      # total position value at formation
    delisted_count = 0
    positions = []            # every position valued, in valuation order

    for sleeve, p in enumerate(portfolios):
        for inv in p.investments:
            t = inv['ticker']
            if t not in df_year.index:
                continue
            price = df_year.loc[t, 'Ending_Price']
            if pd.isna(price) or price <= 0:
                continue
            pos_val = inv['number_of_shares'] * price
            ret_val = df_year.loc[t, 'Next-Years_Return']

            if pd.isna(ret_val):
                # Pre-delist return: actual price change from formation to delisting
                partial_return = 0.0
                if has_delist_price:
                    dp = df_year.loc[t, 'Delist_Price']
                    if pd.notna(dp) and dp >= 0:
                        partial_return = (dp / price) - 1.0
                delisted_pre_profit += pos_val * partial_return
                capital_at_delist = pos_val * (1.0 + partial_return)

                # Post-delist fraction: time remaining in holding year
                fraction = 0.5  # default when date unknown
                if has_delist_date:
                    delist_dt = df_year.loc[t, 'Delist_Date']
                    if pd.notna(delist_dt):
                        if hasattr(delist_dt, 'date'):
                            delist_dt = delist_dt.date()
                        remaining_days = max(0, (holding_end - delist_dt).days)
                        fraction = remaining_days / 365.0
                delisted_value += pos_val
                delisted_post_capital.append((capital_at_delist, fraction))
                delisted_count += 1
                positions.append({'sleeve': sleeve, 'ticker': t, 'shares': inv['number_of_shares'],
                                  'price': float(price), 'value': pos_val, 'delisted': True,
                                  'partial_return': partial_return, 'fraction': fraction})
            else:
                live_positions.append((pos_val, ret_val / 100.0))
                positions.append({'sleeve': sleeve, 'ticker': t, 'shares': inv['number_of_shares'],
                                  'price': float(price), 'value': pos_val, 'delisted': False,
                                  'return': ret_val / 100.0})

    live_value = sum(pv for pv, _ in live_positions)
    total_val = live_value + delisted_value

    if total_val == 0:
        return 0.0, 0, []

    live_profit = sum(pv * ret for pv, ret in live_positions)

    if delisting_strategy == 'hold_cash':
        post_profit = sum(
            risk_free_rate * frac * cap for cap, frac in delisted_post_capital
        )
    elif delisting_strategy == 'reinvest':
        if live_value > 0:
            live_return = sum(pv * ret for pv, ret in live_positions) / live_value
            post_profit = sum(
                frac * cap * live_return
                for cap, frac in delisted_post_capital
            ) if delisted_post_capital else 0.0
        else:
            post_profit = 0.0
    else:  # 'zero_return'
        post_profit = 0.0

    total_profit = live_profit + delisted_pre_profit + post_profit

    # Each delisted position's effective return: its pre-delist return, plus what its
    # capital at delisting earned for the rest of the year under the strategy
    # (the per-position split of delisted_pre_profit + post_profit above).
    if delisting_strategy == 'hold_cash':
        post_rate = risk_free_rate
    elif delisting_strategy == 'reinvest' and live_value > 0:
        post_rate = live_profit / live_value
    else:
        post_rate = 0.0
    for pos in positions:
        if pos['delisted']:
            partial_return = pos.pop('partial_return')
            pos['return'] = partial_return + post_rate * pos.pop('fraction') * (1.0 + partial_return)

    return total_profit / total_val, delisted_count, positions


def run_cohort_comparison(data: pd.DataFrame, 
                          selected_factors: List[str], 
                          factor_directions: Dict[str, str],
                          cohort_pct: float,
                          user_settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Orchestrates a comparative backtest between the chosen factor tilts 
    and their exact opposites, combining factors the way the run did
    (user_settings['factor_combination'], Mixing when absent).
    """
    # 1. Run Top Cohort (User's Strategy)
    res_top = rebalance_portfolio(
        data=data,
        factors=selected_factors,
        factor_directions=factor_directions,
        start_year=user_settings['start_year'],
        end_year=user_settings['end_year'],
        initial_aum=user_settings['initial_aum'],
        benchmark_index=1,
        top_pct=cohort_pct,
        use_market_cap_weight=user_settings['use_market_cap_weight'],
        delisting_strategy=user_settings.get('delisting_strategy', 'zero_return'),
        factor_combination=user_settings.get('factor_combination', 'mixing')
    )

    # 2. Run Bottom Cohort (Inverse Strategy)
    inverse_dirs = {f: ('bottom' if d == 'top' else 'top') for f, d in factor_directions.items()}
    res_bot = rebalance_portfolio(
        data=data,
        factors=selected_factors,
        factor_directions=inverse_dirs,
        start_year=user_settings['start_year'],
        end_year=user_settings['end_year'],
        initial_aum=user_settings['initial_aum'],
        benchmark_index=1,
        top_pct=cohort_pct,
        use_market_cap_weight=user_settings['use_market_cap_weight'],
        delisting_strategy=user_settings.get('delisting_strategy', 'zero_return'),
        factor_combination=user_settings.get('factor_combination', 'mixing')
    )

    top_list = res_top['portfolio_values'] if isinstance(res_top, dict) else res_top
    bot_list = res_bot['portfolio_values'] if isinstance(res_bot, dict) else res_bot

    return top_list, bot_list