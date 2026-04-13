"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/backtest_engine.py
PURPOSE: Backtesting engine with integrated on-the-fly filtering.
VERSION: 2.4.1
"""

import numpy as np
import pandas as pd
import math
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional  # Added Tuple and Optional

from .factor_utils import normalize_series
from .benchmarks import get_benchmark_list
from .portfolio import Portfolio
from src.factor_registry import get_factor_column
from .portfolio_filters import filter_universe
from .performance_metrics import compute_comprehensive_metrics


def build_ranked_stocks_table(
    data: pd.DataFrame,
    factors: List[str],
    factor_directions: Dict[str, str],
    target_year: int,
    top_pct: float = 10.0
) -> pd.DataFrame:
    """
    Builds a best-to-worst ranked stock table for a single rebalance year.

    The ranking is a composite score formed by averaging each selected factor's
    normalized signal after applying its selected direction (top or bottom).
    """
    if data.empty or not factors:
        return pd.DataFrame()

    year_slice = data[data['Year'] == target_year]
    if year_slice.empty:
        return pd.DataFrame()

    if 'Ticker' not in year_slice.columns:
        return pd.DataFrame()

    df_year = year_slice.set_index('Ticker')

    score_components: List[pd.Series] = []
    for factor_col in factors:
        if factor_col not in df_year.columns:
            continue
        higher_is_better = factor_directions.get(factor_col, 'top') == 'top'
        norm_scores = normalize_series(df_year[factor_col], higher_is_better=higher_is_better)
        score_components.append(norm_scores.rename(f"score_{factor_col}"))

    if not score_components:
        return pd.DataFrame()

    scores_df = pd.concat(score_components, axis=1)
    composite_scores = scores_df.mean(axis=1, skipna=True).dropna().sort_values(ascending=False)
    if composite_scores.empty:
        return pd.DataFrame()

    n_select = max(1, math.floor(len(composite_scores) * (top_pct / 100.0)))
    selected_scores = composite_scores.head(n_select)

    ranked_df = pd.DataFrame({
        'Ticker': selected_scores.index,
        'Composite Score': selected_scores.values,
        'Rank': np.arange(1, len(selected_scores) + 1)
    })

    if 'Next-Years_Return' in df_year.columns:
        ranked_df['Next-Year Return %'] = pd.to_numeric(
            df_year.reindex(selected_scores.index)['Next-Years_Return'], errors='coerce'
        ).values

    if 'Scotts_Sector_5' in df_year.columns:
        ranked_df['Sector'] = df_year.reindex(selected_scores.index)['Scotts_Sector_5'].values

    return ranked_df[['Rank', 'Ticker', 'Composite Score'] + [
        c for c in ['Next-Year Return %', 'Sector'] if c in ranked_df.columns
    ]]
    
def calculate_holdings(
    df_year: pd.DataFrame, 
    factor_key: str, 
    aum: float, 
    higher_is_better: bool = True, 
    top_pct: float = 10.0, 
    use_market_cap_weight: bool = False
) -> Portfolio:
    """
    Constructs a cross-sectional portfolio for a specific period.
    
    This function handles the mapping from internal factor keys to database 
    column names and performs the stock selection and weight allocation.
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

    # 3. Score Normalization
    # higher_is_better=True ranks high values at the top (Long)
    # higher_is_better=False ranks low values at the top (Short/Bottom)
    scores = normalize_series(df_year[factor_col], higher_is_better=higher_is_better)
    valid_scores = scores.dropna().sort_values(ascending=False)
    
    if valid_scores.empty:
        return Portfolio(name=f"Empty_{factor_key}")

    # 4. Determine Selection Size
    n_select = max(1, math.floor(len(valid_scores) * (top_pct / 100.0)))
    selected_tickers = valid_scores.head(n_select).index.tolist()
    holdings_data = df_year.loc[selected_tickers]
    
    portfolio = Portfolio(name=f"Portfolio_{factor_key}")

    # 5. Calculate Weights (Market Cap vs. Equal Weight)
    if use_market_cap_weight and 'Market_Capitalization' in holdings_data.columns:
        caps = pd.to_numeric(holdings_data['Market_Capitalization'], errors='coerce').fillna(0)
        if caps.sum() > 0:
            weights = caps / caps.sum()
        else:
            weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)
    else:
        weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)

    # 6. Allocate Positions
    # We verify required columns exist to prevent runtime errors during the loop
    has_price = 'Ending_Price' in holdings_data.columns
    
    for ticker in selected_tickers:
        if has_price:
            price = holdings_data.loc[ticker, 'Ending_Price']
            # Only add investment if price is valid and positive
            if pd.notnull(price) and price > 0:
                shares = (weights[ticker] * aum) / price
                portfolio.add_investment(ticker, shares)
                
    return portfolio

def rebalance_portfolio(data: pd.DataFrame, factors: List[str], factor_directions: Dict[str, str],
                        start_year: int, end_year: int, initial_aum: float,
                        benchmark_index: int = 1, top_pct: float = 10.0,
                        use_market_cap_weight: bool = False,
                        delisting_strategy: str = 'zero_return') -> Dict[str, Any]:
    """Executes simulation using a filtered working copy of the master data.

    Args:
        delisting_strategy: 'zero_return', 'hold_cash', or 'reinvest'.
    """

    # ON-THE-FLY FILTERING: Apply constraints to a copy
    exclude_fossil = st.session_state.get('exclude_fossil_fuels', False)
    selected_sectors = st.session_state.get('selected_sectors', [])

    working_data = filter_universe(data, exclude_fossil, selected_sectors)

    aum = initial_aum
    portfolio_values = [aum]
    portfolio_returns = []
    total_delisted = 0

    # Pre-fetch risk-free rates for hold_cash strategy
    rf_by_year = {}
    if delisting_strategy == 'hold_cash':
        rf_full = get_benchmark_list(4, start_year, end_year)
        for i, yr in enumerate(range(start_year, end_year)):
            rf_by_year[yr] = rf_full[i] if i < len(rf_full) else 0.0

    # Simulation Loop
    for year in range(start_year, end_year):
        df_year = working_data[working_data['Year'] == year].set_index('Ticker')
        if df_year.empty: continue

        factor_aum = aum / len(factors)
        yearly_components = [
            calculate_holdings(df_year, f, factor_aum, (factor_directions.get(f) == 'top'), top_pct, use_market_cap_weight)
            for f in factors
        ]

        year_ret, delisted_count = _calculate_annual_return(
            yearly_components, df_year,
            delisting_strategy=delisting_strategy,
            risk_free_rate=rf_by_year.get(year, 0.0),
            rebalance_year=year
        )
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
    import datetime

    holding_end = datetime.date(rebalance_year + 1, 12, 31)
    has_delist_date = 'Delist_Date' in df_year.columns

    live_positions = []       # (position_value, return)
    delisted_value = 0.0
    delisted_fractions = []   # time-fraction remaining after delist
    delisted_count = 0

    for p in portfolios:
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
                # Compute fraction of holding year remaining after delist
                fraction = 0.5  # default when date unknown
                if has_delist_date:
                    delist_dt = df_year.loc[t, 'Delist_Date']
                    if pd.notna(delist_dt):
                        if hasattr(delist_dt, 'date'):
                            delist_dt = delist_dt.date()
                        remaining_days = max(0, (holding_end - delist_dt).days)
                        fraction = remaining_days / 365.0
                delisted_value += pos_val
                delisted_fractions.append((pos_val, fraction))
                delisted_count += 1
            else:
                live_positions.append((pos_val, ret_val / 100.0))

    live_value = sum(pv for pv, _ in live_positions)
    total_val = live_value + delisted_value

    if total_val == 0:
        return 0.0, 0

    live_profit = sum(pv * ret for pv, ret in live_positions)

    if delisting_strategy == 'hold_cash':
        delisted_profit = sum(
            risk_free_rate * frac * pv for pv, frac in delisted_fractions
        )
    elif delisting_strategy == 'reinvest':
        if live_value > 0:
            delisted_profit = sum(
                frac * (pv / live_value) * sum(lv * lr for lv, lr in live_positions)
                for pv, frac in delisted_fractions
            ) if delisted_fractions else 0.0
        else:
            delisted_profit = 0.0
    else:  # 'zero_return'
        delisted_profit = 0.0

    return (live_profit + delisted_profit) / total_val, delisted_count


def run_cohort_comparison(data: pd.DataFrame, 
                          selected_factors: List[str], 
                          factor_directions: Dict[str, str],
                          cohort_pct: float,
                          user_settings: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Orchestrates a comparative backtest between the chosen factor tilts 
    and their exact opposites.
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
        use_market_cap_weight=user_settings['use_market_cap_weight']
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
        use_market_cap_weight=user_settings['use_market_cap_weight']
    )

    top_list = res_top['portfolio_values'] if isinstance(res_top, dict) else res_top
    bot_list = res_bot['portfolio_values'] if isinstance(res_bot, dict) else res_bot

    return top_list, bot_list