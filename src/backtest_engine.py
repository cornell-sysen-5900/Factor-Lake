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

def calculate_holdings(df_year: pd.DataFrame, factor_col: str, aum: float, 
                       higher_is_better: bool = True, top_pct: float = 10.0, 
                       use_market_cap_weight: bool = False) -> Portfolio:
    """Constructs a cross-sectional portfolio for a specific period."""
    scores = normalize_series(df_year[factor_col], higher_is_better=higher_is_better)
    valid_scores = scores.dropna().sort_values(ascending=False)
    
    if valid_scores.empty:
        return Portfolio(name=f"Empty_{factor_col}")

    n_select = max(1, math.floor(len(valid_scores) * (top_pct / 100.0)))
    selected_tickers = valid_scores.head(n_select).index.tolist()
    holdings_data = df_year.loc[selected_tickers]
    
    portfolio = Portfolio(name=f"Portfolio_{factor_col}")

    if use_market_cap_weight and 'Market_Capitalization' in holdings_data.columns:
        caps = pd.to_numeric(holdings_data['Market_Capitalization'], errors='coerce').fillna(0)
        weights = caps / caps.sum() if caps.sum() > 0 else pd.Series(1.0/len(caps), index=caps.index)
    else:
        weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)

    for ticker in selected_tickers:
        price = holdings_data.loc[ticker, 'Ending_Price']
        if price > 0:
            shares = (weights[ticker] * aum) / price
            portfolio.add_investment(ticker, shares)
    return portfolio

def rebalance_portfolio(data: pd.DataFrame, factors: List[str], factor_directions: Dict[str, str], 
                        start_year: int, end_year: int, initial_aum: float, 
                        benchmark_index: int = 1, top_pct: float = 10.0, 
                        use_market_cap_weight: bool = False) -> Dict[str, Any]:
    """Executes simulation using a filtered working copy of the master data."""
    
    # ON-THE-FLY FILTERING: Apply constraints to a copy
    exclude_fossil = st.session_state.get('exclude_fossil_fuels', False)
    selected_sectors = st.session_state.get('selected_sectors', [])
    
    working_data = filter_universe(data, exclude_fossil, selected_sectors)
    
    aum = initial_aum
    portfolio_values = [aum]
    portfolio_returns = []

    # Simulation Loop
    for year in range(start_year, end_year):
        df_year = working_data[working_data['Year'] == year].set_index('Ticker')
        if df_year.empty: continue
            
        factor_aum = aum / len(factors)
        yearly_components = [
            calculate_holdings(df_year, f, factor_aum, (factor_directions.get(f) == 'top'), top_pct, use_market_cap_weight)
            for f in factors
        ]

        year_ret = _calculate_annual_return(yearly_components, df_year)
        portfolio_returns.append(year_ret)
        aum *= (1 + year_ret)
        portfolio_values.append(aum)

    # Benchmarks
    rf_rates = np.array(get_benchmark_list(4, start_year, end_year))
    bench_rets = np.array(get_benchmark_list(benchmark_index, start_year + 1, end_year + 1)) / 100.0
    growth_rets = np.array(get_benchmark_list(2, start_year + 1, end_year + 1)) / 100.0
    value_rets = np.array(get_benchmark_list(3, start_year + 1, end_year + 1)) / 100.0
    
    min_len = min(len(portfolio_returns), len(bench_rets), len(growth_rets), len(value_rets))
    
    return compute_comprehensive_metrics(
        np.array(portfolio_returns[:min_len]), bench_rets[:min_len], 
        growth_rets[:min_len], value_rets[:min_len], rf_rates[:min_len], 
        portfolio_values[:min_len + 1], start_year, start_year + min_len
    )

def _calculate_annual_return(portfolios: List[Portfolio], df_year: pd.DataFrame) -> float:
    """Realized return logic based on SQL 'Next-Years_Return'."""
    total_val, total_profit = 0.0, 0.0
    for p in portfolios:
        for inv in p.investments:
            t = inv['ticker']
            if t in df_year.index:
                price = df_year.loc[t, 'Ending_Price']
                ret_val = df_year.loc[t, 'Next-Years_Return']
                ret = (ret_val / 100.0) if not pd.isna(ret_val) else 0.0
                pos_val = inv['number_of_shares'] * price
                total_val += pos_val
                total_profit += pos_val * ret
    return total_profit / total_val if total_val > 0 else 0.0


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

    return res_top, res_bot