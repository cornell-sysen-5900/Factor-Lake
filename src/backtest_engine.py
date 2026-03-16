"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/backtest_engine.py
PURPOSE: Backtesting engine with integrated on-the-fly filtering and crash protection.
VERSION: 2.5.0
"""

import numpy as np
import pandas as pd
import math
import streamlit as st
from typing import Dict, List, Any, Tuple, Optional

from .factor_utils import normalize_series
from .benchmarks import get_benchmark_list
from .portfolio import Portfolio
from .portfolio_filters import filter_universe
from .performance_metrics import compute_comprehensive_metrics

def calculate_holdings(df_year: pd.DataFrame, factor_col: str, aum: float, 
                       higher_is_better: bool = True, top_pct: float = 10.0, 
                       use_market_cap_weight: bool = False) -> Portfolio:
    """Constructs a cross-sectional portfolio for a specific period."""
    if factor_col not in df_year.columns:
        return Portfolio(name=f"Empty_{factor_col}")

    # Normalize and sort
    scores = normalize_series(df_year[factor_col], higher_is_better=higher_is_better)
    valid_scores = scores.dropna().sort_values(ascending=False)
    
    if valid_scores.empty:
        return Portfolio(name=f"Empty_{factor_col}")

    # Use math.ceil to ensure we select at least 1 stock if data exists
    n_select = max(1, math.ceil(len(valid_scores) * (top_pct / 100.0)))
    selected_tickers = valid_scores.head(n_select).index.tolist()
    holdings_data = df_year.loc[selected_tickers]
    
    portfolio = Portfolio(name=f"Portfolio_{factor_col}")

    if not selected_tickers:
        return portfolio

    # Weighting Logic
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
    from app.streamlit_config import FACTOR_METADATA

    # 1. Internal Translation: Ensure we use the actual column names from metadata
    actual_factors = []
    actual_directions = {}
    for f in factors:
        col_name = FACTOR_METADATA[f]['column'] if f in FACTOR_METADATA else f
        actual_factors.append(col_name)
        # Check if directions are mapped to pretty name or internal name
        actual_directions[col_name] = factor_directions.get(f, factor_directions.get(col_name, 'top'))

    # 2. Filtering
    exclude_fossil = st.session_state.get('exclude_fossil_fuels', False)
    selected_sectors = st.session_state.get('selected_sectors', [])
    working_data = filter_universe(data, exclude_fossil, selected_sectors)
    
    aum = initial_aum
    portfolio_values = [aum]
    portfolio_returns = []

    # 3. Simulation Loop
    for year in range(start_year, end_year):
        df_year = working_data[working_data['Year'] == year].copy()
        if df_year.empty: 
            portfolio_returns.append(0.0)
            portfolio_values.append(aum)
            continue
        
        df_year = df_year.set_index('Ticker')
        factor_aum = aum / len(actual_factors) if actual_factors else 0
        
        yearly_components = [
            calculate_holdings(
                df_year, 
                f, 
                factor_aum, 
                (actual_directions.get(f) == 'top'), 
                top_pct, 
                use_market_cap_weight
            )
            for f in actual_factors
        ]

        year_ret = _calculate_annual_return(yearly_components, df_year)
        portfolio_returns.append(year_ret)
        aum *= (1 + year_ret)
        portfolio_values.append(aum)

    # 4. Benchmarks & Metrics
    rf_rates = np.array(get_benchmark_list(4, start_year, end_year))
    bench_rets = np.array(get_benchmark_list(benchmark_index, start_year + 1, end_year + 1)) / 100.0
    growth_rets = np.array(get_benchmark_list(2, start_year + 1, end_year + 1)) / 100.0
    value_rets = np.array(get_benchmark_list(3, start_year + 1, end_year + 1)) / 100.0
    
    min_len = min(len(portfolio_returns), len(bench_rets), len(growth_rets), len(value_rets))
    
    return compute_comprehensive_metrics(
        np.array(portfolio_returns[:min_len]), 
        bench_rets[:min_len], 
        growth_rets[:min_len], 
        value_rets[:min_len], 
        rf_rates[:min_len], 
        portfolio_values[:min_len + 1], 
        start_year, 
        start_year + min_len
    )

def _calculate_annual_return(portfolios: List[Portfolio], df_year: pd.DataFrame) -> float:
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

def run_cohort_comparison(data, selected_factors, factor_directions, cohort_pct, user_settings):
    """
    Orchestrates the top vs bottom comparison. 
    Internal translation is handled within rebalance_portfolio.
    """
    # 1. Run Top Cohort (Default directions)
    res_top = rebalance_portfolio(
        data=data,
        factors=selected_factors,
        factor_directions=factor_directions,
        start_year=int(user_settings['start_year']),
        end_year=int(user_settings['end_year']),
        initial_aum=user_settings['initial_aum'],
        benchmark_index=user_settings.get('benchmark_index', 1),
        top_pct=cohort_pct,
        use_market_cap_weight=user_settings.get('use_market_cap_weight', False)
    )

    # 2. Run Bottom Cohort (Invert directions)
    inverse_directions = {
        f: ('bottom' if factor_directions.get(f) == 'top' else 'top') 
        for f in selected_factors
    }

    res_bot = rebalance_portfolio(
        data=data,
        factors=selected_factors,
        factor_directions=inverse_directions,
        start_year=int(user_settings['start_year']),
        end_year=int(user_settings['end_year']),
        initial_aum=user_settings['initial_aum'],
        benchmark_index=user_settings.get('benchmark_index', 1),
        top_pct=cohort_pct,
        use_market_cap_weight=user_settings.get('use_market_cap_weight', False)
    )

    return res_top, res_bot