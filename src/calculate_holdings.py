"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/calculate_holdings.py
PURPOSE: Core backtesting engine for factor-based portfolio construction and risk attribution.
VERSION: 2.2.0
"""

import numpy as np
import pandas as pd
import math
import logging
from typing import Dict, List, Any, Optional
from .factor_utils import normalize_series
from .benchmarks import get_benchmark_list
from .portfolio import Portfolio

logger = logging.getLogger(__name__)

def calculate_holdings(
    df_year: pd.DataFrame,
    factor_col: str,
    aum: float,
    higher_is_better: bool = True,
    top_pct: float = 10.0,
    use_market_cap_weight: bool = False
) -> Portfolio:
    """
    Constructs a cross-sectional portfolio for a specific time period.
    
    Now utilizes the user-toggled polarity to ensure that if 'bottom' is selected,
    the 'higher_is_better' logic in the normalization utility is inverted.
    """
    # Normalize factor values across the cross-section
    # The utility handles the actual inversion logic based on higher_is_better
    scores = normalize_series(df_year[factor_col], higher_is_better=higher_is_better)
    
    # We sort descending because normalize_series ensures 'higher' is always 'better' 
    # after accounting for the direction toggle.
    valid_scores = scores.dropna().sort_values(ascending=False)
    
    if valid_scores.empty:
        return Portfolio(name=f"Empty_{factor_col}")

    # Determine selection size
    n_select = max(1, math.floor(len(valid_scores) * (top_pct / 100.0)))
    selected_tickers = valid_scores.head(n_select).index.tolist()
    holdings_data = df_year.loc[selected_tickers]
    
    portfolio = Portfolio(name=f"Portfolio_{factor_col}")

    # Weighting Strategy
    if use_market_cap_weight and 'Market_Capitalization' in holdings_data.columns:
        caps = pd.to_numeric(holdings_data['Market_Capitalization'], errors='coerce').fillna(0)
        weights = caps / caps.sum() if caps.sum() > 0 else pd.Series(1.0/len(caps), index=caps.index)
    else:
        weights = pd.Series(1.0 / len(selected_tickers), index=selected_tickers)

    # Translate dollar allocations into shares using SQL 'Ending_Price'
    for ticker in selected_tickers:
        price = holdings_data.loc[ticker, 'Ending_Price']
        if price > 0:
            shares = (weights[ticker] * aum) / price
            portfolio.add_investment(ticker, shares)

    return portfolio

def rebalance_portfolio(
    data: pd.DataFrame,
    factors: List[str],
    factor_directions: Dict[str, str],
    start_year: int,
    end_year: int,
    initial_aum: float,
    benchmark_index: int = 1,
    top_pct: float = 10.0,
    use_market_cap_weight: bool = False
) -> Dict[str, Any]:
    """
    Executes a multi-year backtest simulation with integrated risk analytics.
    """
    aum = initial_aum
    portfolio_values = [aum]
    portfolio_returns = []
    
    # Ensure Ticker is the index for faster lookups in the loop
    if data.index.name != 'Ticker':
        data = data.copy()
    
    # 1. Annual Simulation Loop
    for year in range(start_year, end_year):
        df_year = data[data['Year'] == year].set_index('Ticker')
        if df_year.empty:
            continue
            
        factor_aum = aum / len(factors)
        yearly_components = []

        for f_col in factors:
            # Direction is pulled from UI toggle dictionary
            direction = factor_directions.get(f_col, 'top')
            
            p = calculate_holdings(
                df_year, f_col, factor_aum, 
                higher_is_better=(direction == 'top'),
                top_pct=top_pct,
                use_market_cap_weight=use_market_cap_weight
            )
            yearly_components.append(p)

        # realized forward returns (using SQL column: Next-Years_Return)
        year_ret = _calculate_annual_return(yearly_components, df_year)
        portfolio_returns.append(year_ret)
        aum *= (1 + year_ret)
        portfolio_values.append(aum)

    # 2. Benchmark & Risk-Free Alignment
    rf_rates = np.array(get_benchmark_list(4, start_year, end_year))
    bench_returns = np.array(get_benchmark_list(benchmark_index, start_year + 1, end_year + 1)) / 100.0
    
    # Ensure returns lists match in length for stats
    min_len = min(len(portfolio_returns), len(bench_returns))
    
    return _compute_comprehensive_metrics(
        np.array(portfolio_returns[:min_len]), 
        bench_returns[:min_len], 
        rf_rates[:min_len], 
        portfolio_values[:min_len + 1], 
        start_year, 
        start_year + min_len
    )

def _calculate_annual_return(portfolios: List[Portfolio], df_year: pd.DataFrame) -> float:
    """Calculates realized return based on SQL column 'Next-Years_Return'."""
    total_val = 0.0
    total_profit = 0.0
    for p in portfolios:
        for inv in p.investments:
            t = inv['ticker']
            if t in df_year.index:
                price = df_year.loc[t, 'Ending_Price']
                # Accessing the exact SQL column name
                ret_val = df_year.loc[t, 'Next-Years_Return']
                ret = (ret_val / 100.0) if not pd.isna(ret_val) else 0.0
                
                pos_val = inv['number_of_shares'] * price
                total_val += pos_val
                total_profit += pos_val * ret
                
    return total_profit / total_val if total_val > 0 else 0.0

def _compute_comprehensive_metrics(p_ret: np.ndarray, b_ret: np.ndarray, rf: np.ndarray, 
                                   p_vals: List[float], start: int, end: int) -> Dict[str, Any]:
    """Computes Sharpe, IR, Beta, and MDD with sample standard deviation."""
    excess_p = p_ret - rf
    excess_b = b_ret - rf
    active_ret = p_ret - b_ret
    
    # Risk Ratios (ddof=1 for sample volatility)
    sharpe_p = np.mean(excess_p) / np.std(excess_p, ddof=1) if np.std(excess_p) > 0 else 0
    sharpe_b = np.mean(excess_b) / np.std(excess_b, ddof=1) if np.std(excess_b) > 0 else 0
    info_ratio = np.mean(active_ret) / np.std(active_ret, ddof=1) if np.std(active_ret) > 0 else 0
    
    # Regression for Beta
    beta = np.polyfit(excess_b, excess_p, 1)[0] if len(p_ret) > 1 else 1.0

    # Drawdown Calculation
    def get_mdd(values):
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        return np.min(dd) if len(dd) > 0 else 0

    # Benchmark wealth reconstruction
    b_vals = [p_vals[0]]
    for r in b_ret: 
        b_vals.append(b_vals[-1] * (1 + r))

    # Yearly Comparison metadata for UI tables
    yearly_comparisons = []
    for i in range(len(p_ret)):
        yearly_comparisons.append({
            'year': start + i,
            'portfolio_return': p_ret[i] * 100,
            'benchmark_return': b_ret[i] * 100,
            'win': p_ret[i] > b_ret[i]
        })

    return {
        'final_value': p_vals[-1],
        'portfolio_values': p_vals,
        'yearly_returns': (p_ret * 100).tolist(),
        'benchmark_returns': (b_ret * 100).tolist(),
        'years': list(range(start, end + 1)),
        'sharpe_portfolio': sharpe_p,
        'sharpe_benchmark': sharpe_b,
        'information_ratio': info_ratio,
        'portfolio_beta': beta,
        'max_drawdown_portfolio': get_mdd(p_vals),
        'max_drawdown_benchmark': get_mdd(b_vals),
        'win_rate': np.mean(p_ret > b_ret) if len(p_ret) > 0 else 0,
        'yearly_comparisons': yearly_comparisons
    }