import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

def plot_top_bottom_percent(rdata,
                            factors,
                            years,
                            percent=10,
                            show_bottom=True,
                            benchmark_returns=None,
                            benchmark_label='Russell 2000',
                            initial_investment=1000000.0,
                            baseline_portfolio_values=None,
                            precomputed_top=None,
                            precomputed_bot=None):
    """
    Plots the growth of $1 invested for top/bottom cohorts using precomputed 
    results passed from the main application.
    """

    # 1. Extract values from precomputed results or default to initial investment
    top_values = [initial_investment]
    if precomputed_top and 'portfolio_values' in precomputed_top:
        top_values = precomputed_top['portfolio_values']

    bottom_values = None
    if show_bottom:
        bottom_values = [initial_investment]
        if precomputed_bot and 'portfolio_values' in precomputed_bot:
            bottom_values = precomputed_bot['portfolio_values']

    # 2. Setup Benchmark Values
    benchmark_values = None
    if benchmark_returns is not None:
        br = list(benchmark_returns)
        benchmark_values = [initial_investment]
        for i in range(len(years) - 1):
            if i < len(br):
                # Benchmark returns are natively returned in true decimals
                ret_decimal = float(br[i])
                next_val = benchmark_values[-1] * (1 + ret_decimal)
                benchmark_values.append(next_val)
            else:
                benchmark_values.append(benchmark_values[-1])

    # 3. Initialize Plot
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    
    x_positions = list(range(len(years)))

    # Plot Top Cohort
    plt.plot(x_positions, top_values, marker='o', linestyle='-', color='g', 
             label=f'Top {percent}%', linewidth=1.6, markersize=6)

    # Plot Bottom Cohort
    if show_bottom and bottom_values is not None:
        plt.plot(x_positions[:len(bottom_values)], bottom_values, marker='o', linestyle='-', color='m', 
                 label=f'Bottom {percent}%', linewidth=1.6, markersize=6)

    # Plot Benchmark
    if benchmark_values is not None and len(benchmark_values) == len(years):
        plt.plot(x_positions, benchmark_values, marker='s', linestyle='--', color='r', 
                 label=benchmark_label, linewidth=1.2)

    # Plot Main Portfolio Baseline (blue line)
    if baseline_portfolio_values is not None:
        bp = list(baseline_portfolio_values)
        common_len = min(len(years), len(bp))
        # Use a dashed line and transparent marker to prevent eclipsing solid cohort lines when values match exactly
        plt.plot(x_positions[:common_len], bp[:common_len], marker='x', linestyle='--', 
                 color='b', label='Current Portfolio', linewidth=2.0, markersize=7, alpha=0.8)

    # 4. Formatting
    try:
        factor_names = ", ".join([getattr(f, 'column_name', str(f)) for f in factors])
    except Exception:
        factor_names = "Selected Factors"

    plt.title(f"Cohort Analysis: Top vs Bottom {percent}% ({factor_names})")
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    if len(years) <= 20:
        plt.xticks(range(len(years)), years, rotation=45, fontsize=8)
    else:
        # Thin out the ticks to avoid overlapping text (keep about 10-15 ticks evenly spaced)
        step = max(1, len(years) // 12)
        tick_positions = list(range(0, len(years), step))
        tick_labels = [years[i] for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)
        
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))
    
    # Reference line for starting capital
    plt.axhline(initial_investment, color='black', linestyle=':', linewidth=1.0, alpha=0.5)

    plt.legend(loc='upper left')
    plt.tight_layout()
    
    return plt.gcf()