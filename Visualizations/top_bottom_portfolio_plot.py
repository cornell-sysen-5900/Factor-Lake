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
                            growth_returns=None,
                            value_returns=None,
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
                # Convert percentage (e.g., 10.5) to decimal (0.105)
                ret_decimal = float(br[i]) / 100.0
                next_val = benchmark_values[-1] * (1 + ret_decimal)
                benchmark_values.append(next_val)
            else:
                benchmark_values.append(benchmark_values[-1])

    growth_values = None
    if growth_returns is not None:
        gr = list(growth_returns)
        growth_values = [initial_investment]
        for i in range(len(years) - 1):
            if i < len(gr):
                ret_decimal = float(gr[i]) / 100.0
                next_val = growth_values[-1] * (1 + ret_decimal)
                growth_values.append(next_val)
            else:
                growth_values.append(growth_values[-1])

    value_values = None
    if value_returns is not None:
        vr = list(value_returns)
        value_values = [initial_investment]
        for i in range(len(years) - 1):
            if i < len(vr):
                ret_decimal = float(vr[i]) / 100.0
                next_val = value_values[-1] * (1 + ret_decimal)
                value_values.append(next_val)
            else:
                value_values.append(value_values[-1])

    # 3. Initialize Plot
    plt.figure(figsize=(11, 5))
    ax = plt.gca()

    # Plot Top Cohort
    plt.plot(years, top_values, marker='o', linestyle='-', color='g', 
             label=f'Top {percent}%', linewidth=1.6, markersize=6)

    # Plot Bottom Cohort
    if show_bottom and bottom_values is not None:
        plt.plot(years, bottom_values, marker='o', linestyle='-', color='m', 
                 label=f'Bottom {percent}%', linewidth=1.6, markersize=6)

    # Plot Benchmark
    if benchmark_values is not None and len(benchmark_values) == len(years):
        plt.plot(years, benchmark_values, marker='s', linestyle='--', color='r', 
                 label=benchmark_label, linewidth=1.2)

    if growth_values is not None and len(growth_values) == len(years):
        plt.plot(years, growth_values, marker='D', linestyle=':', color='orange',
                 label='Growth Index', linewidth=1.2, alpha=0.7)

    if value_values is not None and len(value_values) == len(years):
        plt.plot(years, value_values, marker='^', linestyle=':', color='g',
                 label='Value Index', linewidth=1.2, alpha=0.7)

    # Plot Main Portfolio Baseline (blue line)
    if baseline_portfolio_values is not None:
        bp = list(baseline_portfolio_values)
        common_len = min(len(years), len(bp))
        plt.plot(years[:common_len], bp[:common_len], marker='o', linestyle='-', 
                 color='b', label='Current Portfolio', linewidth=1.6, markersize=6)

    # 4. Formatting
    try:
        factor_names = ", ".join([getattr(f, 'column_name', str(f)) for f in factors])
    except Exception:
        factor_names = "Selected Factors"

    plt.title(f"Cohort Analysis: Top vs Bottom {percent}% ({factor_names})")
    plt.xlabel('Year')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.xticks(years)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))
    
    # Reference line for starting capital
    plt.axhline(initial_investment, color='black', linestyle=':', linewidth=1.0, alpha=0.5)

    plt.legend(loc='upper left')
    plt.tight_layout()
    
    return plt.gcf()