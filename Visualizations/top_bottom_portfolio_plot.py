"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: Visualizations/top_bottom_portfolio_plot.py
PURPOSE: Institutional-grade cohort spread analysis visualization.
VERSION: 2.1.1
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional, Dict, Any, Union

<<<<<<< HEAD
def plot_top_bottom_percent(
    years: List[int],
    percent: float = 10.0,
    show_bottom: bool = True,
    benchmark_returns: Optional[List[float]] = None,
    benchmark_label: str = 'Russell 2000',
    initial_investment: float = 1000000.0,
    baseline_portfolio_values: Optional[List[float]] = None,
    precomputed_top: Optional[Union[Dict[str, Any], List[float]]] = None,
    precomputed_bot: Optional[Union[Dict[str, Any], List[float]]] = None
) -> plt.Figure:
=======
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
>>>>>>> main
    """
    Constructs a wealth-index chart comparing performance of top and bottom cohorts.
    
    This visualization identifies the predictive power of selected factors by 
    plotting the divergence between the highest and lowest ranked stocks.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    # 1. Benchmark Trajectory Construction
    if benchmark_returns is not None:
        bench_vals = [initial_investment]
        for ret in benchmark_returns:
            # Handle both percentage (8.5) and decimal (0.085) formats
            multiplier = (ret / 100.0) if abs(ret) > 1.0 else ret
            bench_vals.append(bench_vals[-1] * (1 + multiplier))
        
        if len(bench_vals) >= len(years):
            ax.plot(years, bench_vals[:len(years)], label=benchmark_label, color='#d62728', 
                    linestyle='--', linewidth=1.5, alpha=0.7)

<<<<<<< HEAD
    # 2. Bottom Cohort (Lower Visual Weight)
    if show_bottom and precomputed_bot is not None:
        # Resolve either dictionary-style or list-style input
        if isinstance(precomputed_bot, dict):
            bot_vals = precomputed_bot.get('portfolio_values', [initial_investment])
        else:
            bot_vals = precomputed_bot
            
        if len(bot_vals) >= len(years):
            ax.plot(years, bot_vals[:len(years)], label=f'Bottom {percent}%', color='#9467bd', 
                    marker='v', markersize=4, linewidth=1.5, alpha=0.6)
=======
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
>>>>>>> main

    # 3. Top Cohort (High Contrast)
    if precomputed_top is not None:
        # Resolve either dictionary-style or list-style input
        if isinstance(precomputed_top, dict):
            top_vals = precomputed_top.get('portfolio_values', [initial_investment])
        else:
            top_vals = precomputed_top

        if len(top_vals) >= len(years):
            ax.plot(years, top_vals[:len(years)], label=f'Top {percent}%', color='#2ca02c', 
                    marker='^', markersize=5, linewidth=2.0, zorder=4)

    # 4. Active Portfolio Strategy
    if baseline_portfolio_values:
        common_len = min(len(years), len(baseline_portfolio_values))
        ax.plot(years[:common_len], baseline_portfolio_values[:common_len], 
                label='Active Strategy', color='#003366', linewidth=2.5, 
                marker='o', markersize=4, zorder=5)

<<<<<<< HEAD
    # Institutional Chart Formatting
    ax.set_title(f"Factor Efficacy: Top vs. Bottom {percent}% Cohort Spread", 
                  fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Account Value (USD)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
=======
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
>>>>>>> main

    # Axis and Grid Management
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Static Reference for Initial Capital
    ax.axhline(initial_investment, color='#000000', linestyle=':', linewidth=1.2, alpha=0.4)

    ax.legend(loc='upper left', frameon=True, facecolor='white', shadow=True)
    
    plt.tight_layout()
    return fig