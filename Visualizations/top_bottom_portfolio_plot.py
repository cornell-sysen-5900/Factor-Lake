"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/top_bottom_portfolio_plot.py
PURPOSE: Visualization utility for analyzing factor premiums via cohort spreads.
VERSION: 2.0.0
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional, Dict, Any

def plot_top_bottom_percent(
    years: List[int],
    percent: float = 10.0,
    show_bottom: bool = True,
    benchmark_returns: Optional[List[float]] = None,
    benchmark_label: str = 'Russell 2000',
    initial_investment: float = 1000000.0,
    baseline_portfolio_values: Optional[List[float]] = None,
    precomputed_top: Optional[Dict[str, Any]] = None,
    precomputed_bot: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Constructs a wealth-index chart comparing the performance of top and bottom cohorts.

    This visualization allows researchers to observe the effectiveness of a factor 
    by displaying the performance gap (spread) between the 'Top' decile and 
    the 'Bottom' decile. It includes a baseline for the current user-defined 
    portfolio and the selected market benchmark.

    Args:
        years (List[int]): Timeline for the x-axis.
        percent (float): The percentage of the universe represented in each cohort.
        show_bottom (bool): If True, the bottom cohort is rendered for spread analysis.
        benchmark_returns (List[float]): Annual percentage returns for the benchmark.
        benchmark_label (str): Label for the benchmark index.
        initial_investment (float): The starting dollar amount for all series.
        baseline_portfolio_values (List[float]): The values of the active strategy.
        precomputed_top (Dict): Results dictionary for the top cohort.
        precomputed_bot (Dict): Results dictionary for the bottom cohort.

    Returns:
        plt.Figure: A formatted Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    # 1. Setup Benchmark Trajectory
    # Converts annual returns into a cumulative wealth index starting at initial_investment
    if benchmark_returns is not None:
        bench_vals = [initial_investment]
        for ret in benchmark_returns:
            # ret is assumed to be in percentage (e.g., 8.5 for 8.5%)
            bench_vals.append(bench_vals[-1] * (1 + (float(ret) / 100.0)))
        
        if len(bench_vals) == len(years):
            ax.plot(years, bench_vals, label=benchmark_label, color='#d62728', 
                    linestyle='--', linewidth=1.5, alpha=0.8)

    # 2. Plot Cohort Series
    # Render Bottom Cohort first (lower visual priority than Top)
    if show_bottom and precomputed_bot:
        bot_vals = precomputed_bot.get('portfolio_values', [initial_investment])
        if len(bot_vals) == len(years):
            ax.plot(years, bot_vals, label=f'Bottom {percent}%', color='#9467bd', 
                    marker='v', markersize=4, linewidth=1.8, alpha=0.7)

    # Render Top Cohort
    if precomputed_top:
        top_vals = precomputed_top.get('portfolio_values', [initial_investment])
        if len(top_vals) == len(years):
            ax.plot(years, top_vals, label=f'Top {percent}%', color='#2ca02c', 
                    marker='^', markersize=5, linewidth=2.0)

    # 3. Plot Current Strategy Baseline
    if baseline_portfolio_values:
        common_len = min(len(years), len(baseline_portfolio_values))
        ax.plot(years[:common_len], baseline_portfolio_values[:common_len], 
                label='Active Strategy', color='#003366', linewidth=2.5, marker='o', markersize=4)

    # 4. Institutional Formatting
    ax.set_title(f"Cohort Spread Analysis: Top vs. Bottom {percent}%", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Account Value (USD)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)

    # Currency Formatting for Y-axis
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    
    # Visual Hygiene
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Reference line for starting capital
    ax.axhline(initial_investment, color='#000000', linestyle=':', linewidth=1.2, alpha=0.4)

    ax.legend(loc='upper left', frameon=True, facecolor='white')
    
    plt.tight_layout()
    return fig