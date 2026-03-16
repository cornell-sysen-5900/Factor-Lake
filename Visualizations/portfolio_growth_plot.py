"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/portfolio_growth_plot.py
PURPOSE: Standalone plotting utility for comparative wealth-index visualizations.
VERSION: 2.0.0
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional

def plot_portfolio_growth(
    years: List[int], 
    port_vals: List[float], 
    bench_vals: Optional[List[float]] = None
) -> plt.Figure:
    """
    Constructs a time-series line chart comparing strategy growth against a benchmark.
    
    This utility encapsulates Matplotlib rendering logic to ensure visual consistency 
    across the application. It includes automatic currency formatting and grid 
    styling suitable for quantitative financial analysis.

    Args:
        years (List[int]): A sequence of years representing the X-axis timeline.
        port_vals (List[float]): Cumulative portfolio values for the strategy.
        bench_vals (Optional[List[float]]): Cumulative values for the Russell 2000.

    Returns:
        plt.Figure: A formatted Matplotlib figure object.
    """
    # Initialize figure with professional dimensions
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Apply institutional styling to the Strategy line
    ax.plot(
        years, 
        port_vals, 
        label='Factor Strategy', 
        linewidth=2.5, 
        color='#003366',  # Deep Navy
        marker='o', 
        markersize=4,
        alpha=0.9
    )
    
    # Conditional rendering for the Benchmark
    if bench_vals and len(years) == len(bench_vals):
        ax.plot(
            years, 
            bench_vals, 
            label='Russell 2000 Benchmark', 
            linewidth=2.0, 
            linestyle='--', 
            color='#808080',  # Slate Gray
            alpha=0.7
        )
    
    # Axis Formatting
    ax.set_title("Cumulative Portfolio Performance", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Account Value (USD)", fontsize=11)
    ax.set_xlabel("Rebalance Year", fontsize=11)
    
    # Currency Formatting for Y-axis
    formatter = mticker.StrMethodFormatter('${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    # Visual Hygiene
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d1d1d1')
    
    # Ensure X-axis only displays integer years
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig