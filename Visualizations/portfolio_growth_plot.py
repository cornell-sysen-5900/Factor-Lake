"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/portfolio_growth_plot.py
PURPOSE: Standalone plotting utility for comparative wealth-index visualizations.
VERSION: 2.5.0
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from typing import List, Optional

def plot_portfolio_growth(
    years: List[int], 
    port_vals: List[float], 
    bench_vals: Optional[List[float]] = None,
    val_vals: Optional[List[float]] = None,
    gro_vals: Optional[List[float]] = None,
    factor_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Constructs a time-series line chart comparing strategy growth against multiple benchmarks.
    
    This utility provides a high-fidelity visualization of cumulative wealth, 
    supporting overlays for market, value, and growth indices.

    Args:
        years: A list of integers representing the timeline years.
        port_vals: Cumulative dollar values for the strategy portfolio.
        bench_vals: Optional cumulative values for the Russell 2000 index.
        val_vals: Optional cumulative values for the Russell 2000 Value Index.
        gro_vals: Optional cumulative values for the Russell 2000 Growth Index.
        factor_names: A list of active factor labels to be included in the chart title.

    Returns:
        plt.Figure: The generated Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Primary Portfolio Strategy
    ax.plot(
        years, 
        port_vals, 
        label='Portfolio', 
        linewidth=2.5, 
        color='#1f77b4',
        marker='o', 
        markersize=6,
        zorder=5
    )
    
    # 2. Market Benchmark (Russell 2000)
    if bench_vals and len(years) == len(bench_vals):
        ax.plot(
            years, 
            bench_vals, 
            label='Russell 2000', 
            linewidth=2, 
            linestyle='--', 
            color='#ff7f0e',
            marker='s',
            markersize=4,
            alpha=0.8
        )
        
    # 3. Style Benchmark: Value Index
    if val_vals and len(years) == len(val_vals):
        ax.plot(
            years, 
            val_vals, 
            label='Value Index', 
            linewidth=1.5, 
            linestyle=':', 
            color='#2ca02c',
            marker='^',
            markersize=4,
            alpha=0.6
        )

    # 4. Style Benchmark: Growth Index
    if gro_vals and len(years) == len(gro_vals):
        ax.plot(
            years, 
            gro_vals, 
            label='Growth Index', 
            linewidth=1.5, 
            linestyle=':', 
            color='#d62728',
            marker='D',
            markersize=4,
            alpha=0.6
        )
    
    # Title and Label Configuration
    if factor_names:
        clean_names = [name.replace('_', ' ').title() for name in factor_names]
        title_str = f'Cumulative Wealth: {", ".join(clean_names)}'
    else:
        title_str = 'Portfolio Growth Comparison'
        
    ax.set_title(title_str, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Portfolio Value (USD)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    
    # Grid and Legend Aesthetics
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Financial Axis Formatting
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    # Ensure X-axis utilizes integer year ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig