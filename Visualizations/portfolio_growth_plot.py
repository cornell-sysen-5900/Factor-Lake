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
    
    Args:
        years (List[int]): X-axis timeline.
        port_vals (List[float]): Cumulative portfolio values for the strategy.
        bench_vals (Optional[List[float]]): Cumulative values for the Russell 2000.
        val_vals (Optional[List[float]]): Cumulative values for the Russell 2000 Value Index.
        gro_vals (Optional[List[float]]): Cumulative values for the Russell 2000 Growth Index.
        factor_names (Optional[List[str]]): List of active factors for the dynamic title.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Base Portfolio
    ax.plot(
        years, 
        port_vals, 
        label='Portfolio', 
        linewidth=2, 
        color='#1f77b4',
        marker='o', 
        markersize=6
    )
    
    # 2. Main Russell 2000 Benchmark
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
            alpha=0.7
        )
        
    # 3. Value Index Overlay (Green Triangles)
    if val_vals and len(years) == len(val_vals):
        ax.plot(
            years, 
            val_vals, 
            label='Value Index', 
            linewidth=1.5, 
            linestyle=':', 
            color='g',
            marker='^',
            alpha=0.6
        )

    # 4. Growth Index Overlay (Orange Diamonds)
    if gro_vals and len(years) == len(gro_vals):
        ax.plot(
            years, 
            gro_vals, 
            label='Growth Index', 
            linewidth=1.5, 
            linestyle=':', 
            color='orange',
            marker='D',
            alpha=0.6
        )
    
    # Dynamic Title
    if factor_names:
        title_str = f'Portfolio Growth: {", ".join(factor_names)}'
    else:
        title_str = 'Portfolio Growth Over Time'
        
    ax.set_title(title_str, fontsize=14, fontweight='bold')
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    
    # Legend and Grid
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Currency Formatter
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    # Ensure X-axis only displays integer years
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig