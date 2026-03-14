"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/portfolio_growth_plot.py
PURPOSE: Standalone plotting utility for comparative wealth-index visualizations.
VERSION: 1.1.0
"""

import matplotlib.pyplot as plt
from typing import List, Optional

def plot_portfolio_growth(years: List[int], port_vals: List[float], bench_vals: Optional[List[float]] = None):
    """
    Constructs a time-series line chart comparing strategy growth against a benchmark.
    
    This function isolates the Matplotlib rendering logic from the Streamlit framework, 
    allowing for modular reuse. It handles dimension validation internally to ensure 
    that the visual representation remains accurate even if benchmark data is partial.

    Args:
        years (List[int]): A sequence of years representing the X-axis timeline.
        port_vals (List[float]): Cumulative portfolio values for the strategy.
        bench_vals (Optional[List[float]]): Cumulative values for the Russell 2000 
                                            or similar index. Defaults to None.

    Returns:
        matplotlib.figure.Figure: A formatted figure object ready for display or export.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Render Strategy Growth
    ax.plot(years, port_vals, label='Strategy', linewidth=2, color='#1f77b4')
    
    # Render Benchmark Growth only if data is provided and aligned
    if bench_vals and len(years) == len(bench_vals):
        ax.plot(years, bench_vals, label='Russell 2000', linestyle='--', color='#7f7f7f')
    
    # Professional Formatting
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Year")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    return fig