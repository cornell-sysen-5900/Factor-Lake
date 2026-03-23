"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: Visualizations/top_bottom_portfolio_plot.py
PURPOSE: Institutional-grade cohort spread analysis visualization.
VERSION: 2.1.2
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import List, Optional, Dict, Any, Union

def plot_top_bottom_percent(
    years: List[int],
    percent: float = 10.0,
    show_bottom: bool = True,
    benchmark_returns: Optional[List[float]] = None,
    growth_returns: Optional[List[float]] = None,
    value_returns: Optional[List[float]] = None,
    benchmark_label: str = 'Russell 2000',
    initial_investment: float = 1000000.0,
    baseline_portfolio_values: Optional[List[float]] = None,
    precomputed_top: Optional[Union[Dict[str, Any], List[float]]] = None,
    precomputed_bot: Optional[Union[Dict[str, Any], List[float]]] = None
) -> plt.Figure:
    """
    Constructs a wealth-index chart comparing performance of top and bottom cohorts.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    def _get_trajectory(returns: Optional[List[float]]) -> Optional[List[float]]:
        if returns is None: return None
        vals = [initial_investment]
        for ret in returns:
            multiplier = (ret / 100.0) if abs(ret) > 1.0 else ret
            vals.append(vals[-1] * (1 + multiplier))
        return vals[:len(years)]

    # 1. Index Trajectories (Benchmark, Growth, Value)
    indices = [
        (benchmark_returns, benchmark_label, '#d62728', '--', None),
        (growth_returns, 'Growth Index', '#ff7f0e', ':', 'D'),
        (value_returns, 'Value Index', '#17becf', ':', 's')
    ]

    for rets, label, color, style, marker in indices:
        vals = _get_trajectory(rets)
        if vals and len(vals) == len(years):
            ax.plot(years, vals, label=label, color=color, linestyle=style, 
                    marker=marker, markersize=3, linewidth=1.2, alpha=0.6)

    # 2. Bottom Cohort (Lower Visual Weight)
    if show_bottom and precomputed_bot is not None:
        bot_vals = precomputed_bot.get('portfolio_values', [initial_investment]) if isinstance(precomputed_bot, dict) else precomputed_bot
        if len(bot_vals) >= len(years):
            ax.plot(years, bot_vals[:len(years)], label=f'Bottom {percent}%', color='#9467bd', 
                    marker='v', markersize=4, linewidth=1.5, alpha=0.6)

    # 3. Top Cohort (High Contrast)
    if precomputed_top is not None:
        top_vals = precomputed_top.get('portfolio_values', [initial_investment]) if isinstance(precomputed_top, dict) else precomputed_top
        if len(top_vals) >= len(years):
            ax.plot(years, top_vals[:len(years)], label=f'Top {percent}%', color='#2ca02c', 
                    marker='^', markersize=5, linewidth=2.0, zorder=4)

    # 4. Active Portfolio Strategy
    if baseline_portfolio_values:
        common_len = min(len(years), len(baseline_portfolio_values))
        ax.plot(years[:common_len], baseline_portfolio_values[:common_len], 
                label='Active Strategy', color='#003366', linewidth=2.5, 
                marker='o', markersize=4, zorder=5)

    # Institutional Chart Formatting
    ax.set_title(f"Factor Efficacy: Top vs. Bottom {percent}% Cohort Spread", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel("Account Value (USD)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(initial_investment, color='#000000', linestyle='-', linewidth=0.8, alpha=0.2)

    ax.legend(loc='upper left', frameon=True, facecolor='white', shadow=False, fontsize='small')
    plt.tight_layout()
    return fig