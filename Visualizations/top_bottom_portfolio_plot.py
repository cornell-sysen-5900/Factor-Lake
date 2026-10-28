"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: Visualizations/top_bottom_portfolio_plot.py
PURPOSE: Institutional-grade cohort spread analysis visualization,
         rendered as an interactive Plotly figure (zoom, pan, hover values).
VERSION: 3.0.0
"""

import plotly.graph_objects as go
from typing import List, Optional, Dict, Any, Union

from Visualizations.portfolio_growth_plot import date_axis

HOVER_TEMPLATE = '%{fullData.name}: $%{y:,.0f}<extra></extra>'


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
    precomputed_bot: Optional[Union[Dict[str, Any], List[float]]] = None,
    growth_label: str = 'Growth Index',
    value_label: str = 'Value Index',
    dates: bool = False
) -> go.Figure:
    """
    Constructs a wealth-index chart comparing performance of top and bottom cohorts.

    WRDS runs pass their ETFs' labels, and dates=True when `years` are dates.

    Returns:
        go.Figure: The generated Plotly figure object.
    """
    fig = go.Figure()

    def _get_trajectory(series: Optional[List[float]]) -> Optional[List[float]]:
        if series is None:
            return None

        if len(series) == 0:
            return None

        # If the series already looks like wealth levels (same length as years
        # and starts at the initial investment), plot it directly.
        if (
            len(series) == len(years)
            and abs(float(series[0]) - float(initial_investment)) <= max(1.0, 0.001 * float(initial_investment))
        ):
            return series[:len(years)]

        vals = [initial_investment]
        for ret in series:
            multiplier = (ret / 100.0) if abs(ret) > 1.0 else ret
            vals.append(vals[-1] * (1 + multiplier))
        return vals[:len(years)]

    # 1. Index Trajectories (Benchmark, Growth, Value)
    indices = [
        (benchmark_returns, benchmark_label, '#d62728', 'dash', 'x'),
        (growth_returns, growth_label, '#ff7f0e', 'dot', 'diamond'),
        (value_returns, value_label, '#17becf', 'dot', 'square')
    ]

    for rets, label, color, style, marker in indices:
        vals = _get_trajectory(rets)
        if vals and len(vals) == len(years):
            fig.add_trace(go.Scatter(
                x=years, y=vals, name=label,
                mode='lines+markers',
                line=dict(color=color, dash=style, width=1.2),
                marker=dict(symbol=marker, size=3),
                opacity=0.6, hovertemplate=HOVER_TEMPLATE
            ))

    # 2. Bottom Cohort (Lower Visual Weight)
    if show_bottom and precomputed_bot is not None:
        bot_vals = precomputed_bot.get('portfolio_values', [initial_investment]) if isinstance(precomputed_bot, dict) else precomputed_bot
        if len(bot_vals) >= len(years):
            fig.add_trace(go.Scatter(
                x=years, y=bot_vals[:len(years)], name=f'Bottom {percent}%',
                mode='lines+markers',
                line=dict(color='#9467bd', width=1.5),
                marker=dict(symbol='triangle-down', size=4),
                opacity=0.6, hovertemplate=HOVER_TEMPLATE
            ))

    # 3. Top Cohort (High Contrast)
    if precomputed_top is not None:
        top_vals = precomputed_top.get('portfolio_values', [initial_investment]) if isinstance(precomputed_top, dict) else precomputed_top
        if len(top_vals) >= len(years):
            fig.add_trace(go.Scatter(
                x=years, y=top_vals[:len(years)], name=f'Top {percent}%',
                mode='lines+markers',
                line=dict(color='#2ca02c', width=2.0),
                marker=dict(symbol='triangle-up', size=5),
                hovertemplate=HOVER_TEMPLATE
            ))

    # 4. Active Portfolio Strategy
    if baseline_portfolio_values:
        common_len = min(len(years), len(baseline_portfolio_values))
        fig.add_trace(go.Scatter(
            x=years[:common_len], y=baseline_portfolio_values[:common_len],
            name='Active Strategy',
            mode='lines+markers',
            line=dict(color='#003366', width=2.5),
            marker=dict(symbol='circle', size=4),
            hovertemplate=HOVER_TEMPLATE
        ))

    # Institutional Chart Formatting
    fig.update_layout(
        title=dict(text=f'<b>Factor Efficacy: Top vs. Bottom {percent}% Cohort Spread</b>',
                   font=dict(size=14)),
        height=500,
        hovermode='x unified',
        # Legend in the upper left, inside the plot area
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top', font=dict(size=10))
    )

    fig.update_yaxes(
        title=dict(text='Account Value (USD)', font=dict(size=11)),
        tickprefix='$',
        tickformat=',.0f',
        showgrid=True,
        griddash='dash'
    )
    if dates:
        date_axis(fig, font_size=11)
    else:
        fig.update_xaxes(
            title=dict(text='Year', font=dict(size=11)),
            tickformat='d',
            hoverformat='d',
            dtick=1,
            showgrid=True,
            griddash='dash'
        )

    fig.add_hline(y=initial_investment, line_color='#000000', line_dash='solid',
                  line_width=0.8, opacity=0.2)

    return fig
