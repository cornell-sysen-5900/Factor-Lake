"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/portfolio_growth_plot.py
PURPOSE: Standalone plotting utility for comparative wealth-index visualizations,
         rendered as an interactive Plotly figure (zoom, pan, hover values).
VERSION: 3.0.0
"""

import plotly.graph_objects as go
from typing import Dict, List, Optional

HOVER_TEMPLATE = '%{fullData.name}: $%{y:,.0f}<extra></extra>'
# Legacy's benchmark names; WRDS runs pass their ETFs' names instead
BENCHMARK_NAMES = {'benchmark': 'Russell 2000', 'value': 'Value Index', 'growth': 'Growth Index'}


def date_axis(fig: go.Figure, font_size: int) -> go.Figure:
    """
    An x axis of dates (WRDS runs) that looks like Legacy's year axis: a tick at each
    year, with the full date on hover.
    """
    fig.update_xaxes(
        title=dict(text='Date', font=dict(size=font_size)),
        type='date',
        tickformat='%Y',
        hoverformat='%Y-%m-%d',
        dtick='M12',
        showgrid=True,
        griddash='dash'
    )
    return fig


def plot_portfolio_growth(
    years: List[int],
    port_vals: List[float],
    bench_vals: Optional[List[float]] = None,
    val_vals: Optional[List[float]] = None,
    gro_vals: Optional[List[float]] = None,
    factor_names: Optional[List[str]] = None,
    names: Optional[Dict[str, str]] = None,
    dates: bool = False
) -> go.Figure:
    """
    Constructs a time-series line chart comparing strategy growth against multiple benchmarks.

    This utility provides an interactive visualization of cumulative wealth,
    supporting overlays for market, value, and growth indices.

    Args:
        years: A list of integers representing the timeline years.
        port_vals: Cumulative dollar values for the strategy portfolio.
        bench_vals: Optional cumulative values for the Russell 2000 index.
        val_vals: Optional cumulative values for the Russell 2000 Value Index.
        gro_vals: Optional cumulative values for the Russell 2000 Growth Index.
        factor_names: A list of active factor labels to be included in the chart title.
        names: Benchmark names by role ('benchmark', 'value', 'growth'), Legacy's by default.
        dates: True when `years` are dates (WRDS runs): a date axis with a tick per year.

    Returns:
        go.Figure: The generated Plotly figure object.
    """
    fig = go.Figure()
    names = {**BENCHMARK_NAMES, **(names or {})}

    # Benchmarks are added first so the portfolio line is drawn on top of them;
    # legendrank keeps the legend in the original order (Portfolio first).

    # 2. Market Benchmark (Russell 2000)
    if bench_vals and len(years) == len(bench_vals):
        fig.add_trace(go.Scatter(
            x=years,
            y=bench_vals,
            name=names['benchmark'],
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(symbol='square', size=4),
            opacity=0.8,
            legendrank=2,
            hovertemplate=HOVER_TEMPLATE
        ))

    # 3. Style Benchmark: Value Index
    if val_vals and len(years) == len(val_vals):
        fig.add_trace(go.Scatter(
            x=years,
            y=val_vals,
            name=names['value'],
            mode='lines+markers',
            line=dict(color='#2ca02c', width=1.5, dash='dot'),
            marker=dict(symbol='triangle-up', size=4),
            opacity=0.6,
            legendrank=3,
            hovertemplate=HOVER_TEMPLATE
        ))

    # 4. Style Benchmark: Growth Index
    if gro_vals and len(years) == len(gro_vals):
        fig.add_trace(go.Scatter(
            x=years,
            y=gro_vals,
            name=names['growth'],
            mode='lines+markers',
            line=dict(color='#d62728', width=1.5, dash='dot'),
            marker=dict(symbol='diamond', size=4),
            opacity=0.6,
            legendrank=4,
            hovertemplate=HOVER_TEMPLATE
        ))

    # 1. Primary Portfolio Strategy
    fig.add_trace(go.Scatter(
        x=years,
        y=port_vals,
        name='Portfolio',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2.5),
        marker=dict(symbol='circle', size=6),
        legendrank=1,
        hovertemplate=HOVER_TEMPLATE
    ))

    # Title and Label Configuration
    if factor_names:
        clean_names = [name.replace('_', ' ').title() for name in factor_names]
        title_str = f'Cumulative Wealth: {", ".join(clean_names)}'
    else:
        title_str = 'Portfolio Growth Comparison'

    fig.update_layout(
        title=dict(text=f'<b>{title_str}</b>', font=dict(size=14)),
        height=500,
        hovermode='x unified',
        # Legend in the upper left, inside the plot area
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top')
    )

    # Financial Axis Formatting
    fig.update_yaxes(
        title=dict(text='Portfolio Value (USD)', font=dict(size=12)),
        tickprefix='$',
        tickformat=',.0f',
        showgrid=True,
        griddash='dash'
    )

    if dates:
        return date_axis(fig, font_size=12)

    # Ensure X-axis utilizes integer year ticks
    fig.update_xaxes(
        title=dict(text='Year', font=dict(size=12)),
        tickformat='d',
        hoverformat='d',
        dtick=1,
        showgrid=True,
        griddash='dash'
    )

    return fig
