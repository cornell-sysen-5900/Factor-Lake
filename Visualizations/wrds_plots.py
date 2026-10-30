"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: Visualizations/wrds_plots.py
PURPOSE: The one chart only WRDS runs have: each holding period's return, in the style of
         Legacy's growth chart (colors, line styles, legend, axis and hover formats).
VERSION: 2.0.0

WRDS runs draw their growth and cohort charts with Legacy's own functions
(portfolio_growth_plot.py, top_bottom_portfolio_plot.py) on a date axis.
"""

from typing import List

import plotly.graph_objects as go

from Visualizations.portfolio_growth_plot import date_axis, line_mode

PERCENT_HOVER = '%{fullData.name}: %{y:.2f}%<extra></extra>'


def plot_wrds_period_returns(period_ends: List[str], port_returns: List[float], bench_returns: List[float],
                             bench_label: str, frequency_label: str) -> go.Figure:
    """Each holding period's return (percent) at its end date: portfolio bars, the benchmark as a line."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=period_ends, y=port_returns, name='Portfolio', marker_color='#1f77b4',
        legendrank=1, hovertemplate=PERCENT_HOVER
    ))
    # Legacy's benchmark line
    fig.add_trace(go.Scatter(
        x=period_ends, y=bench_returns, name=bench_label, mode=line_mode(len(period_ends)),
        line=dict(color='#ff7f0e', width=2, dash='dash'), marker=dict(symbol='square', size=4),
        opacity=0.8, legendrank=2, hovertemplate=PERCENT_HOVER
    ))
    fig.update_layout(
        title=dict(text=f'<b>{frequency_label} Returns</b>', font=dict(size=14)),
        height=500,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top')
    )
    fig.update_yaxes(
        title=dict(text='Return (%)', font=dict(size=12)),
        ticksuffix='%',
        showgrid=True,
        griddash='dash'
    )
    return date_axis(fig, font_size=12)
