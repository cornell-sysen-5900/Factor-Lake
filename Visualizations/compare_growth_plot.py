"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/compare_growth_plot.py
PURPOSE: The Compare tab's growth chart: each compared analysis's portfolio (after costs)
         and its own benchmark on one date axis, in the style of the Results tab's growth
         chart (portfolio_growth_plot.py).
VERSION: 1.0.0
"""

from typing import Any, Dict, List, Optional, Sequence

import plotly.graph_objects as go

from Visualizations.portfolio_growth_plot import HOVER_TEMPLATE, date_axis, line_mode

# One hue per compared analysis, in the order entered (never by rank, so a sort never
# repaints a line); its benchmark is the same hue, dashed and lighter
COMPARE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')


def plot_compare_growth(series: Sequence[Dict[str, Any]], names: Sequence[str], tags: Sequence[str],
                        order: Optional[List[int]] = None) -> go.Figure:
    """
    Args:
        series: per analysis, in the order entered: {'dates', 'portfolio', 'benchmark',
            'benchmark_name'} (app/compare.growth_series).
        names: per analysis, the legend name of its portfolio line.
        tags: per analysis, a short tag (its short ID) added to its benchmark's legend name.
        order: the positions as sorted (the legend lists them in this order); the order
            entered by default. Colors stay with the position entered.
    """
    fig = go.Figure()
    order = list(range(len(series))) if order is None else list(order)
    rank = {i: place for place, i in enumerate(order)}
    # Benchmarks first, so every portfolio line is drawn on top of them
    for i, s in enumerate(series):
        if s['benchmark'] and len(s['benchmark']) == len(s['dates']):
            fig.add_trace(go.Scatter(
                x=s['dates'], y=s['benchmark'], name=f"{s['benchmark_name']} · {tags[i]}",
                mode=line_mode(len(s['dates'])), legendgroup=str(i), legendrank=10 * rank[i] + 2,
                line=dict(color=COMPARE_COLORS[i], width=1.5, dash='dash'),
                marker=dict(symbol='square', size=4), opacity=0.55, hovertemplate=HOVER_TEMPLATE))
    for i, s in enumerate(series):
        fig.add_trace(go.Scatter(
            x=s['dates'], y=s['portfolio'], name=names[i],
            mode=line_mode(len(s['dates'])), legendgroup=str(i), legendrank=10 * rank[i] + 1,
            line=dict(color=COMPARE_COLORS[i], width=2.5),
            marker=dict(symbol='circle', size=6), hovertemplate=HOVER_TEMPLATE))

    fig.update_layout(
        title=dict(text='<b>Cumulative Wealth: compared analyses</b>', font=dict(size=14)),
        height=500,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
    )
    fig.update_yaxes(
        title=dict(text='Portfolio Value (USD)', font=dict(size=12)),
        tickprefix='$', tickformat=',.0f', showgrid=True, griddash='dash',
    )
    return date_axis(fig, font_size=12)
