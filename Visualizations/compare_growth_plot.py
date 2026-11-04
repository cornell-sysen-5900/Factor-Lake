"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/compare_growth_plot.py
PURPOSE: The Compare tab's growth chart: each compared analysis's portfolio (after costs)
         and each distinct benchmark once, on one date axis, in the style of the Results
         tab's growth chart (portfolio_growth_plot.py).
VERSION: 1.1.0
"""

from typing import Any, Dict, List, Optional, Sequence

import plotly.graph_objects as go

from Visualizations.portfolio_growth_plot import HOVER_TEMPLATE, date_axis, line_mode

# One hue per compared analysis, in the order entered (never by rank, so a sort never
# repaints a line)
COMPARE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
# Benchmarks are neutral grey and dotted, one style per benchmark (by name, so a sort never
# restyles one)
BENCHMARK_STYLES = (('#5f6368', 'dot'), ('#9aa0a6', 'dashdot'), ('#3c4043', 'longdashdot'), ('#bdc1c6', 'dot'))


def plot_compare_growth(series: Sequence[Dict[str, Any]], names: Sequence[str],
                        benchmarks: Sequence[Dict[str, Any]], order: Optional[List[int]] = None,
                        growth_of: Optional[float] = None) -> go.Figure:
    """
    Args:
        series: per analysis, in the order entered: {'dates', 'portfolio', ...}
            (app/compare.chart_series).
        names: per analysis, the legend name of its portfolio line.
        benchmarks: each distinct benchmark once, in the legend's order: {'name', 'dates',
            'values'} (app/compare.benchmark_lines).
        order: the positions as sorted (the legend lists them in this order); the order
            entered by default. Colors stay with the position entered.
        growth_of: the amount every line was scaled to start at (analyses with different
            initial AUMs), named in the title and axis; None for dollar values.
    """
    fig = go.Figure()
    order = list(range(len(series))) if order is None else list(order)
    rank = {i: place for place, i in enumerate(order)}
    styles = {name: BENCHMARK_STYLES[k % len(BENCHMARK_STYLES)]
              for k, name in enumerate(sorted({b['name'] for b in benchmarks}))}
    # Benchmarks first, so every portfolio line is drawn on top of them; listed after the portfolios
    for place, b in enumerate(benchmarks):
        color, dash = styles[b['name']]
        fig.add_trace(go.Scatter(
            x=b['dates'], y=b['values'], name=b['name'], mode='lines',
            legendrank=100 + place, line=dict(color=color, width=1.8, dash=dash),
            hovertemplate=HOVER_TEMPLATE))
    for i, s in enumerate(series):
        fig.add_trace(go.Scatter(
            x=s['dates'], y=s['portfolio'], name=names[i],
            mode=line_mode(len(s['dates'])), legendrank=1 + rank[i],
            line=dict(color=COMPARE_COLORS[i], width=2.5),
            marker=dict(symbol='circle', size=6), hovertemplate=HOVER_TEMPLATE))

    fig.update_layout(
        title=dict(text=(f'<b>Growth of ${growth_of:,.0f}: compared analyses, each scaled to start at '
                         f'${growth_of:,.0f}</b>' if growth_of else '<b>Cumulative Wealth: compared analyses</b>'),
                   font=dict(size=14)),
        height=500,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top'),
    )
    fig.update_yaxes(
        title=dict(text=f'Growth of ${growth_of:,.0f} (USD)' if growth_of else 'Portfolio Value (USD)',
                   font=dict(size=12)),
        tickprefix='$', tickformat=',.0f', showgrid=True, griddash='dash',
    )
    return date_axis(fig, font_size=12)
