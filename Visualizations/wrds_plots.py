"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: Visualizations/wrds_plots.py
PURPOSE: Interactive Plotly charts of WRDS runs on a date axis (rebalance and period end dates):
         growth, period returns, drawdown and the cohort spread.
VERSION: 1.0.0

Legacy's charts (portfolio_growth_plot.py, top_bottom_portfolio_plot.py) use a year axis
and stay as they are; these use the same colors, sizes and layout on dates.
"""

from typing import Dict, List, Optional

import plotly.graph_objects as go

MONEY_HOVER = '%{fullData.name}: $%{y:,.0f}<extra></extra>'
PERCENT_HOVER = '%{fullData.name}: %{y:.2f}%<extra></extra>'
# Colors and line styles of Legacy's growth chart: portfolio, benchmark, value, growth
STYLES = {
    'portfolio': dict(color='#1f77b4', width=2.5),
    'benchmark': dict(color='#ff7f0e', width=2, dash='dash'),
    'value': dict(color='#2ca02c', width=1.5, dash='dot'),
    'growth': dict(color='#d62728', width=1.5, dash='dot'),
}


def _layout(fig: go.Figure, title: str, y_title: str, money: bool, legend_above: bool = False) -> go.Figure:
    """
    Legacy's chart layout on a date axis. The legend sits in the upper left inside the plot,
    as in Legacy, or above the plot (legend_above) for charts whose data reaches that corner.
    """
    legend = (dict(orientation='h', x=0, xanchor='left', y=1.0, yanchor='bottom', font=dict(size=10))
              if legend_above else dict(x=0.01, y=0.99, xanchor='left', yanchor='top', font=dict(size=10)))
    fig.update_layout(title=dict(text=f'<b>{title}</b>', font=dict(size=14)), height=450,
                      hovermode='x unified', legend=legend, margin=dict(t=90) if legend_above else None)
    fig.update_yaxes(title=dict(text=y_title, font=dict(size=12)), showgrid=True, griddash='dash',
                     **(dict(tickprefix='$', tickformat=',.0f') if money else dict(ticksuffix='%')))
    fig.update_xaxes(title=dict(text='Date', font=dict(size=12)), type='date', hoverformat='%Y-%m-%d',
                     showgrid=True, griddash='dash')
    return fig


def _markers(n_points: int) -> str:
    """Lines with markers for short series; lines only for long (monthly) ones."""
    return 'lines+markers' if n_points <= 60 else 'lines'


def plot_wrds_growth(dates: List[str], port_vals: List[float], benchmarks: Dict[str, Optional[List[float]]],
                     labels: Dict[str, str], factor_names: Optional[List[str]] = None) -> go.Figure:
    """
    Portfolio value and the benchmarks grown from the same start, at each period boundary.

    benchmarks and labels are keyed by role: 'benchmark' (IWM), 'value' (IWN), 'growth' (IWO).
    """
    fig = go.Figure()
    mode = _markers(len(dates))
    for rank, role in enumerate(('benchmark', 'value', 'growth'), start=2):
        values = benchmarks.get(role)
        if values and len(values) == len(dates):
            fig.add_trace(go.Scatter(x=dates, y=values, name=labels[role], mode=mode, line=STYLES[role],
                                     opacity=0.8 if role == 'benchmark' else 0.6, legendrank=rank,
                                     hovertemplate=MONEY_HOVER))
    fig.add_trace(go.Scatter(x=dates, y=port_vals, name='Portfolio', mode=mode, line=STYLES['portfolio'],
                             legendrank=1, hovertemplate=MONEY_HOVER))
    title = f"Cumulative Wealth: {', '.join(factor_names)}" if factor_names else 'Portfolio Growth Comparison'
    return _layout(fig, title, 'Portfolio Value (USD)', money=True)


def plot_wrds_period_returns(period_ends: List[str], port_returns: List[float], bench_returns: List[float],
                             bench_label: str, frequency_label: str) -> go.Figure:
    """Each holding period's return (percent): portfolio bars, the benchmark as a line."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=period_ends, y=port_returns, name='Portfolio',
                         marker_color=['#1f77b4' if r >= 0 else '#9ecae1' for r in port_returns],
                         hovertemplate=PERCENT_HOVER))
    fig.add_trace(go.Scatter(x=period_ends, y=bench_returns, name=bench_label, mode=_markers(len(period_ends)),
                             line=STYLES['benchmark'], hovertemplate=PERCENT_HOVER))
    return _layout(fig, f'{frequency_label} Returns', 'Return (%)', money=False, legend_above=True)


def plot_wrds_drawdown(dates: List[str], port_drawdown: List[float], bench_drawdown: Optional[List[float]],
                       bench_label: str) -> go.Figure:
    """Drawdown from the running peak (percent), portfolio and benchmark."""
    fig = go.Figure()
    if bench_drawdown and len(bench_drawdown) == len(dates):
        fig.add_trace(go.Scatter(x=dates, y=[d * 100 for d in bench_drawdown], name=bench_label, mode='lines',
                                 line=STYLES['benchmark'], hovertemplate=PERCENT_HOVER))
    fig.add_trace(go.Scatter(x=dates, y=[d * 100 for d in port_drawdown], name='Portfolio', mode='lines',
                             line=STYLES['portfolio'], fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.15)',
                             hovertemplate=PERCENT_HOVER))
    return _layout(fig, 'Drawdown from Peak', 'Drawdown (%)', money=False, legend_above=True)


def plot_wrds_cohort(dates: List[str], percent: float, top: List[float], bottom: List[float],
                     baseline: Optional[List[float]], benchmark: Optional[List[float]], bench_label: str,
                     initial_investment: float) -> go.Figure:
    """Top vs bottom cohort values on the run's dates, with the strategy and the benchmark (Legacy's colors)."""
    fig = go.Figure()
    mode = _markers(len(dates))
    if benchmark and len(benchmark) == len(dates):
        fig.add_trace(go.Scatter(x=dates, y=benchmark, name=bench_label, mode=mode, opacity=0.6,
                                 line=dict(color='#d62728', dash='dash', width=1.2), hovertemplate=MONEY_HOVER))
    if len(bottom) >= len(dates):
        fig.add_trace(go.Scatter(x=dates, y=bottom[:len(dates)], name=f'Bottom {percent}%', mode=mode, opacity=0.6,
                                 line=dict(color='#9467bd', width=1.5), hovertemplate=MONEY_HOVER))
    if len(top) >= len(dates):
        fig.add_trace(go.Scatter(x=dates, y=top[:len(dates)], name=f'Top {percent}%', mode=mode,
                                 line=dict(color='#2ca02c', width=2.0), hovertemplate=MONEY_HOVER))
    if baseline:
        n = min(len(dates), len(baseline))
        fig.add_trace(go.Scatter(x=dates[:n], y=baseline[:n], name='Active Strategy', mode=mode,
                                 line=dict(color='#003366', width=2.5), hovertemplate=MONEY_HOVER))
    _layout(fig, f'Factor Efficacy: Top vs. Bottom {percent}% Cohort Spread', 'Account Value (USD)', money=True)
    fig.add_hline(y=initial_investment, line_color='#000000', line_dash='solid', line_width=0.8, opacity=0.2)
    return fig
