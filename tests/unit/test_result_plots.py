"""
Tests for the interactive (Plotly) Results charts in Visualizations/.
"""
import plotly.graph_objects as go
import pytest

from Visualizations.portfolio_growth_plot import plot_portfolio_growth
from Visualizations.top_bottom_portfolio_plot import plot_top_bottom_percent

YEARS = [2018, 2019, 2020, 2021]
INITIAL = 1_000_000.0


def _wealth(*mults):
    vals = [INITIAL]
    for m in mults:
        vals.append(vals[-1] * m)
    return vals


PORT = _wealth(1.10, 0.95, 1.20)
BENCH = _wealth(1.05, 0.90, 1.15)
VALUE = _wealth(1.02, 0.97, 1.10)
GROWTH = _wealth(1.08, 0.92, 1.25)


def _traces(fig):
    return {t.name: t for t in fig.data}


@pytest.mark.unit
class TestPlotPortfolioGrowth:
    def test_returns_plotly_figure(self):
        assert isinstance(plot_portfolio_growth(YEARS, PORT), go.Figure)

    def test_portfolio_only_when_benchmarks_omitted(self):
        fig = plot_portfolio_growth(YEARS, PORT)
        assert [t.name for t in fig.data] == ['Portfolio']

    def test_all_series_when_benchmarks_given(self):
        fig = plot_portfolio_growth(YEARS, PORT, bench_vals=BENCH, val_vals=VALUE, gro_vals=GROWTH)
        assert set(_traces(fig)) == {'Portfolio', 'Russell 2000', 'Value Index', 'Growth Index'}

    def test_benchmark_with_mismatched_length_is_skipped(self):
        fig = plot_portfolio_growth(YEARS, PORT, bench_vals=BENCH[:-1], val_vals=VALUE)
        assert set(_traces(fig)) == {'Portfolio', 'Value Index'}

    def test_legend_order_matches_original(self):
        fig = plot_portfolio_growth(YEARS, PORT, bench_vals=BENCH, val_vals=VALUE, gro_vals=GROWTH)
        ranked = sorted(fig.data, key=lambda t: t.legendrank)
        assert [t.name for t in ranked] == ['Portfolio', 'Russell 2000', 'Value Index', 'Growth Index']

    def test_colors_match_original(self):
        fig = plot_portfolio_growth(YEARS, PORT, bench_vals=BENCH, val_vals=VALUE, gro_vals=GROWTH)
        colors = {name: t.line.color for name, t in _traces(fig).items()}
        assert colors == {
            'Portfolio': '#1f77b4',
            'Russell 2000': '#ff7f0e',
            'Value Index': '#2ca02c',
            'Growth Index': '#d62728',
        }

    def test_title_uses_factor_names(self):
        fig = plot_portfolio_growth(YEARS, PORT, factor_names=['roa_pct', 'book_price'])
        assert 'Cumulative Wealth: Roa Pct, Book Price' in fig.layout.title.text

    def test_default_title(self):
        fig = plot_portfolio_growth(YEARS, PORT)
        assert 'Portfolio Growth Comparison' in fig.layout.title.text

    def test_dollar_axis_and_unified_hover(self):
        fig = plot_portfolio_growth(YEARS, PORT)
        assert fig.layout.yaxis.tickprefix == '$'
        assert fig.layout.yaxis.tickformat == ',.0f'
        assert fig.layout.hovermode == 'x unified'
        assert '$%{y:,.0f}' in fig.data[0].hovertemplate


@pytest.mark.unit
class TestPlotTopBottomPercent:
    def test_returns_plotly_figure(self):
        assert isinstance(plot_top_bottom_percent(YEARS), go.Figure)

    def test_no_traces_when_nothing_given(self):
        fig = plot_top_bottom_percent(YEARS)
        assert len(fig.data) == 0

    def test_all_series_when_given(self):
        fig = plot_top_bottom_percent(
            YEARS, percent=10.0,
            benchmark_returns=BENCH, growth_returns=GROWTH, value_returns=VALUE,
            baseline_portfolio_values=PORT, precomputed_top=PORT, precomputed_bot=BENCH,
        )
        assert [t.name for t in fig.data] == [
            'Russell 2000', 'Growth Index', 'Value Index',
            'Bottom 10.0%', 'Top 10.0%', 'Active Strategy',
        ]

    def test_bottom_hidden_when_show_bottom_false(self):
        fig = plot_top_bottom_percent(YEARS, show_bottom=False, precomputed_top=PORT, precomputed_bot=BENCH)
        assert [t.name for t in fig.data] == ['Top 10.0%']

    def test_cohort_shorter_than_years_is_skipped(self):
        fig = plot_top_bottom_percent(YEARS, precomputed_top=PORT[:2], precomputed_bot=BENCH)
        assert [t.name for t in fig.data] == ['Bottom 10.0%']

    def test_colors_match_original(self):
        fig = plot_top_bottom_percent(
            YEARS, benchmark_returns=BENCH, growth_returns=GROWTH, value_returns=VALUE,
            baseline_portfolio_values=PORT, precomputed_top=PORT, precomputed_bot=BENCH,
        )
        colors = {name: t.line.color for name, t in _traces(fig).items()}
        assert colors == {
            'Russell 2000': '#d62728',
            'Growth Index': '#ff7f0e',
            'Value Index': '#17becf',
            'Bottom 10.0%': '#9467bd',
            'Top 10.0%': '#2ca02c',
            'Active Strategy': '#003366',
        }

    def test_index_lines_have_markers(self):
        fig = plot_top_bottom_percent(YEARS, benchmark_returns=BENCH, growth_returns=GROWTH, value_returns=VALUE)
        traces = _traces(fig)
        assert traces['Russell 2000'].mode == 'lines+markers'
        assert {t.marker.size for t in fig.data} == {3}
        symbols = [t.marker.symbol for t in fig.data]
        assert traces['Russell 2000'].marker.symbol == 'x'
        assert len(set(symbols)) == len(symbols)

    def test_precomputed_top_as_list(self):
        fig = plot_top_bottom_percent(YEARS, precomputed_top=PORT)
        assert list(_traces(fig)['Top 10.0%'].y) == PORT

    def test_precomputed_top_as_dict(self):
        fig = plot_top_bottom_percent(YEARS, precomputed_top={'portfolio_values': PORT})
        assert list(_traces(fig)['Top 10.0%'].y) == PORT

    def test_precomputed_dict_without_values_is_skipped(self):
        fig = plot_top_bottom_percent(YEARS, precomputed_top={})
        assert len(fig.data) == 0

    def test_precomputed_longer_than_years_is_truncated(self):
        fig = plot_top_bottom_percent(YEARS, precomputed_top=PORT + [9e9])
        assert list(_traces(fig)['Top 10.0%'].y) == PORT

    def test_wealth_levels_plotted_directly(self):
        fig = plot_top_bottom_percent(YEARS, benchmark_returns=BENCH)
        assert list(_traces(fig)['Russell 2000'].y) == BENCH

    def test_returns_converted_to_wealth(self):
        # Three percent returns -> four wealth points starting at the initial investment
        fig = plot_top_bottom_percent(YEARS, benchmark_returns=[10.0, -5.0, 20.0])
        assert list(_traces(fig)['Russell 2000'].y) == pytest.approx(_wealth(1.10, 0.95, 1.20))

    def test_reference_line_at_initial_investment(self):
        fig = plot_top_bottom_percent(YEARS, initial_investment=INITIAL)
        hlines = [s for s in fig.layout.shapes if s.y0 == s.y1 == INITIAL]
        assert len(hlines) == 1
        assert hlines[0].line.color == '#000000'

    def test_dollar_axis_and_unified_hover(self):
        fig = plot_top_bottom_percent(YEARS, precomputed_top=PORT)
        assert fig.layout.yaxis.tickprefix == '$'
        assert fig.layout.yaxis.tickformat == ',.0f'
        assert fig.layout.hovermode == 'x unified'
        assert '$%{y:,.0f}' in fig.data[0].hovertemplate
