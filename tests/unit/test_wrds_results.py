"""
Tests for the Results tab of WRDS runs (app/components/wrds_results.py, app/wrds_tables.py,
Visualizations/wrds_plots.py): a WRDS run is drawn by Legacy's own render functions, so its
sections, titles, metric names, table headers, widgets and chart styles are compared with
Legacy's Results tab as main rendered it (tests/fixtures/legacy_results_render_main.json),
with only the names that must differ (period for year, the ETFs for Legacy's indices).
Also: holdings and ranked stocks in Legacy's units, reopening a WRDS ID in a new session,
and Legacy's Results tab unchanged from main.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.wrds_tables import (benchmark_label, calendar_year_results, factor_header, frequency_caption,
                             holdings_display_table, period_by_period_columns, ranked_display_table, result_notes,
                             table_number_formats, win_loss_columns)
from src.wrds_engine import rank_stocks, run_wrds_backtest
from scripts.check_backtest_lambda import TOLERANCE, max_difference
from tests.unit.results_snapshot import comparable, legacy_results_snapshot, main_area
from tests.unit.test_backtest_lambda import DELISTING_FILE, UNIVERSE_FILE, _delisting, _parquet, _raw_universe
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab, _open_tab
from tests.unit.test_wrds_saved import _run_wrds, manager, new_session  # noqa: F401 (fixtures)
from tests.unit.wrds_fixtures import RangeS3, month_ends, put_wrds_files, synthetic_frames
from Visualizations.portfolio_growth_plot import plot_portfolio_growth
from Visualizations.top_bottom_portfolio_plot import plot_top_bottom_percent
from Visualizations.wrds_plots import plot_wrds_period_returns

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'
FACTORS = ['f_roe', 'mom_12_1', 'mkt_cap_musd', 'f_book_to_price']
LABELS = {'f_roe': 'ROE', 'mom_12_1': '12-1 Mo Momentum', 'mkt_cap_musd': 'Market Cap', 'f_book_to_price': 'Book/Price'}
DIRECTIONS = {f: 'top' for f in FACTORS}
BOUNDS = dict(first_month='2009-11', last_month='2013-02')
# Legacy's holdings columns around the factor values (backtest_engine's holdings rows)
LEGACY_HOLDINGS_BEFORE = ['Rank', 'Ticker', 'Name', 'Sector', 'Picked By', 'Score']
LEGACY_HOLDINGS_AFTER = ['Weight %', 'Shares', 'Buy Price', 'Dollar Value', 'Period Return %', 'Contribution %',
                         'Delisted']
# The style of a chart, everything but its data and names
TRACE_STYLE = ('type', 'mode', 'line', 'marker', 'opacity', 'legendrank', 'hovertemplate')
LAYOUT_STYLE = ('legend', 'height', 'hovermode', 'yaxis')


@pytest.fixture(scope='module')
def backtest():
    frames = synthetic_frames()
    data = frames.data()
    results, holdings = run_wrds_backtest(data, FACTORS, DIRECTIONS, 'quarterly', 2010, 2012, 1000.0,
                                          factor_labels=LABELS, **BOUNDS)
    return data, results, holdings


@pytest.fixture(scope='module')
def legacy():
    """Legacy's Results tab after a cohort comparison, as main rendered it."""
    return json.loads((FIXTURES / 'legacy_results_render_main.json').read_text(encoding='utf-8'))['run2_cohort']


def _style(trace) -> dict:
    trace = trace if isinstance(trace, dict) else trace.to_plotly_json()
    return {k: trace.get(k) for k in TRACE_STYLE}


def _layout(fig) -> dict:
    layout = fig['layout'] if isinstance(fig, dict) else fig.to_plotly_json()['layout']
    return {k: layout.get(k) for k in LAYOUT_STYLE} | {'title_font': layout['title']['font']}


# ---------------------------------------------------------------------------
# Tables: Legacy's columns, order and units, WRDS display names
# ---------------------------------------------------------------------------

def test_factor_headers_show_their_unit():
    assert factor_header('f_roe', 'ROE') == 'ROE %'
    assert factor_header('mkt_cap_musd', 'Market Cap') == 'Market Cap ($M)'
    assert factor_header('f_book_to_price', 'Book/Price') == 'Book/Price'
    assert factor_header('beta_252d', 'Beta (1-Yr Daily)') == 'Beta (1-Yr Daily)'


def test_holdings_table_has_legacys_columns_order_and_units(backtest):
    _, _, holdings = backtest
    date = holdings['rebalance_date'].iloc[-1]
    raw = holdings[holdings['rebalance_date'] == date].reset_index(drop=True)
    table = holdings_display_table(raw, FACTORS, LABELS)
    assert list(table.columns) == LEGACY_HOLDINGS_BEFORE + ['ROE %', '12-1 Mo Momentum %', 'Market Cap ($M)',
                                                             'Book/Price'] + LEGACY_HOLDINGS_AFTER + [
        'PERMNO', 'Months Without Return']
    np.testing.assert_allclose(table['Weight %'], raw['weight'] * 100)
    np.testing.assert_allclose(table['Period Return %'], raw['period_return'] * 100)
    np.testing.assert_allclose(table['Contribution %'], raw['contribution'] * 100)
    np.testing.assert_allclose(table['ROE %'], raw['f_roe'] * 100)
    np.testing.assert_allclose(table['Market Cap ($M)'], raw['mkt_cap_musd'])
    np.testing.assert_allclose(table['Book/Price'], raw['f_book_to_price'])
    assert table['Weight %'].sum() == pytest.approx(100.0)
    assert set(', '.join(table['Picked By']).split(', ')) <= set(LABELS.values())     # display names
    formats = table_number_formats(table, FACTORS, LABELS)
    assert formats['Weight %'] == '%.2f' and formats['Period Return %'] == '%.2f' and formats['ROE %'] == '%.4f'
    assert formats['Buy Price'] == 'dollar' and formats['PERMNO'] == '%d' and formats['Rank'] == '%d'


@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
def test_ranked_table_follows_the_holdings_tables_order(backtest, combination):
    data, results, _ = backtest
    date = pd.Timestamp(results['rebalance_dates'][2])
    rows = rank_stocks(data, date, FACTORS, DIRECTIONS, combination)
    table = ranked_display_table(rows, FACTORS, LABELS, combination)
    expected = ['Rank', 'Ticker', 'Name', 'Sector', 'Held', 'Score']
    for f in FACTORS:
        expected += [factor_header(f, LABELS[f]), f"{LABELS[f]} Z-Score"] + (
            [f"Rank by {LABELS[f]}"] if combination == 'mixing' else [])
    assert list(table.columns) == expected + ['Weight %', 'Price', 'PERMNO']
    assert table['Weight %'].sum() == pytest.approx(100.0)
    np.testing.assert_allclose(table['ROE %'], rows['f_roe'] * 100)
    assert table_number_formats(table, FACTORS, LABELS)['ROE Z-Score'] == '%.4f'


def test_legacys_table_headers_with_the_period_and_the_etfs(backtest):
    _, results, _ = backtest
    assert period_by_period_columns(results) == ['Period', 'Portfolio Value', 'Period Return', 'IWM Returns',
                                                 'IWO (Growth) Returns', 'IWN (Value) Returns']
    assert period_by_period_columns(results, 'Year', 'YoY Return')[:3] == ['Year', 'Portfolio Value', 'YoY Return']
    assert win_loss_columns(results) == ['Period', 'Portfolio Return', 'IWM Return', 'Outperformed IWM',
                                         'IWO (Growth) Return', 'Outperformed IWO (Growth)', 'IWN (Value) Return',
                                         'Outperformed IWN (Value)']


def test_calendar_years_compound_their_periods(backtest):
    _, results, _ = backtest
    years, by_year = calendar_year_results(results)
    assert years == [2010, 2011, 2012]
    first_year = (np.prod([1 + r / 100 for r in results['yearly_returns'][:4]]) - 1) * 100
    assert by_year['yearly_returns'][0] == pytest.approx(first_year)
    iwm = (np.prod([1 + r / 100 for r in results['benchmark_returns'][4:8]]) - 1) * 100
    assert by_year['benchmark_returns'][1] == pytest.approx(iwm)
    # Values at each year's last period end, from the initial AUM to the final value
    assert by_year['portfolio_values'][0] == 1000.0
    assert by_year['portfolio_values'][-1] == pytest.approx(results['final_value'])
    assert by_year['portfolio_values'][1] == pytest.approx(results['portfolio_values'][4])


def test_frequency_caption_and_notes(backtest):
    _, results, _ = backtest
    assert frequency_caption(results) == (
        "Quarterly returns (4 per year, 12 periods = 3.0 years): volatility, Sharpe and information ratios are "
        "annualized by √4, and win rates count quarterly periods. Benchmark = IWM, Growth = IWO, Value = IWN "
        "(ETF total returns).")
    quiet = {'total_delisted_positions': 0, 'missing_return_stock_months': 0, 'rf_filled_months': [],
             'empty_rebalance_dates': []}
    assert result_notes(quiet) == []
    notes = result_notes(dict(quiet, missing_return_stock_months=3, rf_filled_months=['2026-09'],
                              empty_rebalance_dates=['2011-03-31']))
    assert notes == ["**Stock-months without a return (counted as 0%):** 3",
                     "**Risk-free months filled from the 3-month T-bill / 12:** 2026-09",
                     "**Rebalance dates with nothing to buy (the money earned the risk-free rate):** 2011-03-31"]


# ---------------------------------------------------------------------------
# Charts: Legacy's functions and styles on a date axis
# ---------------------------------------------------------------------------

def _names(results):
    return {role: benchmark_label(results, role) for role in ('benchmark', 'value', 'growth')}


def test_growth_chart_is_legacys_on_dates(backtest):
    _, results, _ = backtest
    values, n = results['portfolio_values'], len(results['years'])
    legacy = plot_portfolio_growth(list(range(2000, 2000 + n)), values, values, values, values, ['ROE'])
    wrds = plot_portfolio_growth(results['years'], values, values, values, values, ['ROE'],
                                 names=_names(results), dates=True)
    assert [t.name for t in legacy.data] == ['Russell 2000', 'Value Index', 'Growth Index', 'Portfolio']
    assert [t.name for t in wrds.data] == ['IWM (ETF, total return)', 'Value: IWN (ETF, total return)',
                                           'Growth: IWO (ETF, total return)', 'Portfolio']
    assert [_style(t) for t in wrds.data] == [_style(t) for t in legacy.data]
    assert _layout(wrds) == _layout(legacy)
    assert list(wrds.data[-1].x) == results['years']
    axis = wrds.layout.xaxis
    assert (axis.type, axis.tickformat, axis.dtick, axis.hoverformat) == ('date', '%Y', 'M12', '%Y-%m-%d')
    assert axis.title.font.size == legacy.layout.xaxis.title.font.size and axis.griddash == 'dash'


def test_cohort_chart_is_legacys_on_dates(backtest):
    _, results, _ = backtest
    values, n = results['portfolio_values'], len(results['years'])
    legacy = plot_top_bottom_percent(list(range(2000, 2000 + n)), 10, True, values, values, values,
                                     initial_investment=1000.0, baseline_portfolio_values=values,
                                     precomputed_top=values, precomputed_bot=values)
    names = _names(results)
    wrds = plot_top_bottom_percent(results['years'], 10, True, values, values, values,
                                   benchmark_label=names['benchmark'], initial_investment=1000.0,
                                   baseline_portfolio_values=values, precomputed_top=values, precomputed_bot=values,
                                   growth_label=names['growth'], value_label=names['value'], dates=True)
    assert [t.name for t in legacy.data][:3] == ['Russell 2000', 'Growth Index', 'Value Index']
    assert [t.name for t in wrds.data] == [names['benchmark'], names['growth'], names['value'], 'Bottom 10%',
                                           'Top 10%', 'Active Strategy']
    assert [_style(t) for t in wrds.data] == [_style(t) for t in legacy.data]
    assert _layout(wrds) == _layout(legacy) and wrds.layout.xaxis.type == 'date'


def test_period_returns_chart_has_legacys_style(backtest):
    _, results, _ = backtest
    legacy = plot_portfolio_growth([2000, 2001], [1, 2], [1, 2], None, None, ['ROE'])
    fig = plot_wrds_period_returns(results['period_end_dates'], results['yearly_returns'],
                                   results['benchmark_returns'], 'IWM (ETF, total return)', 'Quarterly')
    assert fig.data[0].type == 'bar' and fig.data[0].marker.color == legacy.data[-1].line.color
    assert list(fig.data[0].x) == results['period_end_dates']
    benchmark = {k: v for k, v in _style(fig.data[1]).items() if k != 'hovertemplate'}
    assert benchmark == {k: v for k, v in _style(legacy.data[0]).items() if k != 'hovertemplate'}
    assert fig.layout.title.text == '<b>Quarterly Returns</b>' and fig.layout.yaxis.ticksuffix == '%'
    layout, legacy_layout = _layout(fig), _layout(legacy)
    assert {k: layout[k] for k in ('legend', 'height', 'hovermode', 'title_font')} == \
        {k: legacy_layout[k] for k in ('legend', 'height', 'hovermode', 'title_font')}
    assert fig.layout.xaxis.tickformat == '%Y' and fig.layout.xaxis.hoverformat == '%Y-%m-%d'


# ---------------------------------------------------------------------------
# Every view in the app, against Legacy's Results tab as main rendered it
# ---------------------------------------------------------------------------

@pytest.fixture
def s3():
    """Fake S3 with WRDS data that has a missing risk-free month and a month without returns."""
    frames = synthetic_frames()
    frames.market.loc[frames.market['month_end'] == month_ends('2011-02', '2011-02')[0], 'rf'] = np.nan
    frames.returns.loc[frames.returns['month_end'] == month_ends('2011-05', '2011-05')[0], 'ret_m'] = np.nan
    s3 = RangeS3()
    s3.objects['factor-lake-data/' + UNIVERSE_FILE] = (_parquet(_raw_universe()), {})
    s3.objects['factor-lake-data/' + DELISTING_FILE] = (_parquet(_delisting()), {})
    put_wrds_files(s3, frames)
    return s3


def _plot_x_types(at):
    return [json.loads(c.proto.spec)['layout']['xaxis']['type'] for c in at.main.get('plotly_chart')]


def _bold_labels(captions):
    """The bold labels of the captions ('Factors', 'Period', ...), in order, without the holdings' period."""
    return [re.sub(r'^Portfolio return \(.*\)$', 'Portfolio return', label)
            for c in captions for label in re.findall(r'\*\*(.+?):\*\*', c)]


def _is_subsequence(part, whole):
    it = iter(whole)
    return all(x in it for x in part)


def _wrds_names(text):
    """Legacy's names as a quarterly WRDS run shows them."""
    for legacy, wrds in [('Year-by-Year', 'Period-by-Period'), ('Yearly', 'Quarterly'), ('R2000', 'IWM')]:
        text = text.replace(legacy, wrds)
    return text


def _header(csv_text):
    return csv_text.splitlines()[0].split(',')


def test_a_wrds_run_looks_like_legacys_results(new_session, legacy):
    at = new_session()
    run = _run_wrds(at, 'w_roe', 'w_mom_12_1', 'w_mcap')
    at.button(key=f"cohort_btn_{run['id']}").click().run()
    _assert_clean(at)
    results, shown = run['results'], main_area(at)
    assert run['label'] == 'Run 1: WRDS, Quarterly'

    # Sections, titles and metric tiles: Legacy's, with the period and the ETF for its names
    assert shown['header'] == [_wrds_names(h) for h in legacy['header']]
    assert shown['subheader'] == [_wrds_names(h) for h in legacy['subheader']]
    assert [m[0] for m in shown['metric']] == [_wrds_names(m[0]) for m in legacy['metric']]
    tiles = {m[0]: m[1] for m in shown['metric']}
    assert tiles['CAGR'] == f"{results['cagr']:.2%}" and tiles['Beta vs IWM'] == f"{results['portfolio_beta']:.3f}"
    assert tiles['Quarterly Win Rate (vs IWM)'] == f"{results['win_rate']:.2%}"
    assert next(m for m in at.main.metric if m.label == 'Cumulative Outperformance').help == \
        'Against the same initial AUM invested in IWM (ETF, total return).'

    # Captions: Legacy's labels in Legacy's order, plus the data source, the ID and the WRDS counts
    labels = _bold_labels(shown['caption'])
    assert _is_subsequence(_bold_labels(legacy['caption']), labels)
    assert set(labels) - set(_bold_labels(legacy['caption'])) == {
        'Data Source', 'Rebalancing', 'Computed', 'Analysis ID', 'Stock-months without a return (counted as 0%)',
        'Risk-free months filled from the 3-month T-bill / 12'}
    assert "**Delisting Strategy:** CRSP delisting returns included, then the risk-free rate until the next " \
           "rebalance" in shown['caption']
    assert "**Risk-free months filled from the 3-month T-bill / 12:** 2011-02" in shown['caption']
    assert "**Factors:** 12-1 Mo Momentum (Higher is better), ROE (Higher is better), Market Cap (Lower is better)" \
        in shown['caption']

    # Tables: holdings, then Legacy's Year-by-Year and Win/Loss headers with the period and the ETFs
    holdings, periods, win_loss = (_header(t) for t in shown['dataframe'])
    legacy_holdings = [c for c in _header(legacy['dataframe'][0])
                       if c in LEGACY_HOLDINGS_BEFORE + LEGACY_HOLDINGS_AFTER]
    assert _is_subsequence(legacy_holdings, holdings) and holdings[-2:] == ['PERMNO', 'Months Without Return']
    assert periods == period_by_period_columns(results)
    assert win_loss == win_loss_columns(results)
    assert len(_header(legacy['dataframe'][1])) == len(periods)
    assert len(_header(legacy['dataframe'][2])) == len(win_loss)
    period_rows = at.main.dataframe[1].value
    assert list(period_rows['Period'])[0] == f"{results['rebalance_dates'][0]} to {results['period_end_dates'][0]}"
    assert list(at.main.dataframe[2].value['Period']) == list(period_rows['Period'])
    assert shown['table'][0].splitlines()[0] == legacy['table'][0].splitlines()[0]          # cohort stats
    assert [r.split(',')[0] for r in shown['table'][0].splitlines()] == \
        [r.split(',')[0] for r in legacy['table'][0].splitlines()]

    # Widgets: Legacy's, with a date picker and the ranked stocks added
    assert [s[0] for s in shown['selectbox']] == ['Rebalance date']
    assert shown['selectbox'][0][2] == results['rebalance_dates'][-1]                       # newest, as Legacy
    assert [s[0] for s in shown['slider']] == [s[0] for s in legacy['slider']]
    assert _is_subsequence(legacy['button'], shown['button'])
    assert set(shown['button']) - set(legacy['button']) == {'Rank every stock on this date'}
    def uncounted(labels):          # "This session's analyses (2)": main's snapshot had two runs
        return [re.sub(r' \(\d+\)$', '', label) for label in labels]
    assert _is_subsequence(uncounted(legacy['expander']), uncounted(shown['expander']))

    # Charts: Legacy's growth chart, the period returns, Legacy's cohort chart, on dates
    growth, period_returns, cohort = shown['plotly']
    legacy_growth, legacy_cohort = legacy['plotly']
    assert [_style(t) for t in growth['data']] == [_style(t) for t in legacy_growth['data']]
    assert _layout(growth) == _layout(legacy_growth)
    assert [_style(t) for t in cohort['data']] == [_style(t) for t in legacy_cohort['data']]
    assert _layout(cohort) == _layout(legacy_cohort)
    assert [t['name'] for t in cohort['data']][:3] == ['IWM (ETF, total return)', 'Growth: IWO (ETF, total return)',
                                                      'Value: IWN (ETF, total return)']
    assert period_returns['data'][0]['type'] == 'bar'
    assert _plot_x_types(at) == ['date', 'date', 'date'] and cohort['data'][-1]['x'] == results['years']

    # The calendar-year view: Legacy's Year-by-Year table by calendar year
    at.button_group(key=f"wrds_period_view_{run['id']}").set_value('By calendar year').run()
    years = at.main.dataframe[1].value
    assert list(years.columns) == period_by_period_columns(results, 'Year', 'YoY Return')
    assert list(years['Year']) == [2010, 2011, 2012]


def test_holdings_ranked_stocks_and_csvs_of_another_date(new_session):
    at = new_session()
    run = _run_wrds(at, 'w_roe', 'w_mom_12_1', 'w_mcap')
    results = run['results']
    holdings = at.main.dataframe[0].value
    assert {'12-1 Mo Momentum %', 'ROE %', 'Market Cap ($M)', 'Weight %', 'Period Return %'} <= set(holdings.columns)
    assert holdings['Weight %'].sum() == pytest.approx(100.0)
    newest = results['rebalance_dates'][-1]
    caption = (f"**Method:** Mixing | **Holdings:** {len(holdings)} | **Portfolio return ({newest} to "
               f"{results['period_end_dates'][-1]}):** {results['yearly_returns'][-1]:.2f}%")
    assert caption in [c.value for c in at.caption]
    assert [d.proto.label for d in at.main.get('download_button')] == ['Download CSV']

    # Another date: only that date's row group is read
    other = results['rebalance_dates'][3]
    at.selectbox(key=f"ranked_year_{run['id']}").set_value(other).run()
    _assert_clean(at)
    assert any(c.value.startswith("**Method:** Mixing | **Holdings:** ") and f"({other} to" in c.value
               for c in at.caption)

    # Ranked stocks on that date, with the same names and its own CSV
    at.button(key=f"wrds_rank_btn_{run['id']}").click().run()
    _assert_clean(at)
    ranked = at.main.dataframe[1].value
    assert list(ranked.columns[:6]) == ['Rank', 'Ticker', 'Name', 'Sector', 'Held', 'Score']
    assert {'ROE %', 'ROE Z-Score', 'Rank by ROE', 'Market Cap ($M)', 'PERMNO'} <= set(ranked.columns)
    assert len(at.main.get('download_button')) == 2


def test_a_reopened_wrds_id_shows_every_view_in_a_new_session(new_session):
    mine = new_session()
    run = _run_wrds(mine, 'w_roe', 'w_btp')
    theirs = new_session()
    _click_main_tab(theirs, 'Results')
    theirs.text_input(key='open_analysis_id').set_value(run['compute']['hash']).run()
    theirs.button(key='open_analysis_btn').click().run()
    opened = theirs.session_state['saved_runs'][0]
    _open_tab(theirs, opened['label'])
    _assert_clean(theirs)
    assert opened['label'] == 'Run 1: WRDS, Quarterly (opened)'
    assert main_area(theirs)['subheader'] == main_area(mine)['subheader']
    assert main_area(theirs)['header'][-2:] == main_area(mine)['header'][-2:]
    assert _plot_x_types(theirs) == ['date', 'date']
    assert theirs.main.dataframe[0].value.equals(mine.main.dataframe[0].value)       # the same holdings
    assert any(c.value.startswith('**Computed:** Opened from a saved analysis') for c in theirs.caption)


def test_the_data_viewer_says_it_shows_legacy_data_in_wrds_mode(new_session):
    at = new_session()
    at.sidebar.radio(key='data_mode_input').set_value('WRDS').run()
    _click_main_tab(at, 'Data Viewer')
    assert any('The Data Viewer shows the Legacy data, not WRDS' in c.value for c in at.main.caption)
    at.sidebar.radio(key='data_mode_input').set_value('Legacy').run()
    assert not any('The Data Viewer shows the Legacy data' in c.value for c in at.main.caption)


# ---------------------------------------------------------------------------
# Legacy's Results tab is exactly as on main
# ---------------------------------------------------------------------------

def test_legacy_results_tab_renders_exactly_as_on_main():
    expected = json.loads((FIXTURES / 'legacy_results_render_main.json').read_text(encoding='utf-8'))
    actual = json.loads(json.dumps(legacy_results_snapshot(), sort_keys=True))
    # Text, widgets and table cells must be equal; numbers to the Lambda check's relative 1e-9
    # (main's snapshot was rendered on Windows, CI renders on Linux)
    difference, where = max_difference(comparable(actual), comparable(expected))
    assert difference <= TOLERANCE, f"{where} differs from main's rendering (relative difference {difference:.2e})"
