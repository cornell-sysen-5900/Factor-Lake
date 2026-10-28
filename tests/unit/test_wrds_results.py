"""
Tests for the Results tab of WRDS runs (app/components/wrds_results.py, app/wrds_tables.py,
Visualizations/wrds_plots.py): every view with dates, Legacy's units and the WRDS display
names; reopening a WRDS ID in a new session; and Legacy's Results tab unchanged from main.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.wrds_tables import (benchmark_label, calendar_year_table, drawdowns, factor_header,
                             frequency_caption, holdings_display_table, metrics_table, period_table,
                             ranked_display_table, result_notes, table_number_formats)
from src.backtest_service import BacktestService
from src.s3_client import S3DataManager
from src.wrds_engine import rank_stocks, run_wrds_backtest
from tests.unit.results_snapshot import legacy_results_snapshot, without_clock_times
from tests.unit.test_backtest_lambda import DELISTING_FILE, UNIVERSE_FILE, _delisting, _parquet, _raw_universe
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab, _open_tab
from tests.unit.test_wrds_saved import _run_wrds, manager, new_session  # noqa: F401 (fixtures)
from tests.unit.wrds_fixtures import RangeS3, month_ends, put_wrds_files, synthetic_frames
from Visualizations.wrds_plots import plot_wrds_cohort, plot_wrds_drawdown, plot_wrds_growth, plot_wrds_period_returns

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'
FACTORS = ['f_roe', 'mom_12_1', 'mkt_cap_musd', 'f_book_to_price']
LABELS = {'f_roe': 'ROE', 'mom_12_1': '12-1 Mo Momentum', 'mkt_cap_musd': 'Market Cap', 'f_book_to_price': 'Book/Price'}
DIRECTIONS = {f: 'top' for f in FACTORS}
BOUNDS = dict(first_month='2009-11', last_month='2013-02')


@pytest.fixture(scope='module')
def backtest():
    frames = synthetic_frames()
    data = frames.data()
    results, holdings = run_wrds_backtest(data, FACTORS, DIRECTIONS, 'quarterly', 2010, 2012, 1000.0,
                                          factor_labels=LABELS, **BOUNDS)
    return data, results, holdings


# ---------------------------------------------------------------------------
# Tables: Legacy's units and formats, WRDS display names
# ---------------------------------------------------------------------------

def test_factor_headers_show_their_unit():
    assert factor_header('f_roe', 'ROE') == 'ROE %'
    assert factor_header('mkt_cap_musd', 'Market Cap') == 'Market Cap ($M)'
    assert factor_header('f_book_to_price', 'Book/Price') == 'Book/Price'
    assert factor_header('beta_252d', 'Beta (1-Yr Daily)') == 'Beta (1-Yr Daily)'


def test_holdings_table_has_legacys_columns_and_units(backtest):
    _, _, holdings = backtest
    date = holdings['rebalance_date'].iloc[-1]
    raw = holdings[holdings['rebalance_date'] == date].reset_index(drop=True)
    table = holdings_display_table(raw, FACTORS, LABELS)
    assert list(table.columns) == [
        'Rank', 'Ticker', 'PERMNO', 'Name', 'Sector', 'Picked By', 'Score', 'ROE %', '12-1 Mo Momentum %',
        'Market Cap ($M)', 'Book/Price', 'Weight %', 'Shares', 'Buy Price', 'Dollar Value', 'Period Return %',
        'Contribution %', 'Delisted', 'Months Without Return']
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
def test_ranked_table_uses_the_same_names_and_units(backtest, combination):
    data, results, _ = backtest
    date = pd.Timestamp(results['rebalance_dates'][2])
    rows = rank_stocks(data, date, FACTORS, DIRECTIONS, combination)
    table = ranked_display_table(rows, FACTORS, LABELS, combination)
    expected = ['Rank', 'Ticker', 'PERMNO', 'Name', 'Sector', 'Held', 'Weight %', 'Score']
    for f in FACTORS:
        expected += [factor_header(f, LABELS[f]), f"{LABELS[f]} Z-Score"] + (
            [f"Rank by {LABELS[f]}"] if combination == 'mixing' else [])
    assert list(table.columns) == expected + ['Price']
    assert table['Weight %'].sum() == pytest.approx(100.0)
    np.testing.assert_allclose(table['ROE %'], rows['f_roe'] * 100)
    assert table_number_formats(table, FACTORS, LABELS)['ROE Z-Score'] == '%.4f'


def test_period_calendar_and_metric_tables(backtest):
    _, results, _ = backtest
    periods = period_table(results)
    assert list(periods.columns) == ['Period Start', 'Period End', 'Portfolio Value', 'Portfolio Return',
                                     'IWM Return', 'IWN (Value) Return', 'IWO (Growth) Return', 'Risk-Free',
                                     'Beat IWM', 'Beat IWN (Value)', 'Beat IWO (Growth)']
    assert list(periods['Period Start']) == results['rebalance_dates']
    assert list(periods['Period End']) == results['period_end_dates']
    assert periods['Portfolio Return'].iloc[0] == f"{results['yearly_returns'][0]:.2f}%"

    years = calendar_year_table(results)
    assert list(years['Year']) == [2010, 2011, 2012] and list(years['Periods']) == [4, 4, 4]
    first_year = np.prod([1 + r / 100 for r in results['yearly_returns'][:4]]) - 1
    assert years['Portfolio Return'].iloc[0] == f"{first_year * 100:.2f}%"

    metrics = metrics_table(results)
    assert list(metrics.columns) == ['Metric', 'Portfolio', 'IWM', 'IWN (Value)', 'IWO (Growth)']
    by_metric = metrics.set_index('Metric')
    assert by_metric.loc['Sharpe Ratio (annualized)', 'Portfolio'] == f"{results['sharpe_portfolio']:.4f}"
    assert by_metric.loc['CAGR', 'Portfolio'] == f"{results['cagr']:.2%}"
    assert by_metric.loc['Portfolio Beta vs', 'IWM'] == f"{results['portfolio_beta']:.3f}"
    assert by_metric.loc['Period Win Rate vs', 'IWN (Value)'] == f"{results['win_rate_value']:.2%}"
    assert frequency_caption(results).startswith('Quarterly returns (4 per year). CAGR uses calendar years '
                                                 '(12 periods = 3.0 years)')


def test_notes_only_for_counts_that_are_not_zero():
    quiet = {'total_delisted_positions': 0, 'missing_return_stock_months': 0, 'rf_filled_months': [],
             'empty_rebalance_dates': []}
    assert result_notes(quiet) == []
    notes = result_notes(dict(quiet, missing_return_stock_months=3, rf_filled_months=['2026-09'],
                              empty_rebalance_dates=['2011-03-31']))
    assert notes == ["Stock-months without a return (counted as 0%): 3.",
                     "Risk-free months filled from the 3-month T-bill / 12: 2026-09.",
                     "Rebalance dates with nothing to buy (the money earned the risk-free rate): 2011-03-31."]


# ---------------------------------------------------------------------------
# Charts on a date axis
# ---------------------------------------------------------------------------

def test_charts_use_dates_and_the_etf_labels(backtest):
    _, results, _ = backtest
    labels = {role: benchmark_label(results, role) for role in ('benchmark', 'value', 'growth')}
    assert labels == {'benchmark': 'IWM (ETF, total return)', 'value': 'Value: IWN (ETF, total return)',
                      'growth': 'Growth: IWO (ETF, total return)'}
    values = results['portfolio_values']
    growth = plot_wrds_growth(results['years'], values, {'benchmark': values, 'value': values, 'growth': values},
                              labels, ['ROE'])
    assert [t.name for t in growth.data] == [labels['benchmark'], labels['value'], labels['growth'], 'Portfolio']
    assert list(growth.data[-1].x) == results['years'] and growth.layout.xaxis.type == 'date'
    assert growth.layout.yaxis.tickprefix == '$'

    period = plot_wrds_period_returns(results['period_end_dates'], results['yearly_returns'],
                                      results['benchmark_returns'], labels['benchmark'], 'Quarterly')
    assert period.data[0].type == 'bar' and list(period.data[0].x) == results['period_end_dates']
    assert period.layout.title.text == '<b>Quarterly Returns</b>' and period.layout.yaxis.ticksuffix == '%'

    drawdown = plot_wrds_drawdown(results['years'], drawdowns(values), drawdowns(values), labels['benchmark'])
    assert max(drawdown.data[-1].y) == 0 and min(drawdown.data[-1].y) <= 0 and drawdown.layout.xaxis.type == 'date'
    # Drawdowns start at 0% and period returns fill the top: their legends sit above the plot, not on the data
    assert drawdown.layout.legend.orientation == 'h' and drawdown.layout.legend.yanchor == 'bottom'
    assert period.layout.legend.orientation == 'h'
    assert growth.layout.legend.yanchor == 'top' and growth.layout.legend.orientation is None   # Legacy's place

    cohort = plot_wrds_cohort(results['years'], 10, values, values, values, values, labels['benchmark'], 1000.0)
    assert [t.name for t in cohort.data] == [labels['benchmark'], 'Bottom 10%', 'Top 10%', 'Active Strategy']
    assert cohort.layout.xaxis.type == 'date'


def test_long_series_are_drawn_without_markers():
    dates = [d.strftime('%Y-%m-%d') for d in month_ends('2008-04', '2026-09')]
    fig = plot_wrds_growth(dates, list(range(len(dates))), {}, {}, None)
    assert fig.data[0].mode == 'lines'


# ---------------------------------------------------------------------------
# Every view in the app, and reopening an ID
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


def _subheaders(at):
    return [s.value for s in at.main.subheader]


def _plot_x_types(at):
    return [json.loads(c.proto.spec)['layout']['xaxis']['type'] for c in at.main.get('plotly_chart')]


def test_every_view_of_a_wrds_run(new_session):
    at = new_session()
    run = _run_wrds(at, 'w_roe', 'w_mom_12_1', 'w_mcap')
    results = run['results']
    assert run['label'] == 'Run 1: WRDS, Quarterly'
    assert [t.label for t in at.tabs if t.label.startswith('Run ')] == ['Run 1: WRDS, Quarterly']

    assert _subheaders(at) == ['Performance Summary', 'Holdings and Ranks', 'Portfolio Growth Over Time',
                               'Period-by-Period Performance', 'Top vs Bottom Cohort Analysis']
    captions = [c.value for c in at.caption]
    assert "**Factors:** 12-1 Mo Momentum (Higher is better), ROE (Higher is better), Market Cap (Lower is better)" \
        in captions
    assert any(c.startswith('Stock-months without a return (counted as 0%): ') for c in captions)
    assert "Risk-free months filled from the 3-month T-bill / 12: 2011-02." in captions
    assert not any(c.startswith('Rebalance dates with nothing to buy') for c in captions)   # zero: not shown

    # Holdings of the picked date, in Legacy's units, with display names and a CSV download
    holdings = at.main.dataframe[0].value
    assert list(holdings.columns[:6]) == ['Rank', 'Ticker', 'PERMNO', 'Name', 'Sector', 'Picked By']
    assert {'12-1 Mo Momentum %', 'ROE %', 'Market Cap ($M)', 'Weight %', 'Period Return %'} <= set(holdings.columns)
    assert holdings['Weight %'].sum() == pytest.approx(100.0)
    assert any(c.startswith(f"**Method:** Mixing | **Holdings:** {len(holdings)} | **Portfolio return "
                            f"({results['rebalance_dates'][-1]} to") for c in captions)
    downloads = at.main.get('download_button')
    assert [d.proto.label for d in downloads] == ['Download CSV']

    # Charts: growth, period returns, drawdown, all on dates
    assert _plot_x_types(at) == ['date', 'date', 'date']
    # Period table, then the calendar-year view
    periods = at.main.dataframe[1].value
    assert list(periods['Period End']) == results['period_end_dates']
    at.button_group(key=f"wrds_period_view_{run['id']}").set_value('By calendar year').run()
    assert list(at.main.dataframe[1].value['Year']) == [2010, 2011, 2012]
    # Statistics table, annualized by frequency
    stats = at.main.dataframe[-1].value
    assert list(stats.columns) == ['Metric', 'Portfolio', 'IWM', 'IWN (Value)', 'IWO (Growth)']
    assert any(c.startswith('Quarterly returns (4 per year).') for c in [c.value for c in at.caption])

    # Another date: only that date's row group is read
    other = results['rebalance_dates'][3]
    at.selectbox(key=f"wrds_date_{run['id']}").set_value(other).run()
    _assert_clean(at)
    assert any(c.value.startswith(f"**Method:** Mixing | **Holdings:** ") and f"({other} to" in c.value
               for c in at.caption)

    # Ranked stocks on that date, with the same names and its own CSV
    at.button(key=f"wrds_rank_btn_{run['id']}").click().run()
    _assert_clean(at)
    ranked = at.main.dataframe[1].value
    assert list(ranked.columns[:8]) == ['Rank', 'Ticker', 'PERMNO', 'Name', 'Sector', 'Held', 'Weight %', 'Score']
    assert {'ROE %', 'ROE Z-Score', 'Rank by ROE', 'Market Cap ($M)'} <= set(ranked.columns)
    assert len(at.main.get('download_button')) == 2

    # Cohort at the run's frequency on the run's dates
    at.button(key=f"wrds_cohort_btn_{run['id']}").click().run()
    _assert_clean(at)
    cohort_table = at.main.table[0].value
    assert list(cohort_table['Metric']) == ['Total Return (%)', 'Final Value ($)', 'CAGR (%)']
    cohort_spec = json.loads(at.main.get('plotly_chart')[-1].proto.spec)
    assert cohort_spec['layout']['xaxis']['type'] == 'date'
    assert cohort_spec['data'][-1]['x'] == results['years']


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
    assert _subheaders(theirs) == _subheaders(mine)
    assert _plot_x_types(theirs) == ['date', 'date', 'date']
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
    assert without_clock_times(actual) == without_clock_times(expected)
