"""
Tests for the factor combination switch and the per-year Holdings and Ranks table.

Engine tests cover Mixing / Integrating in src/backtest_engine.py and check the
holdings rows against the positions the backtest valued. Helper tests cover the
pure display/CSV/default-year functions in app/holdings_table.py. AppTest tests
drive the real app with the synthetic universe from test_saved_runs.
"""
import io
import math

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.backtest_engine import (
    _calculate_annual_return,
    _dedupe_year_slice_by_ticker,
    calculate_holdings,
    calculate_integrated_holdings,
    composite_scores,
    rebalance_portfolio,
    run_cohort_comparison,
)
from src.benchmarks import get_benchmark_list
from src.factor_utils import normalize_series
from app.holdings_table import (
    build_holdings_display_table,
    default_holdings_year,
    holdings_number_formats,
    holdings_period_return,
    holdings_table_to_csv,
)
from tests.unit.test_mixing_regression import delisting_universe
from tests.unit.test_saved_runs import RUN, _add_factor, _assert_clean, _captions, _click, _open_tab, _run_analysis, at  # noqa: F401

FACTORS = ['ROE_using_9-30_Data', 'ROA']
DIRECTIONS = {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom'}
LABELS = {'ROE_using_9-30_Data': 'ROE using 9/30 Data', 'ROA': 'ROA %'}
METHODS = ('mixing', 'integrating')
STRATEGIES = ('zero_return', 'hold_cash', 'reinvest')
START, END = 2010, 2016


@pytest.fixture(autouse=True)
def _mock_session_state():
    """Ensure st.session_state has the keys filter_universe reads."""
    st.session_state['exclude_fossil_fuels'] = False
    st.session_state['selected_sectors'] = []


@pytest.fixture(scope='module')
def universe():
    return delisting_universe()


def _run(data, method, factors=FACTORS, directions=DIRECTIONS, top_pct=20.0,
         use_mcap=False, strategy='zero_return', start=START, end=END):
    return rebalance_portfolio(
        data, factors, directions, start, end, initial_aum=1000.0, top_pct=top_pct,
        use_market_cap_weight=use_mcap, delisting_strategy=strategy,
        factor_combination=method, factor_labels=LABELS
    )


def _year_slice(data, year):
    return _dedupe_year_slice_by_ticker(data[data['Year'] == year].set_index('Ticker'))


def _expected_composite(df_year, factors=FACTORS, directions=DIRECTIONS):
    """Composite score computed independently of composite_scores."""
    scores = pd.concat([
        normalize_series(df_year[f], higher_is_better=directions[f] == 'top') for f in factors
    ], axis=1)
    return scores.dropna().mean(axis=1).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Engine: Integrating selection
# ---------------------------------------------------------------------------

class TestIntegrating:
    def test_composite_drops_stocks_missing_any_factor(self, universe):
        df_year = _year_slice(universe, 2012)
        missing = df_year.index[df_year[FACTORS].isna().any(axis=1)]
        assert len(missing) > 0                       # the fixture really has gaps
        composite = composite_scores(df_year, FACTORS, DIRECTIONS)
        assert not set(missing) & set(composite.index)
        assert len(composite) == len(df_year) - len(missing)

    def test_composite_is_mean_of_z_scores_best_first(self, universe):
        df_year = _year_slice(universe, 2012)
        expected = _expected_composite(df_year)
        composite = composite_scores(df_year, FACTORS, DIRECTIONS)
        assert composite.index.tolist() == expected.index.tolist()
        assert composite.values.tolist() == pytest.approx(expected.values.tolist())

    def test_buys_top_pct_by_composite_once_each(self, universe):
        df_year = _year_slice(universe, 2012)
        expected = _expected_composite(df_year)
        n = max(1, math.floor(len(expected) * 0.25))
        portfolio = calculate_integrated_holdings(df_year, FACTORS, DIRECTIONS, 1000.0, top_pct=25.0)
        bought = [inv['ticker'] for inv in portfolio.investments]
        priced = [t for t in expected.index[:n] if pd.notna(df_year.loc[t, 'Ending_Price'])
                  and df_year.loc[t, 'Ending_Price'] > 0]
        assert bought == priced
        assert len(bought) == len(set(bought))

    def test_equal_weight_across_whole_list(self, universe):
        df_year = _year_slice(universe, 2013)
        portfolio = calculate_integrated_holdings(df_year, FACTORS, DIRECTIONS, 1000.0, top_pct=25.0)
        values = [inv['number_of_shares'] * df_year.loc[inv['ticker'], 'Ending_Price'] for inv in portfolio.investments]
        assert values == pytest.approx([1000.0 / len(portfolio.scores)] * len(values))

    def test_holdings_rows_are_the_composite_ranking(self, universe):
        res = _run(universe, 'integrating')
        for year, rows in res['holdings_by_year'].items():
            expected = _expected_composite(_year_slice(universe, year))
            assert [r['Ticker'] for r in rows] == [t for t in expected.index if t in {r['Ticker'] for r in rows}]
            for r in rows:
                assert r['Rank'] == expected.index.get_loc(r['Ticker']) + 1
                assert r['Score'] == pytest.approx(expected[r['Ticker']])
            assert 'Picked By' not in rows[0]


# ---------------------------------------------------------------------------
# Engine: one factor gives the same portfolio either way
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('use_mcap', [False, True])
@pytest.mark.parametrize('strategy', STRATEGIES)
def test_one_factor_mixing_equals_integrating(universe, strategy, use_mcap):
    kwargs = dict(factors=['ROA'], directions={'ROA': 'top'}, use_mcap=use_mcap, strategy=strategy)
    mixing, integrating = _run(universe, 'mixing', **kwargs), _run(universe, 'integrating', **kwargs)
    assert mixing['portfolio_values'] == integrating['portfolio_values']
    assert mixing['yearly_returns'] == integrating['yearly_returns']
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'Picked By'} for r in rows]
    for year in mixing['holdings_by_year']:
        assert strip(mixing['holdings_by_year'][year]) == strip(integrating['holdings_by_year'][year])


def test_two_factors_mixing_and_integrating_differ(universe):
    mixing, integrating = _run(universe, 'mixing'), _run(universe, 'integrating')
    assert mixing['portfolio_values'] != integrating['portfolio_values']
    assert mixing['holdings_by_year'][2012] != integrating['holdings_by_year'][2012]


def test_unknown_method_is_rejected(universe):
    with pytest.raises(ValueError):
        _run(universe, 'blending')


# ---------------------------------------------------------------------------
# Engine: Mixing holdings
# ---------------------------------------------------------------------------

def test_stock_in_two_sleeves_is_one_row_with_both_factors():
    # TOP leads both factors; with top_pct=20 each sleeve buys 2 of 10 stocks
    data = pd.DataFrame({
        'Ticker': [f'S{i}' for i in range(9)] + ['TOP'],
        'Year': 2010,
        'Ending_Price': [10.0 + i for i in range(9)] + [50.0],
        'ROE_using_9-30_Data': [0.01 * i for i in range(9)] + [1.0],
        'ROA': [0.09 - 0.01 * i for i in range(9)] + [1.0],
        'Next-Years_Return': [5.0] * 9 + [20.0],
    })
    res = rebalance_portfolio(data, FACTORS, {f: 'top' for f in FACTORS}, 2010, 2011, 1000.0,
                              top_pct=20.0, factor_labels=LABELS)
    rows = res['holdings_by_year'][2010]
    top = [r for r in rows if r['Ticker'] == 'TOP']
    assert len(top) == 1
    assert top[0]['Picked By'] == 'ROE using 9/30 Data, ROA %'
    assert top[0]['Rank'] == 1
    # Each sleeve gets $500 split over 2 stocks: TOP holds $250 in each
    assert top[0]['Dollar Value'] == pytest.approx(500.0)
    assert top[0]['Weight %'] == pytest.approx(50.0)
    assert top[0]['Shares'] == pytest.approx(10.0)
    others = [r for r in rows if r['Ticker'] != 'TOP']
    assert {r['Picked By'] for r in others} == {'ROE using 9/30 Data', 'ROA %'}
    assert [r['Weight %'] for r in others] == pytest.approx([25.0, 25.0])
    assert sum(r['Contribution %'] for r in rows) == pytest.approx(res['yearly_returns'][0])


def test_mixing_rank_and_score_from_best_sleeve(universe):
    res = _run(universe, 'mixing')
    rows = res['holdings_by_year'][2012]
    df_year = _year_slice(universe, 2012)
    ranked = {
        f: normalize_series(df_year[f], higher_is_better=DIRECTIONS[f] == 'top').dropna().sort_values(ascending=False)
        for f in FACTORS
    }
    for r in rows:
        picks = [f for f in FACTORS if LABELS[f] in r['Picked By'].split(', ')]
        best = min(picks, key=lambda f: ranked[f].index.get_loc(r['Ticker']))
        assert r['Rank'] == ranked[best].index.get_loc(r['Ticker']) + 1
        assert r['Score'] == pytest.approx(ranked[best][r['Ticker']])
    assert [r['Rank'] for r in rows] == sorted(r['Rank'] for r in rows)


# ---------------------------------------------------------------------------
# Engine: the table is exactly what the backtest valued
# ---------------------------------------------------------------------------

def _rebuild_year(data, year, method, aum, top_pct, use_mcap):
    """The year's portfolios, rebuilt with the engine's selection functions."""
    df_year = _year_slice(data, year)
    if method == 'integrating':
        return df_year, [calculate_integrated_holdings(df_year, FACTORS, DIRECTIONS, aum, top_pct, use_mcap)]
    return df_year, [
        calculate_holdings(df_year, f, aum / len(FACTORS), DIRECTIONS[f] == 'top', top_pct, use_mcap)
        for f in FACTORS
    ]


@pytest.mark.parametrize('use_mcap', [False, True])
@pytest.mark.parametrize('strategy', STRATEGIES)
@pytest.mark.parametrize('method', METHODS)
def test_holdings_match_backtest_positions(universe, method, strategy, use_mcap):
    res = _run(universe, method, use_mcap=use_mcap, strategy=strategy)
    holdings = res['holdings_by_year']
    assert list(holdings) == res['years'][:-1]
    rf = dict(zip(range(START, END), get_benchmark_list(4, START, END)))
    delisted_seen = 0

    for i, (year, rows) in enumerate(holdings.items()):
        df_year, portfolios = _rebuild_year(universe, year, method, res['portfolio_values'][i], 20.0, use_mcap)
        # The rebuilt portfolios give the backtest's return for the year...
        ret, _ = _calculate_annual_return(portfolios, df_year, strategy,
                                          rf[year] if strategy == 'hold_cash' else 0.0, year)
        assert ret * 100 == pytest.approx(res['yearly_returns'][i], abs=1e-12)

        # ...and the table holds exactly their valued positions, one row per stock
        shares = {}
        for p in portfolios:
            for inv in p.investments:
                if df_year.loc[inv['ticker'], 'Ending_Price'] > 0:
                    shares[inv['ticker']] = shares.get(inv['ticker'], 0.0) + inv['number_of_shares']
        assert sorted(r['Ticker'] for r in rows) == sorted(shares)
        for r in rows:
            assert r['Shares'] == pytest.approx(shares[r['Ticker']])
            assert r['Buy Price'] == df_year.loc[r['Ticker'], 'Ending_Price']
            assert r['Dollar Value'] == pytest.approx(r['Shares'] * r['Buy Price'])
            assert r['Delisted'] == bool(pd.isna(df_year.loc[r['Ticker'], 'Next-Years_Return']))
            if not r['Delisted']:
                assert r['Period Return %'] == pytest.approx(df_year.loc[r['Ticker'], 'Next-Years_Return'])
            assert r['Contribution %'] == pytest.approx(r['Weight %'] * r['Period Return %'] / 100)
            assert r[LABELS['ROA']] == df_year.loc[r['Ticker'], 'ROA'] or (
                np.isnan(r[LABELS['ROA']]) and np.isnan(df_year.loc[r['Ticker'], 'ROA']))
            assert r['Name'] == df_year.loc[r['Ticker'], 'Security_Name']
            assert r['Sector'] == df_year.loc[r['Ticker'], 'Scotts_Sector_5']
        delisted_seen += sum(r['Delisted'] for r in rows)

        assert sum(r['Weight %'] for r in rows) == pytest.approx(100.0, abs=1e-9)
        assert sum(r['Contribution %'] for r in rows) == pytest.approx(res['yearly_returns'][i], abs=1e-9)

    assert delisted_seen > 0


def test_delisted_row_return_includes_what_its_capital_earned_after_delisting():
    """Worked example for each strategy; contributions still add up to the year's return."""
    data = pd.DataFrame({
        'Ticker': ['LIVE', 'DEAD'],
        'Year': 2010,
        'Ending_Price': [100.0, 100.0],
        'ROA': [1.0, 2.0],
        'Next-Years_Return': [20.0, np.nan],
        'Delist_Date': [pd.NaT, pd.Timestamp('2011-07-01')],
        'Delist_Price': [np.nan, 150.0],
    })
    frac = (pd.Timestamp('2011-12-31') - pd.Timestamp('2011-07-01')).days / 365.0
    rf_2010 = get_benchmark_list(4, 2010, 2011)[0]
    # Pre-delist return 50%; the $150 at delisting then earns post_rate for frac of the year
    for strategy, post_rate in [('zero_return', 0.0), ('hold_cash', rf_2010), ('reinvest', 0.20)]:
        res = rebalance_portfolio(data, ['ROA'], {'ROA': 'top'}, 2010, 2011, 200.0, top_pct=100.0,
                                  delisting_strategy=strategy, factor_combination='integrating')
        dead = next(r for r in res['holdings_by_year'][2010] if r['Ticker'] == 'DEAD')
        assert dead['Delisted']
        assert dead['Period Return %'] == pytest.approx(100 * (0.5 + post_rate * frac * 1.5))
        assert sum(r['Contribution %'] for r in res['holdings_by_year'][2010]) == pytest.approx(res['yearly_returns'][0])


def test_holdings_years_follow_the_rebalance_years_kept(universe):
    gap = universe[universe['Year'] != 2013]          # no data for 2013: no return, no holdings
    res = _run(gap, 'integrating')
    assert list(res['holdings_by_year']) == [2010, 2011, 2012, 2014, 2015]
    assert len(res['holdings_by_year']) == len(res['yearly_returns'])
    # 2014's return is the 4th entry, although results['years'] labels it 2013
    assert holdings_period_return(res, 2014) == res['yearly_returns'][3]
    assert sum(r['Contribution %'] for r in res['holdings_by_year'][2014]) == pytest.approx(res['yearly_returns'][3])


# ---------------------------------------------------------------------------
# Engine: cohort comparison
# ---------------------------------------------------------------------------

def test_cohort_comparison_follows_the_switch(universe):
    settings = {'start_year': START, 'end_year': END, 'initial_aum': 1000.0, 'use_market_cap_weight': False}
    mix_top, mix_bot = run_cohort_comparison(universe, FACTORS, DIRECTIONS, 20, settings)
    int_top, int_bot = run_cohort_comparison(universe, FACTORS, DIRECTIONS, 20,
                                             dict(settings, factor_combination='integrating'))
    assert (mix_top, mix_bot) == run_cohort_comparison(universe, FACTORS, DIRECTIONS, 20,
                                                       dict(settings, factor_combination='mixing'))
    assert mix_top != int_top and mix_bot != int_bot
    assert int_top == _run(universe, 'integrating', top_pct=20)['portfolio_values']
    inverse = {f: 'bottom' if d == 'top' else 'top' for f, d in DIRECTIONS.items()}
    assert int_bot == _run(universe, 'integrating', directions=inverse, top_pct=20)['portfolio_values']


# ---------------------------------------------------------------------------
# Results tab helpers
# ---------------------------------------------------------------------------

def _rows(universe, method='mixing', year=2012):
    return _run(universe, method)['holdings_by_year'][year]


class TestHoldingsTableHelpers:
    def test_newest_year_is_default(self):
        assert default_holdings_year([2005, 2023, 2010]) == 2023

    def test_no_default_without_years(self):
        assert default_holdings_year([]) is None

    def test_display_table_columns_for_each_method(self, universe):
        mixing = build_holdings_display_table(_rows(universe, 'mixing'), list(LABELS.values()))
        assert mixing.columns.tolist() == [
            'Rank', 'Ticker', 'Name', 'Sector', 'Picked By', 'Score', 'ROE using 9/30 Data', 'ROA %',
            'Weight %', 'Shares', 'Buy Price', 'Dollar Value', 'Period Return %', 'Contribution %', 'Delisted',
        ]
        integrating = build_holdings_display_table(_rows(universe, 'integrating'), list(LABELS.values()))
        assert integrating.columns.tolist() == [c for c in mixing.columns if c != 'Picked By']

    def test_display_table_converts_without_rounding(self):
        rows = [{'Rank': 1, 'Ticker': 'A', 'Score': '1.234567', 'ROA %': '0.05123456',
                 'Weight %': '12.3456789', 'Period Return %': '12.3456', 'Delisted': 0}]
        df = build_holdings_display_table(rows, ['ROA %'])
        assert df.loc[0, 'Score'] == 1.234567
        assert df.loc[0, 'ROA %'] == 0.05123456
        assert df.loc[0, 'Weight %'] == 12.3456789
        assert df.loc[0, 'Period Return %'] == 12.3456
        assert df.loc[0, 'Delisted'] is np.False_

    def test_number_formats(self):
        assert holdings_number_formats(['ROA %']) == {
            'Rank': '%d', 'Score': '%.4f', 'ROA %': '%.4f', 'Weight %': '%.2f', 'Shares': '%.4f',
            'Buy Price': 'dollar', 'Dollar Value': 'dollar', 'Period Return %': '%.2f', 'Contribution %': '%.2f',
        }

    def test_period_return_is_the_years_backtest_return(self, universe):
        res = _run(universe, 'integrating')
        for i, year in enumerate(res['holdings_by_year']):
            assert holdings_period_return(res, year) == res['yearly_returns'][i]
        assert holdings_period_return(res, 1999) is None

    def test_csv_has_same_rows_and_columns_as_table(self, universe):
        for method in METHODS:
            df = build_holdings_display_table(_rows(universe, method), list(LABELS.values()))
            exported = pd.read_csv(io.BytesIO(holdings_table_to_csv(df)), float_precision='round_trip')
            pd.testing.assert_frame_equal(exported, df, check_exact=True)


# ---------------------------------------------------------------------------
# AppTest: the real Results tab
# ---------------------------------------------------------------------------

def _run_two_factor_backtest(at, method='Mixing'):
    _add_factor(at, 'ROE using 9/30 Data')
    _add_factor(at, 'ROA %')
    at.radio(key='factor_combination_input').set_value(method).run()
    _click(at, 'Load Market Data')
    _click(at, RUN)


@pytest.fixture
def served_files(monkeypatch):
    """Records every file AppTest stores for a download, by file id."""
    from streamlit.runtime.memory_media_file_storage import MemoryFile, MemoryMediaFileStorage
    files = {}
    original = MemoryMediaFileStorage.load_and_get_id

    def record(self, path_or_data, mimetype, kind, filename=None):
        file_id = original(self, path_or_data, mimetype, kind, filename)
        files[file_id] = MemoryFile(content=path_or_data, mimetype=mimetype, kind=kind, filename=filename)
        return file_id

    monkeypatch.setattr(MemoryMediaFileStorage, 'load_and_get_id', record)
    return files


def _downloaded_csv(at, served_files, key):
    """The exact file the Download button with this key hands the browser."""
    button = next(b for b in at.get('download_button') if b.proto.id.endswith(key))
    file_id = button.proto.url.rsplit('/', 1)[1].split('.')[0]
    return served_files[file_id]


def test_year_selector_defaults_to_newest_year(at):
    _run_two_factor_backtest(at)
    _assert_clean(at)
    selector = at.selectbox(key='ranked_year_1')
    results = at.session_state['saved_runs'][0]['results']
    years = sorted(results['holdings_by_year'])
    assert years == list(range(2002, 2024)) == results['years'][:-1]   # synthetic data 2002-2024
    assert selector.value == 2023


def test_table_is_the_backtest_holdings_and_updates_per_year(at):
    _run_two_factor_backtest(at)
    first = at.dataframe[0].value
    assert 'Picked By' in first.columns
    assert first.columns.tolist()[5:7] == ['ROE using 9/30 Data', 'ROA %']

    at.selectbox(key='ranked_year_1').set_value(2010).run()
    _assert_clean(at)
    shown = at.dataframe[0].value
    results = at.session_state['saved_runs'][0]['results']
    assert shown['Ticker'].tolist() == [r['Ticker'] for r in results['holdings_by_year'][2010]]
    assert shown['Ticker'].tolist() != first['Ticker'].tolist()
    assert shown['Contribution %'].sum() == pytest.approx(results['yearly_returns'][2010 - 2002])
    assert any(f"**Method:** Mixing | **Holdings:** {len(shown)}" in c for c in _captions(at))


def test_integrating_run_hides_picked_by_and_shows_the_method(at):
    _run_two_factor_backtest(at, 'Integrating')
    _assert_clean(at)
    run = at.session_state['saved_runs'][0]
    assert run['settings']['factor_combination'] == 'integrating'
    assert run['results']['factor_combination'] == 'integrating'
    table = at.dataframe[0].value
    assert 'Picked By' not in table.columns
    assert table['Ticker'].is_unique
    assert '**Factor Combination:** Integrating (top stocks by composite score)' in _captions(at)
    assert any('**Method:** Integrating' in c for c in _captions(at))


def test_mixing_and_integrating_runs_differ_and_cohort_follows(at):
    _run_two_factor_backtest(at, 'Mixing')               # Run 1
    at.button(key='cohort_btn_1').click().run()
    from tests.unit.test_saved_runs import _click_main_tab
    _click_main_tab(at, 'Analysis')
    at.radio(key='factor_combination_input').set_value('Integrating').run()
    _click(at, RUN)                                      # Run 2
    at.button(key='cohort_btn_2').click().run()
    _assert_clean(at)

    run_2, run_1 = at.session_state['saved_runs']
    assert run_1['settings']['factor_combination'] == 'mixing'
    assert run_2['settings']['factor_combination'] == 'integrating'
    assert run_1['results']['holdings_by_year'][2010] != run_2['results']['holdings_by_year'][2010]
    assert run_1['cohort']['top'] != run_2['cohort']['top']
    assert run_2['cohort']['top'] == run_2['results']['portfolio_values']   # same top % (10)


def test_single_factor_run_is_the_same_for_both_methods(at):
    _add_factor(at, 'ROA %')
    _click(at, 'Load Market Data')
    _click(at, RUN)                                      # Run 1: Mixing
    from tests.unit.test_saved_runs import _click_main_tab
    _click_main_tab(at, 'Analysis')
    at.radio(key='factor_combination_input').set_value('Integrating').run()
    _click(at, RUN)                                      # Run 2: Integrating
    _assert_clean(at)
    run_2, run_1 = at.session_state['saved_runs']
    assert run_1['results']['portfolio_values'] == run_2['results']['portfolio_values']
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'Picked By'} for r in rows]
    for year, rows in run_1['results']['holdings_by_year'].items():
        assert strip(rows) == strip(run_2['results']['holdings_by_year'][year])


def test_downloaded_csv_matches_table_on_screen(at, served_files):
    _run_two_factor_backtest(at)
    at.selectbox(key='ranked_year_1').set_value(2010).run()
    file = _downloaded_csv(at, served_files, 'ranked_csv_1')
    assert file.filename == 'holdings_2010.csv'
    assert file.mimetype == 'text/csv'
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(file.content), float_precision='round_trip'),
                                  at.dataframe[0].value, check_exact=True)


def test_screen_rounds_but_downloaded_csv_is_unrounded(at, served_files):
    import json
    _run_two_factor_backtest(at)
    table = at.dataframe[0]
    formats = {col: cfg['type_config']['format']
               for col, cfg in json.loads(table.proto.columns).items() if 'type_config' in cfg}
    assert formats == holdings_number_formats(['ROE using 9/30 Data', 'ROA %'])

    exported = pd.read_csv(io.BytesIO(_downloaded_csv(at, served_files, 'ranked_csv_1').content), float_precision='round_trip')
    data = at.session_state['saved_runs'][0]['data']
    raw = data[data['Year'] == 2023].set_index('Ticker')
    assert exported['ROA %'].tolist() == raw.loc[exported['Ticker'], 'ROA'].tolist()
    assert any(round(v, 4) != v for v in exported['Score'])


def test_saved_runs_have_independent_year_selectors(at, served_files):
    _run_two_factor_backtest(at)          # Run 1
    _run_analysis(at)                     # Run 2, same factors (back in the Analysis tab)
    run_1 = 'Run 1: ROE using 9/30 Data +1'

    _open_tab(at, run_1)
    at.run()
    _open_tab(at, run_1)
    at.selectbox(key='ranked_year_1').set_value(2010).run()
    assert at.selectbox(key='ranked_year_1').value == 2010

    _open_tab(at, 'Run 2: ROE using 9/30 Data +1')
    at.run()
    _assert_clean(at)
    assert at.selectbox(key='ranked_year_2').value == 2023
    assert _downloaded_csv(at, served_files, 'ranked_csv_2').filename == 'holdings_2023.csv'

    # Run 1 keeps its chosen year after switching away and back
    _open_tab(at, run_1)
    at.run()
    assert at.selectbox(key='ranked_year_1').value == 2010
    assert _downloaded_csv(at, served_files, 'ranked_csv_1').filename == 'holdings_2010.csv'
