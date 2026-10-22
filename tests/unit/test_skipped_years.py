"""
Tests for years the backtest skips because they have no data.

rebalance_portfolio skips a year with no rows. The remaining returns must keep
their real years: in results['years'], the year-by-year comparisons, the
benchmark (and risk-free) alignment, and holdings_by_year.
"""
import numpy as np
import pytest
import streamlit as st

from src.backtest_engine import rebalance_portfolio
from src.benchmarks import get_benchmark_list
from tests.unit.test_mixing_regression import delisting_universe

FACTORS = ['ROE_using_9-30_Data', 'ROA']
DIRECTIONS = {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom'}


@pytest.fixture(autouse=True)
def _mock_session_state():
    """Ensure st.session_state has the keys filter_universe reads."""
    st.session_state['exclude_fossil_fuels'] = False
    st.session_state['selected_sectors'] = []


@pytest.fixture(scope='module')
def universe():
    return delisting_universe()          # 2010-2016


def _run(data, method='mixing', start=2010, end=2016):
    return rebalance_portfolio(data, FACTORS, DIRECTIONS, start, end, initial_aum=1000.0,
                               top_pct=20.0, factor_combination=method)


@pytest.mark.parametrize('method', ['mixing', 'integrating'])
def test_missing_middle_year_is_left_out_of_the_labels(universe, method):
    res = _run(universe[universe['Year'] != 2013], method)
    assert res['years'] == [2010, 2011, 2012, 2014, 2015, 2016]
    assert list(res['holdings_by_year']) == res['years'][:-1]
    assert [c['year'] for c in res['yearly_comparisons']] == res['years'][:-1]
    assert len(res['portfolio_values']) == len(res['years'])


@pytest.mark.parametrize('method', ['mixing', 'integrating'])
def test_each_return_keeps_its_real_year(universe, method):
    # Returns do not depend on AUM, so each year's return equals the gap-free run's
    gap = _run(universe[universe['Year'] != 2013], method)
    full = _run(universe, method)
    by_year = dict(zip(full['years'], full['yearly_returns']))
    assert gap['yearly_returns'] == pytest.approx([by_year[y] for y in gap['years'][:-1]], abs=1e-12)


def test_benchmarks_line_up_with_the_real_years(universe):
    res = _run(universe[universe['Year'] != 2013])
    held = res['years'][:-1]
    # The return from rebalance year y is compared with each index's year y + 1
    for key, index in [('benchmark_returns', 1), ('growth_benchmark_returns', 2), ('value_benchmark_returns', 3)]:
        assert res[key] == pytest.approx([get_benchmark_list(index, y + 1, y + 2)[0] for y in held]), key
    for c, y in zip(res['yearly_comparisons'], held):
        assert c['b_ret'] == pytest.approx(get_benchmark_list(1, y + 1, y + 2)[0])
    # Sharpe uses the risk-free rate of the same rebalance years
    excess = np.array(res['yearly_returns']) / 100 - np.array([get_benchmark_list(4, y, y + 1)[0] for y in held])
    assert res['sharpe_portfolio'] == pytest.approx(np.mean(excess) / np.std(excess, ddof=1))


def test_missing_first_years_start_the_labels_at_the_first_year_with_data(universe):
    res = _run(universe[universe['Year'] >= 2012])
    assert res['years'] == [2012, 2013, 2014, 2015, 2016]
    assert list(res['holdings_by_year']) == res['years'][:-1]
    assert res['benchmark_returns'] == pytest.approx([get_benchmark_list(1, y + 1, y + 2)[0] for y in range(2012, 2016)])


def test_no_gaps_labels_and_benchmarks_unchanged(universe):
    res = _run(universe)
    assert res['years'] == list(range(2010, 2017))
    assert list(res['holdings_by_year']) == res['years'][:-1]
    assert res['benchmark_returns'] == pytest.approx(get_benchmark_list(1, 2011, 2017))
    assert res['value_benchmark_returns'] == pytest.approx(get_benchmark_list(3, 2011, 2017))


def test_no_years_with_data_still_labels_the_start_year(universe):
    res = _run(universe, start=2030, end=2032)
    assert res['years'] == [2030]
    assert res['yearly_returns'] == [] and res['holdings_by_year'] == {}
