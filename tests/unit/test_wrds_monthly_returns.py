"""
Tests for the WRDS engine's month-by-month returns (src/wrds_engine.py, step 6): the
buy-and-hold portfolio's return in each month of a holding period, after the rebalance's
trading cost in the period's first month. Compounded over a period, the months must
equal the period's return, for every frequency, combination and trading cost.
"""
import numpy as np
import pandas as pd
import pytest

from src.wrds_engine import portfolio_monthly_returns, run_periods, run_wrds_backtest
from tests.unit.test_wrds_engine import _quarter, _set_returns
from tests.unit.test_wrds_pinned import FIRST, LAST, pinned_frames
from tests.unit.wrds_fixtures import month_ends, synthetic_frames

FACTORS = ['f_roe', 'mom_12_1', 'f_book_to_price']
DIRECTIONS = {'f_roe': 'top', 'mom_12_1': 'bottom', 'f_book_to_price': 'top'}
RTOL = 1e-12


def _compounded_by_period(results):
    """Each period's months compounded (decimals), grouping the months by the period they fall in."""
    months = pd.to_datetime(results['monthly_dates'])
    ends = pd.to_datetime(results['period_end_dates'])
    period = np.searchsorted(ends.to_numpy(), months.to_numpy())       # month in (end[i-1], end[i]] -> i
    growth = pd.Series(1 + np.array(results['monthly_returns']) / 100).groupby(period).prod()
    return growth.to_numpy() - 1, months


@pytest.fixture(scope='module')
def frames():
    frames = synthetic_frames(n_stocks=40, seed=5)
    # One stock delists mid-quarter with -40% and a few months have no return
    rm = frames.returns
    delist = (rm['permno'] == 10003) & (rm['month_end'] == month_ends('2011-05', '2011-05')[0])
    rm.loc[delist, ['ret_m', 'delisted_in_month']] = [-0.4, True]
    later = (rm['permno'] == 10003) & (rm['month_end'] > month_ends('2011-05', '2011-05')[0])
    gaps = rm.index[(rm['permno'] == 10007) & rm['month_end'].dt.month.isin([2, 8])]
    frames.returns = rm.drop(rm.index[later].union(gaps)).reset_index(drop=True)
    return frames


@pytest.mark.parametrize('frequency', ['monthly', 'quarterly', 'semiannual'])
@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
@pytest.mark.parametrize('bps', [0.0, 40.0])
def test_months_compound_to_the_period_returns(frames, frequency, combination, bps):
    results, _ = run_wrds_backtest(frames.data(), FACTORS, DIRECTIONS, frequency, 2010, 2012, 1000.0,
                                   factor_combination=combination, first_month='2009-11', last_month='2013-02',
                                   trading_cost_bps=bps)
    compounded, months = _compounded_by_period(results)
    np.testing.assert_allclose(compounded * 100, results['yearly_returns'], rtol=RTOL, atol=1e-12)
    # Every month of the backtest, once, in order: from the month after the first rebalance to the last period end
    first, last = pd.Timestamp(results['years'][0]), pd.Timestamp(results['years'][-1])
    expected = month_ends((first + pd.offsets.MonthEnd(1)).strftime('%Y-%m'), last.strftime('%Y-%m'))
    assert list(months) == list(expected)
    assert results['monthly_dates'] == [d.strftime('%Y-%m-%d') for d in expected]
    # And so the months compound to the final value
    growth = np.prod(1 + np.array(results['monthly_returns']) / 100)
    assert 1000.0 * growth == pytest.approx(results['final_value'], rel=1e-11)


def test_buy_and_hold_months_by_hand():
    frames, t0, t1, months = _quarter(2)
    _set_returns(frames, 10001, dict(zip(months, [0.10, 0.10, 0.10])))
    _set_returns(frames, 10002, dict(zip(months, [-0.10, 0.20, 0.0])))
    results, _ = run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4, top_pct=100.0)
    # Equal weights at t0, then each position grows: values 1.0, 1.145, 1.2055 at the month ends
    values = [1.0, 0.5 * 1.1 + 0.5 * 0.9, 0.5 * 1.21 + 0.5 * 1.08, 0.5 * 1.331 + 0.5 * 1.08]
    expected = [values[k + 1] / values[k] - 1 for k in range(3)]
    np.testing.assert_allclose(np.array(results['monthly_returns']) / 100, expected, rtol=RTOL, atol=1e-15)
    assert results['monthly_dates'] == [m.strftime('%Y-%m-%d') for m in months]


def test_the_trading_cost_is_charged_in_the_first_month_only():
    frames, t0, t1, months = _quarter(2)
    _set_returns(frames, 10001, dict(zip(months, [0.10, 0.10, 0.10])))
    _set_returns(frames, 10002, dict(zip(months, [-0.10, 0.20, 0.0])))
    data = frames.data()
    free, _ = run_periods(data, [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4, top_pct=100.0)
    costly, _ = run_periods(data, [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4, top_pct=100.0,
                            trading_cost_bps=50)
    c = 0.005                                    # the first purchase turns over 100%
    gross, net = np.array(free['monthly_returns']) / 100, np.array(costly['monthly_returns']) / 100
    assert net[0] == pytest.approx((1 - c) * (1 + gross[0]) - 1, rel=RTOL, abs=1e-15)
    np.testing.assert_array_equal(net[1:], gross[1:])


def test_after_a_delisting_the_month_earns_the_risk_free_rate():
    frames, t0, t1, months = _quarter(1)
    _set_returns(frames, 10001, {months[0]: -0.5, months[1]: None, months[2]: None}, delisted_month=months[0])
    frames.market.loc[frames.market['month_end'].isin(months), 'rf'] = [0.001, 0.002, 0.003]
    results, _ = run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4, top_pct=100.0)
    np.testing.assert_allclose(np.array(results['monthly_returns']) / 100, [-0.5, 0.002, 0.003], rtol=RTOL)


def test_cash_periods_earn_the_monthly_risk_free_rate_and_still_compound():
    data = pinned_frames().data()
    results, _ = run_wrds_backtest(data, ['f_roe'], {'f_roe': 'top'}, 'quarterly', 2010, 2012, 1000.0, top_pct=50.0,
                                   selected_sectors=['Energy'], first_month=FIRST, last_month=LAST,
                                   trading_cost_bps=40)
    assert results['empty_rebalance_dates']
    compounded, months = _compounded_by_period(results)
    np.testing.assert_allclose(compounded * 100, results['yearly_returns'], rtol=RTOL, atol=1e-12)
    # The second of two empty dates in a row costs nothing: its months are the risk-free rate
    empty = [results['rebalance_dates'].index(d) for d in results['empty_rebalance_dates']]
    cash_months = months[(months > pd.Timestamp(results['years'][empty[1]]))
                         & (months <= pd.Timestamp(results['years'][empty[1] + 1]))]
    rf = data.market['rf'].reindex(cash_months).to_numpy()
    monthly = pd.Series(results['monthly_returns'], index=months).loc[cash_months].to_numpy() / 100
    np.testing.assert_allclose(monthly, rf, rtol=RTOL)


def test_a_portfolio_worth_nothing_stays_at_zero():
    monthly = np.array([[0.0], [1.2], [0.9]])
    np.testing.assert_array_equal(portfolio_monthly_returns(np.array([1.0]), monthly), [-1.0, 0.0, 0.0])
