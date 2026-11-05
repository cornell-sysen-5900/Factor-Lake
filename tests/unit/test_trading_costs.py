"""
Trading costs in both engines (src/trading_costs.py, src/backtest_engine.py, src/wrds_engine.py):
a hand-checked example, delisting cash, Mixing netting, empty rebalance dates, no final sell,
cohorts, and that returns fall by exactly the cost. 0 bps giving main's results is checked by
test_legacy_pinned.py and test_wrds_pinned.py.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import rebalance_portfolio, run_cohort_comparison
from src.trading_costs import cost_results, drifted_weights, net_return, turnover, validate_trading_cost_bps
from src.wrds_engine import run_periods, run_wrds_backtest
from tests.unit.test_mixing_regression import delisting_universe, synthetic_universe
from tests.unit.test_wrds_pinned import FIRST, LAST, pinned_frames
from tests.unit.wrds_fixtures import month_ends, synthetic_frames

RTOL = 1e-12

# ---------------------------------------------------------------------------
# The hand-checked example: 3 stocks, 2 periods, 50 bps, $1,000
#
# Rebalance 1 holds A and B (50% each), bought from cash: turnover 100%, cost $5.
#   A +20%, B -10%: return before costs 5%, after (1 - 0.005) x 1.05 - 1 = 4.475%; value $1,044.75.
#   Drifted: A 0.6 / 1.05 = 4/7, B 0.45 / 1.05 = 3/7.
# Rebalance 2 holds B and C (50% each): sell A 4/7, buy B 1/2 - 3/7 = 1/14, buy C 1/2:
#   turnover 8/7, cost 0.005 x 8/7 of $1,044.75 = $5.97.
#   B +10%, C +30%: return before costs 20%, after (1 - 0.04/7) x 1.2 - 1.
# No final sell: the final value is $1,044.75 x (1 - 0.04/7) x 1.2.
# ---------------------------------------------------------------------------
BPS = 50.0
TURNOVER = [1.0, 8 / 7]
GROSS = [0.05, 0.20]
COST = [0.005, 0.005 * 8 / 7]
NET = [(1 - c) * (1 + g) - 1 for g, c in zip(GROSS, COST)]
VALUES = [1000.0, 1000.0 * (1 + NET[0]), 1000.0 * (1 + NET[0]) * (1 + NET[1])]
SCORES = [{'A': 3.0, 'B': 2.0, 'C': 1.0}, {'A': 1.0, 'B': 3.0, 'C': 2.0}]   # higher is better
RETURNS = [{'A': 0.20, 'B': -0.10, 'C': 0.05}, {'A': 0.0, 'B': 0.10, 'C': 0.30}]
TOP_PCT = 67.0   # floor(3 x 0.67) = 2 stocks


def _legacy_example():
    rows = [{'Year': 2010 + i, 'Ticker': t, 'Ending_Price': 10.0, 'ROA': SCORES[i][t],
             'Next-Years_Return': RETURNS[i][t] * 100.0}
            for i in range(2) for t in 'ABC']
    return pd.DataFrame(rows)


def _wrds_example():
    """The same example on WRDS data: monthly periods, permnos 10001-10003 for A, B, C."""
    frames = synthetic_frames('2010-03', '2010-05', n_stocks=3, membership=1.0)
    dates = list(month_ends('2010-03', '2010-05'))
    panel = frames.panel
    panel = panel[panel['date'].isin(dates[:2])].reset_index(drop=True)
    for i, date in enumerate(dates[:2]):
        for permno, ticker in zip((10001, 10002, 10003), 'ABC'):
            panel.loc[(panel['date'] == date) & (panel['permno'] == permno), 'f_roe'] = SCORES[i][ticker]
    frames.panel = panel
    rm = frames.returns
    for i, month in enumerate(dates[1:]):
        for permno, ticker in zip((10001, 10002, 10003), 'ABC'):
            rm.loc[(rm['month_end'] == month) & (rm['permno'] == permno), 'ret_m'] = RETURNS[i][ticker]
    return frames.data(), dates


def _legacy(bps, **kw):
    return rebalance_portfolio(_legacy_example(), ['ROA'], {'ROA': 'top'}, 2010, 2012, 1000.0,
                               top_pct=TOP_PCT, trading_cost_bps=bps, **kw)


def _wrds(bps):
    data, dates = _wrds_example()
    return run_periods(data, dates, ['f_roe'], {'f_roe': 'top'}, 1000.0, 12, top_pct=TOP_PCT,
                       trading_cost_bps=bps)


def _check_example(results):
    np.testing.assert_allclose(results['turnover_pct'], np.array(TURNOVER) * 100, rtol=RTOL)
    np.testing.assert_allclose(results['trading_cost_pct'], np.array(COST) * 100, rtol=RTOL)
    np.testing.assert_allclose(results['trading_cost_dollars'], [5.0, 1044.75 * 0.005 * 8 / 7], rtol=RTOL)
    np.testing.assert_allclose(results['returns_before_costs'], np.array(GROSS) * 100, rtol=RTOL)
    np.testing.assert_allclose(results['yearly_returns'], np.array(NET) * 100, rtol=RTOL)
    np.testing.assert_allclose(results['portfolio_values'], VALUES, rtol=RTOL)
    np.testing.assert_allclose(results['portfolio_values_before_costs'], [1000.0, 1050.0, 1260.0], rtol=RTOL)
    assert results['final_value'] == pytest.approx(VALUES[-1], rel=RTOL)           # no final sell
    assert results['trading_cost_bps'] == BPS
    assert results['average_turnover_pct'] == pytest.approx(np.mean(TURNOVER) * 100, rel=RTOL)
    assert results['total_trading_cost'] == pytest.approx(5.0 + 1044.75 * 0.005 * 8 / 7, rel=RTOL)


def test_hand_checked_example_in_legacy():
    results = _legacy(BPS)
    _check_example(results)
    # Annual periods: CAGR over 2 years
    assert results['cagr_before_costs'] == pytest.approx(1.26 ** 0.5 - 1, rel=RTOL)
    assert results['cagr_after_costs'] == pytest.approx((VALUES[-1] / 1000) ** 0.5 - 1, rel=RTOL)
    assert results['cost_drag'] == pytest.approx(results['cagr_before_costs'] - results['cagr_after_costs'])
    # Holdings: Traded % is |target - drifted|; the positions are bought with what is left after the cost
    first, second = (pd.DataFrame(results['holdings_by_year'][y]).set_index('Ticker') for y in (2010, 2011))
    np.testing.assert_allclose(first['Traded %'], [50.0, 50.0], rtol=RTOL)
    np.testing.assert_allclose(second.loc[['B', 'C'], 'Traded %'], [100 / 14, 50.0], rtol=RTOL)
    assert first['Dollar Value'].sum() == pytest.approx(995.0, rel=RTOL)
    assert second['Dollar Value'].sum() == pytest.approx(1044.75 * (1 - COST[1]), rel=RTOL)


def test_hand_checked_example_in_wrds():
    results, holdings = _wrds(BPS)
    _check_example(results)
    # Monthly periods: CAGR over 2 months, annualized
    assert results['cagr_before_costs'] == pytest.approx(1.26 ** 6 - 1, rel=RTOL)
    assert results['cagr'] == results['cagr_after_costs'] == pytest.approx((VALUES[-1] / 1000) ** 6 - 1, rel=RTOL)
    first, second = (rows.set_index('permno') for _, rows in holdings.groupby('rebalance_date'))
    np.testing.assert_allclose(first.loc[[10001, 10002], 'traded'], [0.5, 0.5], rtol=RTOL)       # A, B
    np.testing.assert_allclose(second.loc[[10002, 10003], 'traded'], [1 / 14, 0.5], rtol=RTOL)   # B, C
    assert first['dollar_value'].sum() == pytest.approx(995.0, rel=RTOL)
    assert second['dollar_value'].sum() == pytest.approx(1044.75 * (1 - COST[1]), rel=RTOL)


def test_the_example_at_0_bps_has_no_cost_fields_and_no_traded_column():
    legacy = _legacy(0)
    wrds, holdings = _wrds(0)
    for results in (legacy, wrds):
        assert 'trading_cost_bps' not in results and 'turnover_pct' not in results
        np.testing.assert_allclose(results['yearly_returns'], np.array(GROSS) * 100, rtol=RTOL)
    assert 'Traded %' not in legacy['holdings_by_year'][2010][0]
    assert 'traded' not in holdings.columns


# ---------------------------------------------------------------------------
# Delisting cash, Mixing netting, empty rebalance dates
# ---------------------------------------------------------------------------

def test_legacy_delisting_cash_is_reinvested_as_a_charged_buy_without_a_charged_sell():
    """A delists in year 1 (hold_cash); year 2 holds only B. Its cash is no stock, so only the buy is charged."""
    data = pd.DataFrame([
        {'Year': 2010, 'Ticker': 'A', 'Ending_Price': 10.0, 'ROA': 2.0, 'Next-Years_Return': np.nan,
         'Delist_Price': 5.0, 'Delist_Date': pd.Timestamp('2011-06-30')},
        {'Year': 2010, 'Ticker': 'B', 'Ending_Price': 10.0, 'ROA': 1.0, 'Next-Years_Return': 10.0,
         'Delist_Price': np.nan, 'Delist_Date': pd.NaT},
        {'Year': 2011, 'Ticker': 'B', 'Ending_Price': 11.0, 'ROA': 1.0, 'Next-Years_Return': 5.0,
         'Delist_Price': np.nan, 'Delist_Date': pd.NaT},
    ])
    results = rebalance_portfolio(data, ['ROA'], {'ROA': 'top'}, 2010, 2012, 1000.0, top_pct=100.0,
                                  delisting_strategy='hold_cash', trading_cost_bps=100)
    rows = pd.DataFrame(results['holdings_by_year'][2010]).set_index('Ticker')
    a_end = 0.5 * (1 + rows.loc['A', 'Period Return %'] / 100)        # A's cash at the year end
    b_end = 0.5 * 1.10
    cash_share = a_end / (a_end + b_end)
    # Year 2 buys B up to 100%: exactly the cash share. Selling A's position is not charged.
    assert results['turnover_pct'] == pytest.approx([100.0, cash_share * 100], rel=RTOL)
    assert rows.loc['A', 'Delisted']


def test_wrds_delisting_cash_is_reinvested_as_a_charged_buy_without_a_charged_sell():
    frames = synthetic_frames('2010-03', '2010-05', n_stocks=2, membership=1.0)
    dates = list(month_ends('2010-03', '2010-05'))
    panel = frames.panel[frames.panel['date'].isin(dates[:2])]
    frames.panel = panel[~((panel['date'] == dates[1]) & (panel['permno'] == 10001))].reset_index(drop=True)
    rm = frames.returns
    rm.loc[(rm['permno'] == 10001) & (rm['month_end'] == dates[1]), ['ret_m', 'delisted_in_month']] = [-0.5, True]
    rm.drop(rm.index[(rm['permno'] == 10001) & (rm['month_end'] > dates[1])], inplace=True)
    rm.loc[(rm['permno'] == 10002) & (rm['month_end'] == dates[1]), 'ret_m'] = 0.10
    results, _ = run_periods(frames.data(), dates, ['f_roe'], {'f_roe': 'top'}, 1000.0, 12, top_pct=100.0,
                             trading_cost_bps=100)
    cash_share = 0.25 / (0.25 + 0.55)
    assert results['total_delisted_positions'] == 1
    assert results['turnover_pct'] == pytest.approx([100.0, cash_share * 100], rel=RTOL)


def test_mixing_sleeves_are_netted_so_an_overlap_is_not_charged_twice():
    """Two sleeves swap their stocks: the portfolio still holds A and B 50/50, so nothing is traded."""
    rows = [{'Year': 2010 + i, 'Ticker': t, 'Ending_Price': 10.0, 'ROA': roa, 'ROE_using_9-30_Data': roe,
             'Next-Years_Return': 0.0}
            for i, picks in enumerate([{'A': (2, 1), 'B': (1, 2)}, {'A': (1, 2), 'B': (2, 1)}])
            for t, (roa, roe) in picks.items()]
    results = rebalance_portfolio(pd.DataFrame(rows), ['ROA', 'ROE_using_9-30_Data'],
                                  {'ROA': 'top', 'ROE_using_9-30_Data': 'top'}, 2010, 2012, 1000.0,
                                  top_pct=50.0, trading_cost_bps=100)
    assert results['turnover_pct'] == pytest.approx([100.0, 0.0], abs=1e-12)
    # Both sleeves bought A in year 2010's first purchase only once each: 100%, not more
    rows_2011 = pd.DataFrame(results['holdings_by_year'][2011])
    assert rows_2011['Traded %'].tolist() == pytest.approx([0.0, 0.0], abs=1e-12)


def test_wrds_mixing_overlap_is_one_position():
    """Both sleeves pick the same stock: one 100% position, bought once."""
    data, dates = _wrds_example()
    data.panel['mom_12_1'] = data.panel['f_roe']
    results, holdings = run_periods(data, dates, ['f_roe', 'mom_12_1'], {'f_roe': 'top', 'mom_12_1': 'top'},
                                    1000.0, 12, top_pct=34.0, trading_cost_bps=100)
    assert results['turnover_pct'][0] == pytest.approx(100.0, rel=RTOL)
    assert holdings.groupby('rebalance_date')['traded'].sum().iloc[0] == pytest.approx(1.0, rel=RTOL)


def test_empty_rebalance_dates_charge_the_sell_into_cash_and_nothing_for_the_cash_period():
    data = pinned_frames().data()
    results, _ = run_wrds_backtest(data, ['f_roe'], {'f_roe': 'top'}, 'quarterly', 2010, 2012, 1000.0, top_pct=50.0,
                                   selected_sectors=['Energy'], first_month=FIRST, last_month=LAST,
                                   trading_cost_bps=40)
    empty = [results['rebalance_dates'].index(d) for d in results['empty_rebalance_dates']]
    assert len(empty) == 2 and empty[1] == empty[0] + 1
    turnover_pct, rf = results['turnover_pct'], results['rf_returns']
    before, after = empty[0], empty[1] + 1
    # The first empty date sells everything (the drifted stock weights, below 100% only for delisted cash)
    assert 0 < turnover_pct[before] <= 100.0 + 1e-9
    assert turnover_pct[empty[1]] == 0.0                                   # cash stays cash: free
    assert turnover_pct[after] == pytest.approx(100.0, rel=RTOL)           # buying back from cash
    # The cash periods earn the risk-free rate, less the sell on the first one only
    assert results['returns_before_costs'][empty[1]] == pytest.approx(rf[empty[1]], rel=RTOL)
    assert results['yearly_returns'][empty[1]] == pytest.approx(rf[empty[1]], rel=RTOL)
    c = 0.004 * turnover_pct[before] / 100
    assert results['yearly_returns'][before] == pytest.approx(((1 - c) * (1 + rf[before] / 100) - 1) * 100,
                                                              rel=1e-9)


# ---------------------------------------------------------------------------
# Returns fall by exactly the cost; higher costs never give a higher final value; cohorts
# ---------------------------------------------------------------------------

def _legacy_universe_run(bps, **kw):
    return rebalance_portfolio(delisting_universe(), ['ROE_using_9-30_Data', 'ROA', '6-Mo_Momentum'],
                               {'ROE_using_9-30_Data': 'top', 'ROA': 'top', '6-Mo_Momentum': 'bottom'},
                               2010, 2016, 1000.0, trading_cost_bps=bps, **kw)


def _wrds_universe_run(bps, data=None):
    return run_wrds_backtest(data or pinned_frames().data(), ['f_roe', 'mom_12_1'],
                             {'f_roe': 'top', 'mom_12_1': 'top'}, 'monthly', 2010, 2012, 1000.0,
                             first_month=FIRST, last_month=LAST, trading_cost_bps=bps)[0]


def _assert_returns_fall_by_the_cost(results, free):
    gross = np.array(results['returns_before_costs']) / 100
    net = np.array(results['yearly_returns']) / 100
    cost = np.array(results['trading_cost_pct']) / 100
    np.testing.assert_allclose(1 + net, (1 + gross) * (1 - cost), rtol=RTOL)
    np.testing.assert_allclose(cost, results['trading_cost_bps'] / 1e4 * np.array(results['turnover_pct']) / 100,
                               rtol=RTOL)
    # Before costs is the 0 bps run; the turnover does not depend on the cost
    np.testing.assert_allclose(gross * 100, free['yearly_returns'], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(results['portfolio_values_before_costs'], free['portfolio_values'], rtol=1e-9)
    assert results['final_value'] == pytest.approx(1000.0 * np.prod(1 + net), rel=RTOL)
    assert len(results['turnover_pct']) == len(results['yearly_returns'])          # no final sell


@pytest.mark.parametrize('strategy', ['zero_return', 'hold_cash', 'reinvest'])
@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
def test_legacy_returns_fall_by_exactly_the_cost(strategy, combination):
    kw = dict(delisting_strategy=strategy, factor_combination=combination)
    results = _legacy_universe_run(25, **kw)
    _assert_returns_fall_by_the_cost(results, _legacy_universe_run(0, **kw))
    assert results['total_delisted_positions'] > 0
    assert results['turnover_pct'][0] == pytest.approx(100.0, rel=RTOL)


def test_wrds_returns_fall_by_exactly_the_cost():
    data = pinned_frames().data()
    results = _wrds_universe_run(25, data)
    _assert_returns_fall_by_the_cost(results, _wrds_universe_run(0, data))
    assert results['turnover_pct'][0] == pytest.approx(100.0, rel=RTOL)


def test_metrics_are_computed_on_the_returns_after_costs():
    free, costly = _legacy_universe_run(0), _legacy_universe_run(100)
    assert costly['final_value'] < free['final_value']
    assert costly['sharpe_portfolio'] < free['sharpe_portfolio']
    assert costly['yearly_comparisons'][0]['p_ret'] == costly['yearly_returns'][0]
    assert costly['cost_drag'] > 0


def test_higher_costs_never_give_a_higher_final_value():
    levels = [0, 1, 5, 25, 60, 250, 500]
    data = pinned_frames().data()
    legacy = [_legacy_universe_run(bps)['final_value'] for bps in levels]
    wrds = [_wrds_universe_run(bps, data)['final_value'] for bps in levels]
    assert all(a > b for a, b in zip(legacy, legacy[1:]))
    assert all(a > b for a, b in zip(wrds, wrds[1:]))


def test_legacy_cohort_pays_the_runs_cost_on_both_sides():
    settings = dict(start_year=2002, end_year=2012, initial_aum=1000.0, use_market_cap_weight=False)
    args = (synthetic_universe(), ['ROE_using_9-30_Data', 'ROA'], {'ROE_using_9-30_Data': 'top', 'ROA': 'top'}, 20.0)
    free_top, free_bottom = run_cohort_comparison(*args, settings)
    top, bottom = run_cohort_comparison(*args, dict(settings, trading_cost_bps=50))
    assert top[-1] < free_top[-1] and bottom[-1] < free_bottom[-1]
    direct = rebalance_portfolio(args[0], args[1], args[2], 2002, 2012, 1000.0, top_pct=20.0, trading_cost_bps=50)
    assert top == direct['portfolio_values']


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------

def test_turnover_counts_buys_and_sells_of_every_stock():
    total, traded = turnover(pd.Series({'B': 0.5, 'C': 0.5}), pd.Series({'A': 0.4, 'B': 0.4}))
    assert total == pytest.approx(0.4 + 0.1 + 0.5)
    assert traded.to_dict() == pytest.approx({'A': 0.4, 'B': 0.1, 'C': 0.5})
    assert turnover(pd.Series({'A': 1.0}), pd.Series(dtype=float))[0] == 1.0       # from cash


def test_drifted_weights_leave_delisted_cash_out():
    w = drifted_weights(pd.Series({'A': 0.5, 'B': 0.5}), pd.Series({'A': 0.5, 'B': 1.5}),
                        pd.Series({'A': True, 'B': False}))
    assert w.to_dict() == pytest.approx({'B': 0.75})
    assert drifted_weights(pd.Series({'A': 1.0}), pd.Series({'A': 0.0})).empty     # worth nothing


def test_net_return_is_exact_at_zero_cost():
    for r in (0.1, -0.3, 0.123456789, 1e-17, -0.0, 3.0):
        assert net_return(r, 0.0) == r
        assert (1 + r) * (1 - 0.01) - 1 == pytest.approx(net_return(r, 0.01), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize('value, message', [
    (-1, "from 0 to 500"), (500.5, "from 0 to 500"), ('25', "a number"), (True, "a number"),
    (float('nan'), "a number"), (None, "a number"),
])
def test_invalid_costs_are_refused_with_a_readable_message(value, message):
    with pytest.raises(ValueError, match=message):
        validate_trading_cost_bps(value)
    with pytest.raises(ValueError, match=message):
        _legacy(value)


def test_cost_results_with_no_periods():
    fields = cost_results(25, [], [], [1000.0])
    assert fields['average_turnover_pct'] == 0.0 and fields['total_trading_cost'] == 0.0 and fields['cost_drag'] == 0
