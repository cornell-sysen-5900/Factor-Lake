"""
Look-ahead tests for the WRDS engine (DOCS/WRDS_MODE_DESIGN.md, section 3.3): what is
bought at a rebalance date t depends only on data known at t, a holding period's
return only on the months in (t, t'], and columns about the future can never be factors.
"""
import numpy as np
import pandas as pd
import pytest

from src.wrds_data import PANEL_COLUMNS
from src.wrds_engine import run_wrds_backtest
from src.wrds_factors import WRDS_FACTOR_COLUMNS
from tests.unit.wrds_fixtures import synthetic_frames

FACTORS = ['f_roe', 'mom_12_1', 'f_book_to_price']
DIRECTIONS = {'f_roe': 'top', 'mom_12_1': 'bottom', 'f_book_to_price': 'top'}
BOUNDS = dict(first_month='2009-11', last_month='2013-02')
HOLDING_KEYS = ['rebalance_date', 'permno', 'rank', 'score', 'weight']


def _run(frames, combination='mixing', cap=False):
    return run_wrds_backtest(frames.data(), FACTORS, DIRECTIONS, 'quarterly', 2010, 2012, 1000.0,
                             use_market_cap_weight=cap, factor_combination=combination, **BOUNDS)


def _scramble_after(frames, panel_after, months_after, seed=99):
    """Changes every panel row dated after panel_after and all returns and market data after months_after."""
    rng = np.random.default_rng(seed)
    panel = frames.panel
    later = panel['date'] > panel_after
    for col in WRDS_FACTOR_COLUMNS:
        panel.loc[later, col] = rng.normal(size=later.sum())
    panel.loc[later, 'prc'] = rng.uniform(1, 300, later.sum())
    # Later membership changes too: some members leave, a new stock joins
    leave = panel.index[later][::5]
    joined = panel[later].iloc[:3].assign(permno=np.int32(99999), ticker='NEW')
    frames.panel = pd.concat([panel.drop(leave), joined], ignore_index=True)
    rm, mm = frames.returns, frames.market
    rm.loc[rm['month_end'] > months_after, 'ret_m'] = rng.normal(0.5, 0.5, (rm['month_end'] > months_after).sum())
    mm.loc[mm['month_end'] > months_after, 'iwm_total_ret'] = 0.9


@pytest.mark.parametrize('combination, cap', [('mixing', False), ('integrating', True)])
@pytest.mark.parametrize('k', [0, 3, 8])
def test_changing_data_after_t_changes_nothing_at_t(combination, cap, k):
    frames = synthetic_frames()
    results, holdings = _run(frames, combination, cap)
    t = pd.Timestamp(results['rebalance_dates'][k])
    t_next = pd.Timestamp(results['period_end_dates'][k])

    _scramble_after(frames, panel_after=t, months_after=t_next)
    results2, holdings2 = _run(frames, combination, cap)

    # Text columns are categoricals whose category list covers the whole run, so compare values
    upto = lambda h: h[h['rebalance_date'] <= t].reset_index(drop=True).astype(
        {'ticker': object, 'name': object, 'sector': object})
    pd.testing.assert_frame_equal(upto(holdings2), upto(holdings))     # same stocks, ranks, scores, weights, returns
    n = k + 1                                                          # periods ending by t_next
    assert results2['yearly_returns'][:n] == results['yearly_returns'][:n]
    assert results2['benchmark_returns'][:n] == results['benchmark_returns'][:n]
    assert results2['yearly_returns'][n:] != results['yearly_returns'][n:]    # the scramble did reach later periods


def test_returns_in_the_holding_period_change_returns_but_never_what_is_bought():
    frames = synthetic_frames()
    results, holdings = _run(frames)
    t, t_next = (pd.Timestamp(results[k][2]) for k in ('rebalance_dates', 'period_end_dates'))
    rm = frames.returns
    in_period = (rm['month_end'] > t) & (rm['month_end'] <= t_next)
    rm.loc[in_period, 'ret_m'] = -rm.loc[in_period, 'ret_m']
    results2, holdings2 = _run(frames)

    at_t = lambda h: h[h['rebalance_date'] == t]
    pd.testing.assert_frame_equal(at_t(holdings2)[HOLDING_KEYS], at_t(holdings)[HOLDING_KEYS])
    assert results2['yearly_returns'][2] != results['yearly_returns'][2]
    assert results2['yearly_returns'][:2] == results['yearly_returns'][:2]


def test_returns_start_after_t():
    """A return in the rebalance month itself (up to t) never counts in the period from t."""
    frames = synthetic_frames()
    results, _ = _run(frames)
    t = pd.Timestamp(results['rebalance_dates'][1])
    frames.returns.loc[frames.returns['month_end'] == t, 'ret_m'] = 5.0
    results2, holdings2 = _run(frames)
    assert results2['yearly_returns'][1] == results['yearly_returns'][1]


@pytest.mark.parametrize('column', ['ret_next_1m', 'delisted_next_month', 'ret_m', 'delisted_in_month',
                                    'iwm_total_ret', 'prc'])
def test_returns_and_future_columns_can_never_be_factors(column):
    frames = synthetic_frames()
    assert column not in WRDS_FACTOR_COLUMNS
    with pytest.raises(ValueError, match='Not WRDS factors'):
        run_wrds_backtest(frames.data(), [column], {column: 'top'}, 'quarterly', 2010, 2012, 1000.0, **BOUNDS)


def test_the_loader_never_reads_columns_about_the_month_after_t():
    assert not {'ret_next_1m', 'delisted_next_month'} & set(PANEL_COLUMNS)
