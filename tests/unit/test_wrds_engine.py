"""
Tests for the vectorized WRDS engine (src/wrds_engine.py): it matches the Legacy engine
on the same single-period input, and its holding periods, delistings, missing returns,
risk-free fill, benchmarks, filters, ties and rebalance dates follow
DOCS/WRDS_MODE_DESIGN.md (sections 3, 4 and 7).
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import rebalance_portfolio
from src.performance_metrics import compute_comprehensive_metrics
from src.wrds_calendar import FIRST_MONTH, LAST_MONTH, PERIODS_PER_YEAR
from src.wrds_engine import (eligible_universe, factor_zscores, holding_period_growth, run_periods,
                             run_wrds_backtest)
from src.factor_utils import normalize_series
from tests.unit.wrds_fixtures import Frames, month_ends, synthetic_frames

FACTORS = ['f_roe', 'mom_12_1', 'f_book_to_price']
DIRECTIONS = {'f_roe': 'top', 'mom_12_1': 'bottom', 'f_book_to_price': 'top'}
RTOL = 1e-12


# ---------------------------------------------------------------------------
# Matches the Legacy engine on the same single-period input
# ---------------------------------------------------------------------------

def _single_period(seed=3, n=80):
    """One rebalance date and the month after it, every stock priced and returning."""
    frames = synthetic_frames('2010-09', '2010-10', n_stocks=n, seed=seed, membership=1.0)
    t0, t1 = month_ends('2010-09', '2010-10')
    frames.panel = frames.panel[frames.panel['date'] == t0].reset_index(drop=True)
    return frames, t0, t1


def _legacy_frame(frames, t1):
    """The same stocks as Legacy rows: one year whose 'next year' is the month after t0."""
    panel = frames.panel
    ret = frames.returns[frames.returns['month_end'] == t1].set_index('permno')['ret_m']
    return pd.DataFrame({
        'Ticker': panel['ticker'], 'Year': 2010, 'Ending_Price': panel['prc'],
        'Market_Capitalization': panel['mkt_cap_musd'],
        'Next-Years_Return': panel['permno'].map(ret).to_numpy() * 100.0,
        **{f: panel[f] for f in FACTORS},
    })


@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
@pytest.mark.parametrize('cap_weight', [False, True])
@pytest.mark.parametrize('top_pct', [10.0, 25.0])
def test_matches_the_legacy_engine_on_one_period(combination, cap_weight, top_pct):
    frames, t0, t1 = _single_period()
    results, holdings = run_periods(frames.data(), [t0, t1], FACTORS, DIRECTIONS, 1000.0, periods_per_year=1,
                                    top_pct=top_pct, use_market_cap_weight=cap_weight,
                                    factor_combination=combination)
    legacy = rebalance_portfolio(_legacy_frame(frames, t1), FACTORS, DIRECTIONS, 2010, 2011, 1000.0,
                                 top_pct=top_pct, use_market_cap_weight=cap_weight,
                                 factor_combination=combination)
    legacy_rows = pd.DataFrame(legacy['holdings_by_year'][2010]).set_index('Ticker')
    wrds = holdings.set_index('ticker')

    assert set(wrds.index) == set(legacy_rows.index)                       # the same stocks
    legacy_rows = legacy_rows.loc[wrds.index]
    np.testing.assert_allclose(wrds['weight'], legacy_rows['Weight %'] / 100.0, rtol=RTOL)
    np.testing.assert_allclose(wrds['period_return'], legacy_rows['Period Return %'] / 100.0, rtol=RTOL)
    np.testing.assert_allclose(wrds['score'], legacy_rows['Score'], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(wrds['rank'], legacy_rows['Rank'])
    np.testing.assert_allclose(results['yearly_returns'], legacy['yearly_returns'], rtol=RTOL, atol=1e-12)
    np.testing.assert_allclose(results['portfolio_values'], legacy['portfolio_values'], rtol=RTOL)
    if combination == 'mixing':
        assert list(wrds['picked_by']) == list(legacy_rows['Picked By'])


def test_zscores_match_normalize_series_on_every_date():
    frames = synthetic_frames(n_stocks=50)
    universe = frames.panel
    for higher in (True, False):
        z = factor_zscores(universe, 'f_roe', higher)
        for _, rows in universe.groupby('date'):
            expected = normalize_series(rows['f_roe'], higher_is_better=higher)
            np.testing.assert_allclose(z.loc[rows.index], expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Holding periods: drift, delisting, missing returns, risk-free fill, benchmarks
# ---------------------------------------------------------------------------

def _quarter(n_stocks=3):
    """Quarterly data 2010-03 to 2010-06, every stock a member; months Apr, May, Jun are one period."""
    frames = synthetic_frames('2010-03', '2010-06', n_stocks=n_stocks, membership=1.0)
    t0, t1 = month_ends('2010-03', '2010-06')[[0, -1]]
    return frames, t0, t1, list(month_ends('2010-04', '2010-06'))


def _set_returns(frames, permno, values, delisted_month=None):
    """Sets one stock's monthly returns ({month end: return or None for no row})."""
    rm = frames.returns
    for month, value in values.items():
        rows = (rm['month_end'] == month) & (rm['permno'] == permno)
        if value is None:
            rm.drop(rm.index[rows], inplace=True)
        else:
            rm.loc[rows, 'ret_m'] = value
            rm.loc[rows, 'delisted_in_month'] = month == delisted_month


def test_weights_drift_within_the_period():
    frames, t0, t1, months = _quarter(2)
    _set_returns(frames, 10001, dict(zip(months, [0.10, 0.10, 0.10])))
    _set_returns(frames, 10002, dict(zip(months, [-0.10, 0.20, 0.0])))
    results, holdings = run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4,
                                    top_pct=100.0)
    g1, g2 = 1.1 ** 3, 0.9 * 1.2
    # Buy and hold: equal weights at t0, then each position grows by its own compounded return
    assert results['yearly_returns'][0] == pytest.approx((0.5 * g1 + 0.5 * g2 - 1) * 100, rel=1e-12)
    # Not the average of monthly equal-weight returns (that would rebalance every month)
    rebalanced = np.prod([1 + (a + b) / 2 for a, b in [(0.1, -0.1), (0.1, 0.2), (0.1, 0.0)]]) - 1
    assert results['yearly_returns'][0] != pytest.approx(rebalanced * 100, rel=1e-6)
    np.testing.assert_allclose(holdings.set_index('permno')['period_return'].loc[[10001, 10002]],
                               [g1 - 1, g2 - 1], rtol=1e-12)


def test_after_delisting_the_money_earns_the_risk_free_rate():
    frames, t0, t1, months = _quarter(1)
    # Delists in April with a -50% delisting return; no rows afterwards
    _set_returns(frames, 10001, {months[0]: -0.5, months[1]: None, months[2]: None}, delisted_month=months[0])
    frames.market.loc[frames.market['month_end'].isin(months), 'rf'] = [0.001, 0.002, 0.003]
    data = frames.data()
    results, holdings = run_periods(data, [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4, top_pct=100.0)
    expected = 0.5 * 1.002 * 1.003 - 1
    row = holdings.iloc[0]
    assert row['period_return'] == pytest.approx(expected, rel=1e-12)
    assert row['delisted'] and row['missing_return_months'] == 0
    assert results['total_delisted_positions'] == 1 and results['missing_return_stock_months'] == 0


def test_a_missing_return_without_delisting_counts_as_zero_and_is_counted():
    frames, t0, t1, months = _quarter(2)
    _set_returns(frames, 10001, {months[0]: 0.05, months[1]: None, months[2]: 0.02})     # no May row
    _set_returns(frames, 10002, {months[0]: 0.01, months[1]: np.nan, months[2]: np.nan})  # rows without a return
    results, holdings = run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4,
                                    top_pct=100.0)
    by_permno = holdings.set_index('permno')
    assert by_permno.loc[10001, 'period_return'] == pytest.approx(1.05 * 1.02 - 1, rel=1e-12)
    assert by_permno.loc[10002, 'period_return'] == pytest.approx(0.01, rel=1e-12)
    assert list(by_permno.loc[[10001, 10002], 'missing_return_months']) == [1, 2]
    assert results['missing_return_stock_months'] == 3
    assert not by_permno['delisted'].any()


def test_missing_risk_free_months_are_filled_from_the_t_bill_and_flagged():
    frames, t0, t1, months = _quarter(1)
    in_period = frames.market['month_end'].isin(months)
    frames.market.loc[in_period, 'rf'] = [0.001, 0.002, np.nan]
    frames.market.loc[in_period, 'tbill_3m'] = [0.5, 0.5, 0.06]       # only June's is used
    results, _ = run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4, top_pct=100.0)
    assert results['rf_filled_months'] == ['2010-06']
    assert results['rf_returns'][0] == pytest.approx((1.001 * 1.002 * 1.005 - 1) * 100, rel=1e-12)


def test_benchmarks_compound_over_the_holding_period():
    frames, t0, t1, months = _quarter(1)
    in_period = frames.market['month_end'].isin(months)
    frames.market.loc[in_period, 'iwm_total_ret'] = [0.01, -0.02, 0.03]
    frames.market.loc[in_period, 'iwo_total_ret'] = [0.02, 0.02, 0.02]
    frames.market.loc[in_period, 'iwn_total_ret'] = [0.0, 0.0, -0.01]
    results, _ = run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4, top_pct=100.0)
    assert results['benchmark_returns'][0] == pytest.approx((1.01 * 0.98 * 1.03 - 1) * 100, rel=1e-12)
    assert results['growth_benchmark_returns'][0] == pytest.approx((1.02 ** 3 - 1) * 100, rel=1e-12)
    assert results['value_benchmark_returns'][0] == pytest.approx(-1.0, rel=1e-12)
    assert results['benchmark_labels']['benchmark'] == 'IWM (ETF, total return)'


def test_a_missing_benchmark_month_is_refused():
    frames, t0, t1, months = _quarter(1)
    frames.market.loc[frames.market['month_end'] == months[1], 'iwm_total_ret'] = np.nan
    with pytest.raises(ValueError, match='iwm_total_ret'):
        run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 4)


# ---------------------------------------------------------------------------
# Universe, selection and weights
# ---------------------------------------------------------------------------

def test_sector_filter_leaves_out_unclassified_and_fossil_filter_uses_gics_then_sic():
    panel = synthetic_frames(n_stocks=48).panel
    assert 'Unclassified' in set(eligible_universe(panel)['sector_name'])
    filtered = eligible_universe(panel, selected_sectors=['Energy', 'Financials'])
    assert set(filtered['sector_name']) == {'Energy', 'Financials'}

    no_fossil = eligible_universe(panel, restrict_fossil_fuels=True)
    assert not no_fossil['gsubind'].isin([10102020]).any()                              # by GICS
    assert not (no_fossil['gsubind'].isna() & no_fossil['siccd'].eq(1311.0)).any()      # by SIC without GICS
    kept_sic = no_fossil['gsubind'].notna() & no_fossil['siccd'].eq(1311.0)
    assert kept_sic.any()                                                               # GICS wins over SIC


def test_stocks_without_a_price_or_market_cap_cannot_be_bought():
    panel = synthetic_frames(n_stocks=20).panel
    panel.loc[panel.index[:5], 'prc'] = np.nan
    panel.loc[panel.index[5:10], 'mkt_cap_musd'] = 0.0
    assert not eligible_universe(panel).index.isin(panel.index[:10]).any()


def test_ties_are_broken_by_permno_and_at_least_one_stock_is_bought():
    frames, t0, t1 = _single_period(n=20)
    frames.panel['f_roe'] = 1.0                       # every stock ties
    _, holdings = run_periods(frames.data(), [t0, t1], ['f_roe'], {'f_roe': 'top'}, 1000.0, 1, top_pct=1.0)
    assert list(holdings['permno']) == [frames.panel['permno'].min()]


def test_equal_and_cap_weights_and_mixing_sleeves_each_sum_to_one():
    frames = synthetic_frames()
    data = frames.data()
    for cap in (False, True):
        _, holdings = run_wrds_backtest(data, FACTORS, DIRECTIONS, 'quarterly', 2010, 2012, 1000.0,
                                        use_market_cap_weight=cap, first_month='2009-11', last_month='2013-02')
        np.testing.assert_allclose(holdings.groupby('rebalance_date')['weight'].sum(), 1.0, rtol=1e-12)


def test_unknown_factor_sector_or_combination_is_refused():
    frames, t0, t1 = _single_period(n=10)
    data = frames.data()
    for kwargs, message in [({'factors': ['ret_next_1m']}, 'Not WRDS factors'),
                            ({'factor_combination': 'blend'}, 'factor_combination'),
                            ({'selected_sectors': ['Technology']}, 'Unknown GICS sectors')]:
        args = dict(factors=['f_roe'], factor_directions={'f_roe': 'top'}, initial_aum=1000.0,
                    periods_per_year=1)
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            run_periods(data, [t0, t1], **args)


# ---------------------------------------------------------------------------
# Rebalance dates and holding periods
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def full_range_data():
    """Synthetic data over the whole WRDS range, 2008-04 to 2026-09."""
    return synthetic_frames(FIRST_MONTH, LAST_MONTH, n_stocks=12).data()


@pytest.mark.parametrize('frequency, start, end, first, last, period_end, count', [
    ('monthly', 2008, 2026, '2008-04', '2026-08', '2026-09', 221),
    ('quarterly', 2008, 2026, '2008-06', '2026-06', '2026-09', 73),
    ('semiannual', 2008, 2026, '2008-06', '2025-12', '2026-06', 36),
    ('quarterly', 2010, 2012, '2009-12', '2012-09', '2012-12', 12),
    ('monthly', 2015, 2015, '2014-12', '2015-11', '2015-12', 12),
    ('semiannual', 2026, 2026, '2025-12', '2025-12', '2026-06', 1),
])
def test_rebalance_dates_and_holding_periods(full_range_data, frequency, start, end, first, last,
                                             period_end, count):
    results, holdings = run_wrds_backtest(full_range_data, ['f_roe'], {'f_roe': 'top'}, frequency, start, end,
                                          1000.0)
    rebalances = results['rebalance_dates']
    assert len(rebalances) == count == len(results['yearly_returns'])
    assert (rebalances[0][:7], rebalances[-1][:7], results['period_end_dates'][-1][:7]) == (first, last, period_end)
    assert results['period_end_dates'][:-1] == rebalances[1:]          # each period ends at the next rebalance
    # The data's month ends (last business day), not calendar month ends
    expected = month_ends(first, period_end)[::12 // PERIODS_PER_YEAR[frequency]]
    assert results['years'] == [d.strftime('%Y-%m-%d') for d in expected]
    assert sorted(holdings['rebalance_date'].dt.strftime('%Y-%m-%d').unique()) == rebalances
    assert results['periods_per_year'] == PERIODS_PER_YEAR[frequency]


def test_a_rebalance_date_without_panel_rows_is_refused():
    frames = synthetic_frames()
    gone = month_ends('2010-06', '2010-06')[0]
    frames.panel = frames.panel[frames.panel['date'] != gone]
    with pytest.raises(ValueError, match='no rows on the rebalance dates'):
        run_wrds_backtest(frames.data(), ['f_roe'], {'f_roe': 'top'}, 'quarterly', 2010, 2011, 1000.0,
                          first_month='2009-11', last_month='2013-02')


# ---------------------------------------------------------------------------
# Metrics annualized by frequency; Legacy unchanged at the default
# ---------------------------------------------------------------------------

def _metrics(k=None):
    rng = np.random.default_rng(1)
    p, b, g, v = (rng.normal(0.02, 0.05, 12) for _ in range(4))
    rf = rng.uniform(0, 0.01, 12)
    values = list(1000 * np.cumprod(np.r_[1, 1 + p]))
    args = (p, b, g, v, rf, values, 2010, 2022)
    return compute_comprehensive_metrics(*args) if k is None else compute_comprehensive_metrics(*args, periods_per_year=k)


def test_metrics_default_is_legacy_and_k_annualizes_volatility_sharpe_and_ir():
    legacy, k1, k4 = _metrics(), _metrics(1), _metrics(4)
    assert k1 == legacy
    scaled = [k for k in legacy if k.startswith(('vol_', 'sharpe_', 'information_ratio'))]
    assert len(scaled) == 15
    for key in legacy:
        if key in scaled:
            assert k4[key] == pytest.approx(legacy[key] * 2, rel=1e-12)
        else:
            assert k4[key] == legacy[key], key


def test_wrds_results_have_the_app_keys_and_the_wrds_extras():
    frames = synthetic_frames()
    results, _ = run_wrds_backtest(frames.data(), FACTORS, DIRECTIONS, 'quarterly', 2010, 2012, 1000.0,
                                   first_month='2009-11', last_month='2013-02')
    legacy_keys = set(_metrics())
    assert legacy_keys <= set(results)
    n = len(results['yearly_returns'])
    assert results['cagr'] == pytest.approx((results['final_value'] / 1000.0) ** (4 / n) - 1, rel=1e-12)
    assert results['data_mode'] == 'wrds' and results['frequency'] == 'quarterly'
    assert len(results['portfolio_values']) == len(results['years']) == n + 1
