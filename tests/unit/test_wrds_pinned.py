"""
Pinned WRDS results: fixed WRDS backtests on synthetic data must give exactly the results
and holdings they gave on main before trading costs were added, when run at 0 bps.

The fixture (tests/fixtures/wrds_pinned_results.json) was captured with main's code
(faeca5c) by running this file there: it only uses functions main has. Numbers must agree
to a relative 1e-9, as in tests/unit/test_legacy_pinned.py; everything else exactly.
Regenerate it only when a change to WRDS results is intended:

    uv run python -m tests.unit.test_wrds_pinned
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.check_backtest_lambda import TOLERANCE, max_difference, plain_rows
from src.backtest_jobs import encode_result
from src.wrds_engine import run_wrds_backtest
from tests.unit.wrds_fixtures import synthetic_frames

FIXTURE_PATH = Path(__file__).resolve().parents[1] / 'fixtures' / 'wrds_pinned_results.json'
FIRST, LAST = '2009-11', '2013-02'

# name -> (factors, directions, keyword arguments of run_wrds_backtest)
JOBS = {
    'quarterly_mixing_equal': (
        ['f_roe', 'mom_12_1'], {'f_roe': 'top', 'mom_12_1': 'top'},
        dict(frequency='quarterly', start_year=2010, end_year=2012, initial_aum=1000.0)),
    'monthly_integrating_cap_weight': (
        ['f_roe', 'mom_12_1'], {'f_roe': 'top', 'mom_12_1': 'bottom'},
        dict(frequency='monthly', start_year=2010, end_year=2012, initial_aum=5000.0,
             use_market_cap_weight=True, factor_combination='integrating')),
    'semiannual_mixing_fossil_free': (
        ['f_roe', 'mom_12_1'], {'f_roe': 'bottom', 'mom_12_1': 'top'},
        dict(frequency='semiannual', start_year=2010, end_year=2012, initial_aum=1000.0,
             restrict_fossil_fuels=True)),
    # One sector, missing on two dates: they have nothing to buy and earn the risk-free rate
    'quarterly_one_sector_top_half': (
        ['f_roe'], {'f_roe': 'top'},
        dict(frequency='quarterly', start_year=2010, end_year=2012, initial_aum=1000.0, top_pct=50.0,
             selected_sectors=['Energy'])),
}


def pinned_frames():
    """Synthetic WRDS data with delistings (a delisting month, then no more returns) and missing returns."""
    frames = synthetic_frames(FIRST, LAST, n_stocks=40, seed=7, membership=0.8)
    returns = frames.returns
    months = sorted(returns['month_end'].unique())
    for i, permno in enumerate(range(10001, 10041, 5)):
        month = months[6 + 3 * i]
        rows = returns['permno'] == permno
        returns.loc[rows & (returns['month_end'] == month), ['ret_m', 'delisted_in_month']] = [-0.4, True]
        returns.drop(returns.index[rows & (returns['month_end'] > month)], inplace=True)
    returns.loc[returns.sample(frac=0.02, random_state=3).index, 'ret_m'] = np.nan
    # No Energy stock on two quarter ends: a one-sector run has nothing to buy there
    panel = frames.panel
    empty = (panel['sector_name'] == 'Energy') & panel['date'].dt.strftime('%Y-%m').isin(['2010-06', '2010-09'])
    frames.panel = panel[~empty].reset_index(drop=True)
    return frames


def run_pinned(name, data, **extra):
    factors, directions, options = JOBS[name]
    return run_wrds_backtest(data, factors, directions, first_month=FIRST, last_month=LAST, **options, **extra)


def as_json(results, holdings):
    return {'result': json.loads(encode_result(results)), 'holdings': json.loads(json.dumps(plain_rows(holdings)))}


@pytest.fixture(scope='module')
def data():
    return pinned_frames().data()


@pytest.fixture(scope='module')
def pinned():
    return json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', JOBS)
def test_wrds_backtest_at_0_bps_gives_the_results_it_gave_on_main(pinned, data, name):
    got = as_json(*run_pinned(name, data, trading_cost_bps=0))
    difference, where = max_difference(got, pinned[name])
    assert difference <= TOLERANCE, f"{where} differs from main by {difference:.2e} (relative)"


def test_pinned_wrds_jobs_cover_delistings_missing_returns_and_empty_dates(pinned):
    """The fixture really holds what the test claims to compare."""
    results = [v['result'] for v in pinned.values()]
    assert any(r['total_delisted_positions'] > 0 for r in results)
    assert any(r['missing_return_stock_months'] > 0 for r in results)
    assert any(r['empty_rebalance_dates'] for r in results)
    assert {r['frequency'] for r in results} == {'monthly', 'quarterly', 'semiannual'}


if __name__ == '__main__':
    frames_data = pinned_frames().data()
    captured = {name: as_json(*run_pinned(name, frames_data)) for name in JOBS}
    FIXTURE_PATH.write_text(json.dumps(captured, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(f"Wrote {len(captured)} WRDS backtests to {FIXTURE_PATH}")
