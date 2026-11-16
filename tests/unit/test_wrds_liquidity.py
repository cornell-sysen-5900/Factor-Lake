"""
Tests for the liquidity limit's dollar volume (src/wrds_liquidity.py): computed from the
WRDS folder's daily prices one year at a time, attached to the engine's panel, and loaded
by the Lambda runner only for jobs with a limit.
"""
import io
import json

import numpy as np
import pandas as pd
import pytest

from src.backtest_jobs import job_hash
from src.backtest_service import BacktestService
from src.s3_client import S3DataManager
from src.wrds_data import DEFAULT_WRDS_DIR
from src.wrds_liquidity import (
    DAILY_MANIFEST,
    LIQUIDITY_COLUMN,
    NO_VOLUME_MESSAGE,
    add_volume,
    cap_positions,
    read_dollar_volume,
)
from tests.unit.test_wrds_service import FACTORS, PREFIX, wrds_job
from tests.unit.wrds_fixtures import RangeS3, put_wrds_files, synthetic_frames


def daily_prices(permnos, first='2009-06-01', last='2013-03-31', seed=0, gap=True):
    """Business-day prices and volumes, and with gap missing values, a negative volume and a stock absent for a year."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(first, last)
    frames = []
    for permno in permnos:
        d = days[days.year != 2011] if gap and permno == permnos[0] else days
        n = len(d)
        prc, vol = rng.uniform(5, 50, n), rng.uniform(1e4, 1e6, n)
        if gap:          # missing and bad values too
            prc[rng.random(n) < 0.05] = np.nan
            vol[rng.random(n) < 0.05] = np.nan
            vol[rng.random(n) < 0.01] = -99
        frames.append(pd.DataFrame({'date': d, 'permno': np.int32(permno), 'prc': prc, 'vol': vol,
                                    'ticker': 'X', 'ret': 0.0}))
    return pd.concat(frames).sort_values(['permno', 'date'], ignore_index=True)


def daily_files(px):
    """prices_daily/year=YYYY/data.parquet bodies and the manifest, as the pipeline exports them."""
    files = {}
    for year, part in px.groupby(px['date'].dt.year):
        body = io.BytesIO()
        part.to_parquet(body, index=False)
        files[f'prices_daily/year={year}/data.parquet'] = body.getvalue()
    files[DAILY_MANIFEST] = json.dumps({'years': {str(y): {} for y in px['date'].dt.year.unique()}}).encode()
    return files


def one_pass(px):
    """The month-end dollar volume over the whole history at once (pandas rolling, 51 of 63 days)."""
    px = px.copy()
    dv = px['prc'] * px['vol'].where(px['vol'] >= 0)
    px['dv'] = dv.groupby(px['permno']).transform(lambda s: s.rolling(63, min_periods=51).mean())
    month = px['date'].dt.to_period('M')
    month_end = px.groupby(month)['date'].transform('max')
    priced = px[px['prc'] > 0]
    last = priced.groupby([priced['permno'], priced['date'].dt.to_period('M')]).tail(1).index
    out = pd.DataFrame({'date': month_end.loc[last].to_numpy(), 'permno': px.loc[last, 'permno'].to_numpy(),
                        LIQUIDITY_COLUMN: px.loc[last, 'dv'].to_numpy()})
    return out.dropna().sort_values(['date', 'permno'], ignore_index=True)


def test_year_by_year_dollar_volume_matches_one_pass_over_all_years():
    px = daily_prices([10001, 10002, 10003])
    files = daily_files(px)
    got = read_dollar_volume(files.__getitem__, sorted(px['date'].dt.year.unique()))
    want = one_pass(px)
    assert list(got.columns) == ['date', 'permno', LIQUIDITY_COLUMN]
    pd.testing.assert_frame_equal(got[['date', 'permno']], want[['date', 'permno']], check_dtype=False)
    np.testing.assert_allclose(got[LIQUIDITY_COLUMN], want[LIQUIDITY_COLUMN], rtol=1e-6)
    # The windows run across years: January has values, and the stock absent in 2011 picks up its old rows
    assert (got['date'].dt.month == 1).any()
    back = got[(got['permno'] == 10001) & (got['date'] >= '2012-01-01')]
    assert back['date'].min() < pd.Timestamp('2012-03-31')


def test_add_volume_maps_each_row_and_refuses_without_volume():
    volume = pd.Series([1e6, 2e6], index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2010-03-31'), 1), (pd.Timestamp('2010-03-31'), 2)], names=['date', 'permno']), dtype='float32')
    panel = pd.DataFrame({'date': pd.to_datetime(['2010-03-31'] * 3), 'permno': np.array([2, 1, 3], dtype='int32')},
                         index=[7, 8, 9])
    out = add_volume(panel, volume)
    assert list(out.index) == [7, 8, 9]
    np.testing.assert_array_equal(out[LIQUIDITY_COLUMN], [2e6, 1e6, np.nan])
    for missing in (None, volume.iloc[0:0]):
        with pytest.raises(ValueError, match='not loaded'):
            add_volume(panel, missing)
    assert 'No limit' in NO_VOLUME_MESSAGE


@pytest.fixture
def s3():
    frames = synthetic_frames()
    s3 = RangeS3()
    put_wrds_files(s3, frames)
    px = daily_prices(sorted(frames.panel['permno'].unique()), gap=False)    # every stock has volume at every date
    for name, body in daily_files(px).items():
        s3.objects[f'{PREFIX}/{DEFAULT_WRDS_DIR}/{name}'] = (body, {})
    return s3


def _daily_gets(s3):
    return [key for key, _ in s3.gets if '/prices_daily/' in key]


def test_only_jobs_with_a_limit_read_the_daily_prices_once(s3):
    service = BacktestService(S3DataManager(bucket='test-bucket', prefix=PREFIX, client=s3))
    service.run(wrds_job())
    assert _daily_gets(s3) == []
    plain_key = job_hash(wrds_job(), service.wrds.runner.data_version(), service.wrds.code_version)

    limited = wrds_job(liquidity_limit=10, initial_aum=5e6)
    response = service.run(limited)
    reads = len(_daily_gets(s3))
    assert reads == 1 + 5                                       # the manifest and 2009-2013
    service.run(wrds_job(liquidity_limit=5, initial_aum=5e6))
    assert len(_daily_gets(s3)) == reads                        # kept in the container

    # A job without a limit hashes as before; one with a limit also depends on the daily manifest
    assert plain_key == job_hash(wrds_job(), service.wrds.runner.data_version(), service.wrds.code_version)
    assert DAILY_MANIFEST in response['data_version']
    key = response['hash']
    s3.objects[f'{PREFIX}/{DEFAULT_WRDS_DIR}/{DAILY_MANIFEST}'] = (b'{"years": {"2009": {}, "2010": {}, '
                                                                   b'"2011": {}, "2012": {}, "2013": {}}, "v": 2}', {})
    assert service.run(limited)['hash'] != key


# ---------------------------------------------------------------------------
# The rule: min(target, limit x volume), the rest in cash
# ---------------------------------------------------------------------------

def _held():
    """One date's Mixing positions: stock 1 in both sleeves, weights summing to 1."""
    return pd.DataFrame({'sleeve': [0, 0, 1, 1], 'permno': [1, 2, 1, 3], 'weight': [0.25, 0.25, 0.25, 0.25]})


def test_a_capped_position_equals_the_limit_and_the_rest_is_cash():
    value = 1_000_000.0
    volume = pd.Series({1: 2_000_000.0, 2: 1e12, 3: 1e12})       # stock 1: 10% of $2M = $200k < its $500k
    positions, capped, no_volume, cash = cap_positions(_held(), volume, value, 10.0)
    by_stock = positions.groupby('permno')['weight'].sum()
    assert by_stock[1] * value == pytest.approx(0.10 * 2_000_000.0)     # exactly the limit
    assert (by_stock[2], by_stock[3]) == (0.25, 0.25)                   # untouched, not reweighted
    assert positions.loc[positions['permno'] == 1, 'weight'].tolist() == pytest.approx([0.1, 0.1])  # sleeves cut alike
    assert (capped, no_volume) == (1, 0)
    assert cash == pytest.approx(0.5 - 0.2)
    assert cash + by_stock.sum() == pytest.approx(1.0)


def test_a_stock_without_volume_has_no_position_and_its_money_is_cash():
    volume = pd.Series({1: 1e12, 2: np.nan, 3: 0.0})
    positions, capped, no_volume, cash = cap_positions(_held(), volume, 1_000_000.0, 10.0)
    assert set(positions['permno']) == {1}
    assert positions['weight'].sum() == 0.5
    assert (capped, no_volume, cash) == (0, 2, 0.5)
    positions, _, no_volume, cash = cap_positions(_held(), pd.Series(dtype=float), 1_000_000.0, 10.0)
    assert positions is None and no_volume == 3 and cash == 1.0


def test_nothing_capped_leaves_the_positions_and_no_cash():
    positions, capped, no_volume, cash = cap_positions(_held(), pd.Series({1: 1e12, 2: 1e12, 3: 1e12}), 1e6, 5.0)
    pd.testing.assert_frame_equal(positions, _held().astype({'weight': float}))
    assert (capped, no_volume, cash) == (0, 0, 0.0)


def _runner(s3):
    return BacktestService(S3DataManager(bucket='test-bucket', prefix=PREFIX, client=s3)).wrds.runner


def test_small_aum_is_unchanged_from_no_limit(s3):
    runner = _runner(s3)
    plain, plain_holdings = runner.run(wrds_job(initial_aum=1000.0))
    limited, holdings = runner.run(wrds_job(initial_aum=1000.0, liquidity_limit=5))
    summary = limited.pop('liquidity_limit')
    assert summary['cash'] == [0.0] * len(summary['cash']) and sum(summary['capped']) == 0
    limited.pop('factor_risk'), plain.pop('factor_risk')
    assert limited == plain
    pd.testing.assert_frame_equal(holdings.drop(columns=['dollar_volume', 'volume_share']), plain_holdings)


def test_positions_fit_the_limit_and_the_cash_earns_rf(s3):
    results, holdings = _runner(s3).run(wrds_job(liquidity_limit=10, initial_aum=5e7))
    summary = results['liquidity_limit']
    assert holdings['volume_share'].max() <= 0.10 + 1e-9
    assert sum(summary['capped']) > 0 and summary['average_cash'] > 0
    assert summary['max_cash'] == max(summary['cash'])
    assert summary['max_cash_date'] == summary['rebalance_dates'][summary['cash'].index(summary['max_cash'])]
    assert set(FACTORS) <= set(holdings.columns)
    # Each period: the stocks' weights x growth, plus the cash at rf (0 bps, so no cost)
    values = results['portfolio_values']
    for i, date in enumerate(summary['rebalance_dates']):
        rows = holdings[holdings['rebalance_date'] == pd.Timestamp(date)]
        assert rows['weight'].sum() + summary['cash'][i] == pytest.approx(1.0)
        rf = results['rf_returns'][i] / 100
        expected = (rows['weight'] * (1 + rows['period_return'])).sum() + summary['cash'][i] * (1 + rf) - 1
        assert values[i + 1] / values[i] - 1 == pytest.approx(expected, rel=1e-9, abs=1e-12)
    # Monthly returns compound to the period returns
    monthly = np.array(results['monthly_returns']) / 100
    assert np.prod(1 + monthly) == pytest.approx(values[-1] / values[0], rel=1e-9)
