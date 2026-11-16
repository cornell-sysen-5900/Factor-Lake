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
    read_dollar_volume,
)
from tests.unit.test_wrds_service import FACTORS, PREFIX, wrds_job
from tests.unit.wrds_fixtures import RangeS3, put_wrds_files, synthetic_frames


def daily_prices(permnos, first='2009-06-01', last='2013-03-31', seed=0):
    """Business-day prices and volumes, with missing values, a negative volume and a stock absent for a year."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(first, last)
    frames = []
    for permno in permnos:
        d = days[days.year != 2011] if permno == permnos[0] else days
        n = len(d)
        prc, vol = rng.uniform(5, 50, n), rng.uniform(1e4, 1e6, n)
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
    px = daily_prices(sorted(frames.panel['permno'].unique()))
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


def test_positions_fit_the_limit_with_volume_from_the_daily_prices(s3):
    runner = BacktestService(S3DataManager(bucket='test-bucket', prefix=PREFIX, client=s3)).wrds.runner
    results, holdings = runner.run(wrds_job(liquidity_limit=10, initial_aum=2e6))
    assert holdings['volume_share'].max() <= 0.10 + 1e-9
    assert sum(results['liquidity_limit']['dropped']) > 0
    assert set(FACTORS) <= set(holdings.columns)
