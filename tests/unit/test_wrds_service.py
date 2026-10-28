"""
Tests for loading the WRDS data (src/wrds_data.py) and the Lambda-side runner
(src/wrds_service.py), with the WRDS files in a local folder and in a fake S3.
"""
import io

import numpy as np
import pandas as pd
import pytest

from src.s3_client import S3DataManager
from src.wrds_data import (DEFAULT_WRDS_DIR, MARKET_FILE, PANEL_FILE, RETURNS_FILE, read_local_wrds_data,
                           read_s3_wrds_data, wrds_dir)
from src.wrds_engine import run_wrds_backtest
from src.wrds_service import WrdsBacktestRunner
from tests.unit.test_backtest_lambda import FakeS3
from tests.unit.wrds_fixtures import synthetic_frames

PREFIX = 'factor-lake-data'
JOB = {'factors': ['f_roe', 'mom_12_1'], 'factor_directions': {'f_roe': 'top', 'mom_12_1': 'bottom'},
       'factor_labels': {'f_roe': 'ROE', 'mom_12_1': '12-1 Mo Momentum'}, 'frequency': 'quarterly',
       'start_year': 2010, 'end_year': 2012, 'initial_aum': 1000.0, 'use_market_cap_weight': True,
       'factor_combination': 'mixing', 'restrict_fossil_fuels': True, 'selected_sectors': None}


def _file_bytes(frames):
    """The three files as Parquet, with extra columns the loader must not read."""
    panel = frames.panel.assign(ret_next_1m=0.5, delisted_next_month=False, security_extra='x')
    returns = frames.returns.assign(ticker='T', prc=1.0)
    market = frames.market.assign(month=frames.market['month_end'].dt.strftime('%Y-%m'), smb=0.0)
    out = {}
    for name, df in ((PANEL_FILE, panel), (RETURNS_FILE, returns), (MARKET_FILE, market)):
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        out[name] = buffer.getvalue()
    return out


class CountingS3(FakeS3):
    def __init__(self):
        super().__init__()
        self.gets = []

    def get_object(self, Bucket, Key):
        self.gets.append(Key)
        return super().get_object(Bucket, Key)


@pytest.fixture
def frames():
    return synthetic_frames()


@pytest.fixture
def s3(frames):
    s3 = CountingS3()
    for name, body in _file_bytes(frames).items():
        s3.objects[f'{PREFIX}/{DEFAULT_WRDS_DIR}/{name}'] = (body, {})
    return s3


@pytest.fixture
def manager(s3):
    return S3DataManager(bucket='bucket', prefix=PREFIX, client=s3)


@pytest.fixture
def local(frames, tmp_path):
    for name, body in _file_bytes(frames).items():
        (tmp_path / name).write_bytes(body)
    return tmp_path


def test_local_and_s3_give_the_same_data_with_only_the_needed_columns(local, manager):
    a, b = read_local_wrds_data(local), read_s3_wrds_data(manager)
    pd.testing.assert_frame_equal(a.panel, b.panel)
    pd.testing.assert_frame_equal(a.market, b.market)
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.delisted, b.delisted)
    assert not {'ret_next_1m', 'delisted_next_month', 'security_extra'} & set(a.panel.columns)
    assert list(a.market.columns) == ['iwm_total_ret', 'iwn_total_ret', 'iwo_total_ret', 'rf', 'tbill_3m']


def test_return_matrices_cover_every_month_of_either_file(frames, tmp_path):
    # returns_monthly starts a month before market_monthly in the real files
    frames.market = frames.market.iloc[1:]
    for name, body in _file_bytes(frames).items():
        (tmp_path / name).write_bytes(body)
    data = read_local_wrds_data(tmp_path)
    first = frames.returns['month_end'].min()
    assert first < frames.market['month_end'].min() and data.months[0] == first
    assert np.isnan(data.market.loc[first, 'iwm_total_ret'])          # no market data that month
    assert data.returns.shape == (len(data.months), frames.returns['permno'].nunique())


def test_wrds_folder_can_be_overridden(monkeypatch):
    assert wrds_dir() == DEFAULT_WRDS_DIR
    monkeypatch.setenv('FACTOR_LAKE_WRDS_DIR', '/wrds_2027_04/')
    assert wrds_dir() == 'wrds_2027_04'


def test_runner_gives_what_the_engine_gives_on_the_same_files(manager, local):
    results, holdings = WrdsBacktestRunner(manager).run(JOB)
    args = {k: v for k, v in JOB.items() if k not in ('factors', 'factor_directions')}
    expected, expected_holdings = run_wrds_backtest(read_local_wrds_data(local), JOB['factors'],
                                                    JOB['factor_directions'], **args)
    assert results == expected
    pd.testing.assert_frame_equal(holdings, expected_holdings)
    assert set(holdings['picked_by']) <= {'ROE', '12-1 Mo Momentum', 'ROE, 12-1 Mo Momentum'}


def test_runner_reuses_the_data_until_a_file_is_republished(manager, s3, frames):
    runner = WrdsBacktestRunner(manager)
    runner.run(JOB)
    runner.run(dict(JOB, frequency='monthly'))
    assert len(s3.gets) == 3                              # each file read once
    version = runner.data_version()
    assert version.startswith(f'{DEFAULT_WRDS_DIR}|{PANEL_FILE}:')

    frames.market['iwm_total_ret'] += 0.01                # republish one file
    s3.objects[f'{PREFIX}/{DEFAULT_WRDS_DIR}/{MARKET_FILE}'] = (_file_bytes(frames)[MARKET_FILE], {})
    assert runner.data_version() != version
    runner.run(JOB)
    assert len(s3.gets) == 6


def test_runner_refuses_incomplete_or_invalid_jobs(manager):
    runner = WrdsBacktestRunner(manager)
    with pytest.raises(ValueError, match='missing'):
        runner.run({k: v for k, v in JOB.items() if k != 'frequency'})
    with pytest.raises(ValueError, match='Not WRDS factors'):
        runner.run(dict(JOB, factors=['ret_next_1m'], factor_directions={'ret_next_1m': 'top'}))
