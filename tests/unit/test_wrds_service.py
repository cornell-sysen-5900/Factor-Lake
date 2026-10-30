"""
Tests for the WRDS side of the backtest Lambda: loading the WRDS data (src/wrds_data.py),
running jobs (src/wrds_service.py), the holdings Parquet (src/wrds_holdings.py) and the
handler's routing, with the WRDS files in a local folder and in a fake S3.
"""
import json

import numpy as np
import pandas as pd
import pytest

from aws_lambda import backtest_handler
from src.backtest_jobs import decode_saved, job_hash
from src.backtest_service import RESULTS_DIR, BacktestService, result_file
from src.s3_client import S3DataManager
from src.wrds_data import DEFAULT_WRDS_DIR, MARKET_FILE, read_local_wrds_data, read_s3_wrds_data, wrds_dir
from src.wrds_engine import run_wrds_backtest
from src.wrds_holdings import holdings_file, holdings_parquet, read_holdings_date
from src.wrds_jobs import build_wrds_job, engine_arguments
from src.wrds_service import WrdsBacktestRunner
from tests.unit.test_backtest_lambda import (DELISTING_FILE, UNIVERSE_FILE, _delisting, _job as legacy_job,
                                             _parquet, _raw_universe)
from tests.unit.wrds_fixtures import WRDS_SETTINGS, RangeS3, put_wrds_files, synthetic_frames, wrds_file_bytes

BUCKET, PREFIX = 'test-bucket', 'factor-lake-data'
FACTORS = ['f_roe', 'mom_12_1']
DIRECTIONS = {'f_roe': 'top', 'mom_12_1': 'bottom'}
LABELS = {'f_roe': 'ROE', 'mom_12_1': '12-1 Mo Momentum'}


def wrds_job(kind='backtest', **overrides):
    return build_wrds_job(kind, FACTORS, DIRECTIONS, dict(WRDS_SETTINGS, **overrides), factor_labels=LABELS,
                          cohort_pct=20 if kind == 'cohort' else None,
                          rebalance_date='2010-06' if kind == 'ranked' else None)


@pytest.fixture
def frames():
    return synthetic_frames()


@pytest.fixture
def s3(frames):
    """A fake S3 with the Legacy files and the WRDS files."""
    s3 = RangeS3()
    s3.objects[f'{PREFIX}/{UNIVERSE_FILE}'] = (_parquet(_raw_universe()), {})
    s3.objects[f'{PREFIX}/{DELISTING_FILE}'] = (_parquet(_delisting()), {})
    put_wrds_files(s3, frames)
    return s3


@pytest.fixture
def manager(s3):
    return S3DataManager(bucket=BUCKET, prefix=PREFIX, client=s3)


@pytest.fixture
def service(manager):
    return BacktestService(manager)


@pytest.fixture
def local(frames, tmp_path):
    for name, body in wrds_file_bytes(frames).items():
        (tmp_path / name).write_bytes(body)
    return tmp_path


def _key(service, job):
    return job_hash(job, service.wrds.runner.data_version(), service.wrds.code_version)


# ---------------------------------------------------------------------------
# Loading the data
# ---------------------------------------------------------------------------

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
    for name, body in wrds_file_bytes(frames).items():
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
    job = wrds_job(use_market_cap_weight=True, restrict_fossil_fuels=True)
    results, holdings = WrdsBacktestRunner(manager).run(job)
    factors, directions, options = engine_arguments(job)
    expected, expected_holdings = run_wrds_backtest(read_local_wrds_data(local), factors, directions, **options)
    assert results == expected
    pd.testing.assert_frame_equal(holdings, expected_holdings)
    assert set(holdings['picked_by']) <= {'ROE', '12-1 Mo Momentum', 'ROE, 12-1 Mo Momentum'}


def test_runner_reuses_the_data_until_a_file_is_republished(manager, s3, frames):
    runner = WrdsBacktestRunner(manager)
    runner.run(wrds_job())
    runner.run(wrds_job(frequency='monthly'))
    assert len(s3.gets) == 3                              # each file read once
    version = runner.data_version()
    assert version.startswith(f'{DEFAULT_WRDS_DIR}|factor_panel_monthly.parquet:')

    frames.market['iwm_total_ret'] += 0.01                # republish one file
    s3.objects[f'{PREFIX}/{DEFAULT_WRDS_DIR}/{MARKET_FILE}'] = (wrds_file_bytes(frames)[MARKET_FILE], {})
    assert runner.data_version() != version
    runner.run(wrds_job())
    assert len(s3.gets) == 6


# ---------------------------------------------------------------------------
# The service: cache, holdings, kinds
# ---------------------------------------------------------------------------

def test_backtest_saves_its_holdings_before_the_result_and_the_result_lists_the_dates(service, s3):
    job = wrds_job()
    response = service.handle_event(job)
    assert response['status'] == 'ok' and response['cached'] is False
    key = response['hash']
    assert key == _key(service, job)
    puts = [k for k in s3.puts if RESULTS_DIR in k]
    assert puts == [f'{PREFIX}/{holdings_file(RESULTS_DIR, key)}', f'{PREFIX}/{result_file(key)}']

    result = response['result']
    holdings = result['holdings']
    assert holdings['file'] == holdings_file(RESULTS_DIR, key)
    assert holdings['dates'] == result['rebalance_dates'] and len(holdings['rows']) == len(holdings['dates'])
    assert holdings['bytes'] == len(s3.objects[f'{PREFIX}/{holdings["file"]}'][0])
    assert 'holdings_by_year' not in result
    assert response['size_bytes'] < 100_000                    # the result JSON stays small
    saved_job, _ = decode_saved(s3.objects[f'{PREFIX}/{result_file(key)}'][0])
    assert saved_job == job                                   # the ID alone reopens it


def test_the_same_job_again_comes_from_the_cache(service, s3):
    first = service.handle_event(wrds_job())
    puts = len(s3.puts)
    second = service.handle_event(json.loads(json.dumps(wrds_job())))
    assert second['cached'] is True and second['hash'] == first['hash']
    assert second['result'] == first['result'] and len(s3.puts) == puts


def test_a_cached_result_without_its_holdings_is_computed_again(service, s3):
    first = service.handle_event(wrds_job())
    holdings_key = f"{PREFIX}/{first['result']['holdings']['file']}"
    del s3.objects[holdings_key]
    again = service.handle_event(wrds_job())
    assert again['cached'] is False and again['hash'] == first['hash']
    assert holdings_key in s3.objects
    assert again['result']['holdings'] == first['result']['holdings']


def test_without_list_bucket_a_missing_holdings_file_still_means_recompute(manager, frames):
    s3 = RangeS3(can_list=False)
    put_wrds_files(s3, frames)
    service = BacktestService(S3DataManager(bucket=BUCKET, prefix=PREFIX, client=s3))
    original_head = s3.head_object

    def head(Bucket, Key):           # S3 answers 403 for a missing object without s3:ListBucket
        if Key not in s3.objects:
            from botocore.exceptions import ClientError
            raise ClientError({'Error': {'Code': '403'}}, 'HeadObject')
        return original_head(Bucket, Key)
    s3.head_object = head
    first = service.handle_event(wrds_job())
    del s3.objects[f"{PREFIX}/{first['result']['holdings']['file']}"]
    assert service.handle_event(wrds_job())['cached'] is False


def test_holdings_are_read_one_date_at_a_time_with_ranged_gets(service, s3, manager):
    response = service.handle_event(wrds_job(frequency='monthly', factor_combination='integrating'))
    layout = response['result']['holdings']
    size = layout['bytes']
    _, expected = WrdsBacktestRunner(manager).run(wrds_job(frequency='monthly', factor_combination='integrating'))
    for row_group in (0, 5, len(layout['dates']) - 1):
        s3.gets.clear()
        date = layout['dates'][row_group]
        table = read_holdings_date(manager, layout['file'], row_group, date)
        want = expected[expected['rebalance_date'].dt.strftime('%Y-%m-%d') == date].reset_index(drop=True)
        pd.testing.assert_frame_equal(table, want, check_categorical=False)
        assert len(table) == layout['rows'][row_group]
        assert all(r is not None for _, r in s3.gets) and len(s3.gets) <= 3          # footer, row group
        fetched = sum(len(s3.get_object(BUCKET, k, r)['Body'].read()) for k, r in list(s3.gets))
        assert fetched < size                                                         # never the whole file
    with pytest.raises(ValueError, match='is not'):
        read_holdings_date(manager, layout['file'], 0, layout['dates'][1])


def test_holdings_parquet_has_one_row_group_per_date():
    frames = synthetic_frames()
    _, holdings = run_wrds_backtest(frames.data(), FACTORS, DIRECTIONS, 'quarterly', 2010, 2012, 1000.0,
                                    first_month='2009-11', last_month='2013-02')
    body, layout = holdings_parquet(holdings)
    import io
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(io.BytesIO(body)).metadata
    assert meta.num_row_groups == len(layout['dates']) == holdings['rebalance_date'].nunique()
    assert [meta.row_group(i).num_rows for i in range(meta.num_row_groups)] == layout['rows']
    with pytest.raises(ValueError, match='sorted'):
        holdings_parquet(holdings.iloc[::-1])


def test_cohort_gives_the_strategy_and_its_opposite_without_holdings(service, s3, manager):
    response = service.handle_event(wrds_job('cohort'))
    result = response['result']
    assert set(result) == {'top', 'bottom', 'years', 'cohort_pct', 'top_cagr', 'bottom_cagr'}
    assert len(result['top']) == len(result['bottom']) == len(result['years'])
    assert not any(k.endswith('.holdings.parquet') for k in s3.puts)
    data = read_s3_wrds_data(manager)
    factors, directions, options = engine_arguments(wrds_job())
    top, _ = run_wrds_backtest(data, factors, directions, **dict(options, top_pct=20.0))
    assert result['top'] == top['portfolio_values']


def test_ranked_lists_every_stock_on_one_rebalance_date(service, manager):
    result = service.handle_event(wrds_job('ranked'))['result']
    rows = pd.DataFrame(result['rows'], columns=result['columns'])
    assert result['rebalance_date'].startswith('2010-06')
    assert {'permno', 'ticker', 'score', 'rank', 'held', 'weight', 'f_roe', 'z_f_roe', 'rank_f_roe'} <= set(rows)
    ranked = rows[rows['rank'].notna()]
    assert list(ranked['rank']) == list(range(1, len(ranked) + 1))
    assert ranked['score'].is_monotonic_decreasing
    # The held stocks are the backtest's holdings on that date
    _, holdings = WrdsBacktestRunner(manager).run(wrds_job())
    on_date = holdings[holdings['rebalance_date'].dt.strftime('%Y-%m-%d') == result['rebalance_date']]
    assert set(rows.loc[rows['held'], 'permno']) == set(on_date['permno'])


def test_invalid_wrds_jobs_come_back_as_readable_errors(service):
    response = service.handle_event(dict(wrds_job(), frequency='weekly'))
    assert response == {'status': 'error', 'error_type': 'invalid_request',
                        'error': "frequency must be one of ('semiannual', 'quarterly', 'monthly'), not 'weekly'."}


# ---------------------------------------------------------------------------
# Lambda handler routing
# ---------------------------------------------------------------------------

def test_the_handler_routes_wrds_jobs_to_wrds_and_everything_else_to_legacy(service, monkeypatch):
    monkeypatch.setattr(backtest_handler, '_service', service)
    calls = []
    monkeypatch.setattr(service.wrds, 'run', lambda job: calls.append('wrds') or {'status': 'ok'})
    original_legacy = service._load_universe
    monkeypatch.setattr(service, '_load_universe', lambda v: calls.append('legacy') or original_legacy(v))

    assert backtest_handler.handler(wrds_job(), None) == {'status': 'ok'}
    legacy = backtest_handler.handler(legacy_job(), None)
    assert legacy['status'] == 'ok' and 'holdings_by_year' in legacy['result']
    assert calls == ['wrds', 'legacy']


def test_legacy_and_wrds_results_live_side_by_side(service, s3):
    legacy = service.handle_event(legacy_job())
    wrds = service.handle_event(wrds_job())
    assert legacy['hash'] != wrds['hash']
    assert legacy['code_version'] == service.code_version != wrds['code_version']
    assert service.handle_event(legacy_job())['cached'] is True


# ---------------------------------------------------------------------------
# Result expiry covers the holdings
# ---------------------------------------------------------------------------

def test_the_result_expiry_rule_covers_the_holdings_files():
    from scripts.deploy_backtest_lambda import expiry_rule_covers, merge_lifecycle_rules
    rules = merge_lifecycle_rules([], PREFIX, 90)
    holdings_key = f'{PREFIX}/{holdings_file(RESULTS_DIR, "a" * 64)}'
    assert holdings_key.startswith(rules[0]['Filter']['Prefix'])
    assert expiry_rule_covers(rules, PREFIX)
    assert not expiry_rule_covers([dict(rules[0], Status='Disabled')], PREFIX)
    assert not expiry_rule_covers([{'ID': 'x', 'Filter': {'Prefix': 'other/'}, 'Status': 'Enabled',
                                    'Expiration': {'Days': 9}}], PREFIX)
