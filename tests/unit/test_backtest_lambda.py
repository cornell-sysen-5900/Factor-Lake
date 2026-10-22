"""
Tests for running backtests in AWS Lambda (story: "lambda functions").

The Lambda code (src/backtest_service.py) runs in-process against a fake S3, and
the app's Lambda client calls it through a fake Lambda client, so the whole path
app -> Lambda -> S3 cache is exercised without AWS.

Definition of done covered here:
- Results match the current version for the same inputs: a Lambda run in the app
  equals a local run (Load Market Data + Run) for both factor combinations, every
  delisting strategy, and the cohort comparison.
- A repeated run returns the cached result without rerunning the backtest.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from streamlit.testing.v1 import AppTest

import src.backtest_service as backtest_service
from src.backtest_engine import rebalance_portfolio, run_cohort_comparison
from src.backtest_jobs import (build_job, code_version, decode_result, encode_result, job_hash,
                               results_match, run_job)
from src.backtest_service import BacktestService, result_file
from src.lambda_client import BacktestLambdaClient, BacktestLambdaError
from src.s3_client import DELISTING_FILE, UNIVERSE_FILE, S3DataManager
from src.universe_selection import select_universe
from tests.unit.test_saved_runs import (_assert_clean, _click, _click_main_tab, _open_tab,
                                        browser_like_widget_states)

APP_DIR = Path(__file__).resolve().parents[2] / 'app'
BUCKET, PREFIX = 'test-bucket', 'factor-lake-data'
FACTORS = ['ROE_using_9-30_Data', 'ROA', '6-Mo_Momentum']
LABELS = {'ROE_using_9-30_Data': 'ROE using 9/30 Data', 'ROA': 'ROA', '6-Mo_Momentum': '6-Mo Momentum %'}


# ---------------------------------------------------------------------------
# Fakes and data
# ---------------------------------------------------------------------------

class FakeS3:
    """
    The S3 calls the service makes, on an in-memory store. can_list=False behaves like
    a role without s3:ListBucket: missing objects give AccessDenied instead of NoSuchKey.
    """

    def __init__(self, can_list=True):
        self.objects = {}   # key -> (body, metadata)
        self.puts = []
        self.can_list = can_list

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey' if self.can_list else 'AccessDenied'}}, 'GetObject')
        body, metadata = self.objects[Key]
        return {'Body': io.BytesIO(body), 'Metadata': dict(metadata)}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        body = self.objects[Key][0]
        return {'ETag': f'"{hashlib.sha256(body).hexdigest()[:32]}"', 'ContentLength': len(body)}

    def put_object(self, Bucket, Key, Body, ContentType=None, Metadata=None):
        self.objects[Key] = (bytes(Body), dict(Metadata or {}))
        self.puts.append(Key)

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://fake-s3.example/{Params['Key']}?expires={ExpiresIn}"


class FakeLambda:
    """Invokes the service like AWS Lambda would, JSON in and JSON out."""

    def __init__(self, service):
        self.service = service
        self.invocations = 0

    def invoke(self, FunctionName, InvocationType, Payload):
        self.invocations += 1
        response = self.service.handle_event(json.loads(Payload))
        return {'StatusCode': 200, 'Payload': io.BytesIO(json.dumps(response).encode())}


def _raw_universe() -> pd.DataFrame:
    """
    The universe as stored in S3: 40 tickers x 2002-2024 over 4 sectors, with some
    stocks delisting (no next-year return) and some missing factor values.
    """
    rng = np.random.RandomState(11)
    sectors = [('Technology', 'Software'), ('Energy', 'Coal'), ('Healthcare', 'Biotechnology'),
               ('Financials', 'Banks')]
    rows = []
    for year in range(2002, 2025):
        for i in range(40):
            sector, industry = sectors[i % 4]
            rows.append({
                'Ticker-Region': f'T{i:02d}-US',
                'Security_Name': f'Company {i:02d}',
                'Date': f'{year}-09-30',
                'Ending_Price': rng.uniform(5, 120),
                'Next-Years_Return': np.nan if rng.rand() < 0.05 else rng.normal(8, 25),
                'ROE_using_9-30_Data': np.nan if rng.rand() < 0.1 else rng.normal(0.1, 0.05),
                'ROA': rng.normal(0.05, 0.03),
                '6-Mo_Momentum': rng.normal(3, 15),
                'Market_Capitalization': rng.uniform(1e8, 5e9),
                'Scotts_Sector_5': sector,
                'FactSet_Industry': industry,
            })
    return pd.DataFrame(rows)


def _delisting() -> pd.DataFrame:
    return pd.DataFrame({'ticker': [f'T{i:02d}-US' for i in range(0, 40, 3)],
                         'last_date': ['2015-03-31'] * 14, 'last_price': np.linspace(1, 30, 14)})


def _parquet(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


@pytest.fixture
def s3():
    fake = FakeS3()
    fake.put_object(BUCKET, f'{PREFIX}/{UNIVERSE_FILE}', _parquet(_raw_universe()))
    fake.put_object(BUCKET, f'{PREFIX}/{DELISTING_FILE}', _parquet(_delisting()))
    fake.puts.clear()
    return fake


@pytest.fixture
def manager(s3):
    return S3DataManager(bucket=BUCKET, prefix=PREFIX, client=s3)


@pytest.fixture
def service(manager):
    return BacktestService(manager)


def _settings(**overrides):
    settings = {'start_year': 2004, 'end_year': 2020, 'initial_aum': 1000.0, 'use_market_cap_weight': False,
                'delisting_strategy': 'zero_return', 'factor_combination': 'mixing',
                'restrict_fossil_fuels': False, 'selected_sectors': None}
    settings.update(overrides)
    return settings


def _job(kind='backtest', cohort_pct=None, factors=FACTORS, **overrides):
    directions = {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom', '6-Mo_Momentum': 'top'}
    return build_job(kind, factors, directions, _settings(**overrides), factor_labels=LABELS, cohort_pct=cohort_pct)


# ---------------------------------------------------------------------------
# Jobs: inputs, hashing, serialization
# ---------------------------------------------------------------------------

def test_equal_inputs_give_the_same_hash():
    a = _job(selected_sectors=['Energy', 'Technology'], initial_aum=1000)
    b = _job(selected_sectors=['Technology', 'Energy'], initial_aum=1000.0)   # order and int/float don't matter
    assert a == b
    assert job_hash(a, 'data-v1') == job_hash(b, 'data-v1')


@pytest.mark.parametrize('change', [
    {'initial_aum': 2000.0}, {'start_year': 2005}, {'end_year': 2019}, {'factor_combination': 'integrating'},
    {'delisting_strategy': 'hold_cash'}, {'use_market_cap_weight': True}, {'restrict_fossil_fuels': True},
    {'selected_sectors': ['Energy']}, {'factors': ['ROA', 'ROE_using_9-30_Data', '6-Mo_Momentum']},
])
def test_any_input_change_gives_a_new_hash(change):
    assert job_hash(_job(), 'data-v1') != job_hash(_job(**change), 'data-v1')


def test_data_and_code_versions_are_part_of_the_hash():
    job = _job()
    assert job_hash(job, 'data-v1') != job_hash(job, 'data-v2')
    assert job_hash(job, 'data-v1', 'code-a') != job_hash(job, 'data-v1', 'code-b')
    assert len(code_version()) == 16


@pytest.mark.parametrize('bad, message', [
    ({'initial_aum': 0}, 'Initial AUM'),
    ({'factor_combination': 'blending'}, 'factor_combination'),
    ({'delisting_strategy': 'sell'}, 'delisting_strategy'),
])
def test_invalid_jobs_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        _job(**bad)


def test_results_survive_the_json_round_trip(manager):
    result = run_job(manager.fetch_all_data(), _job(delisting_strategy='reinvest'))
    decoded = decode_result('backtest', encode_result(result))
    assert results_match(decoded, result)
    assert list(decoded['holdings_by_year']) == list(result['holdings_by_year'])   # int year keys again
    assert all(isinstance(year, int) for year in decoded['holdings_by_year'])


def test_tied_holdings_are_ordered_by_ticker_whatever_the_float_noise():
    """Seen on AWS: Mixing sleeve leaders tie in score, and the last float bit differs by machine."""
    import src.backtest_engine as engine
    from src.portfolio import Portfolio
    df = pd.DataFrame({'Ending_Price': [10.0, 20.0], 'Next-Years_Return': [5.0, 7.0], 'A': [1.0, 2.0],
                       'B': [3.0, 4.0]}, index=pd.Index(['VGR', 'USNA'], name='Ticker'))
    orders = []
    for noise in (1e-15, -1e-15):                          # VGR's score a hair above, then below, USNA's
        sleeves = [Portfolio('A'), Portfolio('B')]
        sleeves[0].scores = pd.Series({'VGR': 2.7646355838833925 + noise})
        sleeves[1].scores = pd.Series({'USNA': 2.7646355838833925})
        positions = [{'sleeve': 0, 'ticker': 'VGR', 'shares': 1.0, 'price': 10.0, 'value': 10.0,
                      'return': 0.05, 'delisted': False},
                     {'sleeve': 1, 'ticker': 'USNA', 'shares': 1.0, 'price': 20.0, 'value': 20.0,
                      'return': 0.07, 'delisted': False}]
        rows = engine._build_holdings_rows(sleeves, positions, df, ['A', 'B'], 'mixing', {})
        orders.append([r['Ticker'] for r in rows])
    assert orders == [['USNA', 'VGR'], ['USNA', 'VGR']]


def test_run_job_matches_the_app_filtering_and_engine(manager):
    raw = manager.fetch_all_data()
    job = _job(restrict_fossil_fuels=True, selected_sectors=['Energy', 'Technology', 'Financials'],
               factor_combination='integrating', delisting_strategy='hold_cash')
    data = select_universe(raw, 2004, 2020, ['Energy', 'Technology', 'Financials'], True)
    expected = rebalance_portfolio(
        data, FACTORS, factor_directions=job['factor_directions'], start_year=2004, end_year=2020,
        initial_aum=1000.0, delisting_strategy='hold_cash', factor_combination='integrating',
        factor_labels=LABELS)
    assert results_match(run_job(raw, job), expected)


# ---------------------------------------------------------------------------
# The Lambda service: S3 cache, data version, response size
# ---------------------------------------------------------------------------

def test_first_run_computes_and_saves_then_repeats_come_from_the_cache(service, s3, monkeypatch):
    first = service.run(_job())
    assert first['status'] == 'ok' and first['cached'] is False
    saved_key = f'{PREFIX}/{result_file(first["hash"])}'
    assert s3.puts == [saved_key]
    assert s3.objects[saved_key][1]['computed_at'] == first['computed_at']

    def must_not_run(*args, **kwargs):
        raise AssertionError("the backtest ran again")
    monkeypatch.setattr(backtest_service, 'run_job', must_not_run)

    second = service.run(_job())
    assert second['cached'] is True
    assert second['hash'] == first['hash']
    assert second['computed_at'] == first['computed_at']
    assert second['result'] == first['result']
    assert s3.puts == [saved_key]                          # nothing new written


def test_cache_works_for_a_role_without_list_bucket(s3, monkeypatch):
    s3.can_list = False                                    # like the class's Sysen5900LambdaExecRole
    service = BacktestService(S3DataManager(bucket=BUCKET, prefix=PREFIX, client=s3))
    first = service.run(_job())
    assert first['cached'] is False and len(s3.puts) == 1
    monkeypatch.setattr(backtest_service, 'run_job', lambda *a: pytest.fail("the backtest ran again"))
    assert service.run(_job())['cached'] is True


def test_other_cache_read_errors_are_not_hidden(service, s3, monkeypatch):
    def broken_get(Bucket, Key):
        raise ClientError({'Error': {'Code': 'SlowDown'}}, 'GetObject')
    monkeypatch.setattr(s3, 'get_object', broken_get)
    assert service.handle_event(_job())['error_type'] == 'ClientError'


def test_a_new_lambda_container_reuses_results_saved_by_another(manager):
    first = BacktestService(manager).run(_job())
    again = BacktestService(manager).run(_job())           # cold start: no memory of the first run
    assert again['cached'] is True and again['result'] == first['result']


def test_republished_data_gives_fresh_results(service, s3):
    first = service.run(_job())
    s3.put_object(BUCKET, f'{PREFIX}/{UNIVERSE_FILE}', _parquet(_raw_universe().iloc[:-40]))  # drop a year's rows
    second = service.run(_job())
    assert second['cached'] is False and second['hash'] != first['hash']
    assert second['data_version'] != first['data_version']


def test_missing_delisting_file_is_allowed_and_versioned(manager, s3):
    with_file = BacktestService(manager).data_version()
    del s3.objects[f'{PREFIX}/{DELISTING_FILE}']
    without = BacktestService(manager).data_version()
    assert without != with_file and without.endswith(f'{DELISTING_FILE}:missing')
    assert BacktestService(manager).run(_job())['status'] == 'ok'


def test_results_over_the_inline_limit_are_returned_as_an_s3_link(manager, s3, monkeypatch):
    small = BacktestService(manager).run(_job())
    service = BacktestService(manager, inline_limit=small['size_bytes'] - 1)
    response = service.run(_job())
    assert 'result' not in response
    assert response['result_url'].startswith(f'https://fake-s3.example/{PREFIX}/{result_file(response["hash"])}')

    # The app's client downloads it from the link and gets the same result
    monkeypatch.setattr('src.lambda_client._download',
                        lambda url: s3.objects[url.split('example/')[1].split('?')[0]][0])
    result, info = BacktestLambdaClient('fn', client=FakeLambda(service)).run(_job())
    assert info['delivery'] == 's3_link'
    assert results_match(result, decode_result('backtest', small['result']))


def test_errors_come_back_as_messages(service):
    assert service.handle_event({'kind': 'backtest'})['error_type'] == 'invalid_request'
    bad = dict(_job(), factors=['No_Such_Factor'], factor_directions={'No_Such_Factor': 'top'})
    response = service.handle_event(bad)
    assert response['status'] == 'error' and 'No_Such_Factor' in response['error']
    assert service.handle_event({'action': 'ping'}) == {'status': 'ok', 'code_version': code_version()}


def test_lambda_handler_uses_one_service_per_container(monkeypatch, service):
    import aws_lambda.backtest_handler as handler_module
    monkeypatch.setattr(handler_module, '_service', service)
    assert handler_module.handler({'action': 'ping'}, None)['status'] == 'ok'


# ---------------------------------------------------------------------------
# The app's Lambda client
# ---------------------------------------------------------------------------

def test_client_returns_the_result_and_how_it_was_made(service):
    client = BacktestLambdaClient('factor-lake-backtest', client=FakeLambda(service))
    result, info = client.run(_job())
    assert info['cached'] is False and info['delivery'] == 'inline'
    assert set(info) >= {'hash', 'computed_at', 'compute_seconds', 'size_bytes', 'round_trip_seconds'}
    assert results_match(result, run_job(service.manager.fetch_all_data(), _job()))
    assert client.run(_job())[1]['cached'] is True


def test_client_raises_readable_errors(service):
    client = BacktestLambdaClient('fn', client=FakeLambda(service))
    bad = dict(_job(), factors=['No_Such_Factor'], factor_directions={'No_Such_Factor': 'top'})
    with pytest.raises(BacktestLambdaError, match='could not run this backtest'):
        client.run(bad)

    class Crashing:
        def invoke(self, **kwargs):
            return {'FunctionError': 'Unhandled',
                    'Payload': io.BytesIO(b'{"errorMessage": "Task timed out after 600.00 seconds"}')}
    with pytest.raises(BacktestLambdaError, match='timed out'):
        BacktestLambdaClient('fn', client=Crashing()).run(_job())


# ---------------------------------------------------------------------------
# scripts/check_backtest_lambda.py (run against the deployed Lambda in the demo)
# ---------------------------------------------------------------------------

class SkewedLambda(FakeLambda):
    """A Lambda whose final values are off by 0.1%."""

    def invoke(self, FunctionName, InvocationType, Payload):
        response = json.loads(super().invoke(FunctionName, InvocationType, Payload)['Payload'].read())
        result = response.get('result', {})
        if 'final_value' in result:
            result['final_value'] *= 1.001
        return {'StatusCode': 200, 'Payload': io.BytesIO(json.dumps(response).encode())}


@pytest.mark.parametrize('lambda_class, exit_code', [(FakeLambda, 0), (SkewedLambda, 1)])
def test_check_script_compares_lambda_with_local(monkeypatch, manager, capsys, lambda_class, exit_code):
    import scripts.check_backtest_lambda as check
    fake = lambda_class(BacktestService(manager))
    monkeypatch.setattr(check, 'S3DataManager', lambda: manager)
    monkeypatch.setattr(check, 'BacktestLambdaClient', lambda name: BacktestLambdaClient(name, client=fake))
    assert check.main([]) == exit_code
    out = capsys.readouterr().out
    assert out.count('cached=True') == 4                        # every repeat came from the cache
    assert ('All checks passed.' in out) == (exit_code == 0)


# ---------------------------------------------------------------------------
# Deploy script helpers
# ---------------------------------------------------------------------------

def test_lifecycle_rule_is_added_without_dropping_other_rules():
    from scripts.deploy_backtest_lambda import LIFECYCLE_RULE_ID, merge_lifecycle_rules
    other = {'ID': 'archive-old-data', 'Filter': {'Prefix': 'archive/'}, 'Status': 'Enabled',
             'Transitions': [{'Days': 30, 'StorageClass': 'GLACIER'}]}
    old_ours = {'ID': LIFECYCLE_RULE_ID, 'Filter': {'Prefix': 'x/'}, 'Status': 'Enabled', 'Expiration': {'Days': 7}}
    rules = merge_lifecycle_rules([other, old_ours], PREFIX, 90)
    assert rules[0] == other
    assert rules[1] == {'ID': LIFECYCLE_RULE_ID, 'Filter': {'Prefix': f'{PREFIX}/backtest-results/'},
                        'Status': 'Enabled', 'Expiration': {'Days': 90}}


def test_role_instructions_for_an_admin_include_both_policies(capsys):
    from scripts.deploy_backtest_lambda import BASIC_EXECUTION_POLICY, main
    assert main(['--print-role-policy', '--bucket', BUCKET, '--prefix', PREFIX]) == 0
    out = capsys.readouterr().out
    assert 'lambda.amazonaws.com' in out and BASIC_EXECUTION_POLICY in out
    assert f'arn:aws:s3:::{BUCKET}/{PREFIX}/backtest-results/*' in out and '--role-arn' in out


def test_deploy_without_iam_access_explains_the_admin_step(monkeypatch, caplog):
    import boto3
    import scripts.deploy_backtest_lambda as deploy

    class DeniedIam:
        def get_role(self, **kwargs):
            raise ClientError({'Error': {'Code': 'AccessDenied', 'Message': 'explicit deny'}}, 'GetRole')

    monkeypatch.setattr(deploy, 'install_libraries', lambda target: None)   # no download in tests
    monkeypatch.setattr(boto3, 'client', lambda service, **kwargs: DeniedIam() if service == 'iam' else object())
    assert deploy.main([]) == 1
    assert '--print-role-policy' in caplog.text and '--role-arn' in caplog.text


def test_unused_library_parts_are_pruned(tmp_path):
    from scripts.deploy_backtest_lambda import prune_libraries
    keep = ['pyarrow/libarrow.so.2400', 'pyarrow/libarrow_substrait.so.2400', 'pyarrow/libparquet.so.2400',
            'pyarrow/parquet/core.py', 'pandas/core/frame.py', 'numpy/_core/multiarray.py']
    drop = ['pyarrow/libarrow_flight.so.2400', 'pyarrow/libarrow_python_flight.so',
            'pyarrow/_flight.cpython-312-x86_64-linux-gnu.so', 'pyarrow/include/arrow/api.h',
            'pyarrow/tests/test_io.py', 'pandas/tests/test_x.py', 'numpy/_core/tests/test_y.py',
            'pyarrow/lib.pyx', 'pandas/_libs/lib.pxd', 'pandas/__pycache__/x.cpython-312.pyc']
    for name in keep + drop:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b'x')
    prune_libraries(tmp_path)
    assert all((tmp_path / name).exists() for name in keep)
    assert not any((tmp_path / name).exists() for name in drop)


def test_package_over_the_lambda_limit_is_refused(tmp_path, monkeypatch):
    import scripts.deploy_backtest_lambda as deploy
    (tmp_path / 'big.so').write_bytes(b'0' * 2048)
    monkeypatch.setattr(deploy, 'LAMBDA_UNZIPPED_LIMIT', 1024)
    with pytest.raises(SystemExit, match='250 MB'):
        deploy.build_package(tmp_path)


def test_lambda_package_has_the_engine_but_not_supabase():
    import zipfile
    from scripts.deploy_backtest_lambda import build_package
    names = zipfile.ZipFile(io.BytesIO(build_package())).namelist()
    assert 'aws_lambda/backtest_handler.py' in names and 'src/backtest_engine.py' in names
    assert 'src/supabase_client.py' not in names
    assert not any('streamlit' in n or n.startswith('app/') for n in names)


def test_lambda_code_does_not_import_streamlit_or_supabase():
    """The Lambda has no streamlit or supabase package; importing them would fail there."""
    import subprocess  # nosec B404 - runs this Python on a fixed script
    import sys
    code = ("import sys, builtins; real = builtins.__import__\n"
            "def guard(name, *a, **k):\n"
            "    if name.split('.')[0] in ('streamlit', 'supabase'): raise ImportError(name)\n"
            "    return real(name, *a, **k)\n"
            "builtins.__import__ = guard\n"
            "import aws_lambda.backtest_handler, src.backtest_service\n")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)  # nosec B603
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# The app with Lambda backtests on
# ---------------------------------------------------------------------------

@pytest.fixture
def lambda_app(monkeypatch, manager):
    """The real app with FACTOR_LAKE_BACKTEST_LAMBDA set and Lambda served by the in-process service."""
    import streamlit as st
    st.cache_resource.clear()
    fake_lambda = FakeLambda(BacktestService(manager))
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', 'factor-lake-backtest')
    monkeypatch.setattr('src.lambda_client.BacktestLambdaClient.__init__',
                        _client_init_with(fake_lambda))
    monkeypatch.syspath_prepend(str(APP_DIR))
    browser_like_widget_states(monkeypatch)
    at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    at.run()
    at.fake_lambda = fake_lambda
    at.universe = manager.fetch_all_data()
    yield at
    st.cache_resource.clear()


def _client_init_with(fake_lambda):
    def init(self, function_name=None, region=None, client=None):
        self.function_name = function_name
        self.client = fake_lambda
    return init


def _add(at, name):
    from app.streamlit_config import FACTOR_METADATA
    at.button(key=f"factor_add_{FACTOR_METADATA[name]['key']}").click().run()


def _sidebar_radio(at, label, value):
    next(r for r in at.sidebar.radio if r.label == label).set_value(value).run()


def _run(at):
    _click_main_tab(at, 'Analysis')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    return at.session_state['saved_runs'][0]


@pytest.mark.parametrize('combination, strategy, weighting', [
    ('Mixing', 'Zero Return', 'Equal Weight'),
    ('Integrating', 'Hold Cash', 'Market Cap Weight'),
    ('Mixing', 'Reinvest', 'Market Cap Weight'),
    ('Integrating', 'Reinvest', 'Equal Weight'),
])
def test_lambda_results_match_the_app_running_it_locally(lambda_app, combination, strategy, weighting):
    at = lambda_app
    _add(at, 'ROE using 9/30 Data')
    _add(at, 'ROA %')
    _add(at, '6-Mo Momentum %')
    next(r for r in at.sidebar.radio if r.key == 'factor_combination_input').set_value(combination).run()
    _sidebar_radio(at, 'How to treat delisted positions:', strategy)
    _sidebar_radio(at, 'Select weighting method:', weighting)

    # Lambda: no market data is loaded in the app
    lambda_run = _run(at)
    assert lambda_run['data'] is None and lambda_run['compute']['cached'] is False
    assert at.session_state['raw_data'] is None
    assert at.fake_lambda.invocations == 1

    # The same inputs run locally, the way the app worked before this story
    at.toggle(key='use_lambda_input').set_value(False).run()
    at.session_state['raw_data'] = at.universe
    _click_main_tab(at, 'Analysis')
    _click(at, 'Load Market Data')
    local_run = _run(at)
    assert local_run['compute'] is None and at.fake_lambda.invocations == 1

    assert lambda_run['settings']['factor_combination'] == combination.lower()
    assert results_match(lambda_run['results'], local_run['results'])
    assert lambda_run['results']['holdings_by_year']                    # the holdings table has rows


def test_repeated_lambda_run_returns_the_cached_result(lambda_app):
    at = lambda_app
    _add(at, 'ROE using 9/30 Data')
    first = _run(at)
    second = _run(at)
    assert first['compute']['cached'] is False
    assert second['compute']['cached'] is True
    assert second['compute']['hash'] == first['compute']['hash']
    assert results_match(first['results'], second['results'])
    assert any('result cache' in s.value for s in at.success)               # the user is told
    assert any('Computed:** Returned from the AWS Lambda result cache' in c.value for c in at.caption)


def test_lambda_cohort_comparison_matches_local(lambda_app):
    at = lambda_app
    _add(at, 'ROE using 9/30 Data')
    run = _run(at)
    _open_tab(at, run['label'])
    at.button(key=f"cohort_btn_{run['id']}").click().run()
    _assert_clean(at)

    cohort = next(r for r in at.session_state['saved_runs'] if r['id'] == run['id'])['cohort']
    s = run['settings']
    data = select_universe(at.universe, s['start_year'], s['end_year'])
    top, bottom = run_cohort_comparison(data, run['factors'], run['factor_directions'], 10, s)
    assert results_match(cohort['top'], list(top)) and results_match(cohort['bottom'], list(bottom))
    assert cohort['compute']['cached'] is False
    assert at.fake_lambda.invocations == 2


def test_lambda_errors_are_shown_and_nothing_is_saved(lambda_app, monkeypatch):
    at = lambda_app

    def broken(raw, job):
        raise RuntimeError('out of memory')
    monkeypatch.setattr(backtest_service, 'run_job', broken)
    _add(at, 'ROE using 9/30 Data')
    _click(at, 'Run Portfolio Analysis')
    assert any('Backtest Execution Error' in e.value and 'out of memory' in e.value for e in at.error)
    assert at.session_state['saved_runs'] == []
