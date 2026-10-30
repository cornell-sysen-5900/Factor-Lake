"""
The post-deploy check's WRDS part (scripts/check_backtest_lambda.check_wrds): one quarterly
WRDS backtest end to end against an in-process Lambda and a fake S3 with synthetic WRDS
data. It passes on a correct Lambda and fails when the function reads another WRDS folder,
its result differs from the local run, or its holdings file does.
"""
import io
import json

import pandas as pd
import pytest

import scripts.check_backtest_lambda as check
from src.backtest_service import BacktestService
from src.lambda_client import BacktestLambdaClient
from src.wrds_data import WRDS_DIR_ENV, wrds_dir
from src.wrds_holdings import holdings_parquet
from src.wrds_jobs import build_wrds_job
from tests.unit.test_backtest_lambda import FakeLambda
from tests.unit.test_wrds_saved import manager, s3  # noqa: F401 (fixtures)
from tests.unit.wrds_fixtures import WRDS_SETTINGS


class ConfiguredLambda(FakeLambda):
    """A correct Lambda whose configuration names its WRDS folder."""
    folder = wrds_dir()

    def get_function_configuration(self, FunctionName):
        return {'Environment': {'Variables': {WRDS_DIR_ENV: self.folder}}}

    def _response(self, FunctionName, InvocationType, Payload):
        return json.loads(super().invoke(FunctionName, InvocationType, Payload)['Payload'].read())

    @staticmethod
    def _payload(response):
        return {'StatusCode': 200, 'Payload': io.BytesIO(json.dumps(response).encode())}


class OtherFolderLambda(ConfiguredLambda):
    folder = 'wrds_2025_01'


class SkewedWrdsLambda(ConfiguredLambda):
    """Returns a final value off by 1e-4 (relative)."""

    def invoke(self, FunctionName, InvocationType, Payload):
        response = self._response(FunctionName, InvocationType, Payload)
        response['result']['final_value'] *= 1.0001
        return self._payload(response)


class OtherHoldingsLambda(ConfiguredLambda):
    """Saves holdings whose newest date's last weight is off by 1%."""

    def invoke(self, FunctionName, InvocationType, Payload):
        response = self._response(FunctionName, InvocationType, Payload)
        if not response['cached']:
            manager = self.service.manager
            key = manager.object_key(response['result']['holdings']['file'])
            frame = pd.read_parquet(io.BytesIO(manager.client.objects[key][0]))
            frame.loc[frame.index[-1], 'weight'] *= 1.01
            manager.client.objects[key] = (holdings_parquet(frame)[0], {})
        return self._payload(response)


@pytest.fixture
def synthetic_job(monkeypatch):
    """The check's job on the years the synthetic WRDS data covers."""
    real = check.wrds_sample_job

    def job(unique):
        sample = real(unique)
        return build_wrds_job('backtest', sample['factors'], sample['factor_directions'],
                              dict(WRDS_SETTINGS, initial_aum=sample['initial_aum']),
                              factor_labels=sample['factor_labels'])
    monkeypatch.setattr(check, 'wrds_sample_job', job)


def test_the_checked_job_is_a_quarterly_wrds_backtest_over_the_whole_data():
    job = check.wrds_sample_job(unique=False)
    assert (job['data_mode'], job['kind'], job['frequency']) == ('wrds', 'backtest', 'quarterly')
    assert (job['start_year'], job['end_year']) == (2008, 2026)


@pytest.mark.parametrize('lambda_class, failures, failed_line', [
    (ConfiguredLambda, 0, None),
    (OtherFolderLambda, 1, '  folder: wrds_2025_01'),
    (SkewedWrdsLambda, 1, '  match:  NO'),
    (OtherHoldingsLambda, 1, 'match NO'),
])
def test_wrds_check(monkeypatch, manager, synthetic_job, capsys, lambda_class, failures, failed_line):
    fake = lambda_class(BacktestService(manager))
    monkeypatch.setattr(check, 'S3DataManager', lambda: manager)
    assert check.check_wrds(BacktestLambdaClient('factor-lake-backtest', client=fake), unique=True) == failures
    out = capsys.readouterr().out
    assert 'rows read with ranged GETs' in out and 'cached=True' in out
    if failed_line:
        assert failed_line in out
    else:
        assert 'NO' not in out
    holdings_gets = [r for k, r in manager.client.gets if k.endswith('.holdings.parquet')]
    assert holdings_gets and all(r is not None for r in holdings_gets)      # only ranged GETs, as the app
