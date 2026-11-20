"""Preview contract, exact filtering, release gate and bounded storage (synthetic data only)."""
import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from scripts.build_wrds_preview import main as build_main
from src.backtest_service import BacktestService
from src.lambda_client import BacktestLambdaClient, BacktestLambdaError
from src.s3_client import S3DataManager
from src.wrds_data import PANEL_FILE, wrds_dir
from src.wrds_gics import GICS_LEVELS, gics_mask, level_names
from src.wrds_preview import (
    APPROVAL_ENV,
    PREVIEW_COLUMNS,
    WrdsPreviewService,
    build_index,
    encode_index,
    index_filename,
    select_preview,
    validate_request,
)
from tests.unit.wrds_fixtures import RangeS3


def panel():
    rows = []
    for year in (2010, 2011, 2012):
        for day in (1, 2):
            for i, (sector, sub) in enumerate([('Energy', 10102010), ('Financials', 40101010),
                                               ('Financials', 40301010), ('Energy', None),
                                               ('Unclassified', None)]):
                for stock in range(8):
                    row = dict.fromkeys(PREVIEW_COLUMNS, 0.2)
                    row.update(date=pd.Timestamp(year, 1, day), permno=1000 + 10 * i + stock,
                               ticker=f'T{i}{stock}', security_name='<script>unsafe</script>', sector_name=sector,
                               gsubind=sub, siccd=1000, gsector_name=sector, ggroup_name='Group',
                               gind_name='Industry', gsubind_name='Subindustry', f_roa=float('nan'))
                    rows.append(row)
    frame = pd.DataFrame(rows)
    frame['gsubind'] = frame['gsubind'].astype('Int64')
    return frame


def request(gics=None, start=2010, end=2012):
    return {'start_year': start, 'end_year': end, 'gics_filter': gics}


@pytest.mark.parametrize('level', [None, *GICS_LEVELS])
def test_index_equals_filtering_full_panel_before_sampling_for_every_gics_code(level):
    full = panel().sample(frac=1, random_state=42)
    index = build_index([full.iloc[n:n + 17] for n in range(0, len(full), 17)])
    filters = [None] if level is None else [
        {'level': level, 'codes': [code]} for code in level_names(level)] + [
        {'level': level, 'codes': sorted(level_names(level))}]
    for gics in filters:
        result = select_preview(index, request(gics))
        expected = full[gics_mask(full, gics)].sort_values(['date', 'permno'])
        expected = expected.groupby(expected['date'].dt.year).head(5)
        assert [(r['date'], r['permno']) for r in result['rows']] == list(zip(
            expected['date'].dt.strftime('%Y-%m-%d'), expected['permno']))
        assert max(result['years'].values()) <= 5
        assert all(r['f_roa'] is None for r in result['rows'])


def test_sample_is_deterministic_bounded_and_contains_only_allowlisted_fields():
    full = panel().assign(ret_next_1m=99, secret_extra='not permitted')
    first = build_index([full])
    second = build_index([full.sample(frac=1, random_state=2)])
    pd.testing.assert_frame_equal(first, second)
    result = select_preview(first, request(start=2011, end=2011))
    assert len(result['rows']) == 5 and set(result['years']) == {'2011'}
    assert result['columns'] == list(PREVIEW_COLUMNS)
    assert 'ret_next_1m' not in json.dumps(result) and 'secret_extra' not in json.dumps(result)


@pytest.mark.parametrize('bad', [None, {}, request(start=True), request(start=2000), request(end=2027),
                                 request(start=2012, end=2010), request({'level': 'sector', 'codes': []}),
                                 {**request(), 'limit': 5000}, {**request(), 'approved': True}])
def test_invalid_requests_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_request(bad)


class PreviewS3(RangeS3):
    def get_object(self, Bucket, Key, Range=None, IfMatch=None):
        if IfMatch:
            assert self.head_object(Bucket, Key)['ETag'] == IfMatch
        return super().get_object(Bucket, Key, Range)

    def put_object(self, Bucket, Key, Body, ContentType=None, Metadata=None, IfNoneMatch=None):
        if IfNoneMatch == '*' and Key in self.objects:
            raise ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'PutObject')
        return super().put_object(Bucket, Key, Body, ContentType, Metadata)


@pytest.fixture
def store():
    s3 = PreviewS3()
    manager = S3DataManager(bucket='test', prefix='data', client=s3)
    body = io.BytesIO()
    panel().to_parquet(body, index=False)
    s3.objects[manager.object_key(f'{wrds_dir()}/{PANEL_FILE}')] = (body.getvalue(), {})
    return manager


def test_build_is_explicit_versioned_and_never_overwrites(store, capsys):
    original = dict(store.client.objects)
    assert build_main([], store) == 0 and store.client.puts == []
    assert build_main(['--upload'], store) == 0
    assert len(store.client.puts) == 1 and '/preview/v1-' in store.client.puts[0]
    assert all(store.client.objects[k] == v for k, v in original.items())
    with pytest.raises(ClientError):
        build_main(['--upload'], store)
    assert 'unsafe' not in capsys.readouterr().out


def test_gate_is_closed_by_default_and_cannot_be_overridden_by_the_request(store, monkeypatch):
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    service = BacktestService(store)
    result = service.handle_event({'action': 'wrds_preview', 'request': {**request(), 'approved': True}})
    assert result['status'] == 'disabled' and store.client.gets == []
    assert service._wrds is None and service._universe is None


def test_runtime_reads_only_one_small_index_reuses_it_and_invalidates_on_new_data(store, monkeypatch):
    build_main(['--upload'], store)
    store.client.gets.clear()
    store.client.puts.clear()
    monkeypatch.setenv(APPROVAL_ENV, 'true')
    service = WrdsPreviewService(store)
    for code in (10, 40, 35, 10):
        assert service.preview(request({'level': 'sector', 'codes': [code]}))['status'] == 'ok'
    assert len(store.client.gets) == 1 and '/preview/' in store.client.gets[0][0]
    assert store.client.puts == []
    key = store.object_key(f'{wrds_dir()}/{PANEL_FILE}')
    store.client.objects[key] = (b'new panel version', {})
    assert service.preview(request())['status'] == 'unavailable'
    assert service._index is None


def test_backend_errors_are_sanitized_and_no_backtest_is_created(store, monkeypatch):
    monkeypatch.setenv(APPROVAL_ENV, 'true')
    def fail(**kwargs):
        raise RuntimeError('secret credentials and internal path')
    monkeypatch.setattr(store.client, 'head_object', fail)
    service = BacktestService(store)
    result = service.handle_event({'action': 'wrds_preview', 'request': request()})
    assert result['status'] == 'error' and 'secret' not in json.dumps(result)
    assert service._wrds is None and store.client.puts == []


def test_mismatched_index_provenance_fails_closed(store, monkeypatch):
    monkeypatch.setenv(APPROVAL_ENV, 'true')
    head = store.client.head_object(store.bucket, store.object_key(f'{wrds_dir()}/{PANEL_FILE}'))
    etag = head['ETag'].strip('"')
    key = store.object_key(index_filename(wrds_dir(), etag))
    store.client.objects[key] = (encode_index(build_index([panel()]), 'wrong version'), {})
    with pytest.raises(ValueError, match='invalid'):
        WrdsPreviewService(store).preview(request())


class PreviewLambda:
    def __init__(self, service):
        self.service = service
        self.events = []

    def invoke(self, **kwargs):
        event = json.loads(kwargs['Payload'])
        self.events.append(event)
        return {'Payload': io.BytesIO(json.dumps(self.service.handle_event(event)).encode())}


def test_client_uses_preview_action_only_and_handles_disabled_and_ready(store, monkeypatch):
    fake = PreviewLambda(BacktestService(store))
    client = BacktestLambdaClient('dev', client=fake)
    monkeypatch.delenv(APPROVAL_ENV, raising=False)
    assert client.preview(request())['status'] == 'disabled'
    build_main(['--upload'], store)
    monkeypatch.setenv(APPROVAL_ENV, 'true')
    assert len(client.preview(request())['preview']['rows']) == 15
    assert all(e['action'] == 'wrds_preview' for e in fake.events)


def test_client_rejects_overlarge_samples_and_never_returns_raw_errors():
    payload = {'status': 'ok', 'preview': select_preview(build_index([panel()]), request())}
    payload['preview']['rows'] *= 2
    class BadLambda:
        def invoke(self, **kwargs):
            return {'Payload': io.BytesIO(json.dumps(payload).encode())}
    with pytest.raises(BacktestLambdaError):
        BacktestLambdaClient('dev', client=BadLambda()).preview(request())
