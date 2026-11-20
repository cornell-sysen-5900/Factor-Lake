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
    PREVIEW_ROWS,
    WrdsPreviewService,
    build_index,
    encode_index,
    index_filename,
    select_preview,
    validate_request,
)
from tests.unit.wrds_fixtures import RangeS3

CELLS = [('Energy', 10102010), ('Financials', 40101010), ('Financials', 40301010), ('Energy', None),
         ('Unclassified', None)]
UTILITIES = 55101010   # a sub-industry with only two companies, each with many rows


def panel():
    rows = []
    for year in (2010, 2011, 2012):
        for day in (1, 2):
            stocks = [(i, sector, sub, stock) for i, (sector, sub) in enumerate(CELLS) for stock in range(12)
                      # stocks 9-11 of every cell stop after 2011, so older companies fill in
                      if year < 2012 or stock < 9]
            stocks += [(len(CELLS), 'Utilities', UTILITIES, stock) for stock in range(2)]
            for i, sector, sub, stock in stocks:
                row = dict.fromkeys(PREVIEW_COLUMNS, 0.2)
                row.update(date=pd.Timestamp(year, 1, day), permno=1000 + 100 * i + stock,
                           ticker=f'T{i}{stock}', security_name='<script>unsafe</script>', sector_name=sector,
                           gsubind=sub, siccd=1000, gsector_name=sector, ggroup_name='Group',
                           gind_name='Industry', gsubind_name='Subindustry', f_roa=float('nan'))
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame['gsubind'] = frame['gsubind'].astype('Int64')
    return frame


def request(gics=None):
    return {'gics_filter': gics}


def expected_sample(full, gics):
    """Brute force on the full panel: each company's latest row in the filter, newest first."""
    rows = full[gics_mask(full, gics)].sort_values(['date', 'permno'], ascending=[False, True])
    return rows.drop_duplicates('permno').head(PREVIEW_ROWS)


@pytest.mark.parametrize('level', [None, *GICS_LEVELS])
def test_index_equals_filtering_full_panel_before_sampling_for_every_gics_code(level):
    full = panel().sample(frac=1, random_state=42)
    index = build_index([full.iloc[n:n + 17] for n in range(0, len(full), 17)])
    filters = [None] if level is None else [
        {'level': level, 'codes': [code]} for code in level_names(level)] + [
        {'level': level, 'codes': sorted(level_names(level))}]
    for gics in filters:
        result = select_preview(index, request(gics))
        expected = expected_sample(full, gics)
        assert [(r['date'], r['permno']) for r in result['rows']] == list(zip(
            expected['date'].dt.strftime('%Y-%m-%d'), expected['permno']))
        assert len(result['rows']) <= PREVIEW_ROWS
        assert all(r['f_roa'] is None for r in result['rows'])


def test_rows_are_different_companies_never_one_companys_time_series():
    index = build_index([panel()])
    # Two companies with twelve rows each: two rows, not ten rows of the same two
    rows = select_preview(index, request({'level': 'sub_industry', 'codes': [UTILITIES]}))['rows']
    assert [(r['permno'], r['date']) for r in rows] == [(1500, '2012-01-02'), (1501, '2012-01-02')]
    # Twelve companies, three gone after 2011: the nine current ones, then the lowest-permno older one
    rows = select_preview(index, request({'level': 'sub_industry', 'codes': [10102010]}))['rows']
    assert [r['permno'] for r in rows] == list(range(1000, 1010))
    assert [r['date'] for r in rows] == ['2012-01-02'] * 9 + ['2011-01-02']
    # Unfiltered: ten different companies, all on the latest date
    rows = select_preview(index, request())['rows']
    assert len(rows) == PREVIEW_ROWS == len({r['permno'] for r in rows})
    assert {r['date'] for r in rows} == {'2012-01-02'}


def test_sample_is_deterministic_bounded_and_contains_only_allowlisted_fields():
    full = panel().assign(ret_next_1m=99, secret_extra='not permitted')
    first = build_index([full])
    second = build_index([full.sample(frac=1, random_state=2)])
    pd.testing.assert_frame_equal(first, second)
    # One row per company and cell, at most PREVIEW_ROWS per cell
    assert not first.duplicated(['permno', 'sector_name', 'gsubind']).any()
    assert first.groupby(['sector_name', 'gsubind'], dropna=False).size().max() == PREVIEW_ROWS
    result = select_preview(first, request())
    assert len(result['rows']) == PREVIEW_ROWS and result['row_limit'] == PREVIEW_ROWS
    assert result['columns'] == list(PREVIEW_COLUMNS)
    assert 'ret_next_1m' not in json.dumps(result) and 'secret_extra' not in json.dumps(result)


@pytest.mark.parametrize('bad', [None, {}, request({'level': 'sector', 'codes': []}),
                                 {**request(), 'start_year': 2010, 'end_year': 2012},
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
    assert len(store.client.puts) == 1 and '/preview/v2-' in store.client.puts[0]
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
    assert len(client.preview(request())['preview']['rows']) == PREVIEW_ROWS
    assert all(e['action'] == 'wrds_preview' for e in fake.events)


def test_client_rejects_repeated_companies():
    payload = {'status': 'ok', 'preview': select_preview(build_index([panel()]), request())}
    rows = payload['preview']['rows']
    rows[1] = {**rows[0], 'date': '2011-01-02'}   # the same company twice
    class RepeatingLambda:
        def invoke(self, **kwargs):
            return {'Payload': io.BytesIO(json.dumps(payload).encode())}
    with pytest.raises(BacktestLambdaError):
        BacktestLambdaClient('dev', client=RepeatingLambda()).preview(request())


def test_client_rejects_overlarge_samples_and_never_returns_raw_errors():
    payload = {'status': 'ok', 'preview': select_preview(build_index([panel()]), request())}
    payload['preview']['rows'] *= 2
    class BadLambda:
        def invoke(self, **kwargs):
            return {'Payload': io.BytesIO(json.dumps(payload).encode())}
    with pytest.raises(BacktestLambdaError):
        BacktestLambdaClient('dev', client=BadLambda()).preview(request())


def test_legacy_report_covers_every_factor_without_promoting_similar_matches():
    from pathlib import Path

    from app.streamlit_config import FACTOR_METADATA, WRDS_FACTOR_METADATA
    from app.wrds_factor_review import REPLACEMENTS, SAME
    from scripts.legacy_wrds_mapping import report
    generated = report()
    assert Path('DOCS/LEGACY_WRDS_MATCHES.md').read_text() == generated
    rows = [line.split('|')[1:-1] for line in generated.splitlines() if line.startswith('| ')]
    rows = {r[0].strip(): [cell.strip() for cell in r[1:]] for r in rows[1:]}
    assert set(rows) == set(FACTOR_METADATA)
    names = {m['column']: name for name, m in WRDS_FACTOR_METADATA.items()}
    reviewed = {r.legacy_column: r for r in REPLACEMENTS}
    for name, meta in FACTOR_METADATA.items():
        match = reviewed.get(meta['column'])
        assert rows[name][0] == (names[match.wrds_column] if match and match.level == SAME else 'none')
