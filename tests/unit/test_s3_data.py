"""
Tests for loading market data from S3 instead of Supabase.

Uses in-memory fakes for the S3 and Supabase clients, so no AWS or Supabase
access is needed. The real-bucket check lives in tests/integration/test_s3_data.py.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError
from streamlit.testing.v1 import AppTest

from app import streamlit_utils
from scripts import publish_data_to_s3 as publisher
from src.data_standardization import validate_universe
from src.s3_client import DELISTING_FILE, UNIVERSE_FILE, S3DataError, S3DataManager
from src.supabase_client import SupabaseManager

APP_DIR = Path(__file__).resolve().parents[2] / 'app'


# ---------------------------------------------------------------------------
# Fakes and sample data
# ---------------------------------------------------------------------------

class FakeS3Client:
    """Stores objects in a dict and mimics the boto3 calls S3DataManager uses."""

    def __init__(self, raise_on_get=None):
        self.objects = {}
        self.raise_on_get = raise_on_get

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'missing'}}, 'GetObject')
        return {'Body': io.BytesIO(self.objects[(Bucket, Key)])}


class FakeSupabaseClient:
    """Serves raw rows through the client.table().select().range().execute() chain."""

    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        rows = self.tables[name]

        class _Query:
            def select(self, *_):
                return self

            def range(self, start, end):
                self._rows = rows[start:end + 1]
                return self

            def execute(self):
                return type('Response', (), {'data': self._rows})()

        return _Query()


def _raw_universe_rows():
    """Raw rows shaped like the 'Full Precision Test' table, incl. sentinels and nulls."""
    rows = []
    for year in (2002, 2003):
        for i in range(4):
            rows.append({
                'Ticker-Region': f'tk{i}-US' if i else None,
                'Date': f'{year}-09-30',
                'Ending_Price': 10.0 + i,
                'Next-Years_Return': 5.0 * i - 3,
                'ROE_using_9-30_Data': 0.1 * i,
                '12-Mo_Momentum': 1.5 * i,
                'Scotts_Sector_5': 'Technology',
                'FactSet_Industry': 'Software' if i != 2 else '--',
                'EarningsReportedLast': None if i == 3 else f'{year}-08-01',
            })
    return rows


def _raw_delisting_rows():
    return [{'ticker': 'tk1-US', 'ticker_clean': 'TK1', 'last_price': 9.5, 'last_date': '2003-05-01'}]


def _parquet_bytes(rows):
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buffer, index=False)
    return buffer.getvalue()


def _s3_with_data(include_delisting=True):
    client = FakeS3Client()
    manager = S3DataManager(bucket='bucket', prefix='factor-lake-data', client=client)
    client.objects[('bucket', manager.object_key(UNIVERSE_FILE))] = _parquet_bytes(_raw_universe_rows())
    if include_delisting:
        client.objects[('bucket', manager.object_key(DELISTING_FILE))] = _parquet_bytes(_raw_delisting_rows())
    return manager


# ---------------------------------------------------------------------------
# S3DataManager
# ---------------------------------------------------------------------------

def test_s3_load_matches_supabase_load_for_the_same_rows():
    supabase = SupabaseManager.__new__(SupabaseManager)  # skip __init__ (needs credentials)
    supabase.client = FakeSupabaseClient({
        'Full Precision Test': _raw_universe_rows(),
        'last_price_mapping': _raw_delisting_rows(),
    })

    from_supabase = supabase.fetch_all_data()
    from_s3 = _s3_with_data().fetch_all_data()

    pd.testing.assert_frame_equal(from_s3, from_supabase)
    assert {'Ticker', 'Year', 'Delist_Date', 'Delist_Price'} <= set(from_s3.columns)
    assert from_s3['FactSet_Industry'].isna().sum() == 2  # '--' sentinel became NaN


def test_missing_delisting_file_still_loads_the_universe():
    df = _s3_with_data(include_delisting=False).fetch_all_data()
    assert len(df) == 8
    assert 'Delist_Date' not in df.columns


def test_missing_universe_file_raises_with_its_location():
    manager = S3DataManager(bucket='bucket', prefix='factor-lake-data', client=FakeS3Client())
    with pytest.raises(S3DataError, match=r's3://bucket/factor-lake-data/full_precision_test\.parquet \(NoSuchKey\)'):
        manager.fetch_all_data()


def test_missing_credentials_give_a_clear_message():
    manager = S3DataManager(bucket='bucket', client=FakeS3Client(raise_on_get=NoCredentialsError()))
    with pytest.raises(S3DataError, match='AWS_ACCESS_KEY_ID'):
        manager.fetch_all_data()


def test_location_comes_from_environment(monkeypatch):
    monkeypatch.setenv('FACTOR_LAKE_S3_BUCKET', 'other-bucket')
    monkeypatch.setenv('FACTOR_LAKE_S3_PREFIX', '/nested/data/')
    manager = S3DataManager(client=FakeS3Client())
    assert manager.s3_uri(UNIVERSE_FILE) == 's3://other-bucket/nested/data/full_precision_test.parquet'
    assert S3DataManager(prefix='', client=FakeS3Client()).object_key('x.parquet') == 'x.parquet'


# ---------------------------------------------------------------------------
# Validation and source selection
# ---------------------------------------------------------------------------

def test_validate_universe_explains_empty_and_incomplete_data():
    with pytest.raises(ValueError, match='s3 data source returned no rows'):
        validate_universe(pd.DataFrame(), 's3')
    with pytest.raises(ValueError, match=r"missing required columns: \['Year'\]"):
        validate_universe(pd.DataFrame({'Ticker': ['A'], 'Ending_Price': [1.0], 'Next-Years_Return': [2.0]}), 's3')


@pytest.mark.parametrize('env, expected', [
    ({'FACTOR_LAKE_DATA_SOURCE': 'S3'}, 's3'),
    ({'FACTOR_LAKE_DATA_SOURCE': 'supabase', 'AWS_ACCESS_KEY_ID': 'x'}, 'supabase'),
    ({'AWS_ACCESS_KEY_ID': 'x'}, 's3'),
    ({}, 'supabase'),
])
def test_get_data_source(monkeypatch, env, expected):
    monkeypatch.delenv('FACTOR_LAKE_DATA_SOURCE', raising=False)
    monkeypatch.delenv('AWS_ACCESS_KEY_ID', raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert streamlit_utils.get_data_source() == expected


def test_invalid_data_source_is_reported(monkeypatch):
    monkeypatch.setenv('FACTOR_LAKE_DATA_SOURCE', 'mysql')
    with pytest.raises(ValueError, match='FACTOR_LAKE_DATA_SOURCE'):
        streamlit_utils.get_data_source()
    assert streamlit_utils.describe_data_source().startswith('Misconfigured')


def test_shared_universe_is_loaded_once_and_failures_are_not_cached(monkeypatch):
    calls = []
    good_data = _s3_with_data().fetch_all_data()

    def fake_fetch(self):
        calls.append(1)
        if len(calls) == 1:
            return pd.DataFrame()  # e.g. a transient outage
        return good_data

    monkeypatch.setattr(S3DataManager, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(S3DataManager, 'fetch_all_data', fake_fetch)
    streamlit_utils.load_shared_universe.clear()
    try:
        with pytest.raises(ValueError, match='returned no rows'):
            streamlit_utils.load_shared_universe('s3')
        first = streamlit_utils.load_shared_universe('s3')
        second = streamlit_utils.load_shared_universe('s3')
    finally:
        streamlit_utils.load_shared_universe.clear()

    assert len(calls) == 2  # the failure was retried, the success was reused
    assert first is second  # one shared copy for every session


def test_app_shows_a_clear_error_when_the_data_source_fails(monkeypatch):
    def failing_fetch(self):
        raise S3DataError("Could not read s3://bucket/factor-lake-data/full_precision_test.parquet (AccessDenied).")

    monkeypatch.setenv('FACTOR_LAKE_DATA_SOURCE', 's3')
    monkeypatch.setattr(S3DataManager, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(S3DataManager, 'fetch_all_data', failing_fetch)
    monkeypatch.syspath_prepend(str(APP_DIR))
    st.cache_resource.clear()

    at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    at.run()
    assert any('AWS S3' in c.value for c in at.sidebar.caption)
    next(b for b in at.button if b.label == 'Load Market Data').click().run()

    assert not at.exception
    assert any('AccessDenied' in e.value for e in at.error)
    assert not any("'Year'" in e.value for e in at.error)
    assert at.session_state['raw_data'] is None  # nothing bad kept for the session
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Publishing script
# ---------------------------------------------------------------------------

def test_publish_uploads_live_and_archive_copies_and_round_trips():
    client = FakeS3Client()
    manager = S3DataManager(bucket='bucket', prefix='factor-lake-data', client=client)
    tables = {UNIVERSE_FILE: pd.DataFrame(_raw_universe_rows()),
              DELISTING_FILE: pd.DataFrame(_raw_delisting_rows())}

    published = publisher.publish(tables, manager, timestamp='20260101T000000Z')

    assert published[UNIVERSE_FILE] == 's3://bucket/factor-lake-data/full_precision_test.parquet'
    assert ('bucket', 'factor-lake-data/archive/20260101T000000Z/full_precision_test.parquet') in client.objects
    assert ('bucket', 'factor-lake-data/archive/20260101T000000Z/last_price_mapping.parquet') in client.objects
    pd.testing.assert_frame_equal(manager.read_parquet(UNIVERSE_FILE), tables[UNIVERSE_FILE])


def test_publish_dry_run_uploads_nothing():
    client = FakeS3Client()
    manager = S3DataManager(bucket='bucket', client=client)
    publisher.publish({UNIVERSE_FILE: pd.DataFrame(_raw_universe_rows())}, manager, 'ts', dry_run=True)
    assert client.objects == {}


def test_publish_refuses_data_the_app_cannot_use():
    manager = S3DataManager(bucket='bucket', client=FakeS3Client())
    no_dates = pd.DataFrame(_raw_universe_rows()).drop(columns=['Date'])
    with pytest.raises(ValueError, match='Year'):
        publisher.publish({UNIVERSE_FILE: no_dates}, manager, 'ts')

    bad_delisting = pd.DataFrame({'ticker': ['A'], 'last_price': [np.nan]})
    with pytest.raises(ValueError, match='last_date'):
        publisher.publish({UNIVERSE_FILE: pd.DataFrame(_raw_universe_rows()),
                           DELISTING_FILE: bad_delisting}, manager, 'ts')
    assert manager.client.objects == {}


def test_publish_reads_local_files(tmp_path):
    csv_path = tmp_path / 'universe.csv'
    pd.DataFrame(_raw_universe_rows()).to_csv(csv_path, index=False)
    tables = publisher.load_tables(False, str(csv_path), None)
    assert list(tables) == [UNIVERSE_FILE]
    assert len(tables[UNIVERSE_FILE]) == 8
    with pytest.raises(ValueError, match='Unsupported file type'):
        publisher.read_local_table(str(tmp_path / 'universe.json'))


def test_delisting_file_requires_universe_file():
    with pytest.raises(SystemExit):
        publisher.main(['--from-supabase', '--delisting', 'x.csv'])
