"""
Tests for scripts/publish_daily_data_to_s3.py and the daily-data reader in src/s3_client.py.

A small synthetic CRSP-style CSV stands in for russell2000_cleaned.csv, and an in-memory
fake replaces S3, so no AWS access is needed.
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError

from scripts import publish_daily_data_to_s3 as daily
from src.s3_client import DAILY_LAST_PRICES_FILE, DAILY_MANIFEST_FILE, S3DataError, S3DataManager, daily_year_file

HEADER = 'PERMNO,TICKER,DATE,PRC,CFACPR,SHROUT,RET,VOL,COMNAM\n'
ROWS = [
    # permno 10001 trades through 2009 (last row is the latest date in the file)
    '10001,abc,2007-12-31,-10.0,1,1000,0.01,500,Alpha',
    '10001,abc,2008-01-02,10.5,2,1000,0.05,600,Alpha',
    '10001,abc,2008-01-02,99.0,2,1000,0.05,600,Alpha',        # duplicate date: first row wins
    '10001,abc,20080103,11.0,2,1000,C,700,Alpha',            # YYYYMMDD date, CRSP letter code return
    '10001,abc,2009-12-31,12.0,0,1000,0.02,800,Alpha',       # cfacpr 0: adjusted price = price
    # permno 10002 stops trading in mid-2008 (delisted)
    '10002,XYZ,2008-01-02,5.0,1,2000,--,100,Beta',           # placeholder return
    '10002,XYZ,2008-06-30,4.0,1,2000,-0.2,100,Beta',
    # rows that must be dropped
    '10003,QQQ,,7.0,1,10,0.0,1,Gamma',                       # no date
    '10003,QQQ,not-a-date,7.0,1,10,0.0,1,Gamma',             # bad date
    '10003,QQQ,2008-01-02,0,1,10,0.0,1,Gamma',               # price 0 = no price
    '10003,QQQ,2008-01-03,,1,10,0.0,1,Gamma',                # missing price
    ',QQQ,2008-01-04,7.0,1,10,0.0,1,Gamma',                  # no permno
]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / 'russell2000_cleaned.csv'
    path.write_text(HEADER + '\n'.join(ROWS) + '\n')
    return path


class FakeS3Client:
    """Records uploads in order and serves them back, like the boto3 calls S3DataManager uses."""

    def __init__(self):
        self.objects = {}
        self.upload_order = []

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()
        self.upload_order.append(Key)

    def head_object(self, Bucket, Key):
        return {'ContentLength': len(self.objects[(Bucket, Key)])}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'missing'}}, 'GetObject')
        return {'Body': io.BytesIO(self.objects[(Bucket, Key)])}


def _year(out_dir, year):
    return pd.read_parquet(out_dir / daily_year_file(year))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_build_cleans_and_splits_by_year(csv_path, tmp_path):
    out = tmp_path / 'out'
    manifest = daily.build_daily_dataset(csv_path, out, chunk_rows=3)   # small chunks: duplicates span chunks

    assert sorted(manifest['years']) == ['2007', '2008', '2009']
    assert {y: v['rows'] for y, v in manifest['years'].items()} == {'2007': 1, '2008': 4, '2009': 1}
    assert manifest['rows_written'] == 6
    assert manifest['cleaning'] == {'rows_read': 12, 'dropped_missing_date': 2, 'dropped_missing_price': 2,
                                    'dropped_missing_id': 1, 'dropped_duplicates': 1}
    assert manifest['security_key'] == 'permno'
    assert manifest['ignored_csv_columns'] == ['COMNAM']
    assert manifest['date_range'] == ['2007-12-31', '2009-12-31']

    y2008 = _year(out, 2008)
    assert list(y2008.columns) == daily.OUTPUT_COLUMNS
    assert list(zip(y2008['permno'], y2008['date'].dt.strftime('%Y-%m-%d'))) == [
        (10001, '2008-01-02'), (10001, '2008-01-03'), (10002, '2008-01-02'), (10002, '2008-06-30')]  # sorted
    first = y2008.iloc[0]
    assert first['prc'] == 10.5 and first['prc_adj'] == 5.25 and first['mkt_cap'] == 10500.0  # first duplicate kept
    assert np.isnan(y2008.iloc[1]['ret'])          # CRSP code 'C'
    assert np.isnan(y2008.iloc[2]['ret'])          # '--'
    assert y2008['ticker'].tolist() == ['ABC', 'ABC', 'XYZ', 'XYZ']

    assert _year(out, 2007).iloc[0]['prc'] == 10.0                    # negative CRSP price made positive
    assert _year(out, 2009).iloc[0]['prc_adj'] == 12.0                # cfacpr 0 -> unadjusted
    assert not (out / '_staging').exists()


def test_last_prices_mark_delisted_securities(csv_path, tmp_path):
    out = tmp_path / 'out'
    manifest = daily.build_daily_dataset(csv_path, out, chunk_rows=4)
    last = pd.read_parquet(out / DAILY_LAST_PRICES_FILE).set_index('permno')

    assert str(last.loc[10002, 'last_date'].date()) == '2008-06-30'
    assert last.loc[10002, 'last_prc'] == 4.0
    assert bool(last.loc[10002, 'delisted']) is True
    assert bool(last.loc[10001, 'delisted']) is False
    assert manifest['delisted_securities'] == 1


def test_ticker_only_csv_uses_ticker_as_key(tmp_path):
    path = tmp_path / 'tickers.csv'
    path.write_text('date,ticker,prc\n2010-01-04,aaa,1\n2010-01-04,aaa,2\n2010-01-05,bbb,-3\n,ccc,4\n')
    manifest = daily.build_daily_dataset(path, tmp_path / 'out')
    df = _year(tmp_path / 'out', 2010)
    assert manifest['security_key'] == 'ticker'
    assert df['ticker'].tolist() == ['AAA', 'BBB'] and df['prc'].tolist() == [1.0, 3.0]
    assert df['permno'].isna().all()


def test_missing_required_columns_are_reported(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('date,shrout\n2010-01-04,5\n')
    with pytest.raises(ValueError, match=r"needs date, prc and permno or ticker"):
        daily.build_daily_dataset(path, tmp_path / 'out')


# ---------------------------------------------------------------------------
# Publishing and reading back
# ---------------------------------------------------------------------------

def test_publish_uploads_years_then_manifest_and_reads_back(csv_path, tmp_path):
    out = tmp_path / 'out'
    manifest = daily.build_daily_dataset(csv_path, out)
    client = FakeS3Client()
    manager = S3DataManager(bucket='bucket', prefix='factor-lake-data', client=client)

    daily.publish(out, manifest, manager)

    assert client.upload_order == [
        'factor-lake-data/daily_prices/year=2007/data.parquet',
        'factor-lake-data/daily_prices/year=2008/data.parquet',
        'factor-lake-data/daily_prices/year=2009/data.parquet',
        'factor-lake-data/daily_last_prices.parquet',
        'factor-lake-data/daily_prices/_manifest.json',          # last, so readers see a complete set
    ]
    assert json.loads(client.objects[('bucket', 'factor-lake-data/' + DAILY_MANIFEST_FILE)]) == manifest

    both = manager.read_daily_prices(years=[2008, 2009])
    assert len(both) == 5 and set(both['date'].dt.year) == {2008, 2009}
    assert len(manager.read_daily_prices()) == 6
    with pytest.raises(S3DataError, match=r"No daily data published for \[2015\]"):
        manager.read_daily_prices(years=[2015])


def test_reading_daily_data_before_any_publish_is_a_clear_error():
    manager = S3DataManager(bucket='bucket', prefix='factor-lake-data', client=FakeS3Client())
    with pytest.raises(S3DataError, match=r"daily_prices/_manifest\.json \(NoSuchKey\)"):
        manager.read_daily_prices()


def test_dry_run_keeps_files_locally_and_uploads_nothing(csv_path, tmp_path, monkeypatch):
    def no_s3(*args, **kwargs):
        raise AssertionError("dry run must not touch S3")

    monkeypatch.setattr(S3DataManager, '__init__', no_s3)
    out = tmp_path / 'kept'
    assert daily.main([str(csv_path), '--dry-run', '--output-dir', str(out)]) == 0
    assert pq.read_metadata(out / daily_year_file(2008)).num_rows == 4
    assert json.loads((out / DAILY_MANIFEST_FILE).read_text())['rows_written'] == 6


def test_missing_csv_is_reported(tmp_path):
    with pytest.raises(SystemExit):
        daily.main([str(tmp_path / 'nope.csv')])


def test_quoted_home_paths_are_expanded(csv_path, tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(csv_path.parent))
    out = tmp_path / 'kept'
    assert daily.main(['~/' + csv_path.name, '--dry-run', '--output-dir', str(out)]) == 0
    assert (out / DAILY_MANIFEST_FILE).exists()
