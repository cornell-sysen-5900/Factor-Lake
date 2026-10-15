"""
Tests for scripts/publish_daily_data_to_s3.py and the daily-data reader in src/s3_client.py.

A small synthetic CRSP-style CSV stands in for russell2000_cleaned.csv, and an in-memory
fake replaces S3, so no AWS access is needed.
"""
import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError

from scripts import publish_daily_data_to_s3 as daily
from src.s3_client import DAILY_LAST_PRICES_FILE, DAILY_MANIFEST_FILE, S3DataError, S3DataManager, daily_year_file

HEADER = 'PERMNO,TICKER,DATE,PRC,CFACPR,SHROUT,RET,VOL,COMNAM\n'
# Rows are deliberately out of order. With chunk_rows=4 the two 10001/2008-01-02 rows
# (index 5 and 13) fall in different chunks, so "first wins" is checked across chunks.
ROWS = [
    '10001,abc,2009-12-31,12.0,0,1000,0.02,800,Alpha',        # 0  cfacpr 0 -> unadjusted price
    '10002,XYZ,2008-06-30,4.0,4,0,-0.2,100,Beta',            # 1  last row of 10002; shrout 0 -> missing
    '10003,QQQ,,7.0,1,10,0.0,1,Gamma',                       # 2  dropped: no date
    '10001,abc,2007-12-31,-10.0,2,1000,0.01,500,Alpha',      # 3  negative CRSP price
    '10005,SHR,2009-12-31,30.0,1,10,0.0,1,ShareB',           # 4  share class B (same ticker as 10004)
    '10001,abc,2008-01-02,10.5,2,1000,0.05,600,Alpha',       # 5  first of a duplicate pair: kept
    '10003,QQQ,not-a-date,7.0,1,10,0.0,1,Gamma',             # 6  dropped: bad date
    '10006,BND,2009-12-24,7.0,1,10,0.0,1,Edge7',             # 7  ends exactly 7 days early: not ended_early
    '10001,abc,01/04/2008,11.5,2,1000,-99,-99,Alpha',        # 8  other date format; CRSP -99 codes
    '10003,QQQ,2008-01-02,0,1,10,0.0,1,Gamma',               # 9  dropped: price 0 = no price
    '10002,XYZ,2008-01-02,5.0,1,2000,--,100,Beta',           # 10 placeholder return
    '10007,BNE,2009-12-23,7.0,1,10,0.0,1,Edge8',             # 11 ends 8 days early: ended_early
    '10003,QQQ,2008-01-03,,1,10,0.0,1,Gamma',                # 12 dropped: missing price
    '10001,abc,2008-01-02,99.0,2,1000,0.05,600,Alpha',       # 13 later duplicate: dropped
    '10004,SHR,2009-12-31,20.0,1,10,0.0,1,ShareA',           # 14 share class A
    '10001,abc,20080103,11.0,2,1000,C,700,Alpha',            # 15 YYYYMMDD date, CRSP letter code
    ',QQQ,2008-01-04,7.0,1,10,0.0,1,Gamma',                  # 16 dropped: no permno
]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / 'russell2000_cleaned.csv'
    path.write_text(HEADER + '\n'.join(ROWS) + '\n')
    return path


class FakeS3Client:
    """In-memory stand-in for the boto3 calls S3DataManager makes."""

    def __init__(self, size_delta=0, shrink_reads=False, put_error=None):
        self.objects, self.content_types, self.upload_order = {}, {}, []
        self.size_delta, self.shrink_reads, self.put_error = size_delta, shrink_reads, put_error

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()
        self.content_types[Key] = (ExtraArgs or {}).get('ContentType')
        self.upload_order.append(Key)

    def head_object(self, Bucket, Key):
        return {'ContentLength': len(self.objects[(Bucket, Key)]) + self.size_delta}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'missing'}}, 'GetObject')
        body = self.objects[(Bucket, Key)]
        if self.shrink_reads and Key.endswith('data.parquet'):   # simulate a damaged object
            table = pq.read_table(io.BytesIO(body)).slice(0, 0)
            buf = io.BytesIO(); pq.write_table(table, buf); body = buf.getvalue()
        return {'Body': io.BytesIO(body)}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if self.put_error:
            raise ClientError({'Error': {'Code': self.put_error, 'Message': 'no'}}, 'PutObject')
        self.objects[(Bucket, Key)] = Body

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


def _manager(client):
    return S3DataManager(bucket='bucket', prefix='factor-lake-data', client=client)


def _year(out_dir, year):
    return pd.read_parquet(out_dir / daily_year_file(year))


def _keys(df):
    return list(zip(df['permno'], df['date'].dt.strftime('%Y-%m-%d')))


def _row(df, permno, day):
    """The single row for one security on one day."""
    match = df[(df['permno'] == permno) & (df['date'] == pd.Timestamp(day))]
    assert len(match) == 1
    return match.iloc[0]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

@pytest.fixture
def built(csv_path, tmp_path):
    out = tmp_path / 'out'
    return out, daily.build_daily_dataset(csv_path, out, chunk_rows=4)


def test_rows_are_cleaned_counted_and_split_by_year(built):
    out, manifest = built
    assert {y: v['rows'] for y, v in manifest['years'].items()} == {'2007': 1, '2008': 5, '2009': 5}
    assert manifest['rows_written'] == 11
    assert manifest['cleaning'] == {'rows_read': 17, 'dropped_missing_date': 2, 'dropped_before_start': 0,
                                    'dropped_missing_price': 2, 'dropped_missing_id': 1, 'dropped_duplicates': 1,
                                    'ticker_date_conflicts': 0}
    assert manifest['start_date'] is None
    assert manifest['security_key'] == 'permno'
    assert manifest['ignored_csv_columns'] == ['COMNAM']
    assert manifest['date_range'] == ['2007-12-31', '2009-12-31']
    assert manifest['securities'] == 6
    assert {y: v['securities'] for y, v in manifest['years'].items()} == {'2007': 1, '2008': 2, '2009': 5}
    assert list(_year(out, 2008).columns) == daily.OUTPUT_COLUMNS


def test_each_year_is_sorted_by_security_and_date(built):
    out, _ = built
    assert _keys(_year(out, 2008)) == [(10001, '2008-01-02'), (10001, '2008-01-03'), (10001, '2008-01-04'),
                                       (10002, '2008-01-02'), (10002, '2008-06-30')]
    assert [p for p, _ in _keys(_year(out, 2009))] == [10001, 10004, 10005, 10006, 10007]


def test_first_duplicate_wins_across_chunks_and_share_classes_are_kept(built):
    out, _ = built
    y2008 = _year(out, 2008)
    assert y2008.iloc[0]['prc'] == 10.5            # not the later 99.0 row from another chunk
    y2009 = _year(out, 2009)
    shr = y2009[y2009['ticker'] == 'SHR']
    assert sorted(shr['permno']) == [10004, 10005] # same ticker and date, different permnos: both kept


def test_prices_and_derived_columns(built):
    out, _ = built
    neg = _year(out, 2007).iloc[0]                 # CRSP price -10 with cfacpr 2
    assert (neg['prc'], neg['prc_adj'], neg['mkt_cap']) == (10.0, 5.0, 10000.0)
    assert _year(out, 2009).iloc[0]['prc_adj'] == 12.0   # cfacpr 0 -> unadjusted
    y2008 = _year(out, 2008)
    assert np.isnan(_row(y2008, 10002, '2008-06-30')['mkt_cap'])   # shrout 0 is missing
    assert _row(y2008, 10002, '2008-06-30')['prc_adj'] == 1.0       # 4.0 / cfacpr 4
    assert y2008['ticker'].tolist() == ['ABC', 'ABC', 'ABC', 'XYZ', 'XYZ']


def test_missing_value_codes_and_date_formats(built):
    out, _ = built
    y2008 = _year(out, 2008)
    assert np.isnan(_row(y2008, 10001, '2008-01-03')['ret'])        # letter code C (from 20080103)
    jan4 = _row(y2008, 10001, '2008-01-04')                         # parsed from 01/04/2008
    assert np.isnan(jan4['ret']) and np.isnan(jan4['vol'])          # numeric code -99
    assert jan4['prc'] == 11.5
    assert np.isnan(_row(y2008, 10002, '2008-01-02')['ret'])        # '--'


def test_last_prices_and_the_ended_early_boundary(built):
    out, manifest = built
    last = pd.read_parquet(out / DAILY_LAST_PRICES_FILE).set_index('permno')
    assert str(last.loc[10002, 'last_date'].date()) == '2008-06-30'
    assert (last.loc[10002, 'last_prc'], last.loc[10002, 'last_prc_adj']) == (4.0, 1.0)
    flags = last['ended_early'].astype(bool).to_dict()
    assert flags == {10001: False, 10002: True, 10004: False, 10005: False, 10006: False, 10007: True}
    assert manifest['ended_early_securities'] == 2


def test_ticker_only_csv_drops_exact_copies_and_reports_conflicts(tmp_path):
    path = tmp_path / 'tickers.csv'
    path.write_text('date,ticker,prc\n'
                    '2010-01-04, aaa ,1\n'
                    '2010-01-04,aaa,1\n'        # exact copy once cleaned: dropped
                    '2010-01-04,AAA,2\n'        # same ticker/date, different price: kept and reported
                    '2010-01-05,bbb,-3\n'
                    '2010-01-05,N/A,4\n'        # placeholder ticker: no identifier
                    ',ccc,4\n')
    manifest = daily.build_daily_dataset(path, tmp_path / 'out')
    df = _year(tmp_path / 'out', 2010)
    assert manifest['security_key'] == 'ticker'
    assert df['ticker'].tolist() == ['AAA', 'AAA', 'BBB'] and df['prc'].tolist() == [1.0, 2.0, 3.0]
    assert df['prc_adj'].tolist() == [1.0, 2.0, 3.0]
    c = manifest['cleaning']
    assert (c['dropped_duplicates'], c['ticker_date_conflicts'], c['dropped_missing_id'], c['dropped_missing_date']) \
        == (1, 1, 1, 1)
    assert df['permno'].isna().all()


def test_start_date_drops_earlier_rows_and_is_recorded(csv_path, tmp_path):
    manifest = daily.build_daily_dataset(csv_path, tmp_path / 'out', chunk_rows=4, start=date(2008, 1, 3))
    assert sorted(manifest['years']) == ['2008', '2009']
    assert manifest['start_date'] == '2008-01-03'
    # 2007-12-31, the 10001 and 10002 rows on 2008-01-02 (incl. the duplicate) and the price-0 row that day
    assert manifest['cleaning']['dropped_before_start'] == 5
    assert manifest['cleaning']['dropped_missing_price'] == 1     # only the 2008-01-03 row is left to drop
    assert manifest['date_range'][0] == '2008-01-03'


def test_missing_required_columns_are_reported(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('date,shrout\n2010-01-04,5\n')
    with pytest.raises(ValueError, match=r"needs date, prc and permno or ticker"):
        daily.build_daily_dataset(path, tmp_path / 'out')


def test_parts_left_by_an_interrupted_run_are_not_published(csv_path, tmp_path):
    out = tmp_path / 'out'
    stale = out / '_staging' / 'year=1999'
    stale.mkdir(parents=True)
    rows = pd.DataFrame({c: [None] for c in daily.OUTPUT_COLUMNS}).assign(
        permno=[1], ticker=['OLD'], date=[pd.Timestamp('1999-06-30')], prc=[1.0])
    pq.write_table(pa.Table.from_pandas(rows, schema=daily.SCHEMA, preserve_index=False), stale / 'part-00007.parquet')

    manifest = daily.build_daily_dataset(csv_path, out, chunk_rows=4)
    assert '1999' not in manifest['years']
    assert not (out / '_staging').exists()


# ---------------------------------------------------------------------------
# Publishing and reading back
# ---------------------------------------------------------------------------

def test_publish_uploads_years_then_manifest_and_reads_back(built):
    out, manifest = built
    client = FakeS3Client()
    manager = _manager(client)

    daily.publish(out, manifest, manager)

    assert client.upload_order == [
        'factor-lake-data/daily_prices/year=2007/data.parquet',
        'factor-lake-data/daily_prices/year=2008/data.parquet',
        'factor-lake-data/daily_prices/year=2009/data.parquet',
        'factor-lake-data/daily_last_prices.parquet',
        'factor-lake-data/daily_prices/_manifest.json',          # last, so readers see a complete set
    ]
    assert client.content_types['factor-lake-data/daily_prices/year=2008/data.parquet'] == 'application/vnd.apache.parquet'
    assert client.content_types['factor-lake-data/daily_prices/_manifest.json'] == 'application/json'
    assert json.loads(client.objects[('bucket', 'factor-lake-data/' + DAILY_MANIFEST_FILE)]) == manifest

    both = manager.read_daily_prices(years=[2008, 2009])
    assert len(both) == 10 and set(both['date'].dt.year) == {2008, 2009}
    assert len(manager.read_daily_prices()) == 11
    with pytest.raises(S3DataError, match=r"No daily data published for \[2015\]"):
        manager.read_daily_prices(years=[2015])


def test_size_mismatch_stops_before_the_manifest(built):
    out, manifest = built
    client = FakeS3Client(size_delta=1)
    with pytest.raises(RuntimeError, match='size mismatch'):
        daily.publish(out, manifest, _manager(client))
    assert not any(k.endswith('_manifest.json') for k in client.upload_order)


def test_read_back_mismatch_is_reported(built):
    out, manifest = built
    with pytest.raises(RuntimeError, match='Read-back check failed'):
        daily.publish(out, manifest, _manager(FakeS3Client(shrink_reads=True)))


def test_reading_daily_data_before_any_publish_is_a_clear_error():
    with pytest.raises(S3DataError, match=r"daily_prices/_manifest\.json \(NoSuchKey\)"):
        _manager(FakeS3Client()).read_daily_prices()


def test_bad_keys_fail_before_cleaning(csv_path, monkeypatch):
    client = FakeS3Client(put_error='SignatureDoesNotMatch')
    monkeypatch.setattr(daily, 'S3DataManager', lambda **kw: _manager(client))

    def never(*args, **kwargs):
        raise AssertionError("the CSV must not be cleaned when the keys are wrong")

    monkeypatch.setattr(daily, 'build_daily_dataset', never)
    with pytest.raises(S3DataError, match=r"SignatureDoesNotMatch.*re-copy it exactly"):
        daily.main([str(csv_path)])


def test_main_publishes_end_to_end(csv_path, monkeypatch):
    client = FakeS3Client()
    monkeypatch.setattr(daily, 'S3DataManager', lambda **kw: _manager(client))
    assert daily.main([str(csv_path), '--chunk-rows', '4']) == 0
    assert ('bucket', 'factor-lake-data/_write_check') not in client.objects     # preflight cleaned up
    assert json.loads(client.objects[('bucket', 'factor-lake-data/' + DAILY_MANIFEST_FILE)])['rows_written'] == 11


def test_dry_run_keeps_files_locally_and_uploads_nothing(csv_path, tmp_path, monkeypatch):
    def no_s3(**kwargs):
        raise AssertionError("dry run must not touch S3")

    monkeypatch.setattr(daily, 'S3DataManager', no_s3)
    out = tmp_path / 'kept'
    assert daily.main([str(csv_path), '--dry-run', '--output-dir', str(out)]) == 0
    assert pq.read_metadata(out / daily_year_file(2008)).num_rows == 5
    assert json.loads((out / DAILY_MANIFEST_FILE).read_text())['rows_written'] == 11


def test_home_paths_are_expanded_even_when_quoted(csv_path, tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'data.csv').write_bytes(csv_path.read_bytes())
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(tmp_path)
    assert daily.main(['~/data.csv', '--dry-run', '--output-dir', '~/kept']) == 0
    assert (home / 'kept' / DAILY_MANIFEST_FILE).exists()
    assert not (tmp_path / '~').exists()


def test_missing_csv_is_reported(tmp_path):
    with pytest.raises(SystemExit):
        daily.main([str(tmp_path / 'nope.csv')])
