"""Bounded WRDS samples served by Lambda, separately from the backtest engine.

The index holds the first five (date, permno) rows in every year/sector/sub-industry
cell. Every supported GICS filter is a union of these cells, so merging their first
five rows gives exactly the same sample as filtering the full panel first. This
keeps cold requests small without dropping rare sub-industries from the preview.
"""
import hashlib
import io
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .wrds_calendar import FIRST_YEAR, LAST_YEAR
from .wrds_data import PANEL_COLUMNS, PANEL_FILE, wrds_dir
from .wrds_gics import gics_mask, validate_gics_filter

APPROVAL_ENV = 'FACTOR_LAKE_WRDS_PREVIEW_APPROVED'
DISABLED_MESSAGE = 'WRDS sample rows are not enabled yet. Release requires Prof. Steward’s approval.'
NOT_READY_MESSAGE = 'The WRDS preview is not ready for this data version. Ask the team to build its preview index.'
ROWS_PER_YEAR = 5
MAX_INDEX_ROWS = 100_000
MAX_INDEX_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Explicitly exclude next-period returns and all other fields outside the factor catalog.
PREVIEW_COLUMNS = tuple(PANEL_COLUMNS)
INDEX_VERSION = 'v1'


def preview_enabled():
    """Server-side release gate; request fields can never enable it."""
    return os.environ.get(APPROVAL_ENV, '').strip().lower() == 'true'


def index_filename(data_dir, etag):
    digest = hashlib.sha256(etag.encode()).hexdigest()
    return f'{data_dir}/preview/{INDEX_VERSION}-{digest}.parquet'


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {'start_year', 'end_year', 'gics_filter'}:
        raise ValueError('Preview needs start_year, end_year and gics_filter only.')
    start, end = request['start_year'], request['end_year']
    if any(type(y) is not int for y in (start, end)) or not FIRST_YEAR <= start <= end <= LAST_YEAR:
        raise ValueError(f'Preview years must be between {FIRST_YEAR} and {LAST_YEAR}, in ascending order.')
    if request['gics_filter'] is not None:
        validate_gics_filter(request['gics_filter'])


def _first_in_cells(frame):
    frame = frame.sort_values(['date', 'permno'], kind='stable').drop_duplicates(['date', 'permno'])
    groups = [frame['date'].dt.year, frame['sector_name'], frame['gsubind']]
    return frame.groupby(groups, observed=True, dropna=False, sort=False).head(ROWS_PER_YEAR).reset_index(drop=True)


def build_index(batches):
    """Consume projected panel batches; retain only a bounded candidate set."""
    index = None
    for batch in batches:
        frame = batch.to_pandas() if hasattr(batch, 'to_pandas') else batch
        if not set(PREVIEW_COLUMNS) <= set(frame.columns):
            raise ValueError('The WRDS panel is missing preview fields.')
        frame = frame.loc[:, list(PREVIEW_COLUMNS)].copy()
        frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
        frame = frame[frame['date'].notna() & frame['permno'].notna()]
        frame = frame[frame['date'].dt.year.between(FIRST_YEAR, LAST_YEAR)]
        # Normalize categorical keys before grouping; missing GICS remains a real cell.
        frame['sector_name'] = frame['sector_name'].astype('string')
        frame['gsubind'] = pd.to_numeric(frame['gsubind'], errors='coerce').astype('Int64')
        frame = _first_in_cells(frame)
        index = _first_in_cells(pd.concat([index, frame], ignore_index=True)) if index is not None else frame
        if len(index) > MAX_INDEX_ROWS:
            raise ValueError('The preview index exceeds its row limit.')
    if index is None:
        raise ValueError('The WRDS panel contains no batches.')
    return index


def encode_index(index, etag):
    table = pa.Table.from_pandas(index, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b'preview_version': INDEX_VERSION.encode(),
                                           b'source_etag': etag.encode()})
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression='zstd')
    body = buffer.getvalue()
    if len(body) > MAX_INDEX_BYTES:
        raise ValueError('The preview index exceeds its byte limit.')
    return body


def select_preview(index, request):
    validate_request(request)
    years = index['date'].dt.year
    frame = index[years.between(request['start_year'], request['end_year']) &
                  gics_mask(index, request['gics_filter'])]
    frame = frame.sort_values(['date', 'permno'], kind='stable')
    frame = frame.groupby(frame['date'].dt.year, sort=True).head(ROWS_PER_YEAR)
    frame = frame.loc[:, list(PREVIEW_COLUMNS)].copy()
    counts = {str(y): int(n) for y, n in frame.groupby(frame['date'].dt.year).size().items()}
    frame['date'] = frame['date'].dt.strftime('%Y-%m-%d')
    # pandas converts NaN/inf/NA to JSON null; never send non-standard JSON numbers.
    result = {'columns': list(PREVIEW_COLUMNS), 'rows': json.loads(frame.to_json(orient='records')),
              'rows_per_year': ROWS_PER_YEAR,
              'years': {str(y): counts.get(str(y), 0)
                        for y in range(request['start_year'], request['end_year'] + 1)}}
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RESPONSE_BYTES:
        raise ValueError('The preview exceeds its response limit.')
    return result


class WrdsPreviewService:
    """One small index per Lambda container; no per-filter cache and no backtest data."""

    def __init__(self, manager):
        self.manager = manager
        self._index = None

    def preview(self, request):
        if not preview_enabled():
            self._index = None
            return {'status': 'disabled', 'message': DISABLED_MESSAGE}
        validate_request(request)
        manager, data_dir = self.manager, wrds_dir()
        head = manager.client.head_object(Bucket=manager.bucket,
                                          Key=manager.object_key(f'{data_dir}/{PANEL_FILE}'))
        etag = head['ETag'].strip('"')
        filename = index_filename(data_dir, etag)
        if self._index is None or self._index[0] != filename:
            self._index = None
            try:
                response = manager.client.get_object(Bucket=manager.bucket, Key=manager.object_key(filename))
            except Exception as exc:
                if str(getattr(exc, 'response', {}).get('Error', {}).get('Code', '')) in (
                        '404', 'NoSuchKey', 'NotFound'):
                    return {'status': 'unavailable', 'message': NOT_READY_MESSAGE}
                raise
            stream = response['Body']
            try:
                body = stream.read(MAX_INDEX_BYTES + 1)
            finally:
                stream.close()
            if len(body) > MAX_INDEX_BYTES:
                raise ValueError('The preview index exceeds its byte limit.')
            parquet = pq.ParquetFile(io.BytesIO(body))
            metadata = parquet.schema_arrow.metadata or {}
            if (metadata.get(b'source_etag') != etag.encode() or
                    metadata.get(b'preview_version') != INDEX_VERSION.encode() or
                    parquet.metadata.num_rows > MAX_INDEX_ROWS):
                raise ValueError('The preview index is invalid for this data version.')
            index = parquet.read(columns=list(PREVIEW_COLUMNS)).to_pandas()
            self._index = (filename, index)
        return {'status': 'ok', 'preview': select_preview(self._index[1], request)}
