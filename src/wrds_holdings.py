"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_holdings.py
PURPOSE: Stores a WRDS backtest's holdings as Parquet in S3 and reads back one rebalance date.
VERSION: 1.0.0

The Lambda writes backtest-results/<hash>.holdings.parquet with one row group per
rebalance date, before the result JSON (DOCS/WRDS_MODE_DESIGN.md, section 5.5).
The app reads one date with two ranged GETs: the file's footer, then that date's row
group. It reads with its own S3 keys (the same ones that open saved analyses), so a
shared analysis ID can show its holdings at any time, without calling Lambda and
without links that expire.
"""

import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The footer is read with one GET of the file's last bytes (a second GET if it is bigger)
TAIL_BYTES = 64 * 1024
HOLDINGS_SUFFIX = '.holdings.parquet'


def holdings_file(results_dir: str, key: str) -> str:
    """Path (under the data prefix) of a saved result's holdings."""
    return f'{results_dir}/{key}{HOLDINGS_SUFFIX}'


def holdings_parquet(holdings: pd.DataFrame) -> Tuple[bytes, Dict[str, Any]]:
    """
    The holdings (sorted by rebalance_date) as Parquet bytes with one row group per
    rebalance date, and {'dates': [...], 'rows': [...]} listing the row groups in order.
    """
    table = pa.Table.from_pandas(holdings, preserve_index=False)
    dates = holdings['rebalance_date']
    if not dates.is_monotonic_increasing:
        raise ValueError("Holdings must be sorted by rebalance_date.")
    counts = dates.value_counts(sort=False).sort_index()
    buffer = io.BytesIO()
    with pq.ParquetWriter(buffer, table.schema, compression='zstd') as writer:
        start = 0
        for count in counts:
            writer.write_table(table.slice(start, int(count)), row_group_size=int(count))
            start += int(count)
    return buffer.getvalue(), {'dates': [d.strftime('%Y-%m-%d') for d in counts.index],
                               'rows': [int(c) for c in counts]}


class _Ranges:
    """
    A read-only file over byte ranges fetched beforehand: pyarrow reads the footer and
    one row group through it, and a read outside those ranges is an error (so no read
    ever silently downloads more).
    """

    def __init__(self, size: int):
        self.size, self.pos, self.closed = size, 0, False
        self._chunks: List[Tuple[int, bytes]] = []

    def add(self, start: int, body: bytes) -> None:
        self._chunks.append((start, body))

    def has(self, start: int, end: int) -> bool:
        return any(s <= start and end <= s + len(b) for s, b in self._chunks)

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        for start, body in self._chunks:
            if start <= self.pos and end <= start + len(body):
                out = body[self.pos - start:end - start]
                self.pos = end
                return out
        raise IOError(f"Bytes {self.pos}-{end} of the holdings file were not fetched.")

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True

    @property
    def mode(self) -> str:
        return 'rb'


def _get_range(manager, filename: str, byte_range: str) -> Tuple[bytes, int]:
    """(body, object size) of one ranged GET under the data prefix."""
    response = manager.client.get_object(Bucket=manager.bucket, Key=manager.object_key(filename),
                                         Range=f'bytes={byte_range}')
    size = int(str(response['ContentRange']).rsplit('/', 1)[1])
    return response['Body'].read(), size


def read_holdings_date(manager, filename: str, row_group: int, date: Optional[str] = None) -> pd.DataFrame:
    """
    One rebalance date's holdings from a holdings file in S3, with ranged GETs: the
    footer, then only that date's row group. row_group is the date's position in the
    result's holdings['dates']; date (YYYY-MM-DD), when given, is checked against the rows.
    """
    tail, size = _get_range(manager, filename, f'-{TAIL_BYTES}')
    if tail[-4:] != b'PAR1':
        raise ValueError(f"{filename} is not a Parquet file.")
    ranges = _Ranges(size)
    ranges.add(size - len(tail), tail)
    footer = int.from_bytes(tail[-8:-4], 'little') + 8
    if footer > len(tail):
        body, _ = _get_range(manager, filename, f'{size - footer}-{size - 1}')
        ranges.add(size - footer, body)

    parquet = pq.ParquetFile(ranges, pre_buffer=False)
    if not 0 <= row_group < parquet.metadata.num_row_groups:
        raise ValueError(f"{filename} has no row group {row_group}.")
    group = parquet.metadata.row_group(row_group)
    starts = [c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset
              for c in (group.column(i) for i in range(group.num_columns))]
    start = min(starts)
    end = max(s + group.column(i).total_compressed_size for i, s in enumerate(starts))
    if not ranges.has(start, end):
        body, _ = _get_range(manager, filename, f'{start}-{end - 1}')
        ranges.add(start, body)

    holdings = parquet.read_row_group(row_group).to_pandas()
    if date is not None and not holdings.empty and holdings['rebalance_date'].iloc[0].strftime('%Y-%m-%d') != date:
        raise ValueError(f"Row group {row_group} of {filename} is not {date}.")
    return holdings
