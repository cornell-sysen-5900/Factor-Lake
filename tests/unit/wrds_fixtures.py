"""
Synthetic WRDS data for the WRDS engine tests: the three files' columns, with random
values, built from scratch so no licensed data is needed.

Month ends are the last business day of each month (like CRSP's last trading day), not
calendar month ends. Tests change the frames and rebuild WrdsData from them.
"""
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.wrds_data import (DEFAULT_WRDS_DIR, MARKET_COLUMNS, MARKET_FILE, PANEL_COLUMNS, PANEL_FILE,
                           RETURNS_COLUMNS, RETURNS_FILE, WrdsData)
from src.wrds_factors import WRDS_FACTOR_COLUMNS, WRDS_SECTORS
from tests.unit.test_backtest_lambda import FakeS3

# WRDS sidebar settings the synthetic data covers (its months run 2009-11 to 2013-02)
WRDS_SETTINGS = {'data_mode': 'wrds', 'frequency': 'quarterly', 'start_year': 2010, 'end_year': 2012,
                 'initial_aum': 1000.0, 'use_market_cap_weight': False, 'factor_combination': 'mixing',
                 'restrict_fossil_fuels': False, 'selected_sectors': None}


class RangeS3(FakeS3):
    """FakeS3 that also answers ranged GETs like S3 (Range and ContentRange) and records every GET."""

    def __init__(self, can_list=True):
        super().__init__(can_list)
        self.gets = []   # (key, range or None)

    def get_object(self, Bucket, Key, Range=None):
        self.gets.append((Key, Range))
        response = super().get_object(Bucket, Key)
        if Range is None:
            return response
        body = response['Body'].read()
        spec = Range.removeprefix('bytes=')
        if spec.startswith('-'):
            start, end = max(0, len(body) - int(spec[1:])), len(body) - 1
        else:
            first, last = spec.split('-')
            start, end = int(first), min(int(last), len(body) - 1)
        return {'Body': io.BytesIO(body[start:end + 1]), 'Metadata': response['Metadata'],
                'ContentRange': f'bytes {start}-{end}/{len(body)}'}


def wrds_file_bytes(frames: 'Frames') -> dict:
    """The three WRDS files as Parquet bytes, with extra columns the loader must not read."""
    panel = frames.panel.assign(ret_next_1m=0.5, delisted_next_month=False, security_extra='x')
    returns = frames.returns.assign(ticker='T', prc=1.0)
    market = frames.market.assign(month=frames.market['month_end'].dt.strftime('%Y-%m'), smb=0.0)
    out = {}
    for name, df in ((PANEL_FILE, panel), (RETURNS_FILE, returns), (MARKET_FILE, market)):
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        out[name] = buffer.getvalue()
    return out


def put_wrds_files(s3: FakeS3, frames: 'Frames', prefix: str = 'factor-lake-data',
                   data_dir: str = DEFAULT_WRDS_DIR) -> None:
    """Publishes the synthetic WRDS files to a fake S3, as scripts/upload_wrds.py publishes the real ones."""
    for name, body in wrds_file_bytes(frames).items():
        s3.objects[f'{prefix}/{data_dir}/{name}'] = (body, {})


@dataclass
class Frames:
    """The three files as DataFrames, with only the columns the loader reads."""
    panel: pd.DataFrame
    returns: pd.DataFrame
    market: pd.DataFrame

    def data(self) -> WrdsData:
        return WrdsData.build(self.panel, self.returns, self.market)


def month_ends(first: str, last: str) -> pd.DatetimeIndex:
    """Last business day of every month from first to last ('YYYY-MM')."""
    return pd.date_range(pd.Period(first).start_time, pd.Period(last).end_time, freq='BME')


def synthetic_frames(first: str = '2009-11', last: str = '2013-02', n_stocks: int = 60,
                     seed: int = 0, membership: float = 0.9) -> Frames:
    """
    Panel rows for random members (each stock is in the index at a month end with
    probability membership), returns for every stock and month from the month before
    first, and market data for the same months.
    """
    rng = np.random.default_rng(seed)
    dates = month_ends(first, last)
    months = month_ends(str(pd.Period(first) - 1), last)
    sectors = list(WRDS_SECTORS) + ['Unclassified']

    rows = []
    for date in dates:
        for permno in range(10001, 10001 + n_stocks):
            if rng.random() > membership:
                continue
            i = permno - 10001
            sector = sectors[i % len(sectors)]
            row = {'date': date, 'permno': permno, 'ticker': f'T{i:03d}', 'security_name': f'Company {i:03d}',
                   'sector_name': sector,
                   'gsubind': pd.NA if sector == 'Unclassified' else 10102020 if sector == 'Energy' and i % 2
                   else 45103010,
                   # Oil SIC code on an Unclassified stock (fossil by SIC) and on a GICS stock (GICS wins)
                   'siccd': 1311.0 if i % 12 in (0, 11) else 7372.0,
                   'prc': rng.uniform(2, 200), 'mkt_cap_musd': rng.uniform(100, 5000)}
            for f in WRDS_FACTOR_COLUMNS:
                if f != 'mkt_cap_musd':
                    row[f] = rng.normal() if rng.random() > 0.05 else np.nan
            rows.append(row)
    panel = pd.DataFrame(rows)[list(PANEL_COLUMNS)]
    panel['permno'] = panel['permno'].astype('int32')
    panel['gsubind'] = panel['gsubind'].astype('Int64')

    returns = pd.DataFrame([{'month_end': m, 'permno': p, 'ret_m': rng.normal(0.01, 0.08), 'delisted_in_month': False}
                            for m in months for p in range(10001, 10001 + n_stocks)])[list(RETURNS_COLUMNS)]
    returns['permno'] = returns['permno'].astype('int32')
    market = pd.DataFrame({'month_end': months,
                           'iwm_total_ret': rng.normal(0.008, 0.05, len(months)),
                           'iwn_total_ret': rng.normal(0.007, 0.05, len(months)),
                           'iwo_total_ret': rng.normal(0.009, 0.06, len(months)),
                           'rf': rng.uniform(0.0, 0.004, len(months)),
                           'tbill_3m': rng.uniform(0.0, 0.05, len(months))})[list(MARKET_COLUMNS)]
    return Frames(panel, returns, market)
