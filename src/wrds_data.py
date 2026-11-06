"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_data.py
PURPOSE: Loads the columns of the WRDS files a WRDS backtest reads, locally or from S3.
VERSION: 1.0.0

Three files of the WRDS pull (pipeline/DATA_DICTIONARY.md) are read, each as bytes and
then decoded with only the needed columns, the same way from a local folder and from S3:

- factor_panel_monthly.parquet: the universe at each month end, the allowlisted factors,
  price and market cap. Columns about the month after t (ret_next_1m,
  delisted_next_month) are never read.
- returns_monthly.parquet: every stock's monthly return (delisting return included)
  and its delisting month, also after it leaves the index.
- market_monthly.parquet: IWM, IWN and IWO total returns and the risk-free rate.
"""

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from .wrds_factors import WRDS_FACTOR_COLUMNS

# Folder of the WRDS pull under the S3 data prefix (override with FACTOR_LAKE_WRDS_DIR)
DEFAULT_WRDS_DIR = 'wrds_2026_10'
WRDS_DIR_ENV = 'FACTOR_LAKE_WRDS_DIR'

PANEL_FILE = 'factor_panel_monthly.parquet'
RETURNS_FILE = 'returns_monthly.parquet'
MARKET_FILE = 'market_monthly.parquet'
WRDS_FILES = (PANEL_FILE, RETURNS_FILE, MARKET_FILE)

# Panel columns besides the factors: keys, display, filters, price and weight
PANEL_KEY_COLUMNS = ('date', 'permno', 'ticker', 'security_name', 'sector_name', 'gsubind', 'siccd',
                     'prc', 'mkt_cap_musd')
PANEL_COLUMNS = tuple(dict.fromkeys(PANEL_KEY_COLUMNS + WRDS_FACTOR_COLUMNS))
RETURNS_COLUMNS = ('month_end', 'permno', 'ret_m', 'delisted_in_month')
MARKET_COLUMNS = ('month_end', 'iwm_total_ret', 'iwn_total_ret', 'iwo_total_ret', 'rf', 'tbill_3m')


def wrds_dir() -> str:
    """The WRDS folder under the S3 data prefix."""
    return os.environ.get(WRDS_DIR_ENV, '').strip().strip('/') or DEFAULT_WRDS_DIR


@dataclass
class WrdsData:
    """
    The WRDS data a backtest reads, kept for the life of a Lambda container.

    returns and delisted are month x permno matrices (rows: months, the market data's
    month ends; columns: permnos), so a holding period's returns are one slice.
    """
    panel: pd.DataFrame
    months: pd.DatetimeIndex
    permnos: pd.Index
    returns: np.ndarray
    delisted: np.ndarray
    market: pd.DataFrame
    _month_dates: Dict[str, pd.Timestamp] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, panel: pd.DataFrame, returns_monthly: pd.DataFrame, market: pd.DataFrame) -> 'WrdsData':
        """Builds the return matrices from returns_monthly (one row per month end and permno)."""
        panel = panel.copy()
        for col in ('ticker', 'security_name', 'sector_name'):
            panel[col] = panel[col].astype('category')
        # Every month end of either file; a holding period over a month without market data is refused
        months = pd.DatetimeIndex(returns_monthly['month_end'].unique()).union(
            pd.DatetimeIndex(market['month_end'])).sort_values()
        market = market.set_index('month_end').reindex(months)
        permnos = pd.Index(np.sort(returns_monthly['permno'].unique()))
        rows = months.get_indexer(returns_monthly['month_end'])
        cols = permnos.get_indexer(returns_monthly['permno'])
        returns = np.full((len(months), len(permnos)), np.nan)
        returns[rows, cols] = returns_monthly['ret_m'].to_numpy(dtype=float)
        delisted = np.zeros((len(months), len(permnos)), dtype=bool)
        delisted[rows, cols] = returns_monthly['delisted_in_month'].to_numpy(dtype=bool)
        return cls(panel=panel, months=months, permnos=permnos, returns=returns,
                   delisted=delisted, market=market,
                   _month_dates={d.strftime('%Y-%m'): d for d in months})

    def month_date(self, month: str) -> pd.Timestamp:
        """The data's month end (last CRSP trading day) of a 'YYYY-MM' month."""
        try:
            return self._month_dates[month]
        except KeyError:
            raise ValueError(f"The WRDS data has no month {month}.") from None

    def panel_dates(self) -> pd.DatetimeIndex:
        """The panel's dates (month ends), sorted."""
        return pd.DatetimeIndex(self.panel['date'].unique()).sort_values()


def read_wrds_data(read_bytes: Callable[[str], bytes]) -> WrdsData:
    """
    Reads the three WRDS files through read_bytes(filename) and decodes only the needed
    columns. Each file's bytes are released before the next file is read.
    """
    frames = []
    for filename, columns in ((PANEL_FILE, PANEL_COLUMNS), (RETURNS_FILE, RETURNS_COLUMNS),
                              (MARKET_FILE, MARKET_COLUMNS)):
        frames.append(pd.read_parquet(io.BytesIO(read_bytes(filename)), columns=list(columns)))
    return WrdsData.build(*frames)


def read_local_wrds_data(folder: str | Path) -> WrdsData:
    """Reads the WRDS files from a local folder (e.g. the pipeline's processed/final)."""
    folder = Path(folder)
    return read_wrds_data(lambda filename: (folder / filename).read_bytes())


def read_s3_wrds_data(manager, data_dir: Optional[str] = None) -> WrdsData:
    """Reads the WRDS files from S3, under <data prefix>/<data_dir>/ (src/s3_client.S3DataManager)."""
    data_dir = data_dir or wrds_dir()
    return read_wrds_data(lambda filename: manager.read_bytes(f'{data_dir}/{filename}'))
