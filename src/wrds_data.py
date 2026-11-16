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
- market_monthly.parquet: IWM, IWN and IWO total returns, the risk-free rate, and the
  Fama-French five factors and momentum (for src/factor_risk.py).

Only for backtests with a liquidity limit, prices_daily/ gives each stock's month-end
average daily dollar volume (read_s3_dollar_volume, src/wrds_liquidity.py).
"""

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from .wrds_factors import WRDS_FACTOR_COLUMNS
from .wrds_liquidity import LIQUIDITY_COLUMN, daily_years, read_dollar_volume

# Folder of the WRDS pull under the S3 data prefix (override with FACTOR_LAKE_WRDS_DIR)
DEFAULT_WRDS_DIR = 'wrds_2026_11'
WRDS_DIR_ENV = 'FACTOR_LAKE_WRDS_DIR'

PANEL_FILE = 'factor_panel_monthly.parquet'
RETURNS_FILE = 'returns_monthly.parquet'
MARKET_FILE = 'market_monthly.parquet'
WRDS_FILES = (PANEL_FILE, RETURNS_FILE, MARKET_FILE)

# Panel columns besides the factors: keys, display, filters, price and weight, and the
# four GICS names (shown in the holdings)
PANEL_KEY_COLUMNS = ('date', 'permno', 'ticker', 'security_name', 'sector_name', 'gsubind', 'siccd',
                     'prc', 'mkt_cap_musd', 'gsector_name', 'ggroup_name', 'gind_name', 'gsubind_name')
# Text columns kept as categories (few distinct values, a fraction of the memory)
PANEL_CATEGORY_COLUMNS = ('ticker', 'security_name', 'sector_name', 'gsector_name', 'ggroup_name', 'gind_name',
                          'gsubind_name')
PANEL_COLUMNS = tuple(dict.fromkeys(PANEL_KEY_COLUMNS + WRDS_FACTOR_COLUMNS))
RETURNS_COLUMNS = ('month_end', 'permno', 'ret_m', 'delisted_in_month')
MARKET_COLUMNS = ('month_end', 'iwm_total_ret', 'iwn_total_ret', 'iwo_total_ret', 'rf', 'tbill_3m',
                  'mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd')


def wrds_dir() -> str:
    """The WRDS folder under the S3 data prefix."""
    return os.environ.get(WRDS_DIR_ENV, '').strip().strip('/') or DEFAULT_WRDS_DIR


@dataclass
class WrdsData:
    """
    The WRDS data a backtest reads, kept for the life of a Lambda container.

    returns and delisted are month x permno matrices (rows: months, the market data's
    month ends; columns: permnos), so a holding period's returns are one slice.
    dollar_volume, set only once a job with a liquidity limit needs it, is each stock's
    average daily dollar volume indexed by (date, permno) (src/wrds_liquidity.py).
    """
    panel: pd.DataFrame
    months: pd.DatetimeIndex
    permnos: pd.Index
    returns: np.ndarray
    delisted: np.ndarray
    market: pd.DataFrame
    dollar_volume: Optional[pd.Series] = None
    _month_dates: Dict[str, pd.Timestamp] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, panel: pd.DataFrame, returns_monthly: pd.DataFrame, market: pd.DataFrame) -> 'WrdsData':
        """Builds the return matrices from returns_monthly (one row per month end and permno)."""
        panel = panel.copy()
        for col in PANEL_CATEGORY_COLUMNS:
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


def read_s3_dollar_volume(manager, data_dir: Optional[str] = None) -> pd.Series:
    """Each stock's month-end average daily dollar volume from the S3 daily prices, indexed by (date, permno)."""
    data_dir = data_dir or wrds_dir()
    read_bytes = lambda filename: manager.read_bytes(f'{data_dir}/{filename}')  # noqa: E731
    table = read_dollar_volume(read_bytes, daily_years(read_bytes))
    return table.set_index(['date', 'permno'])[LIQUIDITY_COLUMN]
