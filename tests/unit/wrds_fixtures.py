"""
Synthetic WRDS data for the WRDS engine tests: the three files' columns, with random
values, built from scratch so no licensed data is needed.

Month ends are the last business day of each month (like CRSP's last trading day), not
calendar month ends. Tests change the frames and rebuild WrdsData from them.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.wrds_data import MARKET_COLUMNS, PANEL_COLUMNS, RETURNS_COLUMNS, WrdsData
from src.wrds_factors import WRDS_FACTOR_COLUMNS, WRDS_SECTORS


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
