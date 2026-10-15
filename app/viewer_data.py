"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/viewer_data.py
PURPOSE: Pure helpers for the Data Viewer tab: filtering the loaded market data and CSV export.
VERSION: 1.0.0
"""

from datetime import date
from typing import Sequence

import pandas as pd

DATE_COL = 'Date'
SECTOR_COL = 'Scotts_Sector_5'
INDUSTRY_COL = 'FactSet_Industry'
# Identifier columns shown first; every other column keeps its loaded order
LEADING_COLUMNS = ['Ticker', 'Security_Name', DATE_COL]

# session_state key of the holder for the currently loaded viewer table
VIEWER_STATE_KEY = 'data_viewer'


def filter_market_data(df: pd.DataFrame, start: date, end: date,
                       sectors: Sequence[str], industries: Sequence[str]) -> pd.DataFrame:
    """
    Returns the rows dated from start to end (inclusive) in the chosen sectors and
    industries. An empty sector or industry list means no filter on that column.
    The result is a new frame, so freeing it never touches the loaded market data.
    """
    dates = pd.to_datetime(df[DATE_COL], errors='coerce')
    mask = dates.between(pd.Timestamp(start), pd.Timestamp(end))
    if sectors:
        mask &= df[SECTOR_COL].isin(list(sectors))
    if industries:
        mask &= df[INDUSTRY_COL].isin(list(industries))
    leading = [c for c in LEADING_COLUMNS if c in df.columns]
    columns = leading + [c for c in df.columns if c not in leading]
    return df.loc[mask, columns].reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV export of exactly the table shown on screen (same rows, columns and order)."""
    return df.to_csv(index=False).encode('utf-8')
