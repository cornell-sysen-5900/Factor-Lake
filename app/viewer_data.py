"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/viewer_data.py
PURPOSE: Pure helpers for the Data Viewer tab: filtering the loaded market data and CSV export.
VERSION: 1.0.0
"""

from datetime import date
from typing import List, Optional, Sequence

import pandas as pd

DATE_COL = 'Date'
SECTOR_COL = 'Scotts_Sector_5'
INDUSTRY_COL = 'FactSet_Industry'
# Identifier columns shown first; every other column keeps its loaded order
LEADING_COLUMNS = ['Ticker', 'Security_Name', DATE_COL]

# session_state key of the holder for the currently loaded viewer table
VIEWER_STATE_KEY = 'data_viewer'
# Rows shown on screen; the CSV export always has every filtered row
DISPLAY_ROW_LIMIT = 1000


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


def industries_in_sectors(df: pd.DataFrame, sectors: Sequence[str]) -> List[str]:
    """
    Sorted industries to offer: all of them when no sector is chosen, otherwise only
    those that appear in the chosen sectors. [] if df has no industry column.
    """
    if INDUSTRY_COL not in df.columns:
        return []
    industries = df[INDUSTRY_COL]
    if sectors and SECTOR_COL in df.columns:
        industries = industries[df[SECTOR_COL].isin(list(sectors))]
    return sorted(str(v) for v in industries.dropna().unique())


def keep_valid_selections(selected: Sequence[str], options: Sequence[str]) -> List[str]:
    """The selections still offered in options, in their original order."""
    allowed = set(options)
    return [s for s in selected if s in allowed]


def hidden_rows_caption(total_rows: int) -> Optional[str]:
    """Caption for when only the first DISPLAY_ROW_LIMIT rows are shown, else None."""
    if total_rows <= DISPLAY_ROW_LIMIT:
        return None
    return f"Showing {DISPLAY_ROW_LIMIT:,} of {total_rows:,} rows. Download CSV for all rows."


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    CSV export of every filtered row, with the columns in the order shown on screen.
    The screen shows the first DISPLAY_ROW_LIMIT of these rows.
    """
    return df.to_csv(index=False).encode('utf-8')
