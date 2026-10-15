"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/data_viewer_filters.py
PURPOSE: Pure helpers for the Data Viewer tab: filter options, filtering the
         loaded market data, and CSV export. No Streamlit calls and no data
         downloads, so they can be unit tested directly.
VERSION: 1.0.0
"""

from typing import List, Optional, Sequence, Tuple

import pandas as pd

SECTOR_COLUMN = 'Scotts_Sector_5'
# The industry column has been published under both names
INDUSTRY_COLUMNS = ('FactSet_Industry', 'FactSet Industry')


def industry_column(df: pd.DataFrame) -> Optional[str]:
    """The industry column present in df, or None if it has none."""
    return next((c for c in INDUSTRY_COLUMNS if c in df.columns), None)


def year_options(df: pd.DataFrame) -> List[int]:
    """Sorted distinct years in df."""
    return sorted(int(y) for y in df['Year'].dropna().unique())


def category_options(df: pd.DataFrame, column: Optional[str]) -> List[str]:
    """Sorted distinct non-empty values of a column, or [] if it is missing."""
    if column is None or column not in df.columns:
        return []
    return sorted(str(v) for v in df[column].dropna().unique())


def filter_raw_data(
    df: pd.DataFrame,
    years: Tuple[int, int],
    sectors: Sequence[str] = (),
    industries: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Rows of df within the year range (inclusive) and, when given, the chosen
    sectors and industries. An empty sector or industry list means "all".

    Returns a new frame; df itself is never modified (it is the shared copy
    of the market data).
    """
    mask = df['Year'].between(years[0], years[1])
    if sectors:
        mask &= df[SECTOR_COLUMN].isin(sectors)
    ind_col = industry_column(df)
    if industries and ind_col is not None:
        mask &= df[ind_col].isin(industries)
    return df.loc[mask].reset_index(drop=True)


def raw_data_to_csv(df: pd.DataFrame) -> bytes:
    """CSV export of the table exactly as displayed (no index column)."""
    return df.to_csv(index=False).encode('utf-8')


def csv_file_name(years: Tuple[int, int]) -> str:
    return f"raw_data_{years[0]}_{years[1]}.csv"
