"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/data_standardization.py
PURPOSE: Shared cleaning of the raw investment-universe tables, used by every data source.
VERSION: 1.0.0
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns the backtest engine cannot run without
REQUIRED_COLUMNS: List[str] = ['Ticker', 'Year', 'Ending_Price', 'Next-Years_Return']


def standardize_universe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces structural and numerical consistency across the dataset.

    Takes the raw 'Full Precision Test' table (as stored in Supabase or S3)
    and returns the frame the app works with. Safe to run more than once.
    """
    # 1. Column Hygiene
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 2. Identifier Parsing
    if 'Ticker-Region' in df.columns:
        df['Ticker'] = df['Ticker-Region'].str.split('-').str[0].str.strip().str.upper()

    # 3. Temporal Alignment
    if 'Date' in df.columns:
        df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

    # 4. Standardized Null Conversion
    sentinel_values = ['--', 'N/A', '#N/A', 'NULL', 'null', 'nan', '']
    df = df.replace(sentinel_values, np.nan)

    # 5. Dynamic Numeric Conversion
    # Automatically cast columns that contain factor-lake signals or price data
    numeric_keywords = ['Price', 'Return', 'Weight', 'Data', 'ROE', 'ROA', 'Cap']
    for col in df.columns:
        if any(key in col for key in numeric_keywords):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 6. Schema Integrity Validation
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error(f"Integrity Error: Missing critical schema columns: {missing}")

    return df


def prepare_last_price_mapping(lpm: pd.DataFrame) -> pd.DataFrame:
    """Renames the raw last_price_mapping table to the delisting columns the engine reads."""
    if lpm.empty:
        return lpm
    lpm = lpm.rename(columns={'ticker': 'Ticker-Region', 'last_date': 'Delist_Date', 'last_price': 'Delist_Price'})
    lpm['Delist_Date'] = pd.to_datetime(lpm['Delist_Date'], errors='coerce')
    return lpm


def attach_delisting_info(df: pd.DataFrame, lpm: pd.DataFrame) -> pd.DataFrame:
    """Merges delisting dates and prices onto the universe for time-adjusted strategies."""
    if lpm.empty:
        return df
    return df.merge(lpm[['Ticker-Region', 'Delist_Date', 'Delist_Price']], on='Ticker-Region', how='left')


def validate_universe(df: pd.DataFrame, source: str) -> None:
    """
    Raises a readable error when a data source returned nothing usable.

    Without this check an empty download surfaces later as a bare KeyError
    such as 'Year', which tells the user nothing about the real cause.
    """
    if df is None or df.empty:
        raise ValueError(
            f"The {source} data source returned no rows. Check that the data source is "
            f"reachable and its credentials are configured, then refresh the page."
        )
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"The {source} data is missing required columns: {missing}.")
