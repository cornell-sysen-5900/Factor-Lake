"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/market_object.py
PURPOSE: Data standardization and auxiliary transformation utilities.
VERSION: 2.1.0
"""

import pandas as pd
import numpy as np
import logging

# Institutional logging
logger = logging.getLogger(__name__)

def standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes varied database schemas into a consistent internal naming convention.
    
    This function is called by the SupabaseManager or Streamlit Utils to ensure
    that disparate column names from the raw SQL pull are mapped to the 
    standardized labels used by the Factor Registry.
    """
    if df.empty:
        return df

    column_mapping = {
        'ROE_using_9-30_Data': 'ROE (Sept 30)',
        'ROA_using_9-30_Data': 'ROA (Sept 30)',
        'Price_to_Book_Using_9-30_Data': 'Price to Book',
        'Next_FY_Earns-P': 'Next FY Earns/P',
        '12-Mo_Momentum': '12-Mo Momentum',
        '6-Mo_Momentum': '6-Mo Momentum',
        '1-Mo_Momentum': '1-Mo Momentum',
        '1-Yr_Price_Vol': '1-Yr Price Vol',
        'Accruals-Assets': 'Accruals/Assets',
        '1-Yr_Asset_Growth': 'Asset Growth %',
        '1-Yr_CapEX_Growth': 'CapEX Growth %',
        'Book-Price': 'Book/Price',
        'Next-Years_Return': "Next_Year_Return",
        'FactSet_Industry': 'FactSet Industry',
        'Scotts_Sector_5': 'Scotts_Sector_5'
    }

    # Execute mapping and clean whitespace
    df = df.rename(columns=column_mapping)
    df.columns = df.columns.str.strip()
    
    # Derive essential identifying columns if missing
    if 'Ticker' not in df.columns:
        ticker_src = next((c for c in ['Ticker-Region', 'ticker_region', 'symbol'] if c in df.columns), None)
        if ticker_src:
            df['Ticker'] = df[ticker_src].str.split('-').str[0].str.strip().str.upper()
            
    if 'Year' not in df.columns:
        date_src = next((c for c in ['Date', 'date', 'period'] if c in df.columns), None)
        if date_src:
            df['Year'] = pd.to_datetime(df[date_src]).dt.year
        
    return _enforce_numerical_integrity(df)

def _enforce_numerical_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures essential columns are cast to floating point numbers for math operations.
    """
    numeric_targets = [
        'Ending Price', 'Market Capitalization', 'Next_Year_Return',
        'Return on Equity', 'Return on Assets'
    ]
    
    for col in df.columns:
        if any(target in col for target in numeric_targets):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows that cannot be ranked (missing Ticker, Year, or Price)
    df = df.dropna(subset=['Ticker', 'Year', 'Ending Price'])
    return df.drop_duplicates()

def apply_fossil_fuel_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excludes constituents identified within the fossil fuel value chain.
    """
    industry_col = next((c for c in df.columns if 'industry' in c.lower()), None)
    if industry_col:
        excluded_industries = {
            "integratedoil", "oilfieldservicesequipment", 
            "oilgasproduction", "coal", "oilrefiningmarketing"
        }
        # Clean string for comparison
        mask = df[industry_col].str.lower().str.replace(r'[^a-z0-9]', '', regex=True).isin(excluded_industries)
        return df[~mask]
    
    return df