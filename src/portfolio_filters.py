"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/portfolio_filters.py
PURPOSE: Pure functions for ESG exclusions and sector-based universe filtering.
VERSION: 1.0.0
"""

import pandas as pd
import logging
from typing import List

# Institutional logging configuration
logger = logging.getLogger(__name__)

def apply_fossil_fuel_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excludes constituents within the fossil fuel value chain based on industry classification.
    
    This function targets integrated oil, gas production, and coal-related services
    to align the portfolio with ESG-constrained mandates.
    """
    # Locate industry column using a case-insensitive fuzzy match
    industry_col = next((c for c in df.columns if 'industry' in c.lower()), None)
    
    if not industry_col:
        logger.warning("Fossil fuel exclusion requested but no industry column was identified.")
        return df

    # Defined list of industries categorized under the fossil fuel value chain
    excluded_industries = {
        "integratedoil", 
        "oilfieldservicesequipment", 
        "oilgasproduction", 
        "coal", 
        "oilrefiningmarketing"
    }

    # Normalize industry strings for robust set membership comparison
    normalized_series = (
        df[industry_col]
        .astype(str)
        .str.lower()
        .str.replace(r'[^a-z0-9]', '', regex=True)
    )
    
    mask = normalized_series.isin(excluded_industries)
    filtered_df = df[~mask].copy()
    
    excluded_count = len(df) - len(filtered_df)
    if excluded_count > 0:
        logger.info(f"ESG Filter: Removed {excluded_count} fossil fuel constituents.")
        
    return filtered_df

def apply_sector_filters(df: pd.DataFrame, selected_sectors: List[str]) -> pd.DataFrame:
    """
    Restricts the investment universe to specific sectors defined in the registry.
    """
    if not selected_sectors:
        return df

    # Primary sector column identifier for the 2026 schema
    sector_col = 'Scotts_Sector_5'
    
    if sector_col not in df.columns:
        logger.error(f"Critical Error: Sector column '{sector_col}' is missing from the dataset.")
        return df

    # Efficiently filter to include only the requested sectors
    return df[df[sector_col].isin(selected_sectors)].copy()

def filter_universe(df: pd.DataFrame, exclude_fossil: bool, sectors: List[str]) -> pd.DataFrame:
    """
    Orchestrator function applying active constraints to the investment universe.
    """
    if df.empty:
        return df

    # Apply sequential filters to a local copy to preserve original data integrity
    working_df = df.copy()
    
    if exclude_fossil:
        working_df = apply_fossil_fuel_exclusion(working_df)
        
    if sectors:
        working_df = apply_sector_filters(working_df, sectors)
        
    return working_df