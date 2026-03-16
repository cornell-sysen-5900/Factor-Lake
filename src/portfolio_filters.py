"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/portfolio_filters.py
PURPOSE: Pure functions for ESG exclusions and sector-based universe filtering.
VERSION: 1.0.0
"""

import pandas as pd
import logging
from typing import List

# Institutional logging
logger = logging.getLogger(__name__)

def apply_fossil_fuel_exclusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excludes constituents identified within the fossil fuel value chain based 
    on FactSet Industry classifications.
    
    Args:
        df (pd.DataFrame): The universe of stocks for a given period.
        
    Returns:
        pd.DataFrame: A filtered dataframe excluding integrated oil, coal, 
            and related services.
    """
    # Identify industry column (handling potential naming variations)
    industry_col = next((c for c in df.columns if 'industry' in c.lower()), None)
    
    if not industry_col:
        logger.warning("Fossil fuel exclusion requested but industry column not found.")
        return df

    excluded_industries = {
        "integratedoil", 
        "oilfieldservicesequipment", 
        "oilgasproduction", 
        "coal", 
        "oilrefiningmarketing"
    }

    # Clean strings for robust comparison (remove spaces, special chars, and lowercase)
    mask = (
        df[industry_col]
        .astype(str)
        .str.lower()
        .str.replace(r'[^a-z0-9]', '', regex=True)
        .isin(excluded_industries)
    )
    
    filtered_df = df[~mask].copy()
    
    # Log exclusion count if relevant
    excluded_count = len(df) - len(filtered_df)
    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} fossil fuel constituents.")
        
    return filtered_df

def apply_sector_filters(df: pd.DataFrame, selected_sectors: List[str]) -> pd.DataFrame:
    """
    Restricts the investment universe to specific sectors defined in the registry.
    
    Args:
        df (pd.DataFrame): The input market data.
        selected_sectors (List[str]): List of sector names to keep.
        
    Returns:
        pd.DataFrame: Dataframe containing only the selected sectors.
    """
    if not selected_sectors:
        return df

    # We use the standardized sector column from our database schema
    sector_col = 'Scotts_Sector_5'
    
    if sector_col not in df.columns:
        logger.error(f"Sector column '{sector_col}' missing from dataframe.")
        return df

    filtered_df = df[df[sector_col].isin(selected_sectors)].copy()
    
    return filtered_df

def filter_universe(df: pd.DataFrame, exclude_fossil: bool, sectors: List[str]) -> pd.DataFrame:
    """
    Orchestrator function to apply all active constraints to a dataframe copy.
    
    Args:
        df (pd.DataFrame): The raw source data.
        exclude_fossil (bool): Toggle for ESG exclusion.
        sectors (List[str]): List of sectors to include.
        
    Returns:
        pd.DataFrame: The constrained investment universe.
    """
    working_df = df.copy()
    
    if exclude_fossil:
        working_df = apply_fossil_fuel_exclusion(working_df)
        
    if sectors:
        working_df = apply_sector_filters(working_df, sectors)
        
    return working_df