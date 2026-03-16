"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/factor_registry.py
PURPOSE: Centralized registry of factor column names and metadata.
VERSION: 2.0.0
"""

# A simple mapping of internal keys to database column names.
# This eliminates the need for individual classes while maintaining 
# a single source of truth for the engine.

FACTOR_REGISTRY = {
    'momentum_12m': '12-Mo Momentum %',
    'momentum_6m': '6-Mo Momentum %',
    'momentum_1m': '1-Mo Momentum %',
    'roe': 'ROE using 9/30 Data',
    'roa': 'ROA using 9/30 Data',
    'ptb': 'Price to Book Using 9/30 Data',
    'fey': 'Next FY Earns/P',
    'vol': '1-Yr Price Vol %',
    'accruals': 'Accruals/Assets',
    'roa_pct': 'ROA %',
    'asset_growth': '1-Yr Asset Growth %',
    'capex_growth': '1-Yr CapEX Growth %',
    'btp': 'Book/Price'
}

def get_factor_column(factor_key: str) -> str:
    """
    Retrieves the actual database column name for a given factor identifier.
    """
    return FACTOR_REGISTRY.get(factor_key, factor_key)