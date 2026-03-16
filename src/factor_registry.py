"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/factor_registry.py
PURPOSE: Centralized registry mapping internal factor keys to database column names.
VERSION: 2.1.0
"""

# The FACTOR_REGISTRY serves as the single source of truth for factor naming.
# Updated to match the exact database schema (PostgreSQL column names).

FACTOR_REGISTRY = {
    'momentum_12m': '12-Mo_Momentum',
    'momentum_6m': '6-Mo_Momentum',
    'momentum_1m': '1-Mo_Momentum',
    'roe': 'ROE_using_9-30_Data',
    'roa': 'ROA_using_9-30_Data',
    'ptb': 'Price_to_Book_Using_9-30_Data',
    'fey': 'Next_FY_Earns-P',
    'vol': '1-Yr_Price_Vol',
    'accruals': 'Accruals-Assets',
    'roa_pct': 'ROA',
    'asset_growth': '1-Yr_Asset_Growth',
    'capex_growth': '1-Yr_CapEX_Growth',
    'btp': 'Book-Price'
}

def get_factor_column(factor_key: str) -> str:
    """
    Retrieves the standardized database column name for a given factor identifier.
    """
    return FACTOR_REGISTRY.get(factor_key, factor_key)