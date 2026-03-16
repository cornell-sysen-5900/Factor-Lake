"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/factor_registry.py
PURPOSE: Centralized registry mapping internal factor keys to database column names.
VERSION: 2.0.0
"""

# The FACTOR_REGISTRY serves as the single source of truth for factor naming.
# It maps shorthand internal identifiers to the exact string literals 
# required for Supabase SQL queries.

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
    Retrieves the standardized database column name for a given factor identifier.

    Args:
        factor_key: The shorthand internal identifier (e.g., 'roe').

    Returns:
        str: The corresponding database column name. If the key is not found, 
             the original factor_key is returned as a fallback.
    """
    return FACTOR_REGISTRY.get(factor_key, factor_key)