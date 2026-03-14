"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_config.py
PURPOSE: Centralized configuration for sector definitions and UI-to-Engine factor mappings.
VERSION: 1.1.0
"""

from typing import Dict, Any, List

# Import financial factor logic from the source directory.
# These represent the mathematical implementations used by the rebalancing engine.
from src.factor_function import (
    Momentum6m, Momentum12m, Momentum1m, ROE, ROA, 
    P2B, NextFYrEarns, OneYrPriceVol, AccrualsAssets, 
    ROAPercentage, OneYrAssetGrowth, OneYrCapEXGrowth, BookPrice
)

# Standardized sector classifications available for universe filtering.
SECTOR_OPTIONS: List[str] = [
    'Consumer',
    'Technology',
    'Financials',
    'Industrials',
    'Healthcare'
]

"""
FACTOR_MAP:
This dictionary maps human-readable UI labels to their respective Python 
logic classes. By storing the class references directly, the Streamlit 
utilities can instantiate these objects dynamically during execution.
"""
FACTOR_MAP: Dict[str, Any] = {
    'ROE using 9/30 Data': ROE,
    'ROA using 9/30 Data': ROA,
    '12-Mo Momentum %': Momentum12m,
    '6-Mo Momentum %': Momentum6m,
    '1-Mo Momentum %': Momentum1m,
    'Price to Book Using 9/30 Data': P2B,
    'Next FY Earns/P': NextFYrEarns,
    '1-Yr Price Vol %': OneYrPriceVol,
    'Accruals/Assets': AccrualsAssets,
    'ROA %': ROAPercentage,
    '1-Yr Asset Growth %': OneYrAssetGrowth,
    '1-Yr CapEX Growth %': OneYrCapEXGrowth,
    'Book/Price': BookPrice
}