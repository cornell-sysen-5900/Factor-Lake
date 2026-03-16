"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_config.py
PURPOSE: Configuration mapping UI labels to Supabase SQL column names.
VERSION: 2.2.0
"""

from typing import Dict, List, Any

# Standardized sector classifications based on Scotts_Sector_5
SECTOR_OPTIONS: List[str] = [
    'Consumer',
    'Technology',
    'Financials',
    'Industrials',
    'Healthcare',
    'Energy',
    'Materials',
    'Utilities'
]

"""
FACTOR_METADATA:
This registry maps the Streamlit UI selection strings to the exact 
SQL column names found in the Supabase 'Full Precision Test' table.
"""
FACTOR_METADATA: Dict[str, Dict[str, Any]] = {
    'ROE using 9/30 Data': {
        'column': 'ROE_using_9-30_Data',
        'higher_is_better': True
    },
    'ROA using 9/30 Data': {
        'column': 'ROA_using_9-30_Data',
        'higher_is_better': True
    },
    '12-Mo Momentum %': {
        'column': '12-Mo_Momentum',
        'higher_is_better': True
    },
    '6-Mo Momentum %': {
        'column': '6-Mo_Momentum',
        'higher_is_better': True
    },
    '1-Mo Momentum %': {
        'column': '1-Mo_Momentum',
        'higher_is_better': True
    },
    'Price to Book Using 9/30 Data': {
        'column': 'Price_to_Book_Using_9-30_Data',
        'higher_is_better': False
    },
    'Next FY Earns/P': {
        'column': 'Next_FY_Earns-P',
        'higher_is_better': True
    },
    '1-Yr Price Vol %': {
        'column': '1-Yr_Price_Vol',
        'higher_is_better': False
    },
    'Accruals/Assets': {
        'column': 'Accruals-Assets',
        'higher_is_better': False
    },
    'ROA %': {
        'column': 'ROA',
        'higher_is_better': True
    },
    '1-Yr Asset Growth %': {
        'column': '1-Yr_Asset_Growth',
        'higher_is_better': False
    },
    '1-Yr CapEX Growth %': {
        'column': '1-Yr_CapEX_Growth',
        'higher_is_better': False
    },
    'Book/Price': {
        'column': 'Book-Price',
        'higher_is_better': True
    }
}

FACTOR_OPTIONS: List[str] = list(FACTOR_METADATA.keys())