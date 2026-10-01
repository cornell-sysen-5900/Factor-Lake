"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_config.py
PURPOSE: Centralized registry mapping UI labels to backend database column names.
VERSION: 2.3.0
"""

from typing import Dict, List, Any

# Maximum number of backtest runs kept as tabs in Results (oldest dropped first)
MAX_SAVED_RUNS: int = 5

# Standardized sector classifications for universe filtering
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
The single registry for every factor offered in the UI, keyed by its UI label
(the label is what the Analysis tab returns and what run_backtest_logic maps).

Each entry defines:
- key: Short id used in widget keys.
- group: Category shown in the "Add factor" dropdown and under the factor name.
- column: The exact database field name.
- tooltip: Hover text shown next to the factor name.
- higher_is_better: Default direction when the factor is added
  (True = "Higher is better" = 'top', False = "Lower is better" = 'bottom').

Entries are listed in FACTOR_GROUPS order, which is the dropdown order.
"""
FACTOR_METADATA: Dict[str, Dict[str, Any]] = {
    '12-Mo Momentum %': {
        'key': '12m',
        'group': 'Momentum',
        'column': '12-Mo_Momentum',
        'tooltip': "Price change over the past 12 months. Excludes dividends.",
        'higher_is_better': True
    },
    '6-Mo Momentum %': {
        'key': '6m',
        'group': 'Momentum',
        'column': '6-Mo_Momentum',
        'tooltip': "Price change over the past 6 months. Excludes dividends.",
        'higher_is_better': True
    },
    '1-Mo Momentum %': {
        'key': '1m',
        'group': 'Momentum',
        'column': '1-Mo_Momentum',
        'tooltip': "Price change over the past month. Excludes dividends.",
        # Default pending confirmation: short term reversal may argue for Lower.
        'higher_is_better': True
    },
    'Price to Book Using 9/30 Data': {
        'key': 'ptb',
        'group': 'Value',
        'column': 'Price_to_Book_Using_9-30_Data',
        'tooltip': "Price divided by book value per share.",
        'higher_is_better': False
    },
    'Book/Price': {
        'key': 'btp',
        'group': 'Value',
        'column': 'Book-Price',
        'tooltip': "Book equity divided by market cap, from the latest quarter.",
        'higher_is_better': True
    },
    'Next FY Earns/P': {
        'key': 'fey',
        'group': 'Value',
        'column': 'Next_FY_Earns-P',
        'tooltip': "Analyst consensus EPS for next fiscal year divided by price. Needs 3+ analysts.",
        'higher_is_better': True
    },
    'ROE using 9/30 Data': {
        'key': 'roe',
        'group': 'Profitability',
        'column': 'ROE_using_9-30_Data',
        'tooltip': "Net income divided by shareholders' equity.",
        'higher_is_better': True
    },
    'ROA using 9/30 Data': {
        'key': 'roa',
        'group': 'Profitability',
        'column': 'ROA_using_9-30_Data',
        'tooltip': "Net income divided by total assets.",
        'higher_is_better': True
    },
    'ROA %': {
        'key': 'roa_pct',
        'group': 'Profitability',
        'column': 'ROA',
        'tooltip': "Trailing 12 month net income divided by total assets.",
        'higher_is_better': True
    },
    'Accruals/Assets': {
        'key': 'accruals',
        'group': 'Quality',
        'column': 'Accruals-Assets',
        'tooltip': "Net income minus operating cash flow, over total assets. Lower means more cash backed earnings.",
        'higher_is_better': False
    },
    '1-Yr Price Vol %': {
        'key': 'vol',
        'group': 'Quality',
        'column': '1-Yr_Price_Vol',
        'tooltip': "Annualized volatility of daily returns over the past year.",
        'higher_is_better': False
    },
    '1-Yr Asset Growth %': {
        'key': 'asset_growth',
        'group': 'Growth',
        'column': '1-Yr_Asset_Growth',
        'tooltip': "Change in total assets vs one year ago.",
        'higher_is_better': False
    },
    '1-Yr CapEX Growth %': {
        'key': 'capex_growth',
        'group': 'Growth',
        'column': '1-Yr_CapEX_Growth',
        'tooltip': "Change in 12 month capex vs the prior year.",
        'higher_is_better': False
    }
}

# Factor categories in the order the "Add factor" dropdown lists them
FACTOR_GROUPS: List[str] = ['Momentum', 'Value', 'Profitability', 'Quality', 'Growth']

# Derived list of available factors for UI rendering
FACTOR_OPTIONS: List[str] = list(FACTOR_METADATA.keys())
