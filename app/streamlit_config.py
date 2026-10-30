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


# --- WRDS mode (DOCS/WRDS_MODE_DESIGN.md) ---------------------------------------------

# Data source switch in the sidebar. Legacy settings carry no data_mode key, exactly
# as before WRDS mode; WRDS settings have data_mode 'wrds'.
DATA_MODE_OPTIONS: List[str] = ['Legacy', 'WRDS']

# Rebalancing frequency: label -> settings['frequency'] (src/wrds_calendar.FREQUENCIES)
WRDS_FREQUENCY_OPTIONS: Dict[str, str] = {
    'Semi-annual': 'semiannual',
    'Quarterly': 'quarterly',
    'Monthly': 'monthly',
}
WRDS_DEFAULT_FREQUENCY: str = 'Quarterly'

"""
WRDS_FACTOR_METADATA:
The factors offered in WRDS mode, in the same shape as FACTOR_METADATA, keyed by
display name. 'column' is a factor_panel_monthly column (the allowlist the Lambda
checks is src/wrds_factors.WRDS_FACTOR_COLUMNS). Keys start with 'w_' so WRDS
widgets never share a key with Legacy ones. Every factor that defaults to Lower
says why in its tooltip.
"""
WRDS_FACTOR_METADATA: Dict[str, Dict[str, Any]] = {
    # Momentum
    '12-1 Mo Momentum': {
        'key': 'w_mom_12_1', 'group': 'Momentum', 'column': 'mom_12_1', 'higher_is_better': True,
        'tooltip': "Return over the past 12 months, skipping the most recent month.",
    },
    '12-Mo Momentum': {
        'key': 'w_mom_12m', 'group': 'Momentum', 'column': 'mom_12m', 'higher_is_better': True,
        'tooltip': "Return over the past 252 trading days.",
    },
    '6-Mo Momentum': {
        'key': 'w_mom_6m', 'group': 'Momentum', 'column': 'mom_6m', 'higher_is_better': True,
        'tooltip': "Return over the past 126 trading days.",
    },
    '1-Month Return (Reversal)': {
        'key': 'w_mom_1m', 'group': 'Momentum', 'column': 'mom_1m', 'higher_is_better': False,
        'tooltip': "Return over the past 21 trading days. "
                   "Lower is better: last month's losers tend to rebound the next month.",
    },
    # Value
    'Book/Price': {
        'key': 'w_btp', 'group': 'Value', 'column': 'f_book_to_price', 'higher_is_better': True,
        'tooltip': "Book equity divided by the company's market cap today.",
    },
    'Earnings/Price': {
        'key': 'w_ep', 'group': 'Value', 'column': 'f_earnings_to_price', 'higher_is_better': True,
        'tooltip': "Trailing 12-month earnings divided by market cap.",
    },
    'Sales/Price': {
        'key': 'w_sp', 'group': 'Value', 'column': 'f_sales_to_price', 'higher_is_better': True,
        'tooltip': "Trailing 12-month revenue divided by market cap.",
    },
    'FCF Yield': {
        'key': 'w_fcf', 'group': 'Value', 'column': 'f_fcf_yield', 'higher_is_better': True,
        'tooltip': "Trailing 12-month free cash flow divided by market cap.",
    },
    'Dividend Yield': {
        'key': 'w_dy', 'group': 'Value', 'column': 'f_dividend_yield', 'higher_is_better': True,
        'tooltip': "Trailing 12-month dividends divided by market cap.",
    },
    'EBITDA/EV': {
        'key': 'w_ebitda_ev', 'group': 'Value', 'column': 'f_ebitda_to_ev', 'higher_is_better': True,
        'tooltip': "Trailing 12-month EBITDA divided by enterprise value.",
    },
    'EBIT/EV': {
        'key': 'w_ebit_ev', 'group': 'Value', 'column': 'f_ebit_to_ev', 'higher_is_better': True,
        'tooltip': "Trailing 12-month EBIT divided by enterprise value.",
    },
    'Sales/EV': {
        'key': 'w_sales_ev', 'group': 'Value', 'column': 'f_sales_to_ev', 'higher_is_better': True,
        'tooltip': "Trailing 12-month revenue divided by enterprise value.",
    },
    'Forward E/P (FY1)': {
        'key': 'w_fwd_ep', 'group': 'Value', 'column': 'f_forward_ep', 'higher_is_better': True,
        'tooltip': "Analyst consensus EPS for this fiscal year divided by price.",
    },
    'Forward E/P (FY2)': {
        'key': 'w_fwd_ep2', 'group': 'Value', 'column': 'f_forward_ep_fy2', 'higher_is_better': True,
        'tooltip': "Analyst consensus EPS for next fiscal year divided by price.",
    },
    'Forward Cash Flow/P': {
        'key': 'w_fwd_cfp', 'group': 'Value', 'column': 'f_forward_cfp', 'higher_is_better': True,
        'tooltip': "Analyst consensus cash flow per share divided by price.",
    },
    'Forward Sales/EV': {
        'key': 'w_fwd_sales_ev', 'group': 'Value', 'column': 'f_forward_sales_to_ev', 'higher_is_better': True,
        'tooltip': "Analyst consensus sales divided by enterprise value.",
    },
    'Forward EBITDA/EV': {
        'key': 'w_fwd_ebitda_ev', 'group': 'Value', 'column': 'f_forward_ebitda_to_ev', 'higher_is_better': True,
        'tooltip': "Analyst consensus EBITDA divided by enterprise value.",
    },
    # Profitability
    'ROE': {
        'key': 'w_roe', 'group': 'Profitability', 'column': 'f_roe', 'higher_is_better': True,
        'tooltip': "Trailing 12-month net income divided by book equity.",
    },
    'ROA': {
        'key': 'w_roa', 'group': 'Profitability', 'column': 'f_roa', 'higher_is_better': True,
        'tooltip': "Trailing 12-month net income divided by total assets.",
    },
    'Gross Profitability': {
        'key': 'w_gp', 'group': 'Profitability', 'column': 'f_gross_profitability', 'higher_is_better': True,
        'tooltip': "Revenue minus cost of goods sold (trailing 12 months), divided by total assets.",
    },
    # Quality
    'Accruals/Assets': {
        'key': 'w_accruals', 'group': 'Quality', 'column': 'f_accruals_to_assets', 'higher_is_better': False,
        'tooltip': "Net income minus operating cash flow, over total assets. "
                   "Lower is better: earnings backed by cash tend to last.",
    },
    'Debt/Assets': {
        'key': 'w_leverage', 'group': 'Quality', 'column': 'f_leverage', 'higher_is_better': False,
        'tooltip': "Total debt divided by total assets. "
                   "Lower is better: less debt means less risk of financial distress.",
    },
    'Current Ratio': {
        'key': 'w_current', 'group': 'Quality', 'column': 'f_current_ratio', 'higher_is_better': True,
        'tooltip': "Current assets divided by current liabilities.",
    },
    'Net Share Issuance': {
        'key': 'w_issuance', 'group': 'Quality', 'column': 'f_net_share_issuance', 'higher_is_better': False,
        'tooltip': "Change in split-adjusted shares outstanding over the past year. "
                   "Lower is better: companies that issue shares tend to trail those that buy them back.",
    },
    # Growth
    '1-Yr Asset Growth': {
        'key': 'w_asset_growth', 'group': 'Growth', 'column': 'f_asset_growth_1yr', 'higher_is_better': False,
        'tooltip': "Total assets vs four quarters earlier. "
                   "Lower is better: fast-growing companies tend to underperform.",
    },
    '1-Yr Sales Growth': {
        'key': 'w_sales_growth', 'group': 'Growth', 'column': 'f_sales_growth_1yr', 'higher_is_better': True,
        'tooltip': "Trailing 12-month revenue vs a year earlier.",
    },
    '1-Yr CapEx Growth': {
        'key': 'w_capex_growth', 'group': 'Growth', 'column': 'f_capex_growth_1yr', 'higher_is_better': False,
        'tooltip': "Trailing 12-month capex vs a year earlier. "
                   "Lower is better: companies that ramp up investment tend to underperform.",
    },
    'R&D/Sales': {
        'key': 'w_rd', 'group': 'Growth', 'column': 'f_rd_to_sales', 'higher_is_better': True,
        'tooltip': "Trailing 12-month R&D divided by revenue.",
    },
    # Risk
    '12-Mo Volatility': {
        'key': 'w_vol_12m', 'group': 'Risk', 'column': 'vol_close_12m', 'higher_is_better': False,
        'tooltip': "Annualized volatility of daily returns over the past year. "
                   "Lower is better: calmer stocks have earned as much with less risk.",
    },
    '6-Mo Volatility': {
        'key': 'w_vol_6m', 'group': 'Risk', 'column': 'vol_close_6m', 'higher_is_better': False,
        'tooltip': "Annualized volatility of daily returns over the past 6 months. "
                   "Lower is better: calmer stocks have earned as much with less risk.",
    },
    '3-Mo Volatility': {
        'key': 'w_vol_3m', 'group': 'Risk', 'column': 'vol_close_3m', 'higher_is_better': False,
        'tooltip': "Annualized volatility of daily returns over the past 3 months. "
                   "Lower is better: calmer stocks have earned as much with less risk.",
    },
    '1-Mo Volatility (High-Low)': {
        'key': 'w_vol_gk_1m', 'group': 'Risk', 'column': 'vol_gk_1m', 'higher_is_better': False,
        'tooltip': "Garman-Klass volatility from daily open, high, low and close over the past month, annualized. "
                   "Lower is better: calmer stocks have earned as much with less risk.",
    },
    'Beta (1-Yr Daily)': {
        'key': 'w_beta_1y', 'group': 'Risk', 'column': 'beta_252d', 'higher_is_better': False,
        'tooltip': "Market beta from daily excess returns over the past year. "
                   "Lower is better: low-beta stocks have earned more than their risk suggests.",
    },
    'Beta (5-Yr Monthly)': {
        'key': 'w_beta_5y', 'group': 'Risk', 'column': 'beta_60m', 'higher_is_better': False,
        'tooltip': "Market beta from monthly excess returns over the past 5 years. "
                   "Lower is better: low-beta stocks have earned more than their risk suggests.",
    },
    # Analyst
    'Earnings Surprise (SUE)': {
        'key': 'w_sue', 'group': 'Analyst', 'column': 'f_sue', 'higher_is_better': True,
        'tooltip': "Change in quarterly EPS vs a year earlier, scaled by its past volatility.",
    },
    'Analyst EPS Surprise': {
        'key': 'w_surprise', 'group': 'Analyst', 'column': 'f_analyst_surprise', 'higher_is_better': True,
        'tooltip': "Latest reported EPS minus the consensus before it, over price.",
    },
    'EPS Revisions': {
        'key': 'w_revisions', 'group': 'Analyst', 'column': 'f_eps_revision', 'higher_is_better': True,
        'tooltip': "Analysts revising this year's EPS up minus those revising down, over the number of analysts.",
    },
    'Analyst Disagreement': {
        'key': 'w_dispersion', 'group': 'Analyst', 'column': 'f_eps_dispersion', 'higher_is_better': False,
        'tooltip': "Spread of analysts' EPS estimates relative to their mean. "
                   "Lower is better: stocks analysts disagree on tend to underperform.",
    },
    'Long-Term Growth Forecast': {
        'key': 'w_ltg', 'group': 'Analyst', 'column': 'f_ltg', 'higher_is_better': False,
        'tooltip': "Analysts' long-term EPS growth forecast. "
                   "Lower is better: the most optimistic forecasts tend to disappoint.",
    },
    # Size
    'Market Cap': {
        'key': 'w_mcap', 'group': 'Size', 'column': 'mkt_cap_musd', 'higher_is_better': False,
        'tooltip': "Market value of the listed shares. "
                   "Lower is better: smaller companies have historically earned more.",
    },
}

# WRDS factor categories in the order the factor list shows them
WRDS_FACTOR_GROUPS: List[str] = ['Momentum', 'Value', 'Profitability', 'Quality',
                                 'Growth', 'Risk', 'Analyst', 'Size']

WRDS_FACTOR_OPTIONS: List[str] = list(WRDS_FACTOR_METADATA.keys())

# How the holdings and ranked tables show each WRDS factor's raw value. The panel stores
# decimals; like Legacy's "... %" columns, rates and returns are shown in percent
# (x 100, header ending in " %"), market cap in $ millions, and other ratios as they are.
WRDS_PERCENT_FACTORS = {
    'mom_12_1', 'mom_12m', 'mom_6m', 'mom_1m', 'f_earnings_to_price', 'f_fcf_yield', 'f_dividend_yield',
    'f_forward_ep', 'f_forward_ep_fy2', 'f_roe', 'f_roa', 'f_gross_profitability', 'f_accruals_to_assets',
    'f_leverage', 'f_net_share_issuance', 'f_asset_growth_1yr', 'f_sales_growth_1yr', 'f_capex_growth_1yr',
    'f_rd_to_sales', 'vol_close_12m', 'vol_close_6m', 'vol_close_3m', 'vol_gk_1m', 'f_analyst_surprise',
    'f_eps_revision', 'f_ltg',
}
WRDS_MILLIONS_FACTORS = {'mkt_cap_musd'}
