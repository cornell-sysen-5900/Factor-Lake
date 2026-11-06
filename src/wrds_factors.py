"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_factors.py
PURPOSE: The factor_panel_monthly columns a WRDS backtest may rank stocks on.
VERSION: 1.0.0

Every column here was public by the panel date t (see pipeline/DATA_DICTIONARY.md),
so it can be a factor without look-ahead. Columns about the month after t, such as
ret_next_1m and delisted_next_month, are never allowed. The app's display names,
groups and tooltips for these columns are WRDS_FACTOR_METADATA in app/streamlit_config.py.
"""

WRDS_FACTOR_COLUMNS = (
    # Momentum
    'mom_12_1', 'mom_12m', 'mom_6m', 'mom_1m',
    # Value
    'f_book_to_price', 'f_earnings_to_price', 'f_sales_to_price', 'f_fcf_yield', 'f_dividend_yield',
    'f_ebitda_to_ev', 'f_ebit_to_ev', 'f_sales_to_ev', 'f_forward_ep', 'f_forward_ep_fy2',
    'f_forward_cfp', 'f_forward_sales_to_ev', 'f_forward_ebitda_to_ev',
    # Profitability
    'f_roe', 'f_roa', 'f_gross_profitability',
    # Quality
    'f_accruals_to_assets', 'f_leverage', 'f_current_ratio', 'f_net_share_issuance',
    # Growth
    'f_asset_growth_1yr', 'f_sales_growth_1yr', 'f_capex_growth_1yr', 'f_rd_to_sales',
    # Risk
    'vol_close_12m', 'vol_close_6m', 'vol_close_3m', 'vol_gk_1m', 'beta_252d', 'beta_60m',
    # Analyst
    'f_sue', 'f_analyst_surprise', 'f_eps_revision', 'f_eps_dispersion', 'f_ltg',
    # Size
    'mkt_cap_musd',
)

# The 11 GICS sectors of the panel's sector_name column. Stocks with no sector
# ('Unclassified') are only in the universe while the sector filter is off.
WRDS_SECTORS = (
    'Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials',
    'Health Care', 'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Utilities',
)
UNCLASSIFIED_SECTOR = 'Unclassified'
