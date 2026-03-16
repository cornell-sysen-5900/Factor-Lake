"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/factor_doc.py
PURPOSE: Documentation for available factors including economic theses and implementation mappings.
VERSION: 1.1.0
"""

# Each entry maps the display column name used across the codebase to a dict with:
#  - thesis: one- or two-sentence economic thesis
#  - implementation: the exact column name used in the market data
FACTOR_DOCS = {
    'ROE using 9/30 Data': {
        'thesis': 'Firms with higher return on equity generate more profit from shareholder capital, indicating efficient capital allocation and stronger profitability prospects.',
        'implementation': 'ROE using 9/30 Data',
    },
    'ROA using 9/30 Data': {
        'thesis': 'Return on assets measures how efficiently a company uses its assets to generate earnings; higher ROA typically signals better operational efficiency.',
        'implementation': 'ROA using 9/30 Data',
    },
    '12-Mo Momentum %': {
        'thesis': 'Stocks that have performed well over the past 12 months tend to continue to outperform in the near-term due to persistent investor behavior and trend continuation.',
        'implementation': '12-Mo Momentum %',
    },
    '6-Mo Momentum %': {
        'thesis': 'Stocks with strong 6-month performance often continue upward in the short-term; this captures intermediate-term momentum.',
        'implementation': '6-Mo Momentum %',
    },
    '1-Mo Momentum %': {
        'thesis': 'One-month momentum captures very short-term trend continuation; higher recent returns indicate near-term strength.',
        'implementation': '1-Mo Momentum %',
    },
    'Price to Book Using 9/30 Data': {
        'thesis': "A lower price-to-book (P/B) implies the stock is cheaper relative to its book value; economically we expect higher book-to-price (inverse of P/B) to indicate value.",
        'implementation': 'Price to Book Using 9/30 Data',
    },
    'Next FY Earns/P': {
        'thesis': 'Earnings yield (next fiscal year earnings / price) indicates how cheaply the market prices future earnings; higher values suggest more attractive valuation.',
        'implementation': 'Next FY Earns/P',
    },
    '1-Yr Price Vol %': {
        'thesis': 'Higher trailing 1-year price volatility may indicate higher risk or mispricing; depending on strategy, lower volatility is often targeted for defensive stability.',
        'implementation': '1-Yr Price Vol %',
    },
    'Accruals/Assets': {
        'thesis': 'High accruals relative to assets can indicate lower earnings quality; analyzing the non-cash component of earnings helps identify potential future performance mean-reversion.',
        'implementation': 'Accruals/Assets',
    },
    'ROA %': {
        'thesis': 'Return on assets (percentage) measures profitability relative to asset base; higher ROA suggests better operating performance.',
        'implementation': 'ROA %',
    },
    '1-Yr Asset Growth %': {
        'thesis': 'Higher asset growth can signal expansion and investment opportunities, though excessively high growth can occasionally indicate overextension.',
        'implementation': '1-Yr Asset Growth %',
    },
    '1-Yr CapEX Growth %': {
        'thesis': 'Rising capital expenditures can indicate investment in future growth; interpretation depends on context, but higher CapEx growth is often treated as a growth signal.',
        'implementation': '1-Yr CapEX Growth %',
    },
    'Book/Price': {
        'thesis': 'Book-to-price is the inverse of price-to-book and aligns directly with the value thesis: higher book/price means cheaper relative to book value.',
        'implementation': 'Book/Price',
    },
}

def print_factors_doc():
    """
    Prints a concise list of factors with their corresponding economic thesis.
    
    This function iterates through the FACTOR_DOCS registry to provide a human-readable 
    summary of the available financial metrics and the investment logic behind them.
    """
    for i, (name, meta) in enumerate(FACTOR_DOCS.items(), start=1):
        print(f"{i}. {name}\n   Thesis: {meta['thesis']}\n")