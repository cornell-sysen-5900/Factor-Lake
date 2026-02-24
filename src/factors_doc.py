"""
Documentation for available factors: short thesis, implementation column name, and whether higher values are more attractive.

Each entry maps the display column name used across the codebase to a dict with:
 - thesis: one- or two-sentence economic thesis
 - implementation: the exact column name used in the market data (after standardization)
 - higher_is_better: True if larger numbers mean more attractive (normal direction), False if the factor should be inverted

This file is used for documentation and can be imported by CLI helpers or UIs.
"""

FACTOR_DOCS = {
    'ROE using 9/30 Data': {
        'thesis': 'Firms with higher return on equity generate more profit from shareholder capital, indicating efficient capital allocation and stronger profitability prospects.',
        'implementation': 'ROE using 9/30 Data',
        'higher_is_better': True,
    },
    'ROA using 9/30 Data': {
        'thesis': 'Return on assets measures how efficiently a company uses its assets to generate earnings; higher ROA typically signals better operational efficiency.',
        'implementation': 'ROA using 9/30 Data',
        'higher_is_better': True,
    },
    '12-Mo Momentum %': {
        'thesis': 'Stocks that have performed well over the past 12 months tend to continue to outperform in the near-term due to persistent investor behavior and trend continuation.',
        'implementation': '12-Mo Momentum %',
        'higher_is_better': True,
    },
    '6-Mo Momentum %': {
        'thesis': 'Stocks with strong 6-month performance often continue upward in the short-term; this captures intermediate-term momentum.',
        'implementation': '6-Mo Momentum %',
        'higher_is_better': True,
    },
    '1-Mo Momentum %': {
        'thesis': 'One-month momentum captures very short-term trend continuation; higher recent returns indicate near-term strength.',
        'implementation': '1-Mo Momentum %',
        'higher_is_better': True,
    },
    'Price to Book Using 9/30 Data': {
        'thesis': "A lower price-to-book (P/B) implies the stock is cheaper relative to its book value; economically we expect higher book-to-price (inverse of P/B) to indicate value. If P/B is used, it should be inverted so that higher values mean more attractive.",
        'implementation': 'Price to Book Using 9/30 Data',
        'higher_is_better': True,  # Price-to-Book should be inverted for a value factor
    },
    'Next FY Earns/P': {
        'thesis': 'Earnings yield (next fiscal year earnings / price) indicates how cheaply the market prices future earnings; higher values suggest more attractive valuation considering future earnings.',
        'implementation': 'Next FY Earns/P',
        'higher_is_better': True,
    },
    '1-Yr Price Vol %': {
        'thesis': 'Higher trailing 1-year price volatility may indicate higher risk or mispricing; depending on strategy, higher vol can be less attractive for risk-averse portfolios. Here we treat lower volatility as preferable.',
        'implementation': '1-Yr Price Vol %',
        'higher_is_better': True,
    },
    'Accruals/Assets': {
        'thesis': 'High accruals relative to assets can indicate lower earnings quality; lower accrual ratios are generally preferable, so the factor should be inverted (lower is better).',
        'implementation': 'Accruals/Assets',
        'higher_is_better': True,
    },
    'ROA %': {
        'thesis': 'Return on assets (percentage) measures profitability relative to asset base; higher ROA suggests better operating performance.',
        'implementation': 'ROA %',
        'higher_is_better': True,
    },
    '1-Yr Asset Growth %': {
        'thesis': 'Higher asset growth can signal expansion and investment opportunities, though very high growth can be risky. Generally higher is treated as more attractive for growth-oriented factors.',
        'implementation': '1-Yr Asset Growth %',
        'higher_is_better': True,
    },
    '1-Yr CapEX Growth %': {
        'thesis': 'Rising capital expenditures can indicate investment in future growth; interpretation depends on context, but higher CapEx growth is often treated as positive for growth strategies.',
        'implementation': '1-Yr CapEX Growth %',
        'higher_is_better': True,
    },
    'Book/Price': {
        'thesis': 'Book-to-price is the inverse of price-to-book and aligns directly with the value thesis: higher book/price means cheaper relative to book value and thus more attractive.',
        'implementation': 'Book/Price',
        'higher_is_better': True,
    },
}


def print_factors_doc():
    """Print a concise list of factors with thesis and direction."""
    for i, (name, meta) in enumerate(FACTOR_DOCS.items(), start=1):
        direction = 'Higher is better' if meta['higher_is_better'] else 'Lower is better (factor should be inverted)'
        print(f"{i}. {name} - {direction}\n   Thesis: {meta['thesis']}\n")
