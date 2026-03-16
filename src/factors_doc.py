"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/factor_doc.py
PURPOSE: Documentation for available factors including economic theses and implementation mappings.
VERSION: 1.1.0
"""

# The FACTOR_DOCS registry provides the conceptual framework for each quantitative signal.
# Each entry maps the display name to its economic justification and database implementation.
FACTOR_DOCS = {
    'ROE using 9/30 Data': {
        'thesis': 'Firms with higher return on equity generate superior profit from shareholder capital, indicating efficient management and robust profitability prospects.',
        'implementation': 'ROE using 9/30 Data',
    },
    'ROA using 9/30 Data': {
        'thesis': 'Return on assets measures the efficiency with which a company utilizes its asset base to generate earnings; higher values signal operational excellence.',
        'implementation': 'ROA using 9/30 Data',
    },
    '12-Mo Momentum %': {
        'thesis': 'Empirical evidence suggests that securities with high relative strength over the past 12 months tend to persist in their outperformance due to trend continuation.',
        'implementation': '12-Mo Momentum %',
    },
    '6-Mo Momentum %': {
        'thesis': 'Captures intermediate-term price trends; securities with strong 6-month returns often benefit from continued investor interest in the short term.',
        'implementation': '6-Mo Momentum %',
    },
    '1-Mo Momentum %': {
        'thesis': 'Reflects short-term price velocity; high recent returns serve as a proxy for immediate market sentiment and trend strength.',
        'implementation': '1-Mo Momentum %',
    },
    'Price to Book Using 9/30 Data': {
        'thesis': 'A low price-to-book ratio identifies stocks trading at a discount relative to their accounting net worth, a core pillar of the value premium.',
        'implementation': 'Price to Book Using 9/30 Data',
    },
    'Next FY Earns/P': {
        'thesis': 'The forward earnings yield represents the expected return on investment based on projected earnings, identifying valuation discounts on future cash flows.',
        'implementation': 'Next FY Earns/P',
    },
    '1-Yr Price Vol %': {
        'thesis': 'Price volatility serves as a proxy for risk; low-volatility strategies target stability, while high-volatility targets capture higher-beta market exposure.',
        'implementation': '1-Yr Price Vol %',
    },
    'Accruals/Assets': {
        'thesis': 'High accruals relative to assets may signal aggressive accounting or lower earnings quality, suggesting a potential mean-reversion in future performance.',
        'implementation': 'Accruals/Assets',
    },
    'ROA %': {
        'thesis': 'A standardized measure of profitability relative to the total asset base, reflecting the core operating efficiency of the business entity.',
        'implementation': 'ROA %',
    },
    '1-Yr Asset Growth %': {
        'thesis': 'Securities with moderate asset growth often signal expansion; however, excessive growth can precede diminishing returns on invested capital.',
        'implementation': '1-Yr Asset Growth %',
    },
    '1-Yr CapEX Growth %': {
        'thesis': 'Investment in capital expenditures reflects management confidence in future growth and long-term infrastructure development.',
        'implementation': '1-Yr CapEX Growth %',
    },
    'Book/Price': {
        'thesis': 'The direct inverse of price-to-book; higher values identify securities with the most significant valuation margin relative to book equity.',
        'implementation': 'Book/Price',
    },
}

def print_factors_doc() -> None:
    """
    Outputs a structured list of factors and their economic theses to the console.
    
    This function serves as a CLI-based documentation tool for developers 
    and analysts to verify the qualitative logic behind quantitative signals.
    """
    header = f"{'#':<3} | {'Factor Name':<35} | {'Economic Thesis'}"
    print(header)
    print("-" * len(header) * 2)
    
    for i, (name, meta) in enumerate(FACTOR_DOCS.items(), start=1):
        # Using structured printing for a professional, spreadsheet-like appearance
        print(f"{i:<3} | {name:<35} | {meta['thesis']}")