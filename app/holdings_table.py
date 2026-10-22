"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/holdings_table.py
PURPOSE: Pure helpers for the per-year Holdings and Ranks table in the Results
         tab: default year, display table, number formats, period return and
         CSV export. No Streamlit calls, so they can be unit tested directly.
VERSION: 2.0.0
"""

from typing import Any, Dict, List, Optional

import pandas as pd

# Display names for the factor_combination setting
FACTOR_COMBINATION_LABELS = {'mixing': 'Mixing', 'integrating': 'Integrating'}

# Columns converted to numbers for display (factor value columns are added per run)
NUMBER_COLUMNS = ['Score', 'Weight %', 'Shares', 'Buy Price', 'Dollar Value',
                  'Period Return %', 'Contribution %']


def default_holdings_year(years: List[int]) -> Optional[int]:
    """Newest year with holdings, or None when there are none."""
    return max(years) if years else None


def build_holdings_display_table(rows: List[Dict[str, Any]], factor_headers: List[str]) -> pd.DataFrame:
    """
    Turns one year's holdings rows (from the backtest) into the table shown on
    screen and exported.

    Number columns are converted to numbers but kept at full precision; they
    are rounded on screen only (see holdings_number_formats).
    """
    holdings_df = pd.DataFrame(rows)
    for col in NUMBER_COLUMNS + list(factor_headers):
        if col in holdings_df.columns:
            holdings_df[col] = pd.to_numeric(holdings_df[col], errors='coerce')
    if 'Delisted' in holdings_df.columns:
        holdings_df['Delisted'] = holdings_df['Delisted'].astype(bool)
    return holdings_df


def holdings_number_formats(factor_headers: List[str]) -> Dict[str, str]:
    """On-screen number formats for the holdings table columns."""
    formats = {'Rank': '%d', 'Score': '%.4f'}
    formats.update({header: '%.4f' for header in factor_headers})
    formats.update({
        'Weight %': '%.2f',
        'Shares': '%.4f',
        'Buy Price': 'dollar',
        'Dollar Value': 'dollar',
        'Period Return %': '%.2f',
        'Contribution %': '%.2f',
    })
    return formats


def holdings_period_return(results: Dict[str, Any], year: int) -> Optional[float]:
    """
    The backtest's portfolio return (in percent) for the period that starts at
    rebalance year `year`, or None when that year has no return.

    holdings_by_year holds the rebalance years in the same order as
    yearly_returns, so the year's position gives its return.
    """
    years = list(results.get('holdings_by_year') or {})
    returns = results.get('yearly_returns') or []
    if year not in years or years.index(year) >= len(returns):
        return None
    return returns[years.index(year)]


def holdings_table_to_csv(holdings_df: pd.DataFrame) -> bytes:
    """CSV export of the holdings table exactly as displayed (no index column)."""
    return holdings_df.to_csv(index=False).encode('utf-8')
