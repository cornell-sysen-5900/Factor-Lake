"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/ranked_stocks.py
PURPOSE: Pure helpers for the per-year Ranked Stocks table in the Results tab:
         default year, display table, and CSV export. No Streamlit calls, so
         they can be unit tested directly.
VERSION: 1.0.0
"""

from typing import Any, Dict, List, Optional

import pandas as pd


def default_ranked_year(years: List[int]) -> Optional[int]:
    """Newest year with a ranked table, or None when there are none."""
    return max(years) if years else None


def build_ranked_display_table(rows: List[Dict[str, Any]], factor_headers: List[str]) -> pd.DataFrame:
    """
    Turns one year's ranked rows into the table shown on screen (and exported).

    Composite Score is rounded to 4 places, Next-Year Return % to 2, and the
    factor value columns are converted to numbers.
    """
    ranked_df = pd.DataFrame(rows)

    if 'Composite Score' in ranked_df.columns:
        ranked_df['Composite Score'] = pd.to_numeric(
            ranked_df['Composite Score'], errors='coerce'
        ).round(4)

    for header in factor_headers:
        if header in ranked_df.columns:
            ranked_df[header] = pd.to_numeric(ranked_df[header], errors='coerce')

    if 'Next-Year Return %' in ranked_df.columns:
        ranked_df['Next-Year Return %'] = pd.to_numeric(
            ranked_df['Next-Year Return %'], errors='coerce'
        ).round(2)

    return ranked_df


def ranked_table_to_csv(ranked_df: pd.DataFrame) -> bytes:
    """CSV export of the ranked table exactly as displayed (no index column)."""
    return ranked_df.to_csv(index=False).encode('utf-8')
