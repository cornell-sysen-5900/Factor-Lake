"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/liquidity_ui.py
PURPOSE: What the app says about the WRDS liquidity limit: the sidebar's options and
         captions, and the Results tab's line, tiles and per-rebalance table. Text and
         tables only, no Streamlit calls.

The levels and their meaning are src/wrds_liquidity.py's; nothing here repeats a limit.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.wrds_liquidity import (
    LIQUIDITY_LIMITS,
    NO_LIMIT_LABEL,
    describe_liquidity_limit,
    job_liquidity_limit,
    limit_label,
)

# Sidebar: widget key, options (label -> limit in percent, None for no limit) and the default
LIQUIDITY_KEY = "wrds_liquidity_limit_input"
LIQUIDITY_OPTIONS = {NO_LIMIT_LABEL: None, **{limit_label(v): v for v in LIQUIDITY_LIMITS}}
LIQUIDITY_DEFAULT = NO_LIMIT_LABEL
LIQUIDITY_HELP = (
    "The most a position may be, as a share of the stock's average daily dollar volume over the last 3 months, "
    "so the backtest only holds positions it could really "
    "trade. At each rebalance every position is sized at the portfolio's value on that date with the chosen "
    "weighting and cut to the limit if it is bigger. The money that cannot be placed, and the money of stocks "
    "with no volume data, is held in cash until the next rebalance and earns the risk-free rate; the other "
    "positions keep their weights. The bigger the Initial AUM, the more positions are capped."
)


def option_limit(label: str) -> Optional[float]:
    """The limit of a sidebar option in percent, None for No limit."""
    return job_liquidity_limit(LIQUIDITY_OPTIONS[label])


def sidebar_caption(limit: Optional[float]) -> str:
    """The line under the sidebar's choice: what can be held at this limit."""
    limit = job_liquidity_limit(limit)
    if limit is None:
        return "Positions of any size can be held."
    return (f"No position above {limit:g}% of its stock's average daily dollar volume, checked at each "
            "rebalance.")


def results_line(settings: Dict[str, Any]) -> str:
    """The Results header's caption for a WRDS run: its saved limit."""
    return f"**Liquidity Limit:** {describe_liquidity_limit(settings.get('liquidity_limit'))}"


def results_title(summary: Dict[str, Any]) -> str:
    """The heading of a run's liquidity tiles, e.g. 'Liquidity Limit (max 10% of daily volume)'."""
    return f"**Liquidity Limit ({describe_liquidity_limit(summary['limit_pct']).lower()})**"


def results_tiles(summary: Optional[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """
    (label, value, help) of the Results tab's liquidity tiles, from the result's
    liquidity_limit (src/wrds_liquidity.liquidity_summary). [] for a run without a limit.
    """
    if not summary or 'cash' not in summary:
        return []
    most = f"{summary['max_cash']:.1%}" + (f" ({summary['max_cash_date']})" if summary['max_cash'] > 0 else '')
    return [
        ("Avg Cash Share", f"{summary['average_cash']:.1%}",
         "The share of the portfolio held in cash because of the limit, averaged over the rebalances. "
         "It earns the risk-free rate."),
        ("Most Cash", most, "The largest cash share at one rebalance, and its date."),
        ("Avg Positions Capped", f"{np.mean(summary['capped']) if summary['capped'] else 0:,.1f}",
         "Positions cut to the limit at each rebalance, averaged over the rebalances."),
    ]


def capped_table(summary: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    One row per rebalance date: the stocks the strategy picked, how many positions the limit
    capped, how many stocks had no volume data (no position), and the cash share. None for a
    run without a limit.
    """
    if not summary or 'cash' not in summary:
        return None
    return pd.DataFrame({
        'Rebalance Date': summary['rebalance_dates'],
        'Picked': summary['positions'],
        'Capped': summary['capped'],
        'Without Volume Data': summary['without_volume'],
        'Cash': [f"{c:.1%}" for c in summary['cash']],
    })


def ranked_with_holdings(rows: pd.DataFrame, holdings: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    A run's ranked stocks with Held and the weight taken from its holdings on that date:
    with a liquidity limit, what the strategy holds depends on the portfolio's value, which
    a ranked list does not know. Without holdings the rows are returned unchanged.
    """
    if holdings is None:
        return rows
    weights = holdings.groupby('permno')['weight'].sum()
    return rows.assign(held=rows['permno'].isin(weights.index), weight=rows['permno'].map(weights).fillna(0.0))
