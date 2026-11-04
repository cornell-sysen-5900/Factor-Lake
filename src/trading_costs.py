"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/trading_costs.py
PURPOSE: Trading costs in basis points per trade: turnover, cost and the after-cost results fields.
VERSION: 1.0.0

Shared by the Legacy engine (src/backtest_engine.py) and the WRDS engine
(src/wrds_engine.py). At each rebalance:

1. Drifted weights: last period's weights grown by each stock's return over the period,
   as shares of the portfolio's value at the rebalance. Money from a delisted stock is
   cash: it is in the portfolio's value but is no stock (weight 0), so putting it back
   into stocks is a charged buy and the delisting itself is no charged sell. Before the
   first rebalance the portfolio is all cash, so the first purchase is 100% turnover.
2. Turnover = sum over all stocks of |target weight - drifted weight| (sells and buys
   both count; a stock held by several Mixing sleeves is one position).
3. Cost = bps / 10,000 x turnover x portfolio value, paid at the rebalance, before the
   next period's return. The period's return after costs is
   (1 - cost fraction) x (1 + return before costs) - 1, computed as
   r - c x (1 + r) so that at 0 bps it is exactly r, to the last binary digit.

There is no final sell at the end of the backtest. Costs are the same for every stock;
costs by stock size or liquidity are not modelled.
"""

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

# Largest cost the app and the jobs accept, in basis points per trade
MAX_TRADING_COST_BPS = 500.0
BPS = 10_000.0


def validate_trading_cost_bps(value: Any) -> float:
    """The cost as a float. Raises ValueError with a readable message unless it is a number from 0 to 500."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise ValueError(f"Trading cost must be a number of basis points from 0 to "
                         f"{MAX_TRADING_COST_BPS:g}, not {value!r}.")
    if not 0 <= value <= MAX_TRADING_COST_BPS:
        raise ValueError(f"Trading cost must be from 0 to {MAX_TRADING_COST_BPS:g} bps per trade, not {value:g}.")
    return float(value)


def job_trading_cost(job: Mapping[str, Any]) -> float:
    """A job's or settings' trading cost in bps; 0 when absent (jobs saved before trading costs)."""
    return float(job.get('trading_cost_bps') or 0.0)


def turnover(target: pd.Series, drifted: pd.Series) -> Tuple[float, pd.Series]:
    """
    (total turnover, |target - drifted| per stock) for one rebalance. Both are weights
    by stock (shares of the portfolio's value); a stock missing from one counts as 0
    there. Total turnover is a fraction: 1.0 means trading the portfolio's whole value.
    """
    traded = target.sub(drifted, fill_value=0.0).abs()
    return float(traded.sum()), traded


def cost_fraction(trading_cost_bps: float, period_turnover: float) -> float:
    """The cost of one rebalance as a fraction of the portfolio's value."""
    return trading_cost_bps / BPS * period_turnover


def net_return(gross: float, cost: float) -> float:
    """
    A period's return after paying `cost` (a fraction of the value) at its start:
    (1 - cost) x (1 + gross) - 1, written so that cost 0 gives exactly gross.
    """
    return gross - cost * (1.0 + gross)


def drifted_weights(start_weights: pd.Series, growth: pd.Series, delisted: Optional[pd.Series] = None) -> pd.Series:
    """
    Weights at the end of a period: each stock's start weight times its growth (1 +
    period return), as a share of the portfolio's end value. Delisted stocks' money is
    part of the end value but is cash, so they are left out (weight 0). Empty when the
    portfolio is worth nothing.
    """
    end_values = start_weights * growth
    total = float(end_values.sum())
    if not total > 0:
        return pd.Series(dtype=float)
    if delisted is not None:
        end_values = end_values[~delisted.reindex(end_values.index, fill_value=False).astype(bool)]
    return end_values / total


def _cagr(start: float, end: float, periods: int, periods_per_year: int) -> float:
    if periods <= 0 or not start > 0 or end < 0:
        return 0.0
    return (end / start) ** (periods_per_year / periods) - 1.0


def cost_results(trading_cost_bps: float, gross_returns: Sequence[float], turnovers: Sequence[float],
                 values: Sequence[float], periods_per_year: int = 1) -> Dict[str, Any]:
    """
    The trading cost fields of a backtest's results.

    gross_returns: each period's return before costs (decimal); turnovers: each
    rebalance's turnover (fraction); values: the portfolio values after costs (the
    initial value, then each period's end), as the results' portfolio_values.

    Returns trading_cost_bps; per period turnover_pct, trading_cost_pct (of the value at
    the rebalance) and trading_cost_dollars, returns_before_costs (percent, like
    yearly_returns); portfolio_values_before_costs; average_turnover_pct (per rebalance);
    total_trading_cost ($); cagr_before_costs, cagr_after_costs and cost_drag (their
    difference, decimals).
    """
    n = len(gross_returns)
    costs = [cost_fraction(trading_cost_bps, t) for t in turnovers[:n]]
    before = [float(values[0])]
    for r in gross_returns:
        before.append(before[-1] * (1.0 + r))
    dollars = [c * float(v) for c, v in zip(costs, values[:n])]
    cagr_before = _cagr(before[0], before[-1], n, periods_per_year)
    cagr_after = _cagr(float(values[0]), float(values[n]), n, periods_per_year) if n else 0.0
    return {
        'trading_cost_bps': float(trading_cost_bps),
        'turnover_pct': [t * 100.0 for t in turnovers[:n]],
        'trading_cost_pct': [c * 100.0 for c in costs],
        'trading_cost_dollars': dollars,
        'returns_before_costs': [r * 100.0 for r in gross_returns],
        'portfolio_values_before_costs': before,
        'average_turnover_pct': (sum(turnovers[:n]) / n * 100.0) if n else 0.0,
        'total_trading_cost': float(sum(dollars)),
        'cagr_before_costs': cagr_before,
        'cagr_after_costs': cagr_after,
        'cost_drag': cagr_before - cagr_after,
    }


def has_trading_costs(results: Optional[Mapping[str, Any]]) -> bool:
    """True for results computed with a trading cost above 0 (older results have none)."""
    return bool(results) and float(results.get('trading_cost_bps') or 0.0) > 0


def describe_trading_cost(trading_cost_bps: float) -> str:
    """'25 bps', '12.5 bps'."""
    return f"{trading_cost_bps:g} bps"
