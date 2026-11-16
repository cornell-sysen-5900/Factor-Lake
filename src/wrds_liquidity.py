"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_liquidity.py
PURPOSE: The liquidity limit of a WRDS backtest: its levels, and the cap on each position.
VERSION: 2.0.0

A stock's liquidity is dollar_vol_3m: its average daily dollar volume (price x shares
traded) over its last 63 trading days to t, read from the WRDS folder's daily prices
(read_dollar_volume) only when a job has a limit. Early Nasdaq volume is not adjusted for
double counting (the daily files have no exchange), so before about 2004 a Nasdaq stock's
volume can be overstated.

A liquidity limit is one of LIQUIDITY_LIMITS: the most a position may be, as a percent of
that daily volume. None (no limit) holds what the strategy picks, as before the limit.
With a limit, at each rebalance (cap_positions):

1. Each stock's position is its weight (Mixing sleeves netted) times the portfolio's
   value at that date, before the rebalance's trading cost.
2. Stocks without a positive dollar volume, and stocks whose position is over the limit
   times their dollar volume, are dropped.
3. The rest are reweighted with the chosen weighting: within each Mixing sleeve when the
   sleeves hold their own top X% (each sleeve keeps its share of the money; a sleeve left
   empty gives its share to the others), else across all the stocks.
4. Steps 1 to 3 repeat until every position fits (or nothing is left: the date is held in
   cash).

The limit applies at every rebalance, so a held stock that no longer fits is not bought
again, i.e. sold. Positions are bought after the trading cost, so they are never above the
limit.
"""

import io
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# The panel column the limit reads: average daily dollar volume ($ per day)
LIQUIDITY_COLUMN = 'dollar_vol_3m'
# The one setting with the levels: the most a position may be, in percent of the stock's
# average daily dollar volume. Order: loosest first.
LIQUIDITY_LIMITS = (20.0, 10.0, 5.0)
NO_LIMIT_LABEL = 'No limit'

NO_VOLUME_MESSAGE = ("The daily dollar volume (from the WRDS folder's prices_daily) is not loaded, so the "
                     "liquidity limit cannot run. Set Liquidity Limit to No limit.")


# Dollar volume from the WRDS folder's daily prices (prices_daily/year=YYYY/data.parquet):
# the mean of prc x vol over a stock's last LIQUIDITY_WINDOW trading rows, needing
# MIN_OBS_SHARE of them, as pipeline/stages/s02_prices.py computes its rolling windows.
# Only these columns are read, one year at a time.
DAILY_DIR = 'prices_daily'
DAILY_MANIFEST = f'{DAILY_DIR}/_manifest.json'
DAILY_VOLUME_COLUMNS = ('date', 'permno', 'prc', 'vol')
LIQUIDITY_WINDOW = 63
MIN_OBS_SHARE = 0.8
def _rolling_mean(x: pd.Series, g: pd.Series, n: int) -> pd.Series:
    """Mean over the last n rows per permno (fewer before the n-th row), NaN unless
    MIN_OBS_SHARE x n values are present: s02_prices._rolling's mean."""
    s1 = x.fillna(0).groupby(g).cumsum()
    c = x.notna().astype('int32').groupby(g).cumsum()
    first = c.groupby(g).shift(n).isna()
    d1 = (s1 - s1.groupby(g).shift(n).fillna(0)).where(~first, s1)
    k = (c - c.groupby(g).shift(n).fillna(0)).where(~first, c)
    return (d1 / k).where(k >= MIN_OBS_SHARE * n)


def read_dollar_volume(read_bytes: Callable[[str], bytes], years: Iterable[int]) -> pd.DataFrame:
    """
    (date, permno, dollar_vol_3m) at each month end: date is the last trading day of the
    month in the daily data (the panel's date), and the value is the stock's average daily
    dollar volume over its last LIQUIDITY_WINDOW trading rows as of its last row with a
    price in that month. read_bytes(filename) reads one file of the WRDS folder.

    The years are read in order, one at a time; each stock's last LIQUIDITY_WINDOW rows
    are carried into the next year, so the windows are those of the whole history.
    """
    window = LIQUIDITY_WINDOW
    tail = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'), 'permno': pd.Series(dtype='int64'),
                         'dv': pd.Series(dtype=float)})
    kept = []
    for year in sorted(int(y) for y in years):
        px = pd.read_parquet(io.BytesIO(read_bytes(f'{DAILY_DIR}/year={year}/data.parquet')),
                             columns=list(DAILY_VOLUME_COLUMNS))
        px['date'] = pd.to_datetime(px['date']).astype('datetime64[ns]')
        px['permno'] = px['permno'].astype('int64')
        rows = pd.DataFrame({'date': px['date'], 'permno': px['permno'],
                             'dv': px['prc'] * px['vol'].where(px['vol'] >= 0),
                             'priced': px['prc'].notna() & (px['prc'] > 0), 'new': True})
        del px
        rows = pd.concat([tail.assign(priced=False, new=False), rows], ignore_index=True)
        rows = rows.sort_values(['permno', 'date'], kind='stable', ignore_index=True)
        rows['dollar_vol_3m'] = _rolling_mean(rows['dv'], rows['permno'], window)
        tail = rows.groupby('permno', sort=False).tail(window)[['date', 'permno', 'dv']].reset_index(drop=True)
        new = rows[rows['new'].to_numpy()]
        month = new['date'].dt.to_period('M')
        month_end = new['date'].groupby(month).transform('max')       # the month's last trading day
        last = new[new['priced'].to_numpy()].groupby([new['permno'], month]).tail(1).index
        kept.append(pd.DataFrame({'date': month_end.loc[last].to_numpy(), 'permno': new.loc[last, 'permno'].to_numpy(),
                                  'dollar_vol_3m': new.loc[last, 'dollar_vol_3m'].to_numpy()}))
        del rows, new
    table = pd.concat(kept, ignore_index=True) if kept else tail.iloc[0:0, :2].assign(dollar_vol_3m=np.nan)
    table = table.dropna(subset=['dollar_vol_3m'])
    table['permno'] = table['permno'].astype('int32')
    table['dollar_vol_3m'] = table['dollar_vol_3m'].astype('float32')
    return table.sort_values(['date', 'permno'], ignore_index=True)


def daily_years(read_bytes: Callable[[str], bytes]) -> list:
    """The years the WRDS folder's daily prices have (its _manifest.json)."""
    return sorted(int(y) for y in json.loads(read_bytes(DAILY_MANIFEST))['years'])


def job_liquidity_limit(value: Any) -> Optional[float]:
    """
    The limit to apply in percent (one of LIQUIDITY_LIMITS), or None for no limit (a
    missing field or None). Raises ValueError for anything else.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or float(value) not in LIQUIDITY_LIMITS:
        allowed = ', '.join(f'{v:g}' for v in LIQUIDITY_LIMITS)
        raise ValueError(f"liquidity_limit must be one of {allowed} (percent of daily volume) or left out, "
                         f"not {value!r}.")
    return float(value)


def limit_label(value: Any) -> str:
    """'No limit', or e.g. '10%' (the sidebar's options)."""
    limit = job_liquidity_limit(value)
    return NO_LIMIT_LABEL if limit is None else f"{limit:g}%"


def describe_liquidity_limit(value: Any) -> str:
    """'No limit', or e.g. 'Max 10% of daily volume'."""
    limit = job_liquidity_limit(value)
    return NO_LIMIT_LABEL if limit is None else f"Max {limit:g}% of daily volume"


def add_volume(panel: pd.DataFrame, dollar_volume: Optional[pd.Series]) -> pd.DataFrame:
    """
    panel with a LIQUIDITY_COLUMN column: each row's dollar volume from dollar_volume
    (indexed by date and permno; NaN where it has none). Raises ValueError when
    dollar_volume is not loaded or is empty.
    """
    if dollar_volume is None or dollar_volume.empty:
        raise ValueError(NO_VOLUME_MESSAGE)
    permno_type = dollar_volume.index.levels[1].dtype
    key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(panel['date']).astype(dollar_volume.index.levels[0].dtype),
                                     panel['permno'].to_numpy().astype(permno_type)])
    return panel.assign(**{LIQUIDITY_COLUMN: dollar_volume.reindex(key).to_numpy(dtype=float)})


def cap_positions(held: pd.DataFrame, dollar_volume: pd.Series, portfolio_value: float, limit_pct: float,
                  by_sleeve: bool) -> Tuple[Optional[pd.DataFrame], int, int]:
    """
    One rebalance date's positions after the liquidity limit (see the module docstring).

    held: the positions of select_positions on one date (sleeve, permno, weight, ...).
    dollar_volume: each permno's average daily dollar volume at that date.
    by_sleeve: reweight within each sleeve (Mixing with Top X%), else across all stocks.

    Returns (positions, dropped, dropped_without_volume): the positions left with their new
    weights (None when every stock was dropped), and how many different stocks were dropped,
    of which how many for having no dollar volume.
    """
    cap = limit_pct / 100.0
    groups = held['sleeve'] if by_sleeve else pd.Series(0, index=held.index)
    group_share = held['weight'].groupby(groups).sum()
    volume = pd.to_numeric(dollar_volume, errors='coerce')
    positions, dropped, no_volume = held, 0, 0
    while True:
        weights = positions.groupby('permno')['weight'].sum()
        adv = volume.reindex(weights.index).to_numpy(dtype=float)
        missing = ~(adv > 0)
        over = missing | (weights.to_numpy() * portfolio_value > cap * np.nan_to_num(adv))
        if not over.any():
            return positions, dropped, no_volume
        out = weights.index[over]
        dropped += len(out)
        no_volume += int(missing.sum())
        positions = positions[~positions['permno'].isin(out)]
        if positions.empty:
            return None, dropped, no_volume
        # Reweight: each group keeps its share, then shares of emptied groups go to the rest
        kept = positions['weight'].groupby(groups.loc[positions.index]).transform('sum')
        share = groups.loc[positions.index].map(group_share)
        weight = positions['weight'] * share / kept
        positions = positions.assign(weight=weight / weight.sum())


def liquidity_summary(limit_pct: float, rebalance_dates: List[str], candidates: List[int], dropped: List[int],
                      dropped_without_volume: List[int]) -> Dict[str, Any]:
    """
    What the limit dropped, for the result: per rebalance date (in order) the stocks the
    strategy picked before the limit, how many were dropped, and of those how many had no
    dollar volume; with averages and the date that dropped the most.
    """
    candidates_a, dropped_a = np.array(candidates, dtype=float), np.array(dropped, dtype=float)
    share = np.divide(dropped_a * 100.0, candidates_a, out=np.zeros_like(dropped_a), where=candidates_a > 0)
    most = int(np.argmax(dropped_a)) if len(dropped_a) else 0
    return {
        'limit_pct': float(limit_pct),
        'rebalance_dates': list(rebalance_dates),
        'candidates': [int(v) for v in candidates],
        'dropped': [int(v) for v in dropped],
        'dropped_without_volume': [int(v) for v in dropped_without_volume],
        'average_dropped': float(dropped_a.mean()) if len(dropped_a) else 0.0,
        'average_dropped_pct': float(share.mean()) if len(share) else 0.0,
        'max_dropped': int(dropped_a[most]) if len(dropped_a) else 0,
        'max_dropped_date': rebalance_dates[most] if len(rebalance_dates) else None,
        'total_dropped': int(dropped_a.sum()),
    }
