"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_liquidity.py
PURPOSE: The liquidity limit of a WRDS backtest: its levels, and the cap on each position.
VERSION: 3.0.0

A stock's liquidity is dollar_vol_3m: its average daily dollar volume (price x shares
traded) over its last 63 trading days to t, read from the WRDS folder's daily prices
(read_dollar_volume) only when a job has a limit. Early Nasdaq volume is not adjusted for
double counting (the daily files have no exchange), so before about 2004 a Nasdaq stock's
volume can be overstated.

A liquidity limit is one of LIQUIDITY_LIMITS: the most a position may be, as a percent of
that daily volume. None (no limit) holds what the strategy picks, as before the limit.
With a limit, at each rebalance (cap_positions), each stock's position is

    min(target position, limit x its dollar volume)

where the target position is its weight (Mixing sleeves netted) times the portfolio's value
at that date, before the rebalance's trading cost. A stock without a positive dollar volume
gets no position. Nothing is dropped for being too big and nothing is redistributed: the
money that cannot be placed is held in cash for the period and earns the risk-free rate.
Positions are bought after the trading cost, so they are never above the limit.
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


def cap_positions(held: pd.DataFrame, dollar_volume: pd.Series, portfolio_value: float,
                  limit_pct: float) -> Tuple[Optional[pd.DataFrame], int, int, float]:
    """
    One rebalance date's positions after the liquidity limit (see the module docstring).

    held: the positions of select_positions on one date (sleeve, permno, weight, ...), with
    weights summing to 1. dollar_volume: each permno's average daily dollar volume at that
    date. A stock's position is its weight (Mixing sleeves netted) times portfolio_value.

    Returns (positions, capped, without_volume, cash): the positions with their weights cut
    to at most limit_pct% of the dollar volume over portfolio_value (each sleeve's row cut in
    the same proportion; stocks without a positive volume left out; None when nothing is
    left), how many different stocks were cut, how many had no volume, and the share of the
    value left in cash (1 - the weights kept).
    """
    weights = held.groupby('permno')['weight'].sum()
    adv = pd.to_numeric(dollar_volume, errors='coerce').reindex(weights.index).to_numpy(dtype=float)
    has_volume = adv > 0
    room = np.where(has_volume, limit_pct / 100.0 * np.nan_to_num(adv) / portfolio_value, 0.0)
    kept = np.minimum(weights.to_numpy(), room)
    capped = has_volume & (weights.to_numpy() > room)
    scale = pd.Series(np.divide(kept, weights.to_numpy(), out=np.zeros_like(kept), where=weights.to_numpy() > 0),
                      index=weights.index)
    positions = held.assign(weight=held['weight'] * held['permno'].map(scale).to_numpy())
    positions = positions[positions['weight'] > 0]
    cash = max(0.0, 1.0 - float(kept.sum()))
    return (None if positions.empty else positions), int(capped.sum()), int((~has_volume).sum()), cash


def liquidity_summary(limit_pct: float, rebalance_dates: List[str], positions: List[int], capped: List[int],
                      without_volume: List[int], cash: List[float]) -> Dict[str, Any]:
    """
    What the limit did, for the result: per rebalance date (in order) the stocks the
    strategy picked, how many positions were capped, how many stocks had no dollar volume,
    and the share of the portfolio held in cash (decimal); with the average cash share and
    the rebalance with the most cash.
    """
    cash_a = np.array(cash, dtype=float)
    most = int(np.argmax(cash_a)) if len(cash_a) else 0
    return {
        'limit_pct': float(limit_pct),
        'rebalance_dates': list(rebalance_dates),
        'positions': [int(v) for v in positions],
        'capped': [int(v) for v in capped],
        'without_volume': [int(v) for v in without_volume],
        'cash': [float(v) for v in cash],
        'average_cash': float(cash_a.mean()) if len(cash_a) else 0.0,
        'max_cash': float(cash_a[most]) if len(cash_a) else 0.0,
        'max_cash_date': rebalance_dates[most] if len(rebalance_dates) else None,
    }
