"""Read-only single-factor long/short diagnostics and the rule for WRDS default directions.

At month end t, sort eligible stocks by the known factor, form disjoint equal-weight
upper/lower tails, and observe their CRSP return in t+1 (including delistings).
This is a raw-factor diagnostic, not an app multi-factor strategy. It never edits the
app: choose_default() recommends a direction, and the team sets it in streamlit_config.
"""
import math

import numpy as np
import pandas as pd

from .wrds_engine import eligible_universe

# The default-direction rule. These are the only copies of its parameters.
START_YEAR, END_YEAR = 2002, 2026
CONSISTENT_SHARE = 0.70   # a direction that wins at least this share of calendar years is the default
RECENT_MONTHS = 60        # otherwise, the direction whose spread is positive over the latest months
CONSISTENT, MIXED = 'consistent', 'mixed'


def monthly_spreads(panel, returns, directions, *, start_year=START_YEAR, end_year=END_YEAR, tail_pct=10.0):
    """Return diagnostics and monthly series, without mutating either input.

    directions: {allowlisted factor column: True for Higher, False for Lower}.
    Missing next-month stock returns retain their original weight and receive 0%,
    consistent with the app; they are counted and explicitly flagged in the report.
    A month with no return file coverage, <2 signals or a constant signal is omitted
    and counted, rather than being reported as zero evidence. Both tails use the
    same finite-signal universe. Ties break by PERMNO and tails cannot overlap.
    """
    from .wrds_factors import WRDS_FACTOR_COLUMNS
    if not directions or not set(directions) <= set(WRDS_FACTOR_COLUMNS):
        raise ValueError('Choose at least one known WRDS factor.')
    if any(type(value) is not bool for value in directions.values()):
        raise ValueError('Each direction must be True (Higher) or False (Lower).')
    if not (math.isfinite(tail_pct) and 0 < tail_pct <= 50):
        raise ValueError('tail_pct must be above 0 and at most 50 for disjoint tails.')
    if not 1900 <= start_year <= end_year <= 2200:
        raise ValueError('Choose ordered calendar years between 1900 and 2200.')
    panel_months = pd.to_datetime(panel['date']).dt.to_period('M')
    if pd.DataFrame({'month': panel_months, 'permno': panel['permno']}).duplicated().any():
        raise ValueError('Duplicate factor panel month/PERMNO rows.')
    r = returns[['month_end', 'permno', 'ret_m']].copy()
    r['return_month'] = pd.to_datetime(r['month_end']).dt.to_period('M')
    if r.duplicated(['return_month', 'permno']).any():
        raise ValueError('Duplicate monthly return/PERMNO rows.')
    r['ret_m'] = pd.to_numeric(r['ret_m'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if (r['ret_m'].dropna() < -1).any():
        raise ValueError('A stock return cannot be below -100%.')
    # Index once: each tail lookup should touch its own month, not merge the full
    # multi-decade return file again for every factor/month/leg.
    returns_by_month = {month: group.set_index('permno')['ret_m']
                        for month, group in r.groupby('return_month', sort=False)}
    known_months = set(returns_by_month)
    universe = eligible_universe(panel).copy()
    universe['signal_date'] = pd.to_datetime(universe['date'])
    universe['return_month'] = universe['signal_date'].dt.to_period('M') + 1
    # Desired window refers to realized returns, not the formation dates.
    universe = universe[universe['return_month'].dt.year.between(start_year, end_year)]
    return_months = panel_months + 1
    expected_months = set(return_months[return_months.dt.year.between(start_year, end_year)])
    universe = universe[universe['return_month'].isin(known_months)]
    summaries, monthly_rows = {}, []
    for column, higher in directions.items():
        signal = universe[['signal_date', 'return_month', 'permno', column]].copy()
        signal['value'] = pd.to_numeric(signal.pop(column), errors='coerce').replace([np.inf, -np.inf], np.nan)
        signal = signal.dropna(subset=['value'])
        for month, group in signal.groupby('return_month', sort=True):
            if len(group) < 2 or group['value'].nunique() < 2:
                continue
            ordered = group.sort_values(['value', 'permno'], kind='mergesort')
            count = min(len(group) // 2, max(1, math.floor(len(group) * tail_pct / 100.0)))
            low, high = ordered.head(count), ordered.tail(count)
            legs = []
            for selected in (high, low):
                realized = returns_by_month[month].reindex(selected['permno'])
                legs.append((float(realized.fillna(0).mean()), int(realized.isna().sum())))
            high_ret, low_ret = legs[0][0], legs[1][0]
            long_ret, short_ret = (high_ret, low_ret) if higher else (low_ret, high_ret)
            monthly_rows.append({'column': column, 'signal_date': group['signal_date'].max().strftime('%Y-%m-%d'),
                                 'return_month': str(month),
                                 'eligible_signals': len(group), 'stocks_per_leg': count,
                                 'high_return': high_ret, 'low_return': low_ret,
                                 'long_return': long_ret, 'short_return': short_ret,
                                 'spread': long_ret - short_ret,
                                 'missing_returns': legs[0][1] + legs[1][1]})
        rows = [row for row in monthly_rows if row['column'] == column]
        observed = len(rows)
        missing = sum(row['missing_returns'] for row in rows)
        skipped = len(expected_months) - observed
        observed_returns = sum(2 * row['stocks_per_leg'] - row['missing_returns'] for row in rows)
        mean = float(np.mean([row['spread'] for row in rows])) if observed_returns else None
        annualized = mean * 12 if mean is not None else None
        status = ('insufficient data' if mean is None else 'team review' if mean < 0
                  else 'agrees in sample' if mean > 0 else 'neutral in sample')
        # Calendar years of realized returns (partial first/last years included), each the
        # sum of its monthly spreads, consistent with the arithmetic annualized spread.
        yearly = {}
        for row in rows:
            year = row['return_month'][:4]
            yearly[year] = yearly.get(year, 0.0) + row['spread']
        positive_years = sum(value > 0 for value in yearly.values())
        recent = [row['spread'] for row in rows[-RECENT_MONTHS:]]
        summaries[column] = {
            'months': observed, 'skipped_months': skipped, 'missing_returns': missing,
            'first_return_month': rows[0]['return_month'] if rows else None,
            'last_return_month': rows[-1]['return_month'] if rows else None,
            'mean_monthly_spread': mean, 'annualized_spread': annualized,
            'years': len(yearly), 'positive_years': positive_years,
            'negative_years': sum(value < 0 for value in yearly.values()),
            'share_years_positive': positive_years / len(yearly) if yearly and mean is not None else None,
            'recent_months': len(recent),
            'recent_spread': float(np.mean(recent)) * 12 if recent and mean is not None else None,
            'evidence_status': status,
            'coverage_status': ('review coverage' if not observed or missing or skipped
                                else 'complete for available panel months'),
        }
    return summaries, monthly_rows


def choose_default(summary, higher):
    """(higher_is_better, CONSISTENT | MIXED) from a summary computed with direction `higher`, or None.

    1. A direction that wins (positive summed spread) in at least CONSISTENT_SHARE of the calendar
       years is the default: "consistent".
    2. Otherwise the direction with a positive spread over the latest RECENT_MONTHS: "mixed".
       A recent spread of exactly zero keeps the current direction.
    """
    if summary['annualized_spread'] is None or not summary['years']:
        return None
    if summary['positive_years'] / summary['years'] >= CONSISTENT_SHARE:
        return higher, CONSISTENT
    if summary['negative_years'] / summary['years'] >= CONSISTENT_SHARE:
        return not higher, CONSISTENT
    return (higher if summary['recent_spread'] >= 0 else not higher), MIXED
