"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_calendar.py
PURPOSE: Rebalance months of a WRDS backtest, from its frequency and start and end years.
VERSION: 1.0.0

Works on year-months only, so the sidebar can show a run's rebalances without any
data. The Lambda maps each month to the panel's date in that month (the last CRSP
trading day). See DOCS/WRDS_MODE_DESIGN.md, section 3.
"""

from typing import List

# First month of the WRDS factor panel, and the last month with returns
FIRST_MONTH = '2008-04'
LAST_MONTH = '2026-09'
FIRST_YEAR = int(FIRST_MONTH[:4])
LAST_YEAR = int(LAST_MONTH[:4])

FREQUENCIES = ('semiannual', 'quarterly', 'monthly')
REBALANCE_MONTHS = {
    'semiannual': (6, 12),
    'quarterly': (3, 6, 9, 12),
    'monthly': tuple(range(1, 13)),
}
PERIODS_PER_YEAR = {'semiannual': 2, 'quarterly': 4, 'monthly': 12}


def _index(month: str) -> int:
    """'YYYY-MM' -> months since year 0."""
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def _month(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def period_boundaries(frequency: str, start_year: int, end_year: int,
                      first_month: str = FIRST_MONTH, last_month: str = LAST_MONTH) -> List[str]:
    """
    The run's rebalance months followed by the month its last holding period ends,
    as 'YYYY-MM'. Holding period i runs from boundary i to boundary i + 1.

    The backtest covers the calendar years start_year to end_year: it rebalances first
    in December of start_year - 1 and ends in December of end_year. Where the data does
    not reach that far, it starts at the first rebalance month in the data and ends at
    the last one, so there are no partial periods.

    Raises:
        ValueError: unknown frequency, years outside the data, or no complete period.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}, not '{frequency}'.")
    first_year, last_year = int(first_month[:4]), int(last_month[:4])
    start_year, end_year = int(start_year), int(end_year)
    if not first_year <= start_year <= end_year <= last_year:
        raise ValueError(f"Choose years from {first_year} to {last_year}, with the start year "
                         f"not after the end year (got {start_year}-{end_year}).")

    lo = max(_index(f"{start_year - 1}-12"), _index(first_month))
    hi = min(_index(f"{end_year}-12"), _index(last_month))
    months = [_month(i) for i in range(lo, hi + 1) if i % 12 + 1 in REBALANCE_MONTHS[frequency]]
    if len(months) < 2:
        raise ValueError(f"{start_year}-{end_year} has no complete {frequency} holding period "
                         f"in the data ({first_month} to {last_month}).")
    return months


def describe_schedule(frequency: str, start_year: int, end_year: int) -> str:
    """One line for the sidebar, e.g. "12 quarterly rebalances, 2009-12 to 2012-09; holds to 2012-12"."""
    months = period_boundaries(frequency, start_year, end_year)
    label = {'semiannual': 'semi-annual'}.get(frequency, frequency)
    count = len(months) - 1
    return (f"{count} {label} rebalance{'s' if count > 1 else ''}, "
            f"{months[0]} to {months[-2]}; holds to {months[-1]}")
