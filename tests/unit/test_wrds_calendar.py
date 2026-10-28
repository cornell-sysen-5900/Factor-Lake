"""
Tests for the WRDS rebalance calendar (src/wrds_calendar.py): the rules and the
examples of DOCS/WRDS_MODE_DESIGN.md, section 3.2.
"""
import pytest

from src.wrds_calendar import (FIRST_MONTH, FIRST_YEAR, LAST_MONTH, LAST_YEAR, PERIODS_PER_YEAR,
                               REBALANCE_MONTHS, describe_schedule, period_boundaries)


def _span(frequency, start, end):
    months = period_boundaries(frequency, start, end)
    return months[0], months[-2], months[-1], len(months) - 1


@pytest.mark.parametrize('frequency, start, end, expected', [
    # (first rebalance, last rebalance, last period end, number of rebalances)
    ('monthly', 2008, 2026, ('2008-04', '2026-08', '2026-09', 221)),
    ('quarterly', 2008, 2026, ('2008-06', '2026-06', '2026-09', 73)),
    ('semiannual', 2008, 2026, ('2008-06', '2025-12', '2026-06', 36)),
    ('quarterly', 2010, 2012, ('2009-12', '2012-09', '2012-12', 12)),
    ('monthly', 2015, 2015, ('2014-12', '2015-11', '2015-12', 12)),
    ('semiannual', 2015, 2016, ('2014-12', '2016-06', '2016-12', 4)),
])
def test_design_examples(frequency, start, end, expected):
    assert _span(frequency, start, end) == expected


@pytest.mark.parametrize('frequency', REBALANCE_MONTHS)
def test_every_boundary_is_a_rebalance_month_and_periods_are_consecutive(frequency):
    months = period_boundaries(frequency, 2009, 2020)
    assert all(int(m[5:]) in REBALANCE_MONTHS[frequency] for m in months)
    step = 12 // PERIODS_PER_YEAR[frequency]
    as_index = [int(m[:4]) * 12 + int(m[5:]) for m in months]
    assert all(b - a == step for a, b in zip(as_index, as_index[1:]))


@pytest.mark.parametrize('frequency', REBALANCE_MONTHS)
def test_calendar_years_are_covered_inside_the_data(frequency):
    """Starts in December of the year before and ends in December of the end year."""
    months = period_boundaries(frequency, 2012, 2019)
    assert months[0] == '2011-12' and months[-1] == '2019-12'


@pytest.mark.parametrize('frequency', REBALANCE_MONTHS)
def test_no_period_reaches_outside_the_data(frequency):
    months = period_boundaries(frequency, FIRST_YEAR, LAST_YEAR)
    assert FIRST_MONTH <= months[0] and months[-1] <= LAST_MONTH


@pytest.mark.parametrize('frequency, start, end, message', [
    ('weekly', 2010, 2012, 'frequency must be one of'),
    ('monthly', 2007, 2010, 'Choose years from 2008 to 2026'),
    ('monthly', 2010, 2027, 'Choose years from 2008 to 2026'),
    ('monthly', 2012, 2010, 'not after the end year'),
])
def test_invalid_inputs_are_rejected(frequency, start, end, message):
    with pytest.raises(ValueError, match=message):
        period_boundaries(frequency, start, end)


def test_a_range_without_a_complete_period_is_rejected():
    # The data ends in 2026-09, so a semi-annual run of 2026 holds only the first half year
    assert period_boundaries('semiannual', 2026, 2026) == ['2025-12', '2026-06']
    # With data ending before June, not even that half year is complete
    with pytest.raises(ValueError, match='no complete'):
        period_boundaries('semiannual', 2026, 2026, last_month='2026-05')


def test_description_for_the_sidebar():
    assert describe_schedule('quarterly', 2010, 2012) == \
        "12 quarterly rebalances, 2009-12 to 2012-09; holds to 2012-12"
    assert describe_schedule('semiannual', 2026, 2026) == \
        "1 semi-annual rebalance, 2025-12 to 2025-12; holds to 2026-06"
