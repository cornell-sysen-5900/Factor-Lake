"""
Pinned Legacy results: fixed Legacy jobs must give exactly the results they gave on
main before WRDS mode was added (DOCS/WRDS_MODE_DESIGN.md, section 1.2).

Each job is built with build_job and run with run_job, the path both the app's Lambda
backtests and the Lambda itself use, and the whole result (metrics, values, returns
and every holdings row) is compared with tests/fixtures/legacy_pinned_results.json.
Legacy's code version may change during WRDS work; its results may not.

Everything that is not a number (keys, tickers, ranks, flags, row order) must be equal.
Numbers must agree to a relative 1e-9, the tolerance scripts/check_backtest_lambda.py
uses for Lambda vs local: the fixture was captured on Windows and CI runs on Linux,
where a few values (betas, from LAPACK) can differ in the last binary digit.

The fixture was captured from main (c063100). Regenerate it only when a change to
Legacy results is intended:

    uv run python -m tests.unit.test_legacy_pinned
"""
import json
from pathlib import Path

import pytest

from scripts.check_backtest_lambda import TOLERANCE, max_difference
from src.backtest_jobs import build_job, encode_result, run_job
from tests.unit.test_mixing_regression import delisting_universe, synthetic_universe

FIXTURE_PATH = Path(__file__).resolve().parents[1] / 'fixtures' / 'legacy_pinned_results.json'

UNIVERSES = {'synthetic': synthetic_universe, 'delisting': delisting_universe}

# name -> (universe, kind, factors, directions, sidebar settings, cohort_pct)
JOBS = {
    'mixing_equal_zero_return': (
        'synthetic', 'backtest', ['ROE_using_9-30_Data', 'ROA'], {'ROE_using_9-30_Data': 'top', 'ROA': 'top'},
        dict(start_year=2002, end_year=2024, initial_aum=1000.0), None),
    'integrating_mcap_hold_cash_sectors': (
        'delisting', 'backtest', ['ROE_using_9-30_Data', 'ROA'], {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom'},
        dict(start_year=2010, end_year=2016, initial_aum=5000.0, use_market_cap_weight=True,
             delisting_strategy='hold_cash', factor_combination='integrating',
             selected_sectors=['Technology', 'Financials']), None),
    'mixing_mcap_reinvest_three_factors': (
        'delisting', 'backtest', ['ROE_using_9-30_Data', 'ROA', '6-Mo_Momentum'],
        {'ROE_using_9-30_Data': 'top', 'ROA': 'top', '6-Mo_Momentum': 'bottom'},
        dict(start_year=2010, end_year=2016, initial_aum=1000.0, use_market_cap_weight=True,
             delisting_strategy='reinvest'), None),
    'cohort_integrating_20pct': (
        'delisting', 'cohort', ['ROE_using_9-30_Data', '6-Mo_Momentum'],
        {'ROE_using_9-30_Data': 'top', '6-Mo_Momentum': 'top'},
        dict(start_year=2010, end_year=2016, initial_aum=1000.0, factor_combination='integrating'), 20.0),
}


def run_pinned(name):
    universe, kind, factors, directions, settings, cohort_pct = JOBS[name]
    job = build_job(kind, factors, directions, settings, cohort_pct=cohort_pct)
    return job, run_job(UNIVERSES[universe](), job)


@pytest.fixture(scope='module')
def pinned():
    return json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))


@pytest.mark.parametrize('name', JOBS)
def test_legacy_job_gives_the_results_it_gave_on_main(pinned, name):
    job, result = run_pinned(name)
    expected = pinned[name]
    assert job == expected['job']
    difference, where = max_difference(json.loads(encode_result(result)), expected['result'])
    assert difference <= TOLERANCE, f"{where} differs from main by {difference:.2e} (relative)"


def test_pinned_jobs_cover_holdings_and_delistings(pinned):
    """The fixture really holds what the test claims to compare."""
    backtests = [v['result'] for v in pinned.values() if v['job']['kind'] == 'backtest']
    assert all(r['holdings_by_year'] for r in backtests)
    assert any(r['total_delisted_positions'] > 0 for r in backtests)


if __name__ == '__main__':
    captured = {}
    for name in JOBS:
        job, result = run_pinned(name)
        captured[name] = {'job': job, 'result': json.loads(encode_result(result))}
    FIXTURE_PATH.write_text(json.dumps(captured, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(f"Wrote {len(captured)} jobs to {FIXTURE_PATH}")
