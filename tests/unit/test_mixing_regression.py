"""
Regression test: Mixing (the original way of combining factors) must give the
same portfolio values and returns it gave before Integrating was added.

The expected values in tests/fixtures/mixing_baseline.json were captured from
the engine before the factor combination switch existed. To regenerate them
(only when a change to Mixing results is intended), run:

    uv run python -m tests.unit.test_mixing_regression
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.backtest_engine import rebalance_portfolio

BASELINE_PATH = Path(__file__).resolve().parents[1] / 'fixtures' / 'mixing_baseline.json'
STRATEGIES = ('zero_return', 'hold_cash', 'reinvest')


def multi_year_data() -> pd.DataFrame:
    """Same universe as the multi_year_data fixture in test_backtest.py."""
    rng = np.random.RandomState(42)
    rows = []
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
               'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    for year in range(2020, 2023):
        for t in tickers:
            rows.append({
                'Ticker': t,
                'Year': year,
                'Ending_Price': rng.uniform(50, 500),
                '6-Mo_Momentum': rng.uniform(-0.1, 0.4),
                'ROE_using_9-30_Data': rng.uniform(0.05, 0.35),
                'ROA_using_9-30_Data': rng.uniform(0.02, 0.20),
                'Next-Years_Return': rng.uniform(-20, 40),
            })
    return pd.DataFrame(rows)


def synthetic_universe() -> pd.DataFrame:
    """Same universe the AppTest tests load (test_saved_runs._synthetic_universe)."""
    from tests.unit.test_saved_runs import _synthetic_universe
    return _synthetic_universe()


def delisting_universe() -> pd.DataFrame:
    """
    60 tickers x 2010-2016 with delistings (with and without Delist_Date and
    Delist_Price), missing factor values, and missing or zero prices.
    """
    rng = np.random.RandomState(7)
    rows = []
    for year in range(2010, 2017):
        for i in range(60):
            delisted = rng.uniform() < 0.15
            row = {
                'Ticker': f'D{i:02d}',
                'Year': year,
                'Security_Name': f'Company {i:02d}',
                'Scotts_Sector_5': ['Technology', 'Financials', 'Healthcare'][i % 3],
                'Ending_Price': rng.uniform(5, 200),
                'Market_Capitalization': rng.uniform(1e8, 5e9),
                'ROE_using_9-30_Data': rng.normal(0.1, 0.05),
                'ROA': rng.normal(0.05, 0.03),
                '6-Mo_Momentum': rng.normal(0.05, 0.2),
                'Next-Years_Return': np.nan if delisted else rng.normal(8, 25),
                'Delist_Date': pd.NaT,
                'Delist_Price': np.nan,
            }
            if delisted:
                if rng.uniform() < 0.7:
                    row['Delist_Date'] = pd.Timestamp(year + 1, 1, 1) + pd.Timedelta(days=int(rng.randint(0, 365)))
                if rng.uniform() < 0.6:
                    row['Delist_Price'] = row['Ending_Price'] * rng.uniform(0.1, 1.6)
            if rng.uniform() < 0.08:
                row['ROA'] = np.nan
            if rng.uniform() < 0.05:
                row['ROE_using_9-30_Data'] = np.nan
            if rng.uniform() < 0.03:
                row['Ending_Price'] = 0.0 if rng.uniform() < 0.5 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


UNIVERSES = {
    'multi_year': multi_year_data,
    'synthetic': synthetic_universe,
    'delisting': delisting_universe,
}

# (universe, factors, directions, start, end, top_pct)
SETUPS = [
    ('multi_year', ['6-Mo_Momentum'], {'6-Mo_Momentum': 'top'}, 2020, 2022, 10.0),
    ('multi_year', ['6-Mo_Momentum', 'ROE_using_9-30_Data'],
     {'6-Mo_Momentum': 'top', 'ROE_using_9-30_Data': 'top'}, 2020, 2022, 30.0),
    ('multi_year', ['6-Mo_Momentum', 'ROE_using_9-30_Data', 'ROA_using_9-30_Data'],
     {'6-Mo_Momentum': 'top', 'ROE_using_9-30_Data': 'bottom', 'ROA_using_9-30_Data': 'top'}, 2020, 2022, 20.0),
    ('synthetic', ['ROE_using_9-30_Data', 'ROA'],
     {'ROE_using_9-30_Data': 'top', 'ROA': 'top'}, 2002, 2024, 10.0),
    ('delisting', ['ROE_using_9-30_Data', 'ROA'],
     {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom'}, 2010, 2016, 20.0),
    ('delisting', ['ROE_using_9-30_Data', 'ROA', '6-Mo_Momentum'],
     {'ROE_using_9-30_Data': 'top', 'ROA': 'top', '6-Mo_Momentum': 'top'}, 2010, 2016, 25.0),
]

CASES = [
    (setup, use_mcap, strategy)
    for setup in SETUPS
    for use_mcap in (False, True)
    for strategy in STRATEGIES
]


def case_id(setup, use_mcap, strategy) -> str:
    universe, factors, directions, start, end, top_pct = setup
    dirs = ','.join(f"{f}:{directions[f]}" for f in factors)
    weighting = 'mcap' if use_mcap else 'equal'
    return f"{universe}|{dirs}|{start}-{end}|{top_pct}|{weighting}|{strategy}"


def run_case(setup, use_mcap, strategy, **kwargs):
    universe, factors, directions, start, end, top_pct = setup
    st.session_state['exclude_fossil_fuels'] = False
    st.session_state['selected_sectors'] = []
    return rebalance_portfolio(
        UNIVERSES[universe](), factors, directions, start, end, initial_aum=1000.0,
        top_pct=top_pct, use_market_cap_weight=use_mcap, delisting_strategy=strategy, **kwargs
    )


def summarize(results) -> dict:
    return {
        'portfolio_values': [float(v) for v in results['portfolio_values']],
        'yearly_returns': [float(r) for r in results['yearly_returns']],
        'total_delisted_positions': int(results['total_delisted_positions']),
    }


@pytest.fixture(scope='module')
def baseline():
    return json.loads(BASELINE_PATH.read_text(encoding='utf-8'))


@pytest.mark.parametrize('setup,use_mcap,strategy', CASES, ids=[case_id(*c) for c in CASES])
def test_mixing_results_unchanged(baseline, setup, use_mcap, strategy):
    expected = baseline[case_id(setup, use_mcap, strategy)]
    assert summarize(run_case(setup, use_mcap, strategy)) == expected


@pytest.mark.parametrize('setup,use_mcap,strategy', CASES[:6], ids=[case_id(*c) for c in CASES[:6]])
def test_mixing_is_the_default(baseline, setup, use_mcap, strategy):
    explicit = summarize(run_case(setup, use_mcap, strategy, factor_combination='mixing'))
    assert explicit == baseline[case_id(setup, use_mcap, strategy)]


def test_delisting_universe_exercises_delistings(baseline):
    """The fixture really has delisted positions, or the strategies are not being compared."""
    delisted = [v['total_delisted_positions'] for k, v in baseline.items() if k.startswith('delisting|')]
    assert min(delisted) > 0


if __name__ == '__main__':
    captured = {case_id(*c): summarize(run_case(*c)) for c in CASES}
    BASELINE_PATH.write_text(json.dumps(captured, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(f"Wrote {len(captured)} cases to {BASELINE_PATH}")
