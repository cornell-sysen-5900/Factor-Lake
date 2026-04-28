"""
Regression tests for the backtest engine against live Supabase data.

Each test runs rebalance_portfolio with a fixed factor configuration and
asserts that the final portfolio value and overall growth match previously
calibrated constants. A mismatch means either the engine logic changed or
the upstream Supabase data was modified.

Values last calibrated: 2026-04-13 (time-adjusted delisting engine).
"""
from src.supabase_client import SupabaseManager
from src.backtest_engine import rebalance_portfolio
import pytest
import os
import streamlit as st


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_KEY'),
        reason="Requires Supabase credentials (SUPABASE_URL and SUPABASE_KEY)"
    ),
]

START_YEAR = 2002
END_YEAR = 2023
INITIAL_AUM = 1


@pytest.fixture(scope="module")
def supabase_data():
    if not hasattr(st, 'session_state'):
        st.session_state = {}
    st.session_state.setdefault('exclude_fossil_fuels', False)
    st.session_state.setdefault('selected_sectors', [])

    manager = SupabaseManager()
    return manager.fetch_all_data()


def _assert_backtest(data, factors, factor_directions,
                     expected_final_value, expected_growth):
    result = rebalance_portfolio(
        data, factors, factor_directions,
        START_YEAR, END_YEAR, INITIAL_AUM,
    )
    final_aum = result["final_value"]

    assert final_aum == pytest.approx(expected_final_value, abs=0.05), (
        f"Expected final value {expected_final_value}, got {final_aum}"
    )

    overall_growth = (final_aum - INITIAL_AUM) / INITIAL_AUM * 100
    assert overall_growth == pytest.approx(expected_growth, abs=1.0), (
        f"Expected growth {expected_growth}%, got {overall_growth}%"
    )


class TestSingleFactor:
    """Regression: 6-Mo Momentum only."""

    def test_portfolio_growth(self, supabase_data):
        _assert_backtest(
            supabase_data,
            factors=['6-Mo_Momentum'],
            factor_directions={'6-Mo_Momentum': 'top'},
            expected_final_value=6.94,
            expected_growth=594.09,
        )


class TestMultiFactor:
    """Regression: Momentum + ROE + ROA composite."""

    def test_portfolio_growth(self, supabase_data):
        _assert_backtest(
            supabase_data,
            factors=['6-Mo_Momentum', 'ROE_using_9-30_Data', 'ROA_using_9-30_Data'],
            factor_directions={
                '6-Mo_Momentum': 'top',
                'ROE_using_9-30_Data': 'top',
                'ROA_using_9-30_Data': 'top',
            },
            expected_final_value=8.35,
            expected_growth=735.41,
        )
