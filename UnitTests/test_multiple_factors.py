from src.supabase_client import SupabaseManager
from src.backtest_engine import rebalance_portfolio
import unittest
import pytest
import os
import streamlit as st

# TODO(supabase): Once Supabase creds are restored, run this with:
#   SUPABASE_URL=... SUPABASE_KEY=... pytest -m integration UnitTests/test_multiple_factors.py
# Expected regression values ($5.29, 429.07%) were calibrated against the old OOP
# code path with 3 factors. Re-calibrate if normalize_series shifts results.
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_KEY'),
    reason="Requires Supabase credentials (SUPABASE_URL and SUPABASE_KEY)"
)
class TestFactorLakePortfolio(unittest.TestCase):
    def setUp(self):
        if not hasattr(st, 'session_state'):
            st.session_state = {}
        st.session_state.setdefault('exclude_fossil_fuels', False)
        st.session_state.setdefault('selected_sectors', [])

        manager = SupabaseManager()
        self.data = manager.fetch_all_data()

        self.start_year = 2002
        self.end_year = 2023
        self.initial_aum = 1
        self.expected_final_value = 5.29  # supabase data sig digits
        self.expected_growth = 429.07
        self.factors = ['6-Mo_Momentum', 'ROE_using_9-30_Data', 'ROA_using_9-30_Data']
        self.factor_directions = {
            '6-Mo_Momentum': 'top',
            'ROE_using_9-30_Data': 'top',
            'ROA_using_9-30_Data': 'top',
        }

    def test_portfolio_growth(self):
        portfolio_result = rebalance_portfolio(
            self.data, self.factors, self.factor_directions,
            self.start_year, self.end_year, self.initial_aum
        )

        final_aum = portfolio_result["final_value"]

        self.assertAlmostEqual(
            final_aum,
            self.expected_final_value,
            delta=0.01,
            msg=f'Expected portfolio value: ${self.expected_final_value}, but got {final_aum}'
        )

        overall_growth = (final_aum - self.initial_aum) / self.initial_aum * 100
        self.assertAlmostEqual(
            overall_growth,
            self.expected_growth,
            delta=0.1,
            msg=f'Expected overall growth: {self.expected_growth}%, but got {overall_growth}%'
        )

if __name__ == '__main__':
    unittest.main()
