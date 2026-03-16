from src.factor_registry import Momentum6m
from src.market_object import load_data
from src.calculate_holdings import rebalance_portfolio
import unittest
import pandas as pd
import pytest
import os

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_KEY'),
    reason="Requires Supabase credentials (SUPABASE_URL and SUPABASE_KEY)"
)
class TestFactorLakePortfolio(unittest.TestCase):
    def setUp(self):
        self.data = load_data(use_supabase=True)
        
        # Ensure 'Year' column exists for consistency
        if 'Year' not in self.data.columns:
            self.data['Year'] = pd.to_datetime(self.data['Date']).dt.year
        
        self.start_year = 2002
        self.end_year = 2023
        self.initial_aum = 1
        self.expected_final_value = 4.42 #error in supabase data sig digits
        self.expected_growth = 341.94
        self.factors = [Momentum6m()]

    def test_portfolio_growth(self):
        portfolio_result = rebalance_portfolio(self.data, self.factors, self.start_year, self.end_year, self.initial_aum)
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
            msg=f'Expected overall growth: {self.expected_growth}%, but got {overall_growth}%'  # Fixed the closing quote
        )

if __name__ == '__main__':
    unittest.main()
