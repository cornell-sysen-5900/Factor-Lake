"""
Test suite for delisting strategy functionality.
Tests the three delisting modes: zero_return, hold_cash, reinvest.
"""
import pytest
import pandas as pd
import numpy as np

from src.backtest_engine import _calculate_annual_return
from src.portfolio import Portfolio


@pytest.fixture
def df_year_with_delisted():
    """
    Synthetic year-slice with 3 live stocks and 1 delisted (NaN return).
    All stocks priced at $100 for simple math.
    """
    return pd.DataFrame({
        'Ending_Price': [100.0, 100.0, 100.0, 100.0],
        'Next-Years_Return': [10.0, 20.0, 30.0, np.nan],
    }, index=['AAPL', 'MSFT', 'GOOGL', 'DEAD'])


@pytest.fixture
def portfolios_with_delisted():
    """Portfolio holding 1 share of each of the 4 stocks."""
    p = Portfolio(name="test")
    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'DEAD']:
        p.add_investment(ticker, 1.0)
    return [p]


class TestZeroReturn:
    """Delisted capital earns 0%."""

    def test_basic(self, portfolios_with_delisted, df_year_with_delisted):
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='zero_return'
        )
        # Live profit: 100*0.10 + 100*0.20 + 100*0.30 = 60
        # Delisted profit: 0
        # Total value: 400
        # Return: 60/400 = 0.15
        assert ret == pytest.approx(0.15, abs=1e-9)
        assert count == 1

    def test_no_delisted(self, df_year_with_delisted):
        """All-live portfolio should behave the same regardless of strategy."""
        p = Portfolio(name="live_only")
        for t in ['AAPL', 'MSFT', 'GOOGL']:
            p.add_investment(t, 1.0)
        ret, count = _calculate_annual_return(
            [p], df_year_with_delisted,
            delisting_strategy='zero_return'
        )
        # Profit: 10+20+30=60, value=300, return=0.20
        assert ret == pytest.approx(0.20, abs=1e-9)
        assert count == 0


class TestHoldCash:
    """Delisted capital earns risk-free rate."""

    def test_basic(self, portfolios_with_delisted, df_year_with_delisted):
        rf = 0.05  # 5%
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='hold_cash',
            risk_free_rate=rf
        )
        # Live profit: 60
        # Delisted profit: 100 * 0.05 = 5
        # Total: 65 / 400 = 0.1625
        assert ret == pytest.approx(0.1625, abs=1e-9)
        assert count == 1

    def test_zero_rf(self, portfolios_with_delisted, df_year_with_delisted):
        """hold_cash with rf=0 should match zero_return."""
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='hold_cash',
            risk_free_rate=0.0
        )
        assert ret == pytest.approx(0.15, abs=1e-9)
        assert count == 1


class TestReinvest:
    """Delisted capital redistributed pro-rata across surviving positions."""

    def test_basic(self, portfolios_with_delisted, df_year_with_delisted):
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='reinvest'
        )
        # Live positions: 3 stocks at $100 each, equal weight
        # Each live stock gets 1/3 of DEAD's $100 capital
        # Extra profit: (100/300)*100*0.10 + (100/300)*100*0.20 + (100/300)*100*0.30
        #             = (1/3)*(10+20+30) = 20
        # Total profit: 60 (live) + 20 (reinvested) = 80
        # Return: 80 / 400 = 0.20
        assert ret == pytest.approx(0.20, abs=1e-9)
        assert count == 1

    def test_all_delisted(self):
        """If every position is delisted, return should be 0."""
        df = pd.DataFrame({
            'Ending_Price': [100.0, 100.0],
            'Next-Years_Return': [np.nan, np.nan],
        }, index=['A', 'B'])
        p = Portfolio(name="all_dead")
        p.add_investment('A', 1.0)
        p.add_investment('B', 1.0)
        ret, count = _calculate_annual_return(
            [p], df, delisting_strategy='reinvest'
        )
        assert ret == pytest.approx(0.0, abs=1e-9)
        assert count == 2


class TestEdgeCases:
    """Edge cases that apply across all strategies."""

    def test_empty_portfolio(self, df_year_with_delisted):
        """Empty portfolio returns 0."""
        p = Portfolio(name="empty")
        for strategy in ['zero_return', 'hold_cash', 'reinvest']:
            ret, count = _calculate_annual_return(
                [p], df_year_with_delisted,
                delisting_strategy=strategy
            )
            assert ret == 0.0
            assert count == 0

    def test_single_delisted_position(self):
        """Portfolio with only a single delisted stock."""
        df = pd.DataFrame({
            'Ending_Price': [50.0],
            'Next-Years_Return': [np.nan],
        }, index=['DEAD'])
        p = Portfolio(name="solo_dead")
        p.add_investment('DEAD', 2.0)

        # zero_return: 0 profit / 100 total = 0
        ret, _ = _calculate_annual_return([p], df, delisting_strategy='zero_return')
        assert ret == pytest.approx(0.0, abs=1e-9)

        # hold_cash: 100 * 0.03 / 100 = 0.03
        ret, _ = _calculate_annual_return(
            [p], df, delisting_strategy='hold_cash', risk_free_rate=0.03
        )
        assert ret == pytest.approx(0.03, abs=1e-9)

        # reinvest: no live positions to redistribute to → 0
        ret, _ = _calculate_annual_return([p], df, delisting_strategy='reinvest')
        assert ret == pytest.approx(0.0, abs=1e-9)
