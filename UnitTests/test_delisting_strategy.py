"""
Test suite for time-adjusted delisting strategy functionality.
Tests the three delisting modes: zero_return, hold_cash, reinvest.
Time fractions are derived from Delist_Date; missing dates default to 0.5.
"""
import pytest
import datetime
import pandas as pd
import numpy as np

from src.backtest_engine import _calculate_annual_return
from src.portfolio import Portfolio


@pytest.fixture
def df_year_with_delisted():
    """
    Synthetic year-slice (formation year 2010, holding period 2011).
    3 live stocks + 1 delisted mid-year (July 1 => ~183 days remaining => fraction ~0.5014).
    All priced at $100 for simple math.
    """
    return pd.DataFrame({
        'Ending_Price': [100.0, 100.0, 100.0, 100.0],
        'Next-Years_Return': [10.0, 20.0, 30.0, np.nan],
        'Delist_Date': [pd.NaT, pd.NaT, pd.NaT, pd.Timestamp('2011-07-01')],
    }, index=['AAPL', 'MSFT', 'GOOGL', 'DEAD'])


@pytest.fixture
def df_year_no_delist_date():
    """Same as above but without Delist_Date column — should fall back to 0.5."""
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


# --- Fraction helper ---
def _july1_fraction():
    """Fraction of year remaining after July 1 in a 2011 holding period."""
    return (datetime.date(2011, 12, 31) - datetime.date(2011, 7, 1)).days / 365.0


class TestZeroReturn:
    """Delisted capital earns 0% regardless of time fraction."""

    def test_basic(self, portfolios_with_delisted, df_year_with_delisted):
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='zero_return', rebalance_year=2010
        )
        # Live profit: 100*0.10 + 100*0.20 + 100*0.30 = 60
        # Delisted profit: 0
        # Total value: 400  =>  60/400 = 0.15
        assert ret == pytest.approx(0.15, abs=1e-9)
        assert count == 1

    def test_no_delisted(self, df_year_with_delisted):
        p = Portfolio(name="live_only")
        for t in ['AAPL', 'MSFT', 'GOOGL']:
            p.add_investment(t, 1.0)
        ret, count = _calculate_annual_return(
            [p], df_year_with_delisted,
            delisting_strategy='zero_return', rebalance_year=2010
        )
        assert ret == pytest.approx(0.20, abs=1e-9)
        assert count == 0


class TestHoldCash:
    """Delisted capital earns risk-free rate scaled by time fraction."""

    def test_time_adjusted(self, portfolios_with_delisted, df_year_with_delisted):
        rf = 0.05
        frac = _july1_fraction()
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='hold_cash', risk_free_rate=rf, rebalance_year=2010
        )
        # Live profit: 60
        # Delisted profit: 100 * 0.05 * fraction
        # Return: (60 + 100*0.05*frac) / 400
        expected = (60.0 + 100.0 * rf * frac) / 400.0
        assert ret == pytest.approx(expected, abs=1e-9)
        assert count == 1

    def test_fallback_fraction(self, portfolios_with_delisted, df_year_no_delist_date):
        """Without Delist_Date column, fraction defaults to 0.5."""
        rf = 0.10
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_no_delist_date,
            delisting_strategy='hold_cash', risk_free_rate=rf, rebalance_year=2010
        )
        expected = (60.0 + 100.0 * rf * 0.5) / 400.0
        assert ret == pytest.approx(expected, abs=1e-9)
        assert count == 1

    def test_zero_rf(self, portfolios_with_delisted, df_year_with_delisted):
        """hold_cash with rf=0 should match zero_return."""
        ret, _ = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='hold_cash', risk_free_rate=0.0, rebalance_year=2010
        )
        assert ret == pytest.approx(0.15, abs=1e-9)


class TestReinvest:
    """Delisted capital redistributed pro-rata, scaled by time fraction."""

    def test_time_adjusted(self, portfolios_with_delisted, df_year_with_delisted):
        frac = _july1_fraction()
        ret, count = _calculate_annual_return(
            portfolios_with_delisted, df_year_with_delisted,
            delisting_strategy='reinvest', rebalance_year=2010
        )
        # Live profit from own positions: 60
        # Live value: 300
        # Reinvested profit: frac * (100/300) * (live_profit=60) for DEAD's capital
        #   = frac * (100/300) * 60 = frac * 20
        # Total: (60 + frac*20) / 400
        expected = (60.0 + frac * 20.0) / 400.0
        assert ret == pytest.approx(expected, abs=1e-9)
        assert count == 1

    def test_all_delisted(self):
        """If every position is delisted, return should be 0."""
        df = pd.DataFrame({
            'Ending_Price': [100.0, 100.0],
            'Next-Years_Return': [np.nan, np.nan],
            'Delist_Date': [pd.Timestamp('2011-06-01'), pd.NaT],
        }, index=['A', 'B'])
        p = Portfolio(name="all_dead")
        p.add_investment('A', 1.0)
        p.add_investment('B', 1.0)
        ret, count = _calculate_annual_return(
            [p], df, delisting_strategy='reinvest', rebalance_year=2010
        )
        assert ret == pytest.approx(0.0, abs=1e-9)
        assert count == 2


class TestEdgeCases:
    """Edge cases across all strategies."""

    def test_empty_portfolio(self, df_year_with_delisted):
        p = Portfolio(name="empty")
        for strategy in ['zero_return', 'hold_cash', 'reinvest']:
            ret, count = _calculate_annual_return(
                [p], df_year_with_delisted,
                delisting_strategy=strategy, rebalance_year=2010
            )
            assert ret == 0.0
            assert count == 0

    def test_single_delisted_position(self):
        """Portfolio with only a single delisted stock."""
        df = pd.DataFrame({
            'Ending_Price': [50.0],
            'Next-Years_Return': [np.nan],
            'Delist_Date': [pd.Timestamp('2011-10-01')],
        }, index=['DEAD'])
        p = Portfolio(name="solo_dead")
        p.add_investment('DEAD', 2.0)

        frac = (datetime.date(2011, 12, 31) - datetime.date(2011, 10, 1)).days / 365.0

        # zero_return: always 0
        ret, _ = _calculate_annual_return([p], df, delisting_strategy='zero_return', rebalance_year=2010)
        assert ret == pytest.approx(0.0, abs=1e-9)

        # hold_cash: rf * fraction * value / value = rf * fraction
        ret, _ = _calculate_annual_return([p], df, delisting_strategy='hold_cash', risk_free_rate=0.03, rebalance_year=2010)
        assert ret == pytest.approx(0.03 * frac, abs=1e-9)

        # reinvest: no live positions => 0
        ret, _ = _calculate_annual_return([p], df, delisting_strategy='reinvest', rebalance_year=2010)
        assert ret == pytest.approx(0.0, abs=1e-9)

    def test_delist_date_dec31_full_year(self):
        """Stock with Delist_Date on Dec 31 of holding year => fraction ~0."""
        df = pd.DataFrame({
            'Ending_Price': [100.0, 100.0],
            'Next-Years_Return': [20.0, np.nan],
            'Delist_Date': [pd.NaT, pd.Timestamp('2011-12-31')],
        }, index=['LIVE', 'DEAD'])
        p = Portfolio(name="test")
        p.add_investment('LIVE', 1.0)
        p.add_investment('DEAD', 1.0)

        # Fraction remaining = 0 days / 365 = 0
        # hold_cash profit = rf * 0 * 100 = 0, same as zero_return
        ret, count = _calculate_annual_return(
            [p], df, delisting_strategy='hold_cash', risk_free_rate=0.05, rebalance_year=2010
        )
        expected = (100.0 * 0.20 + 0.0) / 200.0  # = 0.10
        assert ret == pytest.approx(expected, abs=1e-9)
        assert count == 1
