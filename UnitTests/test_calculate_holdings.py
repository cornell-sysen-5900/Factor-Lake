"""
Test suite for backtest_engine.py
Tests portfolio construction (calculate_holdings), annual return calculation
(_calculate_annual_return), rebalance loop, and benchmark data retrieval.

Rewritten to match the 2026 architecture:
  - calculate_holdings(df_year, factor_key, aum, ...) operates on a DataFrame
    indexed by Ticker, not a MarketObject.
  - _calculate_annual_return replaces calculate_growth; returns (return, delisted_count).
  - rebalance_portfolio requires factor_directions dict and reads st.session_state.
  - get_benchmark_list replaces get_benchmark_return.
  - Information ratio lives in compute_comprehensive_metrics, not a standalone fn.
"""
import pytest
import pandas as pd
import numpy as np
import streamlit as st

from src.backtest_engine import (
    calculate_holdings,
    _calculate_annual_return,
    rebalance_portfolio,
)
from src.benchmarks import get_benchmark_list
from src.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ten_stock_df():
    """DataFrame indexed by Ticker with 10 stocks, mimicking a single-year slice."""
    df = pd.DataFrame({
        'Ending_Price': [150, 300, 2800, 3200, 350, 700, 450, 140, 220, 145],
        '6-Mo_Momentum': [0.20, 0.25, 0.15, 0.30, 0.18, 0.35, 0.28, 0.12, 0.22, 0.10],
        'ROE_using_9-30_Data': [0.25, 0.30, 0.22, 0.28, 0.20, 0.15, 0.35, 0.18, 0.32, 0.16],
        'Next-Years_Return': [10, 15, -5, 20, 8, 25, 30, -2, 12, 5],
        'Market_Capitalization': [2e9, 1.5e9, 3e9, 5e9, 1e9, 4e9, 2.5e9, 0.8e9, 1.2e9, 0.6e9],
    }, index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'])
    df.index.name = 'Ticker'
    return df


@pytest.fixture
def multi_year_data():
    """Long-format DataFrame with Ticker/Year columns for 3 years × 10 stocks.
    Next-Years_Return is seeded deterministically."""
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


@pytest.fixture(autouse=True)
def _mock_session_state():
    """Ensure st.session_state has the keys filter_universe reads."""
    st.session_state['exclude_fossil_fuels'] = False
    st.session_state['selected_sectors'] = []


# ---------------------------------------------------------------------------
# calculate_holdings
# ---------------------------------------------------------------------------

class TestCalculateHoldings:
    """Tests for calculate_holdings(df_year, factor_key, aum, ...)."""

    def test_basic_construction(self, ten_stock_df):
        """Should return a non-empty Portfolio."""
        portfolio = calculate_holdings(ten_stock_df, '6-Mo_Momentum', 10_000)
        assert len(portfolio.investments) > 0

    def test_top_10_percent_selects_one(self, ten_stock_df):
        """With 10 stocks and top_pct=10, exactly 1 stock should be selected."""
        portfolio = calculate_holdings(ten_stock_df, '6-Mo_Momentum', 10_000, top_pct=10.0)
        assert len(portfolio.investments) == 1

    def test_total_value_equals_aum(self, ten_stock_df):
        """Total position value should approximate AUM."""
        aum = 10_000
        portfolio = calculate_holdings(ten_stock_df, '6-Mo_Momentum', aum, top_pct=10.0)
        total = sum(
            inv['number_of_shares'] * ten_stock_df.loc[inv['ticker'], 'Ending_Price']
            for inv in portfolio.investments
        )
        assert total == pytest.approx(aum, rel=0.01)

    def test_selects_highest_factor(self, ten_stock_df):
        """The stock with the highest raw factor value should be selected at top_pct=10."""
        portfolio = calculate_holdings(ten_stock_df, '6-Mo_Momentum', 10_000, top_pct=10.0)
        selected = [inv['ticker'] for inv in portfolio.investments]
        # TSLA has highest 6-Mo_Momentum (0.35)
        assert 'TSLA' in selected

    def test_bottom_direction(self, ten_stock_df):
        """higher_is_better=False should select the lowest-scoring stock."""
        portfolio = calculate_holdings(
            ten_stock_df, '6-Mo_Momentum', 10_000,
            higher_is_better=False, top_pct=10.0
        )
        selected = [inv['ticker'] for inv in portfolio.investments]
        # WMT has lowest 6-Mo_Momentum (0.10)
        assert 'WMT' in selected

    def test_missing_factor_values(self):
        """Stocks with NaN factor values should be excluded."""
        df = pd.DataFrame({
            'Ending_Price': [150, 300, 2800],
            '6-Mo_Momentum': [0.20, np.nan, 0.15],
            'Next-Years_Return': [10, 15, -5],
        }, index=['AAPL', 'MSFT', 'GOOGL'])
        df.index.name = 'Ticker'

        portfolio = calculate_holdings(df, '6-Mo_Momentum', 10_000, top_pct=50.0)
        selected = [inv['ticker'] for inv in portfolio.investments]
        assert 'MSFT' not in selected

    def test_market_cap_weighting(self, ten_stock_df):
        """Market-cap weighting should allocate more to larger-cap stocks."""
        portfolio = calculate_holdings(
            ten_stock_df, '6-Mo_Momentum', 100_000,
            top_pct=30.0, use_market_cap_weight=True
        )
        # With cap weighting, different stocks get different share counts
        shares = {inv['ticker']: inv['number_of_shares'] for inv in portfolio.investments}
        assert len(shares) > 0

    def test_unknown_factor_key_raises(self, ten_stock_df):
        """A factor key that doesn't resolve to any column should raise KeyError."""
        with pytest.raises(KeyError):
            calculate_holdings(ten_stock_df, 'totally_bogus_factor', 10_000)


# ---------------------------------------------------------------------------
# _calculate_annual_return
# ---------------------------------------------------------------------------

class TestCalculateAnnualReturn:
    """Tests for _calculate_annual_return (replaces old calculate_growth).

    NOTE: On main, _calculate_annual_return returns a single float.
    The delisting-strategy-v2 branch extends this to return (float, int)
    with a delisting_strategy kwarg. These tests target the main API;
    delisting-specific tests live in test_delisting_strategy.py.
    """

    def _make_portfolio(self, holdings):
        """Helper: create a Portfolio from a list of (ticker, shares) tuples."""
        p = Portfolio(name='test')
        for ticker, shares in holdings:
            p.add_investment(ticker, shares)
        return p

    def test_positive_return(self):
        """All live positions with positive returns → positive aggregate return."""
        df = pd.DataFrame({
            'Ending_Price': [100.0, 200.0],
            'Next-Years_Return': [20.0, 10.0],  # 20% and 10%
        }, index=['AAPL', 'MSFT'])

        portfolios = [self._make_portfolio([('AAPL', 10), ('MSFT', 5)])]
        ret, _ = _calculate_annual_return(portfolios, df)

        # AAPL: 1000 * 0.20 = 200, MSFT: 1000 * 0.10 = 100
        # Total val = 2000, profit = 300, return = 15%
        assert ret == pytest.approx(0.15, rel=0.01)

    def test_negative_return(self):
        """Negative Next-Years_Return values → negative aggregate return."""
        df = pd.DataFrame({
            'Ending_Price': [100.0],
            'Next-Years_Return': [-25.0],
        }, index=['AAPL'])

        portfolios = [self._make_portfolio([('AAPL', 10)])]
        ret, _ = _calculate_annual_return(portfolios, df)

        assert ret == pytest.approx(-0.25, rel=0.01)

    def test_delisted_stock_treated_as_zero(self):
        """NaN return (delisted) should contribute 0% on main (implicit zero_return)."""
        df = pd.DataFrame({
            'Ending_Price': [100.0, 100.0],
            'Next-Years_Return': [20.0, np.nan],
        }, index=['AAPL', 'DEAD'])

        portfolios = [self._make_portfolio([('AAPL', 10), ('DEAD', 10)])]
        ret, count = _calculate_annual_return(portfolios, df)

        # AAPL: 1000 * 0.20 = 200 profit, DEAD: 1000 * 0.0 = 0 profit
        # Total val = 2000, return = 200/2000 = 10%
        assert ret == pytest.approx(0.10, rel=0.01)
        assert count == 1

    def test_missing_ticker_ignored(self):
        """Tickers not in df_year should be silently skipped."""
        df = pd.DataFrame({
            'Ending_Price': [100.0],
            'Next-Years_Return': [10.0],
        }, index=['AAPL'])

        portfolios = [self._make_portfolio([('AAPL', 10), ('GONE', 5)])]
        ret, _ = _calculate_annual_return(portfolios, df)

        assert ret == pytest.approx(0.10, rel=0.01)

    def test_empty_portfolio(self):
        """Empty portfolio should return 0.0."""
        df = pd.DataFrame({
            'Ending_Price': [100.0],
            'Next-Years_Return': [10.0],
        }, index=['AAPL'])

        portfolios = [self._make_portfolio([])]
        ret, count = _calculate_annual_return(portfolios, df)
        assert ret == 0.0
        assert count == 0


# ---------------------------------------------------------------------------
# rebalance_portfolio (integration-level, synthetic data)
# ---------------------------------------------------------------------------

class TestRebalancePortfolio:
    """Tests for the full rebalance_portfolio loop using synthetic data."""

    def test_basic_run(self, multi_year_data):
        """Should complete and return a dict with expected keys."""
        results = rebalance_portfolio(
            multi_year_data,
            factors=['6-Mo_Momentum'],
            factor_directions={'6-Mo_Momentum': 'top'},
            start_year=2020, end_year=2022,
            initial_aum=1.0,
        )
        assert isinstance(results, dict)
        assert 'final_value' in results
        assert 'yearly_returns' in results
        assert 'portfolio_values' in results
        assert results['final_value'] > 0

    def test_multiple_factors(self, multi_year_data):
        """Multiple factors should run without error."""
        results = rebalance_portfolio(
            multi_year_data,
            factors=['6-Mo_Momentum', 'ROE_using_9-30_Data'],
            factor_directions={
                '6-Mo_Momentum': 'top',
                'ROE_using_9-30_Data': 'top',
            },
            start_year=2020, end_year=2022,
            initial_aum=1.0,
        )
        assert results['final_value'] > 0
        assert len(results['yearly_returns']) == 2

    def test_portfolio_values_length(self, multi_year_data):
        """portfolio_values should have n_years + 1 entries (initial + each year-end)."""
        results = rebalance_portfolio(
            multi_year_data,
            factors=['6-Mo_Momentum'],
            factor_directions={'6-Mo_Momentum': 'top'},
            start_year=2020, end_year=2022,
            initial_aum=1.0,
        )
        assert len(results['portfolio_values']) == 3  # initial + 2 year-ends

    def test_deterministic(self, multi_year_data):
        """Same inputs should produce same outputs (no random state leakage)."""
        kwargs = dict(
            factors=['6-Mo_Momentum'],
            factor_directions={'6-Mo_Momentum': 'top'},
            start_year=2020, end_year=2022,
            initial_aum=1.0,
        )
        r1 = rebalance_portfolio(multi_year_data, **kwargs)
        r2 = rebalance_portfolio(multi_year_data, **kwargs)
        assert r1['final_value'] == r2['final_value']


# ---------------------------------------------------------------------------
# Benchmarks (get_benchmark_list replaces get_benchmark_return)
# ---------------------------------------------------------------------------

class TestBenchmarks:
    """Tests for get_benchmark_list from benchmarks.py."""

    def test_known_year_russell2000(self):
        """Russell 2000 returns for known years should match hardcoded data."""
        # index 1 = Russell 2000; 2002 return is -9.30
        vals = get_benchmark_list(1, 2002, 2003)
        assert len(vals) == 1
        assert vals[0] == -9.30

    def test_risk_free_rates(self):
        """Risk-free rate for 2002 should be 0.0156."""
        vals = get_benchmark_list(4, 2002, 2003)
        assert len(vals) == 1
        assert vals[0] == pytest.approx(0.0156)

    def test_multi_year_range(self):
        """Requesting multiple years should return correct count."""
        vals = get_benchmark_list(1, 2002, 2005)
        assert len(vals) == 3  # 2002, 2003, 2004

    def test_unknown_year_returns_zero(self):
        """Years outside the data range should return 0.0 (fallback)."""
        vals = get_benchmark_list(1, 1990, 1991)
        assert vals == [0.0]

    def test_invalid_index_raises(self):
        """Invalid benchmark index should raise ValueError."""
        with pytest.raises(ValueError):
            get_benchmark_list(99, 2002, 2003)
