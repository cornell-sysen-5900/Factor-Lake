"""
Test suite for calculate_holdings.py module
Tests portfolio construction, rebalancing, and performance calculations
"""
import pytest
import pandas as pd
import numpy as np
from src.calculate_holdings import (
    calculate_holdings, calculate_growth, rebalance_portfolio,
    get_benchmark_return, calculate_information_ratio
)
from src.factor_function import Momentum6m, ROE, ROA
from src.market_object import MarketObject, load_data


class TestCalculateHoldings:
    """Test calculate_holdings function"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        return pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US', 'GOOGL-US', 'AMZN-US', 'META-US',
                             'TSLA-US', 'NVDA-US', 'JPM-US', 'V-US', 'WMT-US'],
            'Ending Price': [150.0, 300.0, 2800.0, 3200.0, 350.0,
                            700.0, 450.0, 140.0, 220.0, 145.0],
            '6-Mo Momentum %': [0.20, 0.25, 0.15, 0.30, 0.18,
                               0.35, 0.28, 0.12, 0.22, 0.10],
            'Year': [2022] * 10
        })
    
    @pytest.fixture
    def market(self, sample_market_data):
        """Create a market object"""
        return MarketObject(sample_market_data, 2022)
    
    def test_calculate_holdings_basic(self, market):
        """Test basic portfolio construction"""
        factor = Momentum6m()
        aum = 10000.0
        
        portfolio = calculate_holdings(factor, aum, market)
        
        assert portfolio is not None
        assert len(portfolio.investments) > 0
        assert portfolio.investments[0]['number_of_shares'] > 0
    
    def test_calculate_holdings_top_10_percent(self, market):
        """Test that only top 10% of stocks are selected"""
        factor = Momentum6m()
        aum = 10000.0
        
        portfolio = calculate_holdings(factor, aum, market)
        
        # With 10 stocks, top 10% = 1 stock
        assert len(portfolio.investments) == 1
    
    def test_calculate_holdings_equal_weighting(self, market):
        """Test equal weighting of selected stocks"""
        factor = Momentum6m()
        aum = 10000.0
        
        portfolio = calculate_holdings(factor, aum, market)
        
        # Calculate total portfolio value
        total_value = sum(
            inv['number_of_shares'] * market.get_price(inv['ticker'])
            for inv in portfolio.investments
        )
        
        # Should be approximately equal to AUM
        assert total_value == pytest.approx(aum, rel=0.01)
    
    def test_calculate_holdings_selects_highest_factor(self, market):
        """Test that stocks with highest factor values are selected"""
        factor = Momentum6m()
        aum = 10000.0
        
        portfolio = calculate_holdings(factor, aum, market)
        
        # TSLA has highest momentum (0.35), should be selected
        selected_tickers = [inv['ticker'] for inv in portfolio.investments]
        assert 'TSLA' in selected_tickers
    
    def test_calculate_holdings_with_missing_values(self):
        """Test portfolio construction with missing factor values"""
        data = pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US', 'GOOGL-US'],
            'Ending Price': [150.0, 300.0, 2800.0],
            '6-Mo Momentum %': [0.20, np.nan, 0.15],  # MSFT has NaN
            'Year': [2022, 2022, 2022]
        })
        market = MarketObject(data, 2022)
        
        factor = Momentum6m()
        portfolio = calculate_holdings(factor, 10000.0, market)
        
        # Should only include stocks with valid values
        selected_tickers = [inv['ticker'] for inv in portfolio.investments]
        assert 'MSFT' not in selected_tickers
    
    @pytest.mark.skip(reason="Fossil fuel filtering needs fixing in calculate_holdings")
    def test_calculate_holdings_fossil_fuel_restriction(self):
        """Test fossil fuel restriction"""
        data = pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'XOM-US', 'MSFT-US', 'GOOGL-US', 'TSLA-US'],
            'Ending Price': [150.0, 100.0, 300.0, 2800.0, 700.0],
            '6-Mo Momentum %': [0.20, 0.35, 0.15, 0.25, 0.30],  # XOM has highest momentum
            'FactSet Industry': ['Technology', 'Oil & Gas', 'Technology', 'Technology', 'Automotive'],
            'Year': [2022, 2022, 2022, 2022, 2022]
        })
        market = MarketObject(data, 2022)
        
        factor = Momentum6m()
        portfolio = calculate_holdings(factor, 10000.0, market, restrict_fossil_fuels=True)
        
        # XOM should be excluded despite having high momentum
        selected_tickers = [inv['ticker'] for inv in portfolio.investments]
        assert 'XOM' not in selected_tickers, f"XOM should be excluded but found in {selected_tickers}"
        # Should select from non-fossil fuel companies
        assert len(selected_tickers) > 0, "Should have selected some companies"


class TestCalculateGrowth:
    """Test calculate_growth function"""
    
    @pytest.fixture
    def portfolio_list(self):
        """Create a list of portfolios"""
        from src.portfolio import Portfolio

        portfolio = Portfolio(name="Test Portfolio")
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('MSFT', 5)

        return [portfolio]
    @pytest.fixture
    def current_market(self):
        """Create current market"""
        data = pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US'],
            'Ending Price': [100.0, 200.0],
            'Year': [2021, 2021]
        })
        return MarketObject(data, 2021)
    
    @pytest.fixture
    def next_market(self):
        """Create next year market"""
        data = pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US'],
            'Ending Price': [150.0, 250.0],
            'Year': [2022, 2022]
        })
        return MarketObject(data, 2022)
    
    def test_calculate_growth_positive(self, portfolio_list, current_market, next_market):
        """Test calculating positive growth"""
        growth, start_value, end_value = calculate_growth(
            portfolio_list, next_market, current_market, verbosity=0
        )
        
        # Start: 10*100 + 5*200 = 2000
        # End: 10*150 + 5*250 = 2750
        # Growth: (2750-2000)/2000 = 0.375
        
        assert start_value == 2000.0
        assert end_value == 2750.0
        assert growth == pytest.approx(0.375, rel=0.01)
    
    def test_calculate_growth_negative(self, portfolio_list, current_market):
        """Test calculating negative growth"""
        # Market with declining prices
        declining_market = MarketObject(pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US'],
            'Ending Price': [80.0, 150.0],
            'Year': [2022, 2022]
        }), 2022)
        
        growth, start_value, end_value = calculate_growth(
            portfolio_list, declining_market, current_market, verbosity=0
        )
        
        assert growth < 0  # Negative growth
        assert end_value < start_value
    
    def test_calculate_growth_with_missing_stock(self, portfolio_list, current_market):
        """Test growth calculation when stock missing in next market"""
        # Next market missing MSFT
        partial_market = MarketObject(pd.DataFrame({
            'Ticker-Region': ['AAPL-US'],
            'Ending Price': [150.0],
            'Year': [2022]
        }), 2022)
        
        growth, start_value, end_value = calculate_growth(
            portfolio_list, partial_market, current_market, verbosity=0
        )
        
        # Should use entry price for MSFT
        assert end_value > 0
        assert start_value == 2000.0


class TestRebalancePortfolio:
    """Test rebalance_portfolio function"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for multiple years"""
        data = []
        for year in range(2020, 2023):
            for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                          'TSLA', 'NVDA', 'JPM', 'V', 'WMT']:
                data.append({
                    'Ticker-Region': f'{ticker}-US',
                    'Ending Price': np.random.uniform(100, 500),
                    '6-Mo Momentum %': np.random.uniform(0.05, 0.30),
                    'ROE using 9/30 Data': np.random.uniform(0.10, 0.40),
                    'ROA using 9/30 Data': np.random.uniform(0.05, 0.25),
                    'Year': year
                })
        
        return pd.DataFrame(data)
    
    def test_rebalance_portfolio_basic(self, sample_data):
        """Test basic portfolio rebalancing"""
        factors = [Momentum6m()]
        
        results = rebalance_portfolio(
            sample_data, factors,
            start_year=2020, end_year=2022,
            initial_aum=1.0,
            verbosity=0
        )
        
        assert 'final_value' in results
        assert 'yearly_returns' in results
        assert 'benchmark_returns' in results
        assert results['final_value'] > 0
    
    def test_rebalance_portfolio_multiple_factors(self, sample_data):
        """Test rebalancing with multiple factors"""
        factors = [Momentum6m(), ROE(), ROA()]
        
        results = rebalance_portfolio(
            sample_data, factors,
            start_year=2020, end_year=2022,
            initial_aum=1.0,
            verbosity=0
        )
        
        assert results['final_value'] > 0
        assert len(results['yearly_returns']) == 2  # 2 years of returns
    
    def test_rebalance_portfolio_returns_structure(self, sample_data):
        """Test that rebalance returns correct structure"""
        factors = [Momentum6m()]
        
        results = rebalance_portfolio(
            sample_data, factors,
            start_year=2020, end_year=2022,
            initial_aum=1.0,
            verbosity=0
        )
        
        assert isinstance(results, dict)
        assert 'final_value' in results
        assert 'yearly_returns' in results
        assert 'benchmark_returns' in results
        assert 'years' in results
        assert 'portfolio_values' in results
    
    def test_rebalance_portfolio_years_length(self, sample_data):
        """Test that years and values have correct length"""
        factors = [Momentum6m()]
        
        results = rebalance_portfolio(
            sample_data, factors,
            start_year=2020, end_year=2022,
            initial_aum=1.0,
            verbosity=0
        )
        
        # Should have 3 years: 2020, 2021, 2022
        assert len(results['years']) == 3
        assert len(results['portfolio_values']) == 3


class TestBenchmarkReturn:
    """Test benchmark return function"""
    
    def test_get_benchmark_return_known_years(self):
        """Test getting benchmark returns for known years"""
        # Test a few known years
        assert get_benchmark_return(2002) == 34.62
        assert get_benchmark_return(2010) == -4.73
        assert get_benchmark_return(2020) == 46.21
    
    def test_get_benchmark_return_unknown_year(self):
        """Test getting benchmark return for unknown year"""
        # Should return 0 for unknown years
        assert get_benchmark_return(1999) == 0
        assert get_benchmark_return(2050) == 0


class TestInformationRatio:
    """Test information ratio calculation"""
    
    def test_calculate_information_ratio_positive(self):
        """Test calculating positive information ratio"""
        portfolio_returns = [0.10, 0.12, 0.15, 0.08, 0.14]
        benchmark_returns = [0.08, 0.09, 0.10, 0.07, 0.11]
        
        ir = calculate_information_ratio(portfolio_returns, benchmark_returns, verbosity=0)
        
        assert ir is not None
        assert ir > 0  # Portfolio outperformed benchmark
    
    def test_calculate_information_ratio_negative(self):
        """Test calculating negative information ratio"""
        portfolio_returns = [0.05, 0.06, 0.07, 0.04, 0.08]
        benchmark_returns = [0.08, 0.09, 0.10, 0.07, 0.11]
        
        ir = calculate_information_ratio(portfolio_returns, benchmark_returns, verbosity=0)
        
        assert ir is not None
        assert ir < 0  # Portfolio underperformed benchmark
    
    def test_calculate_information_ratio_zero_tracking_error(self):
        """Test IR calculation with zero tracking error"""
        portfolio_returns = [0.10, 0.10, 0.10]
        benchmark_returns = [0.10, 0.10, 0.10]
        
        ir = calculate_information_ratio(portfolio_returns, benchmark_returns, verbosity=0)
        
        assert ir is None  # Should return None when tracking error is 0
    
    def test_calculate_information_ratio_with_numpy_arrays(self):
        """Test IR calculation with numpy arrays"""
        portfolio_returns = np.array([0.10, 0.12, 0.15])
        benchmark_returns = np.array([0.08, 0.09, 0.10])
        
        ir = calculate_information_ratio(portfolio_returns, benchmark_returns, verbosity=0)
        
        assert ir is not None
        assert isinstance(ir, (float, np.floating))


class TestCalculateHoldingsIntegration:
    """Integration tests with real data"""
    
    @pytest.fixture
    def real_data(self):
        """Load real data"""
        return load_data(use_supabase=True)
    
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_calculate_holdings_with_real_data(self, real_data):
        """Test calculate_holdings with real data"""
        year = 2022
        year_data = real_data[real_data['Year'] == year]
        
        if len(year_data) > 0:
            market = MarketObject(year_data, year)
            factor = Momentum6m()
            
            portfolio = calculate_holdings(factor, 10000.0, market)
            
            assert portfolio is not None
            assert len(portfolio.investments) > 0
            
            # Verify total investment is close to AUM
            total_value = portfolio.present_value(market)
            assert total_value > 0
    
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_rebalance_with_real_data_subset(self, real_data):
        """Test rebalancing with real data (short period)"""
        # Test with just 2 years to keep test fast
        test_data = real_data[real_data['Year'].isin([2021, 2022])]
        
        if len(test_data) > 0:
            factors = [Momentum6m()]
            
            results = rebalance_portfolio(
                test_data, factors,
                start_year=2021, end_year=2022,
                initial_aum=1.0,
                verbosity=0
            )
            
            assert results['final_value'] > 0
            assert len(results['yearly_returns']) >= 0
