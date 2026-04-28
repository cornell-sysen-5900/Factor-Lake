"""
Test suite for portfolio.py module
Tests portfolio creation, management, and calculations
"""
import pytest
import pandas as pd
from src.portfolio import Portfolio


class TestPortfolioCreation:
    """Test portfolio initialization and creation"""
    
    def test_create_empty_portfolio(self):
        """Test creating an empty portfolio"""
        portfolio = Portfolio(name="Test Portfolio")
        
        assert portfolio.name == "Test Portfolio"
        assert portfolio.investments == []
        assert len(portfolio.investments) == 0
    
    def test_create_portfolio_with_investments(self):
        """Test creating portfolio with initial investments"""
        investments = [
            {'ticker': 'AAPL', 'number_of_shares': 10},
            {'ticker': 'MSFT', 'number_of_shares': 5}
        ]
        
        portfolio = Portfolio(name="Test Portfolio", investments=investments)
        
        assert portfolio.name == "Test Portfolio"
        assert len(portfolio.investments) == 2
        assert portfolio.investments[0]['ticker'] == 'AAPL'
        assert portfolio.investments[1]['ticker'] == 'MSFT'


class TestPortfolioManagement:
    """Test adding and removing investments"""
    
    @pytest.fixture
    def portfolio(self):
        """Create a portfolio for testing"""
        return Portfolio(name="Test Portfolio")
    
    def test_add_single_investment(self, portfolio):
        """Test adding a single investment"""
        portfolio.add_investment('AAPL', 10)
        
        assert len(portfolio.investments) == 1
        assert portfolio.investments[0]['ticker'] == 'AAPL'
        assert portfolio.investments[0]['number_of_shares'] == 10
    
    def test_add_multiple_investments(self, portfolio):
        """Test adding multiple investments"""
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('MSFT', 5)
        portfolio.add_investment('GOOGL', 2)
        
        assert len(portfolio.investments) == 3
        
        tickers = [inv['ticker'] for inv in portfolio.investments]
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers
        assert 'GOOGL' in tickers
    
    def test_add_fractional_shares(self, portfolio):
        """Test adding fractional shares"""
        portfolio.add_investment('AAPL', 10.5)
        
        assert portfolio.investments[0]['number_of_shares'] == 10.5
    
    def test_remove_investment(self, portfolio):
        """Test removing an investment"""
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('MSFT', 5)
        
        portfolio.remove_investment('AAPL')
        
        assert len(portfolio.investments) == 1
        assert portfolio.investments[0]['ticker'] == 'MSFT'
    
    def test_remove_nonexistent_investment(self, portfolio):
        """Test removing an investment that doesn't exist"""
        portfolio.add_investment('AAPL', 10)
        
        portfolio.remove_investment('INVALID')
        
        # Should not raise error, portfolio unchanged
        assert len(portfolio.investments) == 1
        assert portfolio.investments[0]['ticker'] == 'AAPL'
    
    def test_remove_all_investments(self, portfolio):
        """Test removing all investments"""
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('MSFT', 5)
        
        portfolio.remove_investment('AAPL')
        portfolio.remove_investment('MSFT')
        
        assert len(portfolio.investments) == 0


class TestPortfolioValuation:
    """Test portfolio value calculations"""

    @pytest.fixture
    def market_data(self):
        """Create sample market data indexed by ticker with Ending_Price column"""
        return pd.DataFrame({
            'Ending_Price': [150.0, 300.0, 2800.0],
        }, index=['AAPL', 'MSFT', 'GOOGL'])

    @pytest.fixture
    def portfolio(self):
        """Create a portfolio with investments"""
        portfolio = Portfolio(name="Test Portfolio")
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('MSFT', 5)
        return portfolio

    def test_present_value_calculation(self, portfolio, market_data):
        """Test calculating present value of portfolio"""
        value = portfolio.present_value(market_data)

        # AAPL: 10 shares * $150 = $1,500
        # MSFT: 5 shares * $300 = $1,500
        # Total: $3,000
        assert value == 3000.0

    def test_present_value_empty_portfolio(self, market_data):
        """Test present value of empty portfolio"""
        portfolio = Portfolio(name="Empty Portfolio")
        value = portfolio.present_value(market_data)

        assert value == 0.0

    def test_present_value_with_missing_stock(self, market_data):
        """Test present value when stock not in market"""
        portfolio = Portfolio(name="Test Portfolio")
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('INVALID', 5)  # Not in market

        value = portfolio.present_value(market_data)

        # Should only count AAPL
        assert value == 1500.0

    def test_present_value_with_fractional_shares(self, market_data):
        """Test present value with fractional shares"""
        portfolio = Portfolio(name="Test Portfolio")
        portfolio.add_investment('AAPL', 10.5)

        value = portfolio.present_value(market_data)

        # 10.5 shares * $150 = $1,575
        assert value == 1575.0


class TestPortfolioReturns:
    """Test return calculations"""
    
    @pytest.fixture
    def portfolio(self):
        """Create a portfolio for testing"""
        return Portfolio(name="Test Portfolio")
    
    def test_calculate_return_positive(self, portfolio):
        """Test calculating positive return"""
        t1_value = 1000.0
        t2_value = 1200.0
        
        return_pct = portfolio.calculate_return(t1_value, t2_value)
        
        assert return_pct == 20.0  # 20% gain
    
    def test_calculate_return_negative(self, portfolio):
        """Test calculating negative return"""
        t1_value = 1000.0
        t2_value = 800.0
        
        return_pct = portfolio.calculate_return(t1_value, t2_value)
        
        assert return_pct == -20.0  # 20% loss
    
    def test_calculate_return_zero_change(self, portfolio):
        """Test calculating zero return"""
        t1_value = 1000.0
        t2_value = 1000.0
        
        return_pct = portfolio.calculate_return(t1_value, t2_value)
        
        assert return_pct == 0.0
    
    def test_calculate_return_zero_t1(self, portfolio):
        """Test that zero t1 value returns 0.0"""
        assert portfolio.calculate_return(0, 1000.0) == 0.0
    
    def test_calculate_return_large_gain(self, portfolio):
        """Test calculating large return"""
        t1_value = 1000.0
        t2_value = 5000.0
        
        return_pct = portfolio.calculate_return(t1_value, t2_value)
        
        assert return_pct == 400.0  # 400% gain


class TestPortfolioIntegration:
    """Integration tests with real scenarios"""

    def test_portfolio_lifecycle(self):
        """Test complete portfolio lifecycle"""
        # Create portfolio
        portfolio = Portfolio(name="Growth Portfolio")

        # Add investments
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('MSFT', 5)
        portfolio.add_investment('GOOGL', 2)

        assert len(portfolio.investments) == 3

        # Create market data indexed by ticker
        market_data = pd.DataFrame({
            'Ending_Price': [150.0, 300.0, 2800.0],
        }, index=['AAPL', 'MSFT', 'GOOGL'])

        # Calculate value
        initial_value = portfolio.present_value(market_data)
        assert initial_value == 10 * 150 + 5 * 300 + 2 * 2800  # 9,600

        # Modify portfolio
        portfolio.remove_investment('GOOGL')
        portfolio.add_investment('AAPL', 5)  # Add more AAPL

        # Recalculate value
        new_value = portfolio.present_value(market_data)
        assert new_value == 15 * 150 + 5 * 300  # 3,750

        # Calculate return
        return_pct = portfolio.calculate_return(initial_value, new_value)
        assert return_pct < 0  # Value decreased

    def test_portfolio_rebalancing_scenario(self):
        """Test a realistic rebalancing scenario"""
        # Initial portfolio
        portfolio = Portfolio(name="Rebalanced Portfolio")
        portfolio.add_investment('AAPL', 10)
        portfolio.add_investment('MSFT', 10)

        # Year 1 market
        market_y1 = pd.DataFrame({
            'Ending_Price': [100.0, 200.0],
        }, index=['AAPL', 'MSFT'])

        value_y1 = portfolio.present_value(market_y1)
        assert value_y1 == 3000.0  # 10*100 + 10*200

        # Year 2 market (prices changed)
        market_y2 = pd.DataFrame({
            'Ending_Price': [150.0, 250.0],
        }, index=['AAPL', 'MSFT'])

        value_y2 = portfolio.present_value(market_y2)
        assert value_y2 == 4000.0  # 10*150 + 10*250

        # Calculate return
        annual_return = portfolio.calculate_return(value_y1, value_y2)
        assert annual_return == pytest.approx(33.33, rel=0.01)  # ~33.33% return
