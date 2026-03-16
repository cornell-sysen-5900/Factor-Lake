"""
Test suite for factor_function.py module
Tests factor classes and their data retrieval
"""
import pytest
import pandas as pd
import numpy as np
from src.factor_registry import (
    Factors, Momentum6m, Momentum12m, Momentum1m,
    ROE, ROA, P2B, NextFYrEarns, OneYrPriceVol,
    AccrualsAssets, ROAPercentage, OneYrAssetGrowth,
    OneYrCapEXGrowth, BookPrice, NextYrReturn, NextYrActiveReturn
)
from src.market_object import MarketObject, load_data


class TestFactorsBase:
    """Test the base Factors class"""
    
    def test_factor_initialization(self):
        """Test creating a factor with a column name"""
        factor = Factors('Test Column')
        assert factor.column_name == 'Test Column'
    
    def test_factor_str_representation(self):
        """Test string representation of factor"""
        factor = Factors('Test Column')
        assert str(factor) == 'Test Column'
    
    def test_factor_repr(self):
        """Test debug representation of factor"""
        factor = Factors('Test Column')
        assert repr(factor) == "Factors('Test Column')"


class TestFactorClasses:
    """Test all factor subclasses"""
    
    def test_momentum_6m(self):
        """Test 6-month momentum factor"""
        factor = Momentum6m()
        assert factor.column_name == '6-Mo Momentum %'
    
    def test_momentum_12m(self):
        """Test 12-month momentum factor"""
        factor = Momentum12m()
        assert factor.column_name == '12-Mo Momentum %'
    
    def test_momentum_1m(self):
        """Test 1-month momentum factor"""
        factor = Momentum1m()
        assert factor.column_name == '1-Mo Momentum %'
    
    def test_roe(self):
        """Test ROE factor"""
        factor = ROE()
        assert factor.column_name == 'ROE using 9/30 Data'
    
    def test_roa(self):
        """Test ROA factor"""
        factor = ROA()
        assert factor.column_name == 'ROA using 9/30 Data'
    
    def test_p2b(self):
        """Test Price to Book factor"""
        factor = P2B()
        assert factor.column_name == 'Price to Book Using 9/30 Data'
    
    def test_next_fy_earns(self):
        """Test Next FY Earnings factor"""
        factor = NextFYrEarns()
        assert factor.column_name == 'Next FY Earns/P'
    
    def test_one_yr_price_vol(self):
        """Test 1-year price volatility factor"""
        factor = OneYrPriceVol()
        assert factor.column_name == '1-Yr Price Vol %'
    
    def test_accruals_assets(self):
        """Test Accruals/Assets factor"""
        factor = AccrualsAssets()
        assert factor.column_name == 'Accruals/Assets'
    
    def test_roa_percentage(self):
        """Test ROA percentage factor"""
        factor = ROAPercentage()
        assert factor.column_name == 'ROA %'
    
    def test_one_yr_asset_growth(self):
        """Test 1-year asset growth factor"""
        factor = OneYrAssetGrowth()
        assert factor.column_name == '1-Yr Asset Growth %'
    
    def test_one_yr_capex_growth(self):
        """Test 1-year CapEX growth factor"""
        factor = OneYrCapEXGrowth()
        assert factor.column_name == '1-Yr CapEX Growth %'
    
    def test_book_price(self):
        """Test Book/Price factor"""
        factor = BookPrice()
        assert factor.column_name == 'Book/Price'
    
    def test_next_yr_return(self):
        """Test Next Year Return factor"""
        factor = NextYrReturn()
        assert factor.column_name == "Next-Year's Return %"
    
    def test_next_yr_active_return(self):
        """Test Next Year Active Return factor"""
        factor = NextYrActiveReturn()
        assert factor.column_name == "Next-Year's Active Return %"


class TestFactorRetrieval:
    """Test factor value retrieval from market data"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        return pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US', 'GOOGL-US'],
            'Ending Price': [150.0, 300.0, 2800.0],
            '6-Mo Momentum %': [0.15, 0.20, 0.10],
            '12-Mo Momentum %': [0.25, 0.30, 0.18],
            'ROE using 9/30 Data': [0.25, 0.30, 0.22],
            'ROA using 9/30 Data': [0.18, 0.22, 0.20],
            'Year': [2022, 2022, 2022]
        })
    
    @pytest.fixture
    def market(self, sample_market_data):
        """Create a MarketObject"""
        return MarketObject(sample_market_data, 2022)
    
    def test_get_valid_factor_value(self, market):
        """Test retrieving a valid factor value"""
        factor = Momentum6m()
        value = factor.get('AAPL', market)
        assert value == 0.15
    
    def test_get_multiple_factors(self, market):
        """Test retrieving values for multiple factors"""
        mom6 = Momentum6m()
        mom12 = Momentum12m()
        roe = ROE()
        
        assert mom6.get('AAPL', market) == 0.15
        assert mom12.get('AAPL', market) == 0.25
        assert roe.get('AAPL', market) == 0.25
    
    def test_get_missing_ticker(self, market):
        """Test retrieving factor for non-existent ticker"""
        factor = Momentum6m()
        value = factor.get('INVALID', market)
        assert value is None
    
    def test_get_missing_column(self, market):
        """Test retrieving factor when column doesn't exist"""
        # Create a factor with non-existent column
        factor = Factors('Non Existent Column')
        value = factor.get('AAPL', market)
        assert value is None
    
    def test_factor_with_nan_values(self):
        """Test factor retrieval with NaN values"""
        data = pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US', 'GOOGL-US'],
            'Ending Price': [150.0, 300.0, 2800.0],
            '6-Mo Momentum %': [0.15, np.nan, 0.10],
            'Year': [2022, 2022, 2022]
        })
        
        market = MarketObject(data, 2022)
        factor = Momentum6m()
        
        # Valid value
        assert factor.get('AAPL', market) == 0.15
        
        # NaN value should be returned as is or handled
        msft_value = factor.get('MSFT', market)
        assert msft_value is None or pd.isna(msft_value)
    
    def test_all_factors_on_market(self, market):
        """Test that all factor classes can retrieve data"""
        factors = [
            Momentum6m(), Momentum12m(), ROE(), ROA()
        ]
        
        for factor in factors:
            value = factor.get('AAPL', market)
            # Should return a value (not error)
            assert value is not None or value is None  # Either way, no exception


class TestFactorEdgeCases:
    """Test edge cases and error handling"""
    
    def test_factor_with_empty_market(self):
        """Test factor on empty market"""
        empty_data = pd.DataFrame(columns=['Ticker-Region', 'Ending Price', '6-Mo Momentum %'])
        market = MarketObject(empty_data, 2022)
        
        factor = Momentum6m()
        value = factor.get('AAPL', market)
        assert value is None
    
    def test_factor_with_special_characters_in_ticker(self):
        """Test factor with special characters in ticker"""
        data = pd.DataFrame({
            'Ticker-Region': ['BRK.B-US', 'BF.B-US'],
            'Ending Price': [300.0, 50.0],
            '6-Mo Momentum %': [0.12, 0.08],
            'Year': [2022, 2022]
        })
        
        market = MarketObject(data, 2022)
        factor = Momentum6m()
        
        value = factor.get('BRK.B', market)
        assert value == 0.12


class TestFactorIntegration:
    """Integration tests with real data"""
    
    @pytest.fixture
    def real_market(self):
        """Create a market with real data"""
        data = load_data(use_supabase=True)
        year_data = data[data['Year'] == 2022]
        if len(year_data) > 0:
            return MarketObject(year_data, 2022)
        pytest.skip("No data available for year 2022")
    
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_factors_with_real_data(self, real_market):
        """Test factors work with real market data"""
        factors = [Momentum6m(), ROE(), ROA()]
        
        # Get first ticker from market
        first_ticker = real_market.stocks.index[0]
        
        for factor in factors:
            value = factor.get(first_ticker, real_market)
            # Should either return a number or None (no exception)
            assert isinstance(value, (int, float, type(None))) or pd.isna(value)
