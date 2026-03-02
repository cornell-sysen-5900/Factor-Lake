"""
Test suite for market_object.py module
Tests data loading, market object creation, and price retrieval
"""
import pytest
import pandas as pd
import numpy as np
from src.market_object import MarketObject, load_data, _standardize_column_names


class TestDataLoading:
    """Test data loading functionality"""
    
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_load_data_supabase(self):
        """Test loading data from Supabase"""
        data = load_data(use_supabase=True)
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty, "Loaded data should not be empty"
        assert 'Year' in data.columns, "Data should contain 'Year' column"
        assert 'Ending Price' in data.columns, "Data should contain 'Ending Price' column"
        
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_load_data_columns(self):
        """Test that loaded data has required columns"""
        data = load_data(use_supabase=True)
        
        required_columns = [
            'Year', 'Ending Price', 'Ticker-Region',
            '6-Mo Momentum %', 'ROE using 9/30 Data'
        ]
        
        for col in required_columns:
            assert col in data.columns, f"Missing required column: {col}"
    
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_load_data_types(self):
        """Test data types of loaded data"""
        data = load_data(use_supabase=True)
        
        # Year should be numeric
        assert pd.api.types.is_numeric_dtype(data['Year']), "Year should be numeric"
        
        # Ending Price should be numeric
        assert pd.api.types.is_numeric_dtype(data['Ending Price']), "Ending Price should be numeric"
        
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_fossil_fuel_restriction(self):
        """Test fossil fuel filtering"""
        data_unrestricted = load_data(use_supabase=True, restrict_fossil_fuels=False)
        data_restricted = load_data(use_supabase=True, restrict_fossil_fuels=True)
        
        # Restricted data should have fewer or equal rows
        assert len(data_restricted) <= len(data_unrestricted), \
            "Restricted data should have fewer or equal rows than unrestricted"


class TestColumnStandardization:
    """Test column name standardization"""
    
    def test_standardize_column_names(self):
        """Test that Supabase column names are correctly standardized"""
        # Create sample data with Supabase-style column names
        test_data = pd.DataFrame({
            'ID': [1, 2],
            'Security_Name': ['Test A', 'Test B'],
            'Ticker-Region': ['AAPL-US', 'MSFT-US'],
            '6-Mo_Momentum': [0.10, 0.15],
            'ROE_using_9-30_Data': [0.20, 0.25],
            '1-Yr_Price_Vol': [0.30, 0.35],
            'Next_FY_Earns-P': [0.05, 0.08]
        })
        
        standardized = _standardize_column_names(test_data)
        
        # Check that key columns are standardized correctly
        assert '6-Mo Momentum %' in standardized.columns
        assert 'ROE using 9/30 Data' in standardized.columns
        assert '1-Yr Price Vol %' in standardized.columns
        assert 'Next FY Earns/P' in standardized.columns


class TestMarketObject:
    """Test MarketObject class functionality"""
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        return pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US', 'GOOGL-US'],
            'Ending Price': [150.0, 300.0, 2800.0],
            '6-Mo Momentum %': [0.15, 0.20, 0.10],
            'ROE using 9/30 Data': [0.25, 0.30, 0.22],
            'Year': [2022, 2022, 2022]
        })
    
    @pytest.fixture
    def market(self, sample_market_data):
        """Create a MarketObject instance"""
        return MarketObject(sample_market_data, 2022)
    
    def test_market_creation(self, market):
        """Test MarketObject initialization"""
        assert market.t == 2022
        assert isinstance(market.stocks, pd.DataFrame)
        assert len(market.stocks) == 3
    
    def test_market_ticker_index(self, market):
        """Test that market stocks are indexed by ticker"""
        assert 'AAPL' in market.stocks.index
        assert 'MSFT' in market.stocks.index
        assert 'GOOGL' in market.stocks.index
    
    def test_get_price(self, market):
        """Test price retrieval"""
        price = market.get_price('AAPL')
        assert price == 150.0
        
        price = market.get_price('MSFT')
        assert price == 300.0
    
    def test_get_price_missing_ticker(self, market):
        """Test price retrieval for non-existent ticker"""
        price = market.get_price('INVALID')
        assert price is None
    
    def test_market_with_missing_values(self):
        """Test MarketObject handles missing values correctly"""
        data = pd.DataFrame({
            'Ticker-Region': ['AAPL-US', 'MSFT-US', 'GOOGL-US'],
            'Ending Price': [150.0, np.nan, 2800.0],
            '6-Mo Momentum %': [0.15, 0.20, np.nan],
            'ROE using 9/30 Data': [0.25, np.nan, 0.22],
            'Year': [2022, 2022, 2022]
        })
        
        market = MarketObject(data, 2022)
        
        # Should handle NaN prices gracefully
        price = market.get_price('MSFT')
        assert price is None or pd.isna(price)


class TestMarketObjectIntegration:
    """Integration tests using real data"""
    
    @pytest.fixture
    def real_data(self):
        """Load real data for integration tests"""
        return load_data(use_supabase=True)
    
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_market_object_with_real_data(self, real_data):
        """Test MarketObject with real loaded data"""
        # Get data for a specific year
        year = 2022
        year_data = real_data[real_data['Year'] == year]
        
        if len(year_data) > 0:
            market = MarketObject(year_data, year)
            
            assert market.t == year
            assert len(market.stocks) > 0
            
            # Test that we can retrieve prices for stocks in the market
            first_ticker = market.stocks.index[0]
            price = market.get_price(first_ticker)
            assert price is not None or pd.isna(price)
    
    @pytest.mark.skip(reason="Requires Supabase credentials")
    def test_multiple_years(self, real_data):
        """Test creating markets for multiple years"""
        years = real_data['Year'].unique()[:3]  # Test first 3 years
        
        for year in years:
            year_data = real_data[real_data['Year'] == year]
            if len(year_data) > 0:
                market = MarketObject(year_data, int(year))
                assert market.t == int(year)
                assert len(market.stocks) > 0
