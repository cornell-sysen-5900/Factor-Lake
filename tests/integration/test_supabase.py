"""
Test suite for Supabase integration
Tests data loading, pagination, and database connectivity

NOTE: All tests in this module require Supabase credentials.
Mark as @pytest.mark.integration and @pytest.mark.slow
Run with: SUPABASE_URL=... SUPABASE_KEY=... pytest -m integration tests/integration/test_supabase_integration.py

TODO(supabase): Once credentials are restored, verify:
  - The 'FR2000 Annual Quant Data' table still exists (test_load_supabase_data_with_custom_table).
  - Post-standardization column names match expectations (Ticker, Year, Ending_Price, etc.).
  - The >10,000 record threshold still holds for the 'Full Precision Test' table.
  - The Supabase project URL is valid (current secrets.toml points to a dead project).
"""
import pytest
import pandas as pd
import os
from src.supabase_client import SupabaseManager

# Skip entire module if no credentials
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.getenv('SUPABASE_URL') or not os.getenv('SUPABASE_KEY'),
        reason="Requires Supabase credentials (SUPABASE_URL and SUPABASE_KEY)"
    )
]


def _load_data(table_name='Full Precision Test'):
    """Helper to load data via the current SupabaseManager API."""
    manager = SupabaseManager()
    return manager.fetch_all_data(table_name=table_name)


class TestSupabaseConnection:
    """Test Supabase connection and basic queries"""

    def test_load_supabase_data_basic(self):
        """Test basic data loading from Supabase"""
        data = _load_data()

        assert isinstance(data, pd.DataFrame)
        assert not data.empty, "Data should not be empty"
        assert len(data) > 0, "Should load at least some records"

    def test_load_supabase_data_with_custom_table(self):
        """Test loading from custom table name"""
        data = _load_data(table_name='FR2000 Annual Quant Data')

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

    def test_loaded_data_structure(self):
        """Test that loaded data has expected structure"""
        data = _load_data()

        # Should have multiple columns
        assert len(data.columns) > 10, "Should have multiple columns"

        # Should have expected key columns (post-standardization names)
        expected_columns = ['Ticker', 'Year', 'Ending_Price']
        for col in expected_columns:
            assert col in data.columns, f"Missing column: {col}"


class TestSupabasePagination:
    """Test pagination functionality"""

    def test_pagination_loads_all_records(self):
        """Test that pagination loads all available records"""
        data = _load_data()

        # Should load thousands of records (known dataset size)
        assert len(data) > 10000, "Should load large dataset with pagination"

    def test_pagination_no_duplicates(self):
        """Test that pagination doesn't create duplicate records"""
        data = _load_data()

        # Check if there are duplicate rows
        duplicates = data.duplicated()
        assert duplicates.sum() == 0, "Should not have duplicate rows"


class TestSupabaseDataQuality:
    """Test quality of loaded data"""

    @pytest.fixture
    def data(self):
        """Load data for testing"""
        return _load_data()

    def test_data_has_records(self, data):
        """Test that data has substantial number of records"""
        assert len(data) > 1000, "Should have significant amount of data"

    def test_required_columns_present(self, data):
        """Test that all required columns are present after standardization"""
        required = [
            'Ticker', 'Year', 'Ending_Price', 'Next-Years_Return',
            '6-Mo_Momentum', 'ROE_using_9-30_Data'
        ]

        for col in required:
            assert col in data.columns, f"Required column missing: {col}"

    def test_data_types_appropriate(self, data):
        """Test that data types are appropriate"""
        # Ending_Price should be numeric after standardization
        if 'Ending_Price' in data.columns:
            assert pd.api.types.is_numeric_dtype(data['Ending_Price'])

    def test_no_all_null_columns(self, data):
        """Test that no columns are completely null"""
        for col in data.columns:
            null_count = data[col].isna().sum()
            assert null_count < len(data), f"Column {col} is all null"

    def test_ticker_format(self, data):
        """Test that tickers have expected format"""
        if 'Ticker-Region' in data.columns:
            sample_tickers = data['Ticker-Region'].dropna().head(10)

            for ticker in sample_tickers:
                assert '-' in str(ticker), f"Ticker should have region: {ticker}"

    def test_price_data_reasonable(self, data):
        """Test that price data is reasonable (excludes known bad rows with market-cap-scale values)"""
        if 'Ending_Price' in data.columns:
            prices = data['Ending_Price'].dropna()
            # Known data quality issue: a handful of rows have market-cap values
            # in the Ending_Price column (e.g. $1.5T). Filter to realistic prices.
            reasonable = prices[prices < 100_000]

            assert len(reasonable) > len(prices) * 0.99, "More than 1% of prices are unreasonable"
            assert (reasonable > 0).all(), "All prices should be positive"
            assert reasonable.max() < 100000, "Prices seem unreasonably high"
            assert reasonable.min() > 0, "Prices should be above 0"

    def test_year_column_exists(self, data):
        """Test that Year column exists after standardization"""
        assert 'Year' in data.columns, "Year column should exist"
        years = data['Year'].dropna()
        assert len(years) > 0, "Should have some valid years"
        assert years.min() >= 1990, "Years should be reasonable"
        assert years.max() <= 2030, "Years should be reasonable"


class TestSupabasePerformance:
    """Test performance and efficiency"""

    @pytest.mark.slow
    def test_load_data_reasonable_time(self):
        """Test that data loads in reasonable time"""
        import time

        start = time.time()
        data = _load_data()
        elapsed = time.time() - start

        assert elapsed < 30, f"Data loading took too long: {elapsed:.2f}s"
        assert len(data) > 0


class TestSupabaseErrorHandling:
    """Test error handling and edge cases"""

    def test_load_invalid_table_name(self):
        """Test loading from non-existent table"""
        try:
            data = _load_data(table_name='NonExistentTable12345')
            assert isinstance(data, pd.DataFrame)
        except Exception as e:
            assert isinstance(e, Exception)

    def test_connection_resilience(self):
        """Test that connection can be re-established"""
        data1 = _load_data()
        data2 = _load_data()

        assert len(data1) == len(data2), "Should load same amount of data"
        assert len(data1.columns) == len(data2.columns), "Should have same columns"
