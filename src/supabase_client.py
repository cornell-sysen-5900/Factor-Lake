"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/supabase_client.py
PURPOSE: Silent, high-performance data ingestion with schema-aligned standardization.
VERSION: 2.4.0
"""

import os
import logging
import pandas as pd
import numpy as np
from supabase import create_client, Client

# Suppress external library verbosity to maintain clean terminal output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class SupabaseManager:
    """
    Orchestrates bulk data ingestion and preprocessing for the investment universe.
    
    This manager interfaces with Supabase to retrieve longitudinal market data, 
    applying necessary transformations to ensure the dataset conforms to the 
    analytical requirements of the backtesting engine.
    """

    def __init__(self):
        """Initializes the Supabase client using authenticated environment variables."""
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_KEY')

        if not url or not key:
            raise RuntimeError("Cloud configuration failure: SUPABASE_URL or SUPABASE_KEY not found.")
        
        self.client: Client = create_client(url, key)

    def fetch_all_data(self, table_name: str = 'Full Precision Test') -> pd.DataFrame:
        """
        Retrieves the complete dataset from the specified table via iterative pagination.
        
        This method utilizes range-based queries to circumvent database response 
        limits, ensuring comprehensive data coverage for multi-year simulations.
        """
        page_size = 1000
        offset = 0
        all_rows = []

        try:
            while True:
                response = self.client.table(table_name).select('*').range(offset, offset + page_size - 1).execute()
                batch = response.data if hasattr(response, 'data') else []

                if not batch:
                    break
                
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                
                offset += page_size
        except Exception as e:
            logger.error(f"Network Ingestion Failure: {str(e)}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        
        if df.empty:
            logger.warning("Ingestion process completed but returned an empty dataset.")
            return df

        return self._standardize_dataframe(df)

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforces structural and numerical consistency across the dataset.
        """
        # 1. Column Hygiene
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # 2. Identifier Parsing
        if 'Ticker-Region' in df.columns:
            df['Ticker'] = df['Ticker-Region'].str.split('-').str[0].str.strip().str.upper()

        # 3. Temporal Alignment
        if 'Date' in df.columns:
            df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

        # 4. Standardized Null Conversion
        sentinel_values = ['--', 'N/A', '#N/A', 'NULL', 'null', 'nan', '']
        df = df.replace(sentinel_values, np.nan)

        # 5. Dynamic Numeric Conversion
        # Automatically cast columns that contain factor-lake signals or price data
        numeric_keywords = ['Price', 'Return', 'Weight', 'Data', 'ROE', 'ROA', 'Cap']
        for col in df.columns:
            if any(key in col for key in numeric_keywords):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Schema Integrity Validation
        required_cols = ['Ticker', 'Year', 'Ending_Price', 'Next-Years_Return']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            logger.error(f"Integrity Error: Missing critical schema columns: {missing}")
            
        return df