"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/supabase_client.py
PURPOSE: Silent, high-performance data ingestion with schema-aligned standardization.
VERSION: 2.5.0
"""

import os
import logging
import pandas as pd
from supabase import create_client, Client

from .data_standardization import (
    attach_delisting_info,
    prepare_last_price_mapping,
    standardize_universe,
)

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

        df = standardize_universe(df)

        # Merge delisting dates from last_price_mapping for time-adjusted strategies
        return attach_delisting_info(df, self.fetch_last_price_mapping())

    def fetch_table(self, table_name: str) -> pd.DataFrame:
        """
        Downloads one table exactly as stored (no standardization), paginating
        past the response limit. Raises on failure instead of returning an empty frame.

        Used to export tables, e.g. by scripts/publish_data_to_s3.py.
        """
        page_size = 1000
        offset = 0
        all_rows = []
        while True:
            response = self.client.table(table_name).select('*').range(offset, offset + page_size - 1).execute()
            batch = response.data if hasattr(response, 'data') else []
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return pd.DataFrame(all_rows)

    def fetch_last_price_mapping(self) -> pd.DataFrame:
        """Retrieves the last_price_mapping table containing delisting dates and prices."""
        page_size = 1000
        offset = 0
        all_rows = []

        try:
            while True:
                response = self.client.table('last_price_mapping').select('*').range(offset, offset + page_size - 1).execute()
                batch = response.data if hasattr(response, 'data') else []
                if not batch:
                    break
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
        except Exception as e:
            logger.error(f"Failed to fetch last_price_mapping: {str(e)}")
            return pd.DataFrame()

        if not all_rows:
            return pd.DataFrame()

        return prepare_last_price_mapping(pd.DataFrame(all_rows))

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforces structural and numerical consistency across the dataset.
        Kept for compatibility; the logic lives in src/data_standardization.py
        so every data source cleans the data the same way.
        """
        return standardize_universe(df)
