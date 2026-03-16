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

# Silence external library verbosity (mutes the httpx GET logs)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)

# Institutional logging for the application
logger = logging.getLogger(__name__)

class SupabaseManager:
    """
    Manages bulk ingestion and standardization of market data from Supabase.
    """

    def __init__(self):
        """Initializes the connection using validated environment variables."""
        url = os.environ.get('SUPABASE_URL')
        key = os.environ.get('SUPABASE_KEY')

        if not url or not key:
            raise RuntimeError("Cloud configuration missing: SUPABASE_URL or SUPABASE_KEY not found.")
        
        self.client: Client = create_client(url, key)

    def fetch_all_data(self, table_name: str = 'Full Precision Test') -> pd.DataFrame:
        """
        Ingests the entire target table using iterative range-based pagination.
        Executes silently to prevent terminal clutter.
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

        df = pd.DataFrame(all_rows)
        
        if df.empty:
            logger.warning("Supabase ingestion returned an empty dataset.")
            return df

        return self._standardize_dataframe(df)

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforces structural consistency based on actual Supabase SQL schema.
        """
        # 1. Hygiene
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # 2. Ticker Normalization
        if 'Ticker-Region' in df.columns:
            df['Ticker'] = df['Ticker-Region'].str.split('-').str[0].str.strip().str.upper()

        # 3. Temporal Standardization
        if 'Date' in df.columns:
            df['Year'] = pd.to_datetime(df['Date']).dt.year

        # 4. Global Null Handling
        df = df.replace(['--', 'N/A', '#N/A', 'NULL', 'null', 'nan', ''], np.nan)

        # 5. Strict Numeric Casting (Aligned with SQL Schema Output)
        numeric_targets = [
            'Ending_Price', 'Next-Years_Return', 'Market_Capitalization', 
            'Russell_2000_Port_Weight', 'ROE_using_9-30_Data', 'ROA_using_9-30_Data'
        ]
        
        for col in df.columns:
            if col in numeric_targets:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 6. Schema Validation: Match actual SQL column names
        required = ['Ticker', 'Year', 'Ending_Price', 'Next-Years_Return']
        missing = [r for r in required if r not in df.columns]
        
        if missing:
            logger.error(f"SCHEMA MISMATCH: Missing critical SQL columns: {missing}")
            
        return df