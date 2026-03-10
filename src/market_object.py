import pandas as pd
import numpy as np
from .supabase_client import load_supabase_data, load_constituents_from_supabase
import os

def load_data(restrict_fossil_fuels=False, use_supabase=True, table_name='Full Precision Test', show_loading_progress=True, data_path=None, excel_sheet='Data', sectors=None, data_frequency='Yearly', start_year=None, end_year=None):
    """
    Load market data from either Supabase or Excel file (fallback).
    
    Args:
        restrict_fossil_fuels (bool): Whether to exclude fossil fuel companies
        use_supabase (bool): If True, use Supabase; if False, use Excel fallback
        table_name (str): Name of Supabase table containing market data
        show_loading_progress (bool): Whether to show loading progress messages
    
    Returns:
        pandas.DataFrame: Market data
    """
    
    if use_supabase:
        try:
            # Resolve which Supabase table to use (param > env var > sensible default)
            if data_frequency == 'Daily':
                effective_table = 'daily_prices'
            elif data_frequency == 'Monthly':
                effective_table = 'monthly_prices'
            else:
                effective_table = table_name or os.environ.get('SUPABASE_TABLE') or 'Full Precision Test'

            # Load data from Supabase
            if show_loading_progress:
                print(f"Using Supabase table: '{effective_table}'")
            rdata = load_supabase_data(effective_table, show_progress=show_loading_progress, sectors=sectors, start_year=start_year, end_year=end_year, data_frequency=data_frequency)
            
            if rdata.empty:
                print("Warning: No data loaded from Supabase. Check your table and connection.")
                return rdata
            
            # Standardize column names to match existing code expectations
            rdata = _standardize_column_names(rdata)
            
            # New return logic: 0 if null next year return
            rdata['Next_Year_Return'] = rdata['Next_Year_Return'].fillna(0)

            # Add Period column for sub-annual rebalancing
            if data_frequency in ['Monthly', 'Daily'] and 'Date' in rdata.columns:
                if data_frequency == 'Daily':
                    # Each trading day is its own rebalancing period
                    rdata['Period'] = pd.to_datetime(rdata['Date']).dt.strftime('%Y-%m-%d')
                else:
                    rdata['Period'] = pd.to_datetime(rdata['Date']).dt.to_period('M').astype(str)

            # Filter to Russell 2000 constituents using Supabase
            if data_frequency in ['Monthly', 'Daily']:
                rdata = _apply_constituent_filter_supabase(rdata, start_year=start_year, end_year=end_year)

            # Apply sector restriction logic (post-standardization)
            if restrict_fossil_fuels:
                industry_col = 'FactSet Industry'
                if industry_col in rdata.columns:
                    before = rdata.copy()
                    rdata[industry_col] = rdata[industry_col].astype(str).str.lower()
                    fossil_keywords = ['oil', 'gas', 'coal', 'energy', 'fossil']
                    mask = rdata[industry_col].apply(lambda x: not any(kw in x for kw in fossil_keywords))
                    rdata = rdata[mask]
                    # Report removals (tickers)
                    if 'Ticker' in before.columns and 'Ticker' in rdata.columns:
                        removed = sorted(set(before['Ticker']) - set(rdata['Ticker']))
                        if removed:
                            print(f"Fossil filter removed {len(removed)} tickers (Supabase): {', '.join(removed[:25])}{' ...' if len(removed) > 25 else ''}")
                        else:
                            print("Fossil filter removed 0 tickers (Supabase)")
                else:
                    print("Warning: 'FactSet Industry' column not found. Fossil fuel filtering skipped.")

            # If sectors are provided, apply client-side filter as safety-net (in case server-side failed)
            if sectors:
                rdata = _apply_sector_filter(rdata, sectors, context_label="Supabase")

            # Remove duplicate rows and rows with missing essential data (prices/tickers/dates)
            try:
                before_total = len(rdata)
                # Drop exact duplicate rows
                rdata = rdata.drop_duplicates()
                dup_removed = before_total - len(rdata)

                # Filter out rows missing essential data (uses same logic as for Excel fallback)
                rdata_before_filter = rdata.copy()
                rdata = _filter_essential_data(rdata)
                nulls_removed = len(rdata_before_filter) - len(rdata)

                if dup_removed > 0 or nulls_removed > 0:
                    print(f"Supabase load: removed {dup_removed} duplicate rows and {nulls_removed} rows with missing essential data (out of {before_total} rows).")
            except Exception:
                # Non-fatal: continue without failing the load
                pass

            print(f"Successfully loaded {len(rdata)} records from Supabase")
            # Quick sanity check: distribution by Year after standardization
            if 'Year' in rdata.columns:
                try:
                    year_counts = rdata['Year'].value_counts().sort_index()
                    # Print a compact summary
                    print("Rows per Year (Supabase):", ", ".join([f"{int(y)}: {int(c)}" for y, c in year_counts.items()]))
                except Exception:
                    pass
            return rdata
            
        except Exception as e:
            print(f"Error loading from Supabase: {e}")
            raise RuntimeError(f"Failed to load data from Supabase. Please check your Supabase configuration and secrets. Error: {e}")
    
    if not use_supabase:
        # Fallback to local file (Excel or CSV). Accepts a data_path (CSV/Excel path OR file-like object).
        try:
            from google.colab import drive  # type: ignore
            in_colab = True
        except Exception:
            in_colab = False

        # If no explicit data_path provided, try the historical default for Colab
        if data_path is None and in_colab:
            #data_path = '/content/drive/My Drive/Cayuga Fund Factor Lake/FR2000 Annual Quant Data FOR RETURN SIMULATION.xlsx'
            data_path = '/content/drive/MyDrive/Cayuga Fund Factor Lake/Full Precision Test_rows.csv'
        if data_path is None:
            print("Excel/CSV fallback unavailable: provide data_path when not using Supabase or run in Colab.")
            raise RuntimeError("Excel/CSV fallback unavailable: provide data_path when not using Supabase or run in Colab.")

        # Mount Drive in Colab if necessary (only for string paths)
        if in_colab and isinstance(data_path, str) and not os.path.exists('/content/drive'):
            try:
                print("Mounting Google Drive...")
                drive.mount('/content/drive')
            except Exception:
                pass

        try:
            # Check if data_path is a file-like object (e.g., Streamlit UploadedFile) or a string path
            is_file_like = hasattr(data_path, 'read')
            
            if is_file_like:
                # Handle file-like objects (e.g., from Streamlit file_uploader)
                file_name = getattr(data_path, 'name', 'uploaded_file')
                print(f"Loading data from uploaded file: {file_name}")
                
                if file_name.lower().endswith('.csv'):
                    rdata = pd.read_csv(data_path)
                else:
                    rdata = pd.read_excel(data_path, sheet_name=excel_sheet, header=2, skiprows=[3, 4])
            else:
                # Handle string paths
                print(f"Loading data file from: {data_path}")
                lp = str(data_path).lower()
                if lp.endswith('.csv'):
                    rdata = pd.read_csv(data_path)
                else:
                    rdata = pd.read_excel(data_path, sheet_name=excel_sheet, header=2, skiprows=[3, 4])

            # Normalize column names and remove duplicate columns
            rdata.columns = rdata.columns.str.strip()
            rdata = rdata.loc[:, ~rdata.columns.duplicated(keep='first')]

            # Standardize to the same column names we expect from Supabase
            rdata = _standardize_column_names(rdata)
            
            # New return logic: 0 if null next year return
            rdata['Next_Year_Return'] = rdata['Next_Year_Return'].fillna(0)

            # Apply sector restriction logic (post-standardization)
            if restrict_fossil_fuels:
                industry_col = 'FactSet Industry'
                if industry_col in rdata.columns:
                    before = rdata.copy()
                    rdata[industry_col] = rdata[industry_col].astype(str).str.lower()
                    fossil_keywords = ['oil', 'gas', 'coal', 'energy', 'fossil']
                    mask = rdata[industry_col].apply(lambda x: not any(kw in x for kw in fossil_keywords))
                    rdata = rdata[mask]
                    # Report removals (tickers)
                    if 'Ticker' in before.columns and 'Ticker' in rdata.columns:
                        removed = sorted(set(before['Ticker']) - set(rdata['Ticker']))
                        if removed:
                            print(f"Fossil filter removed {len(removed)} tickers (Excel/CSV): {', '.join(removed[:25])}{' ...' if len(removed) > 25 else ''}")
                        else:
                            print("Fossil filter removed 0 tickers (Excel/CSV)")
                else:
                    print("Warning: 'FactSet Industry' column not found. Fossil fuel filtering skipped.")

            # If sectors are provided, apply client-side filter
            if sectors:
                rdata = _apply_sector_filter(rdata, sectors, context_label="File")

            # Remove duplicate rows and rows with missing essential data (prices/tickers/dates)
            try:
                before_total = len(rdata)
                # Drop exact duplicate rows
                rdata = rdata.drop_duplicates()
                dup_removed = before_total - len(rdata)

                # Filter out rows missing essential data
                rdata_before_filter = rdata.copy()
                rdata = _filter_essential_data(rdata)
                nulls_removed = len(rdata_before_filter) - len(rdata)

                if dup_removed > 0 or nulls_removed > 0:
                    print(f"File load: removed {dup_removed} duplicate rows and {nulls_removed} rows with missing essential data (out of {before_total} rows).")
            except Exception:
                pass

            print(f"Successfully loaded {len(rdata)} records from file")
            # Quick sanity check: distribution by Year after standardization
            if 'Year' in rdata.columns:
                try:
                    year_counts = rdata['Year'].value_counts().sort_index()
                    print("Rows per Year (File):", ", ".join([f"{int(y)}: {int(c)}" for y, c in year_counts.items()]))
                except Exception:
                    pass
            return rdata

        except Exception as e:
            print(f"Error loading data file: {e}")
            raise


def _standardize_column_names(df):
    """
    Hardcoded column name mapping from Supabase format to factor code expectations.
    This maps the EXACT column names from Supabase to what the factor functions expect.
    """
    # HARDCODED mapping - Supabase column names -> Expected column names
    column_mapping = {
        # Core columns
        'ID': 'ID',
        'Security_Name': 'Security Name',
        'Ticker-Region': 'Ticker-Region',
        'Russell_2000_Port_Weight': 'Russell 2000 Port. Weight',
        'Ending_Price': 'Ending Price',
        'Market_Capitalization': 'Market Capitalization',
        'Date': 'Date',
        'FactSet_Industry': 'FactSet Industry',
        'Scotts_Sector_5': "Scott's Sector (5)",

        # Factor columns - EXACT mapping from Supabase
        'ROE_using_9-30_Data': 'ROE using 9/30 Data',
        'ROA_using_9-30_Data': 'ROA using 9/30 Data',
        'Price_to_Book_Using_9-30_Data': 'Price to Book Using 9/30 Data',
        'Next_FY_Earns-P': 'Next FY Earns/P',
        '12-Mo_Momentum': '12-Mo Momentum %',
        '6-Mo_Momentum': '6-Mo Momentum %',
        '1-Mo_Momentum': '1-Mo Momentum %',
        '1-Yr_Price_Vol': '1-Yr Price Vol %',
        'Accruals-Assets': 'Accruals/Assets',
        'ROA': 'ROA %',
        '1-Yr_Asset_Growth': '1-Yr Asset Growth %',
        '1-Yr_CapEX_Growth': '1-Yr CapEX Growth %',
        'Book-Price': 'Book/Price',
        'Next-Years_Return': "Next_Year_Return",
        'Next-Years_Active_Return': "Next-Year's Active Return %",
        
        # New Point-in-Time Daily/Monthly columns
        'permno': 'Ticker',
        'date': 'Date',
        'prc': 'Ending Price',
        'mkt_cap': 'Market Capitalization',
        'ret': 'Next_Year_Return', # Daily/Monthly use ret as the forward period return
        'mom_1m': '1-Mo Momentum %',
        'mom_6m': '6-Mo Momentum %',
        'mom_12m': '12-Mo Momentum %',

        # Financial data columns
        'NI_Millions': 'NI, $Millions',
        'OpCF_Millions': 'OpCF, $Millions',
        'Latest_Assets_Millions': 'Latest Assets, $Millions',
        'Prior_Years_Assets_Millions': "Prior Year's Assets, $Millions",
        'Book_Value_Per_Share': 'Book Value Per Share $',
        'CapEx_Millions': 'CapEx, $Millions',
        'Prior_Years_CapEx_Millions': "Prior Year's CapEx, $Millions",
        'Earnings_Surprise': 'Earnings Surprise %',
        'EarningsReportedLast': 'Earnings Reported Last',
        'Avg_Daily_3-Mo_Volume_Mills': 'Avg Daily 3-Mo Volume Mills $',
    }

    # Apply column name mapping
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist (with fallback logic)
    if 'Ticker' not in df.columns:
        if 'Ticker-Region' in df.columns:
            df['Ticker'] = df['Ticker-Region'].str.split('-').str[0].str.strip()
        elif 'ticker_region' in df.columns:
            df['Ticker'] = df['ticker_region'].str.split('-').str[0].str.strip()
    
    if 'Year' not in df.columns:
        if 'Date' in df.columns:
            df['Year'] = pd.to_datetime(df['Date']).dt.year
        elif 'date' in df.columns:
            df['Year'] = pd.to_datetime(df['date']).dt.year
    
    # Ensure Ticker-Region exists if we have ticker_region in lowercase
    if 'Ticker-Region' not in df.columns and 'ticker_region' in df.columns:
        df['Ticker-Region'] = df['ticker_region']
    
    # If Ticker-Region is still missing but we have Ticker (like from permno), generate it
    if 'Ticker-Region' not in df.columns and 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].astype(str)
        df['Ticker-Region'] = df['Ticker'] + '-US'
    
    # Convert decimal returns (e.g. 0.12) to percentage returns (e.g. 12.0)
    # The legacy engine's calculate_growth divides Next_Year_Return by 100,
    # so it expects percentage format. Supabase ret is in decimal format.
    if 'Next_Year_Return' in df.columns:
        nyr = pd.to_numeric(df['Next_Year_Return'], errors='coerce')
        # If the max absolute value is < 20.0, it's almost certainly decimal format
        # (A 2000% return in decimal is 20.0, whereas in percentage it would be 2000.0)
        if nyr.dropna().abs().max() < 20.0:
            df['Next_Year_Return'] = nyr * 100
    
    return df



def _apply_sector_filter(df: pd.DataFrame, sectors, context_label: str = "") -> pd.DataFrame:
    """
    Filter a DataFrame to only include selected sectors.

    Args:
        df: DataFrame containing at least the column "Scott's Sector (5)" after standardization
        sectors: list of sector names to include
        context_label: short label for logging context (e.g., "Supabase" or "File")

    Returns:
        Filtered DataFrame
    """
    if not sectors:
        return df
    col = "Scott's Sector (5)"
    if col not in df.columns:
        print(f"Warning: '{col}' column not found. Sector filtering skipped ({context_label}).")
        return df
    before = len(df)
    filtered = df[df[col].isin(sectors)].copy()
    removed = before - len(filtered)
    print(f"Sector filter kept {len(filtered)} rows and removed {removed} ({context_label}).")
    return filtered

def _apply_constituent_filter_supabase(df, start_year=None, end_year=None):
    """
    Filter a DataFrame to only include Russell 2000 constituents
    using the Supabase iwm_constituents table.
    
    The iwm_constituents table has columns:
        permno, index_effective, index_through, ticker, sector, is_fossil_fuel
    A stock is a valid constituent if its [index_effective, index_through] window
    overlaps with the row's date.
    """
    try:
        constituents = load_constituents_from_supabase(
            start_year=start_year, end_year=end_year, show_progress=True
        )
        if constituents.empty:
            print("Warning: No constituents loaded from Supabase. Skipping filter.")
            return df
        
        # Build a set of valid permnos with their date ranges
        constituents['index_effective'] = pd.to_datetime(constituents['index_effective'])
        constituents['index_through'] = pd.to_datetime(constituents['index_through'])
        
        # For each row in df, check if its permno was a constituent on that date
        df_temp = df.copy()
        df_temp['_permno_int'] = pd.to_numeric(df_temp['Ticker'], errors='coerce').astype('Int64')
        df_temp['_date'] = pd.to_datetime(df_temp['Date'])
        
        # Merge on permno, then filter by date range
        constituents['_permno_int'] = constituents['permno'].astype('Int64')
        merged = pd.merge(
            df_temp, 
            constituents[['_permno_int', 'index_effective', 'index_through']],
            on='_permno_int', 
            how='inner'
        )
        
        # Keep rows where the data date falls within the constituent membership window
        mask = (merged['_date'] >= merged['index_effective']) & (merged['_date'] <= merged['index_through'])
        filtered = merged[mask].copy()
        
        # Drop helper columns and deduplicate (a permno may match multiple constituent windows)
        filtered = filtered.drop(columns=['_permno_int', '_date', 'index_effective', 'index_through'])
        filtered = filtered.drop_duplicates()
        
        removed = len(df) - len(filtered)
        print(f"Supabase constituent filter kept {len(filtered)} rows and removed {removed}.")
        return filtered
    except Exception as e:
        print(f"Warning: Could not apply Supabase constituent filter: {e}")
        return df

def _filter_essential_data(df):
    """
    Filter out rows with missing essential data like pricing information.
    """
    if df.empty:
        return df
        
    initial_count = len(df)
    
    # Remove rows where Ending Price is missing or invalid
    price_col = None
    if 'Ending Price' in df.columns:
        price_col = 'Ending Price'
    elif 'Ending_Price' in df.columns:
        price_col = 'Ending_Price'
    if price_col:
        # Coerce to numeric in case values are strings
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df[df[price_col].notna() & (df[price_col] > 0)]
    
    # Remove rows where Ticker is missing
    if 'Ticker' in df.columns:
        df = df[df['Ticker'].notna() & (df['Ticker'] != '') & (df['Ticker'] != '--')]
    elif 'Ticker-Region' in df.columns:
        df = df[df['Ticker-Region'].notna() & (df['Ticker-Region'] != '') & (df['Ticker-Region'] != '--')]
    
    # Remove rows where Date/Year is missing
    if 'Year' in df.columns:
        df = df[df['Year'].notna()]
    elif 'Date' in df.columns:
        df = df[df['Date'].notna()]
    
    filtered_count = len(df)
    removed_count = initial_count - filtered_count
    
    if removed_count > 0:
        print(f"Filtered out {removed_count} rows with missing essential data (price, ticker, or date)")
    
    return df

 

"""
Note: A duplicate Supabase loading implementation existed below which directly
queried a hard-coded table and bypassed the standardized column mapping. It has
been removed to ensure a single, robust data-loading path via SupabaseDataClient
above. The supported entry point is load_data(restrict_fossil_fuels=False,
use_supabase=True, table_name='All').
"""

class MarketObject():
    def __init__(self, data, t, verbosity=1):
        """
        data(DataFrame): Market data with columns like 'Ticker', 'Ending Price', etc.
        t (int): Year of market data.
        verbosity (int): Controls level of printed output. 0 = silent, 1 = normal, 2+ = verbose.
        """
        # Remove duplicated column names
        data.columns = data.columns.str.strip()
        data = data.loc[:, ~data.columns.duplicated(keep='first')]

        # Ensure 'Ticker' and 'Year' columns are present
        if 'Ticker' not in data.columns and 'Ticker-Region' in data.columns:
            data['Ticker'] = data['Ticker-Region'].str.split('-').str[0].str.strip()
        if 'Year' not in data.columns and 'Date' in data.columns:
            data['Year'] = pd.to_datetime(data['Date']).dt.year

        # Define relevant columns
        available_factors = [
            'ROE using 9/30 Data', 'ROA using 9/30 Data', '12-Mo Momentum %', '1-Mo Momentum %',
            'Price to Book Using 9/30 Data', 'Next FY Earns/P', '1-Yr Price Vol %', 'Accruals/Assets',
            'ROA %', '1-Yr Asset Growth %', '1-Yr CapEX Growth %', 'Book/Price',
            "Next_Year_Return", "Next-Year's Active Return %"
        ]
        # Keep Ticker-Region so we can index uniquely when present
        # Include Market Capitalization for cap-weighted portfolios
        volatility_metrics = ["vol_gk_1m", "vol_gk_3m", "vol_gk_6m", "vol_gk_12m", "vol_close_1m", "vol_close_3m", "vol_close_6m", "vol_close_12m"]
        keep_cols = ['Ticker-Region', 'Ticker', 'Ending Price', 'Year', '6-Mo Momentum %', 'FactSet Industry', 'Market Capitalization'] + available_factors + volatility_metrics

        # Filter and clean data
        data = data[[col for col in keep_cols if col in data.columns]].copy()
        data.replace({'--': None, 'N/A': None, '#N/A': None, '': None}, inplace=True)
        
        # Convert numeric columns to proper numeric types
        numeric_columns = ['Ending Price', 'Market Capitalization'] + [col for col in available_factors if col in data.columns]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Prefer 'Ticker' index for compatibility with 'main'; fallback to 'Ticker-Region'
        index_col = 'Ticker' if 'Ticker' in data.columns else ('Ticker-Region' if 'Ticker-Region' in data.columns else None)
        if index_col:
            try:
                # Drop duplicate tickers, keeping the last occurrence (most recent)
                data = data.drop_duplicates(subset=[index_col], keep='last')
                data.set_index(index_col, inplace=True)
            except Exception:
                pass

        self.stocks = data
        self.t = t
        self.verbosity = verbosity

    def get_price(self, ticker):
        # Try both 'Ending Price' and 'Ending_Price' for compatibility
        for price_col in ['Ending Price', 'Ending_Price']:
            try:
                price = self.stocks.loc[ticker, price_col]
                if isinstance(price, (pd.Series, np.ndarray)):
                    if hasattr(price, 'dropna') and not price.dropna().empty:
                        price = price.dropna().iloc[0]
                    else:
                        price = price.iloc[0] if hasattr(price, 'iloc') and len(price) > 0 else None
                if pd.isna(price) or price is None or price <= 0:
                    if self.verbosity >= 2:
                        print(f"{ticker} - invalid price ({price}) for {self.t} - SKIPPING")
                    return None
                return price
            except KeyError:
                continue
        if self.verbosity >= 2:
            print(f"{ticker} - not found in market data for {self.t} - SKIPPING")
        return None
