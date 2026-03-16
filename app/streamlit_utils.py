"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_utils.py
PURPOSE: Utility functions for session management, authentication, and data orchestration.
VERSION: 2.2.0
"""

import os
import sys
import hmac
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Optional

def initialize_environment() -> None:
    """
    Configures the application runtime environment by establishing necessary system paths.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent 
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if not os.environ.get("STREAMLIT_RUNTIME_ENV"):
        env_path = project_root / '.env'
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key, value.strip("'\""))

def check_password(secrets_map: Dict[str, Any]) -> bool:
    """
    Implements a secure gatekeeper mechanism using HMAC-based credential verification.
    """
    configured_password = secrets_map.get("password") or os.environ.get("ADMIN_PASSWORD")

    if not configured_password:
        st.error("Authentication secret is missing. Please check system configuration.")
        return False

    if st.session_state.get("password_correct"):
        return True

    def validate_input():
        user_input = st.session_state.get("pwd_entry", "")
        if hmac.compare_digest(user_input, configured_password):
            st.session_state["password_correct"] = True
            del st.session_state["pwd_entry"]
        else:
            st.session_state["password_correct"] = False

    st.text_input("Administrator Password", type="password", on_change=validate_input, key="pwd_entry")
    
    if st.session_state.get("password_correct") == False:
        st.error("Invalid credentials. Access denied.")
    
    return False

def initialize_session_state() -> None:
    """
    Initializes the global state container with default values for tracking application data.
    """
    default_state: Dict[str, Any] = {
        'data_loaded': False,
        'raw_data': None,
        'rdata': None,
        'results': None,
        'selected_factors': [],
        'factor_directions': {}
    }
    
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def load_and_process_data(user_settings: Dict[str, Any]) -> None:
    """
    Orchestrates the retrieval and refinement of market data from the cloud backend.
    """
    from src.supabase_client import SupabaseManager
    import pandas as pd

    try:
        # Step 1: Fetch raw universe if not already cached
        if st.session_state.raw_data is None:
            manager = SupabaseManager()
            st.session_state.raw_data = manager.fetch_all_data()

        df = st.session_state.raw_data.copy()

        # Step 2: Apply Sector Filters
        if user_settings.get('selected_sectors'):
            df = df[df['Scotts_Sector_5'].isin(user_settings['selected_sectors'])]

        # Step 3: ESG Exclusionary Screening (Fossil Fuels)
        if user_settings.get('restrict_fossil_fuels'):
            excluded = {"integratedoil", "oilfieldservicesequipment", "oilgasproduction", "coal", "oilrefiningmarketing"}
            # Schema-safe column lookup
            industry_col = 'FactSet_Industry' if 'FactSet_Industry' in df.columns else 'FactSet Industry'
            
            if industry_col in df.columns:
                # Optimized cleaning for string matching
                clean_ind = df[industry_col].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
                df = df[~clean_ind.isin(excluded)]

        # Step 4: Temporal Constraints
        start_yr, end_yr = int(user_settings['start_year']), int(user_settings['end_year'])
        df = df[(df['Year'] >= start_yr) & (df['Year'] <= end_yr)]

        st.session_state.rdata = df
        st.session_state.data_loaded = True
        st.success(f"Universe Refined: {len(df):,} records ready for analysis.")

    except Exception as e:
        st.error(f"Data Orchestration Error: {str(e)}")

def run_backtest_logic(user_settings: Dict[str, Any], 
                        factor_names: List[str], 
                        factor_dirs: Dict[str, str]) -> None:
    """
    Triggers the core rebalancing engine to calculate historical portfolio performance.
    """
    from src.calculate_holdings import rebalance_portfolio
    from app.streamlit_config import FACTOR_METADATA

    if not factor_names:
        st.warning("Strategy Configuration: Please select at least one factor.")
        return

    try:
        # Map UI labels to internal SQL column names
        # e.g., '6-Mo Momentum %' -> '6-Mo_Momentum'
        internal_factor_cols = []
        internal_directions = {}

        for f_label in factor_names:
            if f_label in FACTOR_METADATA:
                sql_col = FACTOR_METADATA[f_label]['column']
                internal_factor_cols.append(sql_col)
                # Ensure the direction toggle is mapped to the SQL column name
                internal_directions[sql_col] = factor_dirs.get(f_label, 'top')
            else:
                st.error(f"Metadata Missing: No SQL mapping found for factor '{f_label}'.")
                return

        # Execute Backtest
        results = rebalance_portfolio(
            st.session_state.rdata,
            internal_factor_cols,
            factor_directions=internal_directions,
            start_year=int(user_settings['start_year']),
            end_year=int(user_settings['end_year']),
            initial_aum=user_settings['initial_aum'],
            benchmark_index=user_settings.get('benchmark_index', 1),
            top_pct=user_settings.get('top_pct', 10.0),
            use_market_cap_weight=user_settings.get('use_market_cap_weight', False)
        )
        
        st.session_state.results = results
        st.session_state.selected_factors = factor_names
        st.session_state.factor_directions = factor_dirs
        
    except Exception as e:
        st.error(f"Backtest Execution Error: {str(e)}")
        st.exception(e)