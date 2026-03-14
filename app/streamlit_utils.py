"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/utils/streamlit_utils.py
PURPOSE: Utility functions for session management, authentication, and data orchestration.
VERSION: 1.1.0
"""

import os
import sys
import hmac
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List

def initialize_environment() -> None:
    """
    Configures the Python path and loads environment variables.
    
    This function ensures that the 'src' directory is added to sys.path 
    for module resolution and loads a local .env file if the application 
    is running in a local development environment.
    
    Input:
        None
    Output:
        None: Modifies sys.path and os.environ in-place.
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    # Ensure src is importable
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Load local .env if not running on Streamlit Cloud
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
    Renders a password input field and verifies credentials via HMAC comparison.
    
    Input:
        secrets_map (Dict[str, Any]): A dictionary containing the 'password' key.
    Output:
        bool: True if the session is authenticated, False otherwise.
    """
    configured_password = secrets_map.get("password") or os.environ.get("ADMIN_PASSWORD")

    if not configured_password:
        st.error("Authentication secret is missing. Please check your configuration.")
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

    st.text_input(
        "Administrator Password", 
        type="password", 
        on_change=validate_input, 
        key="pwd_entry"
    )
    
    if st.session_state.get("password_correct") == False:
        st.error("Invalid credentials. Access denied.")
    
    st.caption("Contact the system administrator if you require access.")
    return False


def initialize_session_state() -> None:
    """
    Ensures all necessary keys exist in st.session_state with default values.
    
    Input:
        None
    Output:
        None: Modifies st.session_state in-place.
    """
    default_state: Dict[str, Any] = {
        'data_loaded': False,
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
    Orchestrates data acquisition from Supabase and performs initial cleaning.
    
    Input:
        user_settings (Dict[str, Any]): Parameters including 'start_year', 'end_year', 
                                        and 'selected_sectors'.
    Output:
        None: Updates st.session_state.rdata and sets data_loaded to True.
    """
    from src.market_object import load_data
    import pandas as pd
    import streamlit_config as config

    with st.spinner("Loading market data..."):
        try:
            sectors_to_use = user_settings['selected_sectors'] if user_settings['sector_filter_enabled'] else None

            rdata = load_data(
                restrict_fossil_fuels=user_settings['restrict_fossil_fuels'],
                use_supabase=True,
                data_path=None,
                show_loading_progress=user_settings['show_loading'],
                sectors=sectors_to_use
            )

            # Standardize formats
            rdata['Ticker'] = rdata['Ticker-Region'].dropna().apply(lambda x: x.split('-')[0].strip())
            rdata['Year'] = pd.to_datetime(rdata['Date']).dt.year

            start_yr, end_yr = int(user_settings['start_year']), int(user_settings['end_year'])
            rdata = rdata[(rdata['Year'] >= start_yr) & (rdata['Year'] <= end_yr)]

            # Map column variations
            cols_to_keep = ['Ticker', 'Year', 'Next_Year_Return']
            col_mappings = {
                'Ending Price': ['Ending Price', 'Ending_Price'],
                'Market Capitalization': ['Market Capitalization', 'Market_Capitalization']
            }
            
            for target, options in col_mappings.items():
                for opt in options:
                    if opt in rdata.columns:
                        rdata[target] = rdata[opt]
                        cols_to_keep.append(target)
                        break

            for factor_label in config.FACTOR_MAP.keys():
                if factor_label in rdata.columns:
                    cols_to_keep.append(factor_label)

            st.session_state.rdata = rdata[list(set(cols_to_keep))]
            st.session_state.data_loaded = True
            st.success(f"Data loaded successfully! {len(rdata)} records found.")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")


def run_backtest_logic(user_settings: Dict[str, Any], 
                       factor_names: List[str], 
                       factor_dirs: Dict[str, str]) -> None:
    """
    Executes the portfolio rebalancing engine and stores the backtest results.
    
    Input:
        user_settings (Dict[str, Any]): UI parameters (AUM, Weighting, etc.).
        factor_names (List[str]): List of active factor labels.
        factor_dirs (Dict[str, str]): Mapping of factors to 'top' or 'bottom' tilt.
    Output:
        None: Updates st.session_state.results with the calculation output.
    """
    from src.calculate_holdings import rebalance_portfolio
    import streamlit_config as config

    if not factor_names:
        st.warning("Please select at least one factor before running the analysis.")
        return

    with st.spinner("Running portfolio backtest..."):
        try:
            factor_objects = [config.FACTOR_MAP[name]() for name in factor_names]
            
            results = rebalance_portfolio(
                st.session_state.rdata,
                factor_objects,
                start_year=int(user_settings['start_year']),
                end_year=int(user_settings['end_year']),
                initial_aum=user_settings['initial_aum'],
                verbosity=user_settings['verbosity_level'],
                restrict_fossil_fuels=user_settings['restrict_fossil_fuels'],
                use_market_cap_weight=user_settings['use_market_cap_weight'],
                factor_directions=factor_dirs
            )
            
            st.session_state.results = results
            st.session_state.selected_factors = factor_names
            st.session_state.factor_directions = factor_dirs
            st.success("Analysis complete! Check the Results tab.")
            
        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")
            st.exception(e)