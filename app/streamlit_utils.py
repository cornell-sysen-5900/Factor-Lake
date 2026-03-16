"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_utils.py
PURPOSE: Utility functions for session management, authentication, and data orchestration.
VERSION: 3.1.0
"""

import os
import sys
import hmac
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List, Optional

def initialize_environment() -> None:
    """Restores the system pathing logic from the working commit."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
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
    """Restores the HMAC gatekeeper from the working version."""
    configured_password = secrets_map.get("password") or os.environ.get("ADMIN_PASSWORD")
    if not configured_password:
        st.error("Authentication secret is missing.")
        return False

    if st.session_state.get("password_correct"):
        return True

    def validate_input():
        if hmac.compare_digest(st.session_state.get("pwd_entry", ""), configured_password):
            st.session_state["password_correct"] = True
            del st.session_state["pwd_entry"]
        else:
            st.session_state["password_correct"] = False

    st.text_input("Administrator Password", type="password", on_change=validate_input, key="pwd_entry")
    if st.session_state.get("password_correct") == False:
        st.error("Invalid credentials.")
    return False

def initialize_session_state() -> None:
    """Maintains the initialization logic from the working commit."""
    default_state = {
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
    """Updated to use the refactored Supabase client while keeping working logic."""
    from src.supabase_client import SupabaseManager
    try:
        if st.session_state.raw_data is None:
            manager = SupabaseManager()
            st.session_state.raw_data = manager.fetch_all_data()

        df = st.session_state.raw_data.copy()

        if user_settings.get('selected_sectors'):
            df = df[df['Scotts_Sector_5'].isin(user_settings['selected_sectors'])]

        if user_settings.get('restrict_fossil_fuels'):
            excluded = {"integratedoil", "oilfieldservicesequipment", "oilgasproduction", "coal", "oilrefiningmarketing"}
            industry_col = 'FactSet_Industry' if 'FactSet_Industry' in df.columns else 'FactSet Industry'
            if industry_col in df.columns:
                clean_ind = df[industry_col].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
                df = df[~clean_ind.isin(excluded)]

        start_yr, end_yr = int(user_settings['start_year']), int(user_settings['end_year'])
        df = df[(df['Year'] >= start_yr) & (df['Year'] <= end_yr)]

        st.session_state.rdata = df
        st.session_state.data_loaded = True
        st.success(f"Universe Refined: {len(df):,} records ready.")
    except Exception as e:
        st.error(f"Data Orchestration Error: {str(e)}")

def run_backtest_logic(user_settings: Dict[str, Any], 
                       factor_names: List[str], 
                       factor_dirs: Dict[str, str]) -> None:
    """Uses the new refactored engine while keeping the working state-binding logic."""
    from src.backtest_engine import rebalance_portfolio
    from app.streamlit_config import FACTOR_METADATA

    try:
        internal_factor_cols = [FACTOR_METADATA[f]['column'] for f in factor_names if f in FACTOR_METADATA]
        internal_directions = {FACTOR_METADATA[f]['column']: factor_dirs.get(f, 'top') for f in factor_names if f in FACTOR_METADATA}

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
    except Exception as e:
        st.error(f"Backtest Execution Error: {str(e)}")