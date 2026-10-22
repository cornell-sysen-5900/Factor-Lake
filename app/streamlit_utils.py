"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_utils.py
PURPOSE: Utility functions for session management, authentication, and data orchestration.
VERSION: 3.4.0
"""

import os
import sys
import hmac
import pandas as pd
import streamlit as st
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Where market data can come from (see get_data_source)
DATA_SOURCES = ('s3', 'supabase')
# How long the shared copy of the market data is reused before it is downloaded again
DATA_CACHE_TTL = timedelta(hours=6)
# Name of the backtest Lambda; when set, backtests run in AWS Lambda (see lambda_backtests_enabled)
BACKTEST_LAMBDA_ENV = 'FACTOR_LAKE_BACKTEST_LAMBDA'

def initialize_environment() -> None:
    """
    Configures the system path and environment variables.
    
    Ensures the project root is accessible to the Python interpreter and 
    loads local environment variables if a .env file is present.
    """
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

# PASSWORD AUTHENTICATION (currently disabled for open access)
# To re-enable password protection:
# 1. Uncomment the function below
# 2. Uncomment the check_password call in streamlit_app.py
# 3. Ensure 'password' is set in .streamlit/secrets.toml or ADMIN_PASSWORD env var
#
# def check_password(secrets_map: Dict[str, Any]) -> bool:
#     """
#     Implements an HMAC-based authentication gatekeeper.
#     
#     Validates user credentials against configured secrets before 
#     allowing access to the application logic.
#     """
#     configured_password = secrets_map.get("password") or os.environ.get("ADMIN_PASSWORD")
#     if not configured_password:
#         st.error("Authentication secret is missing. Please check configuration.")
#         return False
#
#     if st.session_state.get("password_correct"):
#         return True
#
#     def validate_input():
#         if hmac.compare_digest(st.session_state.get("pwd_entry", ""), configured_password):
#             st.session_state["password_correct"] = True
#             del st.session_state["pwd_entry"]
#         else:
#             st.session_state["password_correct"] = False
#
#     st.text_input("Administrator Password", type="password", on_change=validate_input, key="pwd_entry")
#     if st.session_state.get("password_correct") == False:
#         st.error("Invalid credentials.")
#     return False
#
# Placeholder function (always returns True when password is disabled)
def check_password(secrets_map: Dict[str, Any]) -> bool:
    """Placeholder for disabled password check. Always returns True for open access."""
    return True

def initialize_session_state() -> None:
    """
    Initializes the global Streamlit session state object.
    
    Sets default values for data containers, factor selections, and 
    analysis results to ensure cross-component stability.
    """
    default_state = {
        'data_loaded': False,
        'raw_data': None,
        'rdata': None,
        'loaded_filters': None,
        'saved_runs': [],
        'next_run_id': 1,
        # Every analysis run or opened in this session (see add_to_history), oldest first
        'analysis_history': [],
    }
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_data_source() -> str:
    """
    Picks where the market data is loaded from.

    FACTOR_LAKE_DATA_SOURCE ("s3" or "supabase") wins when it is set. Otherwise
    S3 is used whenever AWS credentials are configured, and Supabase is the fallback.
    """
    source = os.environ.get('FACTOR_LAKE_DATA_SOURCE', '').strip().lower()
    if source:
        if source not in DATA_SOURCES:
            raise ValueError(f"FACTOR_LAKE_DATA_SOURCE must be one of {DATA_SOURCES}, not '{source}'.")
        return source
    return 's3' if os.environ.get('AWS_ACCESS_KEY_ID') else 'supabase'

def describe_data_source() -> str:
    """Human-readable name of the configured data source, for the sidebar."""
    labels = {'s3': 'AWS S3 (Parquet files)', 'supabase': 'Supabase (Cloud Relational Database)'}
    try:
        return labels[get_data_source()]
    except ValueError as e:
        return f"Misconfigured ({e})"

@st.cache_resource(ttl=DATA_CACHE_TTL, show_spinner=False)
def load_shared_universe(source: str) -> pd.DataFrame:
    """
    Downloads the full universe once and shares it with every user session.

    Loading per session made each visitor wait for (and hold) a separate copy.
    A failed download raises and is not cached, so the next click retries.
    The returned frame is shared between sessions: copy it before modifying it.
    """
    from src.data_standardization import validate_universe
    if source == 's3':
        from src.s3_client import S3DataManager
        df = S3DataManager().fetch_all_data()
    else:
        from src.supabase_client import SupabaseManager
        df = SupabaseManager().fetch_all_data()
    validate_universe(df, source)
    return df

def load_and_process_data(user_settings: Dict[str, Any]) -> None:
    """
    Orchestrates data fetching and universe filtering.
    
    Applies ESG restrictions, sector filters, and date constraints 
    to the raw dataset retrieved from the configured data source.

    raw_data is shared by every session (st.cache_resource), so it is never
    modified, and it is not copied whole: all filters form one row mask and only
    the kept rows are copied. When nothing is filtered out, rdata shares
    raw_data's columns, so treat rdata as read-only too.
    """
    from src.universe_selection import select_universe
    try:
        if st.session_state.raw_data is None:
            st.session_state.raw_data = load_shared_universe(get_data_source())

        # Year, sector and ESG fossil fuel filters (the backtest Lambda applies the same ones)
        df = select_universe(
            st.session_state.raw_data,
            int(user_settings['start_year']), int(user_settings['end_year']),
            selected_sectors=user_settings.get('selected_sectors'),
            restrict_fossil_fuels=bool(user_settings.get('restrict_fossil_fuels')),
        )

        st.session_state.rdata = df
        st.session_state.loaded_filters = _universe_filters(user_settings)
        st.session_state.data_loaded = True
        st.success(f"Universe Refined: {len(df):,} records prepared.")
    except Exception as e:
        st.error(f"Data Orchestration Error: {str(e)}")

def _universe_filters(user_settings: Dict[str, Any]) -> Dict[str, Any]:
    """The sidebar's universe filters, as shown in the Results tab captions."""
    return {
        'restrict_fossil_fuels': bool(user_settings.get('restrict_fossil_fuels')),
        'selected_sectors': list(user_settings.get('selected_sectors') or []),
    }

def get_backtest_lambda_name() -> str:
    """The backtest Lambda's name (FACTOR_LAKE_BACKTEST_LAMBDA), or '' when none is configured."""
    return os.environ.get(BACKTEST_LAMBDA_ENV, '').strip()

def lambda_backtests_enabled(user_settings: Dict[str, Any]) -> bool:
    """
    True when backtests run in AWS Lambda: a Lambda is configured and the sidebar's
    "Run backtests in AWS Lambda" switch is on. The app then never downloads the
    market data for a backtest; Lambda reads it from S3 instead.
    """
    return bool(get_backtest_lambda_name()) and bool(user_settings.get('use_lambda', True))

@st.cache_resource(show_spinner=False)
def get_lambda_client(function_name: str):
    """One Lambda client shared by every session."""
    from src.lambda_client import BacktestLambdaClient
    return BacktestLambdaClient(function_name)

def describe_compute(info: Optional[Dict[str, Any]]) -> str:
    """One line on where a result came from, e.g. "computed in AWS Lambda in 8.2 s"."""
    if not info:
        return "computed in this app"
    if info.get('opened'):
        text = "opened from a saved analysis"
        if info.get('computed_at'):
            text += f" (first computed {info['computed_at'].replace('T', ' ').rstrip('Z')} UTC)"
        return text
    if info.get('cached'):
        text = f"returned from the AWS Lambda result cache in {info['round_trip_seconds']:.1f} s"
        if info.get('computed_at'):
            text += f" (first computed {info['computed_at'].replace('T', ' ').rstrip('Z')} UTC)"
    else:
        text = (f"computed in AWS Lambda in {info['round_trip_seconds']:.1f} s "
                f"(backtest {info.get('compute_seconds') or 0:.1f} s)")
    if info.get('delivery') == 's3_link':
        text += ", downloaded from an S3 link"
    return text

def run_cohort(run: Dict[str, Any], cohort_pct: float):
    """
    Top vs bottom cohort portfolio values for a saved run, computed where the run
    was: in Lambda for Lambda runs (run['data'] is None), else locally on the run's data.

    Returns:
        (top values, bottom values, compute info or None for local runs)
    """
    from src.backtest_engine import run_cohort_comparison
    if run.get('data') is not None:
        top, bottom = run_cohort_comparison(
            data=run['data'],
            selected_factors=run['factors'],
            factor_directions=run['factor_directions'],
            cohort_pct=cohort_pct,
            user_settings=run['settings'],
        )
        return list(top), list(bottom), None

    from src.backtest_jobs import build_job
    function_name = run['compute'].get('function_name') or get_backtest_lambda_name()
    if not function_name:
        raise RuntimeError("Cohort comparisons for an opened analysis run in AWS Lambda, "
                           f"and no Lambda is configured ({BACKTEST_LAMBDA_ENV}).")
    job = build_job('cohort', run['factors'], run['factor_directions'], run['settings'],
                    factor_labels=dict(zip(run['factors'], run['factor_labels'])), cohort_pct=cohort_pct)
    result, info = get_lambda_client(function_name).run(job)
    return result['top'], result['bottom'], info

def run_backtest_logic(user_settings: Dict[str, Any], 
                       factor_names: List[str], 
                       factor_dirs: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Executes the portfolio backtesting engine and saves the run.

    Translates UI-friendly labels to internal database columns, runs the
    backtest, and saves a snapshot of the run (results plus the factors,
    directions, settings and data it used) as the newest entry in
    st.session_state.saved_runs.

    With Lambda backtests on (lambda_backtests_enabled), the inputs are sent to the
    backtest Lambda instead, which reads the data from S3; the run then keeps no data
    (run['data'] is None) and run['compute'] says how the Lambda produced the result.

    Returns:
        The saved run, or None if the run was rejected or failed (nothing is saved).
    """
    from src.backtest_engine import rebalance_portfolio
    from src.backtest_jobs import build_job
    from app.streamlit_config import FACTOR_METADATA, MAX_SAVED_RUNS
    from app.saved_runs import add_saved_run, build_run_label

    factor_labels = [f for f in factor_names if f in FACTOR_METADATA]
    if not factor_labels:
        st.warning("Select at least one factor before running the analysis.")
        return None
    if float(user_settings['initial_aum']) <= 0:
        st.warning("Initial AUM must be greater than $0.")
        return None

    try:
        # Mapping UI labels to internal SQL columns
        internal_factor_cols = [FACTOR_METADATA[f]['column'] for f in factor_labels]
        internal_directions = {
            FACTOR_METADATA[f]['column']: factor_dirs.get(f, 'top')
            for f in factor_labels
        }
        column_labels = {FACTOR_METADATA[f]['column']: f for f in factor_labels}

        if lambda_backtests_enabled(user_settings):
            # AWS Lambda reads the data from S3, filters it like Load Market Data and runs the same engine
            job = build_job('backtest', internal_factor_cols, internal_directions, user_settings,
                            factor_labels=column_labels)
            results, compute = get_lambda_client(get_backtest_lambda_name()).run(job)
            data, universe = None, _universe_filters(user_settings)
        else:
            data, compute = st.session_state.rdata, None
            universe = dict(st.session_state.get('loaded_filters') or {})
            # Portfolio simulation execution
            results = rebalance_portfolio(
                data,
                internal_factor_cols,
                factor_directions=internal_directions,
                start_year=int(user_settings['start_year']),
                end_year=int(user_settings['end_year']),
                initial_aum=user_settings['initial_aum'],
                benchmark_index=user_settings.get('benchmark_index', 1),
                top_pct=user_settings.get('top_pct', 10.0),
                use_market_cap_weight=user_settings.get('use_market_cap_weight', False),
                delisting_strategy=user_settings.get('delisting_strategy', 'zero_return'),
                factor_combination=user_settings.get('factor_combination', 'mixing'),
                factor_labels=column_labels
            )
    except Exception as e:
        st.error(f"Backtest Execution Error: {str(e)}")
        return None

    if not results.get('yearly_returns'):
        st.warning(
            "The backtest produced no yearly returns. "
            "Choose an End Year after the Start Year and check the loaded data covers that period."
        )
        return None

    # Save a snapshot so later sidebar/factor changes never alter this run
    run_id = st.session_state.next_run_id
    run = {
        'id': run_id,
        'label': build_run_label(run_id, factor_labels),
        'factor_labels': factor_labels,
        'factors': internal_factor_cols,
        'factor_directions': internal_directions,
        'settings': dict(user_settings),
        'universe': universe,
        'data': data,
        'results': results,
        'compute': compute,
        'cohort': None,
    }
    st.session_state.next_run_id = run_id + 1
    st.session_state.saved_runs = add_saved_run(
        st.session_state.saved_runs, run, MAX_SAVED_RUNS
    )
    add_to_history(run)
    return run

def add_to_history(run: Dict[str, Any]) -> None:
    """
    Records a run in this session's history (st.session_state.analysis_history). It
    lives only in the session, so it lists only this user's own runs, including ones
    no longer shown as Results tabs (only the last MAX_SAVED_RUNS are kept there).
    """
    from datetime import datetime
    compute = run.get('compute') or {}
    st.session_state.analysis_history = st.session_state.get('analysis_history', []) + [{
        'label': run['label'],
        'analysis_id': compute.get('hash'),
        'how': 'opened' if compute.get('opened') else 'computed in Lambda' if compute else 'computed in this app',
        'time': datetime.now().strftime('%H:%M:%S'),
    }]

def open_saved_analysis(text: str) -> Dict[str, Any]:
    """
    Opens a saved analysis by its ID as a new Results tab, with the same results,
    factors and settings as the original run, and adds it to the session history.

    Raises:
        SavedAnalysisError: with a message for the user (not an ID, expired, ...).
    """
    from src.saved_analyses import load_saved_analysis
    from app.streamlit_config import MAX_SAVED_RUNS
    from app.saved_runs import add_saved_run, build_run_label

    saved = load_saved_analysis(text, get_results_store())
    job = saved['job']
    factor_labels = [job['factor_labels'].get(f, f) for f in job['factors']]
    run_id = st.session_state.next_run_id
    run = {
        'id': run_id,
        'label': f"{build_run_label(run_id, factor_labels)} (opened)",
        'factor_labels': factor_labels,
        'factors': list(job['factors']),
        'factor_directions': dict(job['factor_directions']),
        'settings': {k: job[k] for k in (
            'start_year', 'end_year', 'initial_aum', 'benchmark_index', 'top_pct', 'use_market_cap_weight',
            'delisting_strategy', 'factor_combination', 'restrict_fossil_fuels', 'selected_sectors')},
        'universe': {'restrict_fossil_fuels': job['restrict_fossil_fuels'],
                     'selected_sectors': list(job['selected_sectors'])},
        'data': None,
        'results': saved['result'],
        'compute': {'hash': saved['id'], 'opened': True, 'cached': True,
                    'computed_at': saved['computed_at'], 'function_name': get_backtest_lambda_name()},
        'cohort': None,
    }
    st.session_state.next_run_id = run_id + 1
    st.session_state.saved_runs = add_saved_run(st.session_state.saved_runs, run, MAX_SAVED_RUNS)
    add_to_history(run)
    return run

@st.cache_resource(show_spinner=False)
def get_results_store():
    """The S3 location saved analyses are read from (the market data's bucket and prefix)."""
    from src.s3_client import S3DataManager
    return S3DataManager()
