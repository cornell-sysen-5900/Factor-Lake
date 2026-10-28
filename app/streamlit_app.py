"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_app.py
PURPOSE: Main application entry point orchestrating UI components and analysis logic.
VERSION: 3.6.0
"""

import os

# Use the system allocator for Arrow. st.dataframe converts every table it shows
# to Arrow, and Arrow's default allocator (mimalloc) keeps that memory cached
# after the table is gone, so clearing large tables (e.g. in the Data Viewer)
# did not lower process memory. pyarrow reads this once, when it is first
# imported, so it must be set before anything imports pandas or pyarrow.
os.environ.setdefault('ARROW_DEFAULT_MEMORY_POOL', 'system')

import sys
from pathlib import Path

# Ensure top-level packages (e.g., `src`, `Visualizations`) are importable
# before component modules are imported by Streamlit runtime.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit_utils as utils
import streamlit_css as css
import streamlit_config as config

import components.sidebar as sidebar
from components.factor_selection import render_factor_selection, reset_factor_selection
from components.results_tab import render_analysis_history, render_open_saved_analysis, render_saved_runs
from components.data_viewer import render_data_viewer_tab, clear_viewer_data
from components.about import render_about_tab

# Global Page Configuration
st.set_page_config(
    page_title="Factor-Lake Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Initialization
utils.initialize_environment()
utils.initialize_session_state()
css.inject_styles()

MAIN_TABS = ["Analysis", "Results", "Data Viewer", "About"]
# The open main tab (a plain session value, updated on every run)
MAIN_TABS_KEY = "main_tabs"
# Widget key of the main tab bar, plus a version that changes each time the app opens a
# tab itself. The browser keeps the tab the user last clicked and sends it on every rerun;
# it does not follow a tab opened from Python. Under the same key, the first click after
# a run that opened Results would rerun on the last clicked tab (and be lost there).
# A new key is a new tab bar, with no clicked tab yet.
MAIN_TABS_WIDGET_KEY = "main_tabs_bar"
MAIN_TABS_VERSION_KEY = "main_tabs_bar_version"
# Set after a successful run: the confirmation to show once the Results tab opens
RUN_COMPLETE_KEY = "run_complete_message"

def _main_tabs_widget_key() -> str:
    """The current widget key of the main tab bar (see MAIN_TABS_WIDGET_KEY)."""
    return f"{MAIN_TABS_WIDGET_KEY}_{st.session_state.get(MAIN_TABS_VERSION_KEY, 0)}"


def _open_main_tab(label: str) -> None:
    """Opens a main tab from Python. Call before the tab bar is drawn."""
    st.session_state[MAIN_TABS_VERSION_KEY] = st.session_state.get(MAIN_TABS_VERSION_KEY, 0) + 1
    st.session_state[_main_tabs_widget_key()] = label


def _on_main_tab_change() -> None:
    """on_change callback: leaving the Data Viewer releases the table it loaded."""
    if st.session_state[_main_tabs_widget_key()] != "Data Viewer":
        clear_viewer_data()


def _render_legacy_run(user_settings, use_lambda, selected_factor_names, factor_directions) -> None:
    """Load Market Data and Run Portfolio Analysis for Legacy, exactly as before WRDS mode."""
    # Step 1: Data Acquisition (with Lambda backtests, only the Data Viewer needs it)
    if use_lambda:
        st.caption("Backtests run in AWS Lambda, which reads the market data from S3. "
                   "Load Market Data is only needed for the Data Viewer.")
    if st.button("Load Market Data", type="secondary" if use_lambda else "primary",
                 width="stretch", disabled=not selected_factor_names):
        with st.spinner("Accessing cloud database..."):
            utils.load_and_process_data(user_settings)

    # Step 2: Execution
    if use_lambda or st.session_state.get('data_loaded', False):
        if st.button("Run Portfolio Analysis", type="primary", width="stretch",
                     disabled=not selected_factor_names):
            spinner = ("Running the backtest in AWS Lambda..." if use_lambda
                       else "Executing backtest simulation...")
            with st.spinner(spinner):
                run = utils.run_backtest_logic(
                    user_settings,
                    selected_factor_names,
                    factor_directions
                )

            if run is not None:
                message = f"Analysis complete. Saved as \"{run['label']}\" in the Results tab."
                if run.get('compute'):
                    message += f" Result {utils.describe_compute(run['compute'])}."
                st.session_state[RUN_COMPLETE_KEY] = message
                st.rerun()


def _render_wrds_run(user_settings, selected_factor_names, factor_directions) -> None:
    """WRDS runs only in AWS Lambda, which reads the WRDS data from S3, so there is nothing to load."""
    st.caption("WRDS backtests run in AWS Lambda on the WRDS data in S3. "
               "No market data is loaded in the app.")
    if st.button("Run Portfolio Analysis", type="primary", width="stretch",
                 disabled=not selected_factor_names, key="run_wrds_analysis"):
        with st.spinner("Running the WRDS backtest in AWS Lambda..."):
            run = utils.run_wrds_backtest_logic(user_settings, selected_factor_names, factor_directions)
        if run is not None:
            st.session_state[RUN_COMPLETE_KEY] = (f"Analysis complete. Saved as \"{run['label']}\" in the Results tab. "
                                                  f"Result {utils.describe_compute(run['compute'])}.")
            st.rerun()


def main():
    """
    Orchestrates the primary application flow, including security authentication,
    parameter configuration, and multi-tab rendering.
    """
    # PASSWORD AUTHENTICATION (disabled for open access)
    # To re-enable: uncomment below and keep password in secrets/env
    # if not utils.check_password(st.secrets):
    #     st.stop()
        
    st.markdown(
        '<div class="main-header-container">' +
        '<div class="main-header">Factor-Lake Portfolio Analysis</div>' +
        '<a href="https://cornell-sysen-5900.github.io/Factor-Lake/" target="_blank" class="docs-icon-link" title="View Documentation">📚</a>' +
        '</div>',
        unsafe_allow_html=True
    )
    
    # Render Global Sidebar and capture configuration
    user_settings = sidebar.render_sidebar(config.SECTOR_OPTIONS, utils.describe_data_source(),
                                           utils.get_backtest_lambda_name(),
                                           on_data_mode_change=reset_factor_selection)
    use_lambda = utils.lambda_backtests_enabled(user_settings)
    # Legacy settings have no data_mode key
    data_mode = user_settings.get('data_mode', 'legacy')

    # A finished run opens the Results tab. The active tab can only be set
    # before st.tabs is drawn, so the run requests it and reruns (see below).
    if RUN_COMPLETE_KEY in st.session_state and st.session_state.get(MAIN_TABS_KEY) != "Results":
        _open_main_tab("Results")

    # Define primary application layout. Only the open tab is rendered.
    main_tabs = st.tabs(MAIN_TABS, key=_main_tabs_widget_key(), on_change=_on_main_tab_change)
    tab_analysis, tab_results, tab_viewer, tab_about = main_tabs
    st.session_state[MAIN_TABS_KEY] = next(label for tab, label in zip(main_tabs, MAIN_TABS) if tab.open)

    if tab_analysis.open:
        with tab_analysis:
            # Configuration phase: Capture factors and directions
            selected_factor_names, factor_directions = render_factor_selection(data_mode)

            if data_mode == 'wrds':
                _render_wrds_run(user_settings, selected_factor_names, factor_directions)
            else:
                _render_legacy_run(user_settings, use_lambda, selected_factor_names, factor_directions)

    if tab_results.open:
        with tab_results:
            # Performance Visualization phase
            if RUN_COMPLETE_KEY in st.session_state:
                st.success(st.session_state.pop(RUN_COMPLETE_KEY))
            # Open a saved analysis by its ID (it becomes the newest run tab), and this session's history
            render_open_saved_analysis(has_runs=bool(st.session_state.get('saved_runs')))
            render_analysis_history(st.session_state.get('analysis_history', []))
            if st.session_state.get('saved_runs'):
                render_saved_runs(st.session_state.saved_runs)
            else:
                st.info("Performance metrics will appear here after a successful analysis run.")

    if tab_viewer.open:
        with tab_viewer:
            # Raw data inspection and CSV export (reuses the data loaded in Analysis)
            render_data_viewer_tab()

    if tab_about.open:
        with tab_about:
            # Documentation phase
            render_about_tab()

if __name__ == "__main__":
    main()
