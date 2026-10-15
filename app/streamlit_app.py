"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_app.py
PURPOSE: Main application entry point orchestrating UI components and analysis logic.
VERSION: 3.5.0
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
from components.factor_selection import render_factor_selection
from components.results_tab import render_saved_runs
from components.data_viewer import render_data_viewer, clear_data_viewer_slice
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
MAIN_TABS_KEY = "main_tabs"
# Set after a successful run: the confirmation to show once the Results tab opens
RUN_COMPLETE_KEY = "run_complete_message"


def _on_main_tab_change() -> None:
    """on_change callback: leaving the Data Viewer releases the rows it loaded."""
    if st.session_state[MAIN_TABS_KEY] != "Data Viewer":
        clear_data_viewer_slice()


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
    user_settings = sidebar.render_sidebar(config.SECTOR_OPTIONS, utils.describe_data_source())
    
    # A finished run opens the Results tab. The active tab can only be set
    # before st.tabs is drawn, so the run requests it and reruns (see below).
    if RUN_COMPLETE_KEY in st.session_state and st.session_state.get(MAIN_TABS_KEY) != "Results":
        st.session_state[MAIN_TABS_KEY] = "Results"

    # Define primary application layout. Only the open tab is rendered.
    tab_analysis, tab_results, tab_data_viewer, tab_about = st.tabs(
        MAIN_TABS, key=MAIN_TABS_KEY, on_change=_on_main_tab_change
    )

    if tab_analysis.open:
        with tab_analysis:
            # Configuration phase: Capture factors and directions
            selected_factor_names, factor_directions = render_factor_selection()
        
            # Step 1: Data Acquisition
            if st.button("Load Market Data", type="primary", use_container_width=True,
                         disabled=not selected_factor_names):
                with st.spinner("Accessing cloud database..."):
                    utils.load_and_process_data(user_settings)

            # Step 2: Execution
            if st.session_state.get('data_loaded', False):
                if st.button("Run Portfolio Analysis", type="primary", use_container_width=True):
                    with st.spinner("Executing backtest simulation..."):
                        run = utils.run_backtest_logic(
                            user_settings, 
                            selected_factor_names, 
                            factor_directions
                        )
                
                    if run is not None:
                        st.session_state[RUN_COMPLETE_KEY] = (
                            f"Analysis complete. Saved as \"{run['label']}\" in the Results tab."
                        )
                        st.rerun()

    if tab_results.open:
        with tab_results:
            # Performance Visualization phase
            if RUN_COMPLETE_KEY in st.session_state:
                st.success(st.session_state.pop(RUN_COMPLETE_KEY))
            if st.session_state.get('saved_runs'):
                render_saved_runs(st.session_state.saved_runs)
            else:
                st.info("Performance metrics will appear here after a successful analysis run.")
    
    if tab_data_viewer.open:
        with tab_data_viewer:
            # Raw data inspection: reads the already-loaded market data only
            render_data_viewer()

    if tab_about.open:
        with tab_about:
            # Documentation phase
            render_about_tab()

if __name__ == "__main__":
    main()
