"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_app.py
PURPOSE: Main application entry point and tab orchestration.
VERSION: 2.2.0
"""

import streamlit as st
import app.streamlit_utils as utils
import app.streamlit_css as css
import app.streamlit_config as config

# Component Imports
import components.sidebar as sidebar
from components.factor_selection import render_factor_selection
from components.results_view import render_results_tab
from components.about import render_about_tab

# --- INITIALIZATION ---

# Mandatory configuration for the Streamlit environment
st.set_page_config(
    page_title="Factor-Lake Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize system paths, session state, and environment variables
# This ensures Supabase keys are loaded before any buttons are pressed
utils.initialize_environment()
utils.initialize_session_state()

# Inject professional CSS styling
css.inject_styles()

# Global configuration constants
SECTOR_OPTIONS = config.SECTOR_OPTIONS

def main():
    """
    Primary execution loop for the Factor-Lake application.
    """
    
    # 1. Security Authentication Gate
    if not utils.check_password(st.secrets):
        st.stop()
        
    # 2. Application Header
    st.markdown('<div class="main-header">Factor-Lake Portfolio Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This platform provides a high-performance environment for quantitative factor 
    backtesting. Configure your strategy parameters below to analyze historical 
    performance relative to Russell 2000 benchmarks.
    """)
    
    # 3. Sidebar Configuration (Universe & Constraints)
    # Returns a dictionary of settings: sectors, fossil fuel toggle, dates, AUM, etc.
    user_settings = sidebar.render_sidebar(SECTOR_OPTIONS)

    # 4. Modular Tabbed Navigation
    tab_analysis, tab_results, tab_about = st.tabs(["Analysis", "Results", "About"])

    with tab_analysis:
        # 4.1. Factor Selection and Tilt Definition
        # Returns: (List of factor labels, Dict of labels to 'top'/'bottom')
        selected_factor_names, factor_directions = render_factor_selection()
        
        if selected_factor_names:
            st.success(f"Active Factor Selection: {len(selected_factor_names)} factor(s)")
        else:
            st.warning("Configuration Required: Select at least one factor to proceed.")

        st.divider()

        # 4.2. Data Acquisition Loop
        # Uses the streamlined loader to fetch standardized data from Supabase
        # Modified to ensure user settings are passed correctly for sector filtering
        if st.button("Load Market Data", type="primary", use_container_width=True):
            with st.spinner("Executing vectorized data ingestion from Cloud..."):
                utils.load_and_process_data(user_settings)

        # 4.3. Backtest Execution Loop
        # Only enabled once 'data_loaded' is True in session state
        if st.session_state.get('data_loaded', False):
            if st.button("Run Portfolio Analysis", type="primary", use_container_width=True):
                with st.spinner("Calculating annual rebalancing and risk metrics..."):
                    utils.run_backtest_logic(
                        user_settings, 
                        selected_factor_names, 
                        factor_directions
                    )
                    
                    if st.session_state.get('results'):
                        st.balloons() # Optional: Visual cue for successful backtest
                        st.success("Analysis complete. Review performance in the Results tab.")
    
    with tab_results:
        # 4.4. Performance and Risk Attribution
        if st.session_state.get('results') is not None:
            render_results_tab(st.session_state.results, user_settings)
        else:
            st.info("No active results. Execute an analysis in the 'Analysis' tab to view metrics.")
    
    with tab_about:
        # 4.5. Methodology and Documentation
        render_about_tab()

if __name__ == "__main__":
    main()