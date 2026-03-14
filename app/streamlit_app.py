"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app.py
PURPOSE: Main application entry point and tab orchestration.
VERSION: 1.1.0
"""

import streamlit as st
import streamlit_utils as utils
import streamlit_css as css
import streamlit_config as config

# Component Imports
import components.sidebar as sidebar
from components.factor_selection import render_factor_selection
from components.results_view import render_results_tab
from components.about import render_about_tab

# --- INITIALIZATION ---

# Mandatory: Must be the first Streamlit command executed
st.set_page_config(
    page_title="Factor-Lake Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize system paths, environment variables, and session state persistence
utils.initialize_environment()
utils.initialize_session_state()

# Inject custom professional styling
css.inject_styles()

# Global Constants
SECTOR_OPTIONS = config.SECTOR_OPTIONS

def main():
    """
    Primary execution loop for the Factor-Lake application.
    
    This function manages the high-level application lifecycle, including 
    security authentication, global configuration via the sidebar, and 
    the modular rendering of the Analysis, Results, and About tabs.
    
    Output:
        None: Manages the Streamlit frontend state and visual output.
    """
    
    # 1. Authentication Gate
    if not utils.check_password(st.secrets):
        st.stop()
        
    # 2. Page Header
    st.markdown('<div class="main-header">Factor-Lake Portfolio Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Factor-Lake Portfolio Analysis tool. This platform provides an interactive 
    environment for quantitative factor backtesting. Use the interface below to:
    - Define strategy parameters and factor tilts.
    - Apply ESG exclusionary filters and sector-specific constraints.
    - Analyze historical performance relative to the Russell 2000 benchmark.
    """)
    
    # 3. Sidebar Configuration
    user_settings = sidebar.render_sidebar(SECTOR_OPTIONS)

    # 4. Tabbed Interface Navigation
    tab1, tab2, tab3 = st.tabs(["Analysis", "Results", "About"])

    with tab1:
        # Render Factor Selection UI
        selected_factor_names, factor_directions = render_factor_selection()
        
        if selected_factor_names:
            st.success(f"Active Factor Selection: {len(selected_factor_names)} factor(s)")
        else:
            st.warning("Strategy Configuration: Please select at least one factor to proceed.")

        st.divider()

        # Data Acquisition Block
        if st.button("Load Market Data", type="primary"):
            utils.load_and_process_data(user_settings)

        # Backtest Execution Block
        if st.session_state.data_loaded:
            if st.button("Run Portfolio Analysis", type="primary"):
                utils.run_backtest_logic(user_settings, selected_factor_names, factor_directions)
    
    with tab2:
        # Performance Visualization and Statistical Analysis
        if st.session_state.get('results') is not None:
            render_results_tab(st.session_state.results, user_settings)
        else:
            st.info("Performance data is currently unavailable. Please execute an analysis in the 'Analysis' tab.")
    
    with tab3:
        # Investment Thesis and Methodology Documentation
        render_about_tab()

if __name__ == "__main__":
    main()