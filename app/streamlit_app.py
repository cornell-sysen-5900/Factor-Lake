"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_app.py
PURPOSE: Main application entry point orchestrating UI components and analysis logic.
VERSION: 3.3.0
"""

import streamlit as st
import app.streamlit_utils as utils
import app.streamlit_css as css
import app.streamlit_config as config

import components.sidebar as sidebar
from components.factor_selection import render_factor_selection
from app.components.results_tab import render_results_tab
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

def main():
    """
    Orchestrates the primary application flow, including security authentication,
    parameter configuration, and multi-tab rendering.
    """
    if not utils.check_password(st.secrets):
        st.stop()
        
    st.markdown('<div class="main-header">Factor-Lake Portfolio Analysis</div>', unsafe_allow_html=True)
    
    # Render Global Sidebar and capture configuration
    user_settings = sidebar.render_sidebar(config.SECTOR_OPTIONS)
    
    # Define primary application layout
    tab_analysis, tab_results, tab_about = st.tabs(["Analysis", "Results", "About"])

    with tab_analysis:
        # Configuration phase: Capture factors and directions
        selected_factor_names, factor_directions = render_factor_selection()
        st.session_state['selected_factor_names'] = selected_factor_names
        st.session_state['factor_directions'] = factor_directions
        
        # Step 1: Data Acquisition
        if st.button("Load Market Data", type="primary", use_container_width=True):
            with st.spinner("Accessing cloud database..."):
                utils.load_and_process_data(user_settings)

        # Step 2: Execution
        if st.session_state.get('data_loaded', False):
            if st.button("Run Portfolio Analysis", type="primary", use_container_width=True):
                with st.spinner("Executing backtest simulation..."):
                    utils.run_backtest_logic(
                        user_settings, 
                        selected_factor_names, 
                        factor_directions
                    )
                
                if st.session_state.get('results') is not None:
                    st.success("Analysis complete. Review performance in the Results tab.")
                    # Trigger automatic tab switch to Results via JavaScript injection
                    st.markdown("""
                        <script>
                        window.parent.document.querySelectorAll('button[data-baseweb="tab"]')[1].click();
                        </script>
                        """, unsafe_allow_html=True)

    with tab_results:
        # Performance Visualization phase
        if st.session_state.get('results') is not None:
            render_results_tab(st.session_state.results, user_settings)
        else:
            st.info("Performance metrics will appear here after a successful analysis run.")
    
    with tab_about:
        # Documentation phase
        render_about_tab()

if __name__ == "__main__":
    main()