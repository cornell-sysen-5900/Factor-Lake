"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_app.py
PURPOSE: Main application entry point with cleaned analysis flow.
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

st.set_page_config(
    page_title="Factor-Lake Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

utils.initialize_environment()
utils.initialize_session_state()
css.inject_styles()

def main():
    if not utils.check_password(st.secrets):
        st.stop()
        
    st.markdown('<div class="main-header">Factor-Lake Portfolio Analysis</div>', unsafe_allow_html=True)
    
    user_settings = sidebar.render_sidebar(config.SECTOR_OPTIONS)
    tab_analysis, tab_results, tab_about = st.tabs(["Analysis", "Results", "About"])

    with tab_analysis:
        # Capture selections and store them in session_state immediately
        selected_factor_names, factor_directions = render_factor_selection()
        st.session_state['selected_factor_names'] = selected_factor_names
        st.session_state['factor_directions'] = factor_directions
        st.divider()

        if st.button("Load Market Data", type="primary", use_container_width=True):
            with st.spinner("Loading Data..."):
                utils.load_and_process_data(user_settings)

        if st.session_state.get('data_loaded', False):
            if st.button("Run Portfolio Analysis", type="primary", use_container_width=True):
                with st.spinner("Calculating..."):
                    utils.run_backtest_logic(
                        user_settings, 
                        selected_factor_names, 
                        factor_directions
                    )
                
                if st.session_state.get('results') is not None:
                    st.success("Analysis complete. Review performance in the Results tab.")
                    st.markdown("""
                        <script>
                        window.parent.document.querySelectorAll('button[data-baseweb="tab"]')[1].click();
                        </script>
                        """, unsafe_allow_html=True)

    with tab_results:
        if st.session_state.get('results') is not None:
            # The component now has access to stored factors via session_state
            render_results_tab(st.session_state.results, user_settings)
        else:
            st.info("No active results. Execute an analysis in the 'Analysis' tab.")
    
    with tab_about:
        render_about_tab()

if __name__ == "__main__":
    main()