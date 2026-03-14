"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/styles/css.py
PURPOSE: Global CSS injection to customize the Streamlit UI, branding, and responsiveness.
VERSION: 1.1.0
"""

import streamlit as st

# Define styles as a constant to avoid re-creating the string on every rerun.
# This approach optimizes memory usage during high-frequency session refreshes.
CSS_STYLE = """
<style>
    /* Main header styling with professional typography and drop shadow */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Styled alert boxes for consistency across success/error messages */
    .stAlert {
        margin-top: 1rem;
        border-radius: 10px;
    }
    
    /* Interactive button states with subtle hover elevation */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar configuration panel: Grey background with Cyan accent border */
    [data-testid="stSidebar"] {
        background-color: #DEDFE0;
        border-right: 4px solid #3BD2E3;
        padding-left: 8px;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1f77b4;
    }
    
    /* Standardized metric value formatting for financial KPIs */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Custom container styling for detailed cards and information blocks */
    .custom-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: #31333F;
    }
    
    /* Clean UI: Hiding Streamlit's default branding and footer elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive breakpoints for mobile and tablet compatibility */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
    }
</style>
"""

def inject_styles() -> None:
    """
    Renders custom CSS definitions into the Streamlit application's frontend.
    
    This function utilizes unsafe_allow_html to inject a raw <style> block, 
    allowing for deep customization of the DOM elements that Streamlit 
    does not expose through its standard API.
    
    Output:
        None: Renders the CSS payload directly to the user's browser.
    """
    st.markdown(CSS_STYLE, unsafe_allow_html=True)