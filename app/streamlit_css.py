"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_css.py
PURPOSE: Global CSS injection to customize the Streamlit UI, branding, and responsiveness.
VERSION: 1.1.0
"""

import streamlit as st

# Global CSS definitions for a unified professional interface.
CSS_STYLE = """
<style>
    /* Header container with icon */
    .main-header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        margin-bottom: 2rem;
    }
    
    /* Typography and header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Documentation icon link */
    .docs-icon-link {
        font-size: 2rem;
        margin-left: 1.5rem;
        text-decoration: none;
        cursor: pointer;
        transition: transform 0.3s ease, opacity 0.3s ease;
        opacity: 0.7;
    }
    
    .docs-icon-link:hover {
        transform: scale(1.2) rotate(5deg);
        opacity: 1;
    }
    
    /* Global component styling */
    .stAlert {
        margin-top: 1rem;
        border-radius: 10px;
    }
    
    /* Interactive element transitions */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Navigation panel aesthetic (Grey with Cyan border) */
    [data-testid="stSidebar"] {
        background-color: #DEDFE0;
        border-right: 4px solid #3BD2E3;
        padding-left: 8px;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1f77b4;
    }
    
    /* Financial KPI display formatting */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Information container cards */
    .custom-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: #31333F;
    }
    
    /* Branding: Hiding standard application headers and footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive breakpoints */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        .main-header-container {
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .docs-icon-link {
            margin-left: 0;
            font-size: 1.5rem;
        }
    }
</style>
"""

def inject_styles() -> None:
    """
    Renders custom CSS definitions into the application's frontend.
    
    This function injects raw CSS to customize UI elements and ensure 
    a consistent visual experience across different screen sizes.
    """
    st.markdown(CSS_STYLE, unsafe_allow_html=True)