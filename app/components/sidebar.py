"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/sidebar.py
PURPOSE: Configuration interface for global backtest parameters and ESG filters.
VERSION: 2.4.0
"""

import streamlit as st
from typing import Dict, Any, List

def render_sidebar(sector_options: List[str]) -> Dict[str, Any]:
    """
    Constructs the application sidebar to capture user configuration settings.
    
    This interface manages global constraints, including date ranges, initial 
    capitalization, sector exposures, and weighting methodologies.
    
    Returns:
        Dict[str, Any]: Standardized dictionary of user-defined parameters.
    """
    with st.sidebar:
        st.header("Configuration")
        
        # Data Source Information
        st.subheader("Data Source")
        st.caption("Primary Source: Supabase (Cloud Relational Database)")

        st.divider()
        
        # ESG & Ethical Filtering
        st.subheader("ESG Filters")
        restrict_fossil_fuels = st.checkbox(
            "Restrict Fossil Fuel Companies",
            value=False,
            help="Exclude companies involved in oil, gas, coal, and fossil energy production."
        )
        
        st.divider()
        
        # Allocation Methodology
        st.subheader("Portfolio Weighting")
        weighting_method = st.radio(
            "Select weighting method:",
            options=["Equal Weight", "Market Cap Weight"],
            index=0,
            help="Equal Weight: Uniform dollar distribution. Market Cap Weight: Proportional to market capitalization."
        )
        use_market_cap_weight = (weighting_method == "Market Cap Weight")
        
        if use_market_cap_weight:
            st.info("Portfolio will be weighted by market capitalization, aligning with the Russell 2000 methodology.")
        
        st.divider()
        
        # Sector Exposure Configuration
        st.subheader("Sector Selection")
        sector_filter_enabled = st.checkbox("Enable Sector Filter", value=False)
        selected_sectors = []
        if sector_filter_enabled:
            selected_sectors = st.multiselect(
                "Include following sectors:",
                options=sector_options,
                default=sector_options,
                help="Only tickers within these selected sectors will be eligible for portfolio inclusion."
            )
            
        st.divider()
        
        # Analysis Temporal Range
        st.subheader("Analysis Period")
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=2002,
                max_value=2023,
                value=2002,
                step=1,
                format="%d",
                key="start_year_input"
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=2002,
                max_value=2024,
                value=2024,
                step=1,
                format="%d",
                key="end_year_input"
            )

        # Logic to ensure chronological consistency
        if start_year > end_year:
            st.warning('Start Year exceeded End Year. Synchronizing dates...')
            end_year = start_year
            
        st.divider()
        
        # Capitalization
        st.subheader("Initial Investment")
        initial_aum = st.number_input(
            "Initial AUM ($)",
            min_value=0.0,
            max_value=1000000000.0,
            value=1000.0,
            step=100.0,
            format="%.0f",
            help="The starting dollar value of the portfolio at the beginning of the backtest."
        )

    return {
        "restrict_fossil_fuels": restrict_fossil_fuels,
        "use_market_cap_weight": use_market_cap_weight,
        "selected_sectors": selected_sectors if sector_filter_enabled else None,
        "sector_filter_enabled": sector_filter_enabled,
        "start_year": start_year,
        "end_year": end_year,
        "initial_aum": initial_aum
    }