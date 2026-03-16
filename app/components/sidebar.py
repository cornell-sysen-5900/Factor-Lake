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
    
    Returns a standardized dictionary of inputs for the data orchestration 
    and backtesting engines.
    """
    with st.sidebar:
        st.header("Configuration")
        
        # Data Source
        st.subheader("Data Source")
        st.caption("Data source: Supabase (cloud)")
        use_supabase = True
        excel_file = None
        uploaded_file = None

        st.divider()
        
        # Fossil Fuel Restriction
        st.subheader("ESG Filters")
        restrict_fossil_fuels = st.checkbox(
            "Restrict Fossil Fuel Companies",
            value=False,
            help="Exclude oil, gas, coal, and fossil energy companies"
        )
        
        st.divider()
        
        # Portfolio Weighting Method
        st.subheader("Portfolio Weighting")
        weighting_method = st.radio(
            "Select weighting method:",
            options=["Equal Weight", "Market Cap Weight"],
            index=0,
            help="Equal Weight: Each stock gets equal dollar investment. Market Cap Weight: Weight by market capitalization (similar to Russell 2000)"
        )
        use_market_cap_weight = (weighting_method == "Market Cap Weight")
        
        if use_market_cap_weight:
            st.info("Market Cap Weighting: Stocks will be weighted by their market capitalization, similar to the Russell 2000 index. This reduces turnover costs and aligns with market performance.")
        
        st.divider()
        
        # Sector Selection
        st.subheader("Sector Selection")
        sector_filter_enabled = st.checkbox("Enable Sector Filter", value=False)
        selected_sectors = []
        if sector_filter_enabled:
            selected_sectors = st.multiselect(
                "Select sectors to include:",
                options=sector_options,
                default=sector_options,
                help="Choose which sectors to include in the analysis"
            )
            
        st.divider()
        
        # Date Range
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
                key="start_year_input",
                help="Select the first year for analysis (2002-2023)"
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=2002,
                max_value=2024,
                value=2024,
                step=1,
                format="%d",
                key="end_year_input",
                help="Select the last year for analysis (2002-2023)"
            )

        # Ensure start_year <= end_year; auto-correct end year if needed
        try:
            if int(st.session_state.get('start_year_input', 2002)) > int(st.session_state.get('end_year_input', 2023)):
                st.warning('Start Year was greater than End Year — adjusting End Year to match Start Year')
                st.session_state['end_year_input'] = int(st.session_state['start_year_input'])
                end_year = start_year
        except Exception:
            pass
            
        st.divider()
        
        # Initial Investment
        st.subheader("Initial Investment")
        initial_aum = st.number_input(
            "Initial AUM ($)",
            min_value=0.0,
            max_value=1000000000.0,
            value=1000.0,
            step=100.0,
            format="%.0f",
            help="Starting portfolio value in dollars"
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