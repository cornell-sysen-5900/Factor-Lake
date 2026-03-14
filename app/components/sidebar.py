"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/sidebar.py
PURPOSE: Configuration interface for global backtest parameters and ESG filters.
VERSION: 1.1.0
"""

import streamlit as st
from typing import Dict, Any, List

def render_sidebar(sector_options: List[str]) -> Dict[str, Any]:
    """
    Renders the configuration sidebar and aggregates all user input parameters.
    
    This interface manages global settings including data source metadata, 
    ESG exclusionary filters, portfolio weighting methodologies, and 
    chronological constraints for the backtesting engine.

    Input:
        sector_options (List[str]): A list of unique sector classifications 
                                    extracted from the configuration.
    Output:
        Dict[str, Any]: A validated dictionary containing all user-defined 
                        configurations for the current session.
    """
    with st.sidebar:
        st.header("Configuration")
        
        # 1. Data Source Metadata
        st.subheader("Data Source")
        st.caption("Data source: Supabase (cloud-hosted)")
        
        st.divider()

        # 2. ESG Restrictions
        st.subheader("ESG Filters")
        restrict_fossil_fuels = st.checkbox(
            "Restrict Fossil Fuel Companies",
            value=False,
            help="Exclude oil, gas, coal, and fossil energy companies based on sector codes."
        )
        
        st.divider()

        # 3. Portfolio Weighting Methodology
        st.subheader("Portfolio Weighting")
        weighting_method = st.radio(
            "Select weighting method:",
            options=["Equal Weight", "Market Cap Weight"],
            index=0,
            help="Equal Weight: Uniform dollar distribution. Market Cap Weight: Proportional to market capitalization."
        )
        use_market_cap_weight = (weighting_method == "Market Cap Weight")
        
        if use_market_cap_weight:
            st.info("📊 Market Cap Weighting: Allocations will be relative to firm size, approximating a Russell 2000 distribution.")
        
        st.divider()

        # 4. Sector Constraints
        st.subheader("Sector Selection")
        sector_filter_enabled = st.checkbox("Enable Sector Filter", value=False)
        selected_sectors = []
        if sector_filter_enabled:
            selected_sectors = st.multiselect(
                "Select sectors to include:",
                options=sector_options,
                default=sector_options,
                help="Only tickers belonging to the selected sectors will enter the factor universe."
            )
            
        st.divider()

        # 5. Chronological Slicing
        st.subheader("Analysis Period")
        col1, col2 = st.columns(2)
        
        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=2002,
                max_value=2024,
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

        # Validation logic to ensure chronological integrity
        if start_year > end_year:
            st.warning('Start Year precedes End Year. Adjusting range...')
            end_year = start_year

        st.divider()

        # 6. Capital Allocation
        st.subheader("Initial Investment")
        initial_aum = st.number_input(
            "Initial AUM ($)",
            min_value=0.0,
            max_value=1000000000.0,
            value=1000000.0,
            step=100000.0,
            format="%.0f"
        )
        
        st.divider()

        # 7. Engine Verbosity
        st.subheader("Output Detail")
        verbosity_level = st.select_slider(
            "Verbosity Level",
            options=[0, 1, 2, 3],
            value=1,
            format_func=lambda x: ["Silent", "Basic", "Detailed", "Debug"][x]
        )
        show_loading = st.checkbox("Show data loading progress", value=True)

    # Compile the validated settings into a unified configuration object
    return {
        "restrict_fossil_fuels": restrict_fossil_fuels,
        "use_market_cap_weight": use_market_cap_weight,
        "selected_sectors": selected_sectors,
        "sector_filter_enabled": sector_filter_enabled,
        "start_year": start_year,
        "end_year": end_year,
        "initial_aum": initial_aum,
        "verbosity_level": verbosity_level,
        "show_loading": show_loading
    }