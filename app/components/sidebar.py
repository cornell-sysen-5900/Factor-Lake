"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/sidebar.py
PURPOSE: Configuration interface for global backtest parameters and ESG filters.
VERSION: 2.0.0
"""

import streamlit as st
from typing import Dict, Any, List

def render_sidebar(sector_options: List[str]) -> Dict[str, Any]:
    """
    Constructs the application sidebar to capture and validate user configuration settings.

    This function builds the interactive control panel that governs the behavior of 
    the backtesting engine. It facilitates the selection of ESG constraints, portfolio 
    weighting schemes, and temporal ranges, consolidating these inputs into a 
    standardized dictionary for downstream data orchestration.

    Args:
        sector_options (List[str]): A list of unique sector classifications 
            available for filtering within the current dataset.

    Returns:
        Dict[str, Any]: A collection of validated user inputs required to 
            initialize the portfolio analysis.
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
                max_value=2026,
                value=2002,
                step=1,
                format="%d",
                key="start_year_input"
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=2002,
                max_value=2026,
                value=2024,
                step=1,
                format="%d",
                key="end_year_input"
            )

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

    return {
        "restrict_fossil_fuels": restrict_fossil_fuels,
        "use_market_cap_weight": use_market_cap_weight,
        "selected_sectors": selected_sectors,
        "sector_filter_enabled": sector_filter_enabled,
        "start_year": start_year,
        "end_year": end_year,
        "initial_aum": initial_aum
    }