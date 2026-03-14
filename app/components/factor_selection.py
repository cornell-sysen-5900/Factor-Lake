"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/factor_selection.py
PURPOSE: UI component for multi-factor selection and signal direction (tilt) configuration.
VERSION: 1.1.0
"""

import streamlit as st
from typing import Dict, List, Tuple, Any

def render_factor_selection() -> Tuple[List[str], Dict[str, str]]:
    """
    Renders an interactive selection interface for various equity factors.
    
    Users can select multiple factors and specify the "tilt" direction 
    (e.g., High to Low or Low to High). This configuration is critical for 
    strategies like Value (Low Price-to-Book) or Momentum (High Return).

    Returns:
        Tuple[List[str], Dict[str, str]]: 
            - A list of the display names for all selected factors.
            - A dictionary mapping factor names to their chosen direction ('top' or 'bottom').
    """
    st.header("Factor Selection")
    st.write("Select one or more factors to define your portfolio strategy:")

    def factor_category(factors_config: List[Tuple[str, str]]):
        """
        Internal helper to render a vertical list of factor checkboxes with 
        associated direction toggles.
        """
        selections = {}
        directions = {}
        for name, key in factors_config:
            fcol1, fcol2 = st.columns([3, 2])
            with fcol1:
                checked = st.checkbox(name, key=key)
            with fcol2:
                if checked:
                    is_bottom = st.toggle(
                        "Low to High", 
                        key=f"{key}_dir", 
                        value=False,
                        help="OFF = High to Low (positive tilt), ON = Low to High (negative tilt)"
                    )
                else:
                    is_bottom = False
            
            selections[name] = checked
            directions[name] = 'bottom' if is_bottom else 'top'
        return selections, directions

    # Layout: Two columns of factor categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Momentum Factors")
        f1, d1 = factor_category([
            ('12-Mo Momentum %', '12m'), 
            ('6-Mo Momentum %', '6m'), 
            ('1-Mo Momentum %', '1m')
        ])
        st.subheader("Profitability Factors")
        f2, d2 = factor_category([
            ('ROE using 9/30 Data', 'roe'), 
            ('ROA using 9/30 Data', 'roa'), 
            ('ROA %', 'roa_pct')
        ])
        st.subheader("Growth Factors")
        f3, d3 = factor_category([
            ('1-Yr Asset Growth %', 'asset_growth'), 
            ('1-Yr CapEX Growth %', 'capex_growth')
        ])
    
    with col2:
        st.subheader("Value Factors")
        f4, d4 = factor_category([
            ('Price to Book Using 9/30 Data', 'ptb'), 
            ('Book/Price', 'btp'), 
            ('Next FY Earns/P', 'fey')
        ])
        st.subheader("Quality Factors")
        f5, d5 = factor_category([
            ('Accruals/Assets', 'accruals'), 
            ('1-Yr Price Vol %', 'vol')
        ])

    # Combine results from all categories
    all_factors = {**f1, **f2, **f3, **f4, **f5}
    all_dirs = {**d1, **d2, **d3, **d4, **d5}
    
    # Filter only for factors that are actively checked
    selected_names = [name for name, selected in all_factors.items() if selected]
    selected_dirs = {name: all_dirs[name] for name in selected_names}

    return selected_names, selected_dirs