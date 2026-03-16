"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/factor_selection.py
PURPOSE: UI component for multi-factor selection and signal direction (tilt) configuration.
VERSION: 2.0.0
"""

import streamlit as st
from typing import Dict, List, Tuple, Any

def render_factor_selection() -> Tuple[List[str], Dict[str, str]]:
    """
    Renders an interactive interface for configuring equity factor exposures and signal tilts.

    This component allows users to construct a multi-factor strategy by selecting 
    specific financial metrics and defining their directional impact on the 
    ranking model. By toggling between 'High to Low' and 'Low to High' orientations, 
    the interface provides the necessary flexibility to implement diverse investment 
    styles, such as Value, Momentum, or Quality.

    Returns:
        Tuple[List[str], Dict[str, str]]: 
            - A list of display names for all factors currently selected by the user.
            - A dictionary mapping those names to their intended signal direction ('top' or 'bottom').
    """
    st.header("Factor Selection")
    st.write("Select one or more factors to define your portfolio strategy:")

    def factor_category(factors_config: List[Tuple[str, str]]):
        """
        Internal utility to generate a vertical alignment of factor checkboxes and direction toggles.

        This helper manages the state of individual factor selections within a 
        categorical group. It utilizes a columnar sub-layout to present the factor 
        label alongside a toggle switch that defines whether the engine should 
        prioritize the highest or lowest numerical values for that specific metric.
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

    # Layout: Two columns of factor categories to optimize screen real estate
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
        st.subheader("") # Aesthetic spacing to align with left column
        f5, d5 = factor_category([
            ('Accruals/Assets', 'accruals'), 
            ('1-Yr Price Vol %', 'vol')
        ])

    # Aggregating data from all rendered categories
    all_factors = {**f1, **f2, **f3, **f4, **f5}
    all_dirs = {**d1, **d2, **d3, **d4, **d5}
    
    # Isolate active selections for the backtesting engine
    selected_names = [name for name, selected in all_factors.items() if selected]
    selected_dirs = {name: all_dirs[name] for name in selected_names}

    return selected_names, selected_dirs