"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/factor_selection.py
PURPOSE: UI component for multi-factor selection and signal direction (tilt) configuration.
VERSION: 2.3.0
"""

import streamlit as st
from typing import Dict, List, Tuple

def render_factor_selection() -> Tuple[List[str], Dict[str, str]]:
    """
    Renders the interactive interface for configuring equity factor exposures.

    This component allows users to select specific financial metrics and define 
    their directional impact (top-down or bottom-up ranking) on the model 
    via toggle switches.

    Returns:
        Tuple[List[str], Dict[str, str]]: A list of selected factor names and 
            a dictionary mapping those names to their ranking direction ('top' or 'bottom').
    """
    st.header("Factor Selection")
    st.write("Select one or more factors for your portfolio strategy:")

    def factor_category(factors_config: List[Tuple[str, str]]):
        """
        Manages the state of individual factor selections within a categorical group.
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
                        help="OFF = High to Low (Positive Tilt), ON = Low to High (Negative Tilt)"
                    )
                else:
                    is_bottom = False
            
            selections[name] = checked
            directions[name] = 'bottom' if is_bottom else 'top'
        
        return selections, directions

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

    # Aggregating all rendered categories
    all_factors = {**f1, **f2, **f3, **f4, **f5}
    all_dirs = {**d1, **d2, **d3, **d4, **d5}
    
    selected_names = [name for name, selected in all_factors.items() if selected]
    selected_dirs = {name: all_dirs[name] for name in selected_names}

    st.write("---")
    
    if selected_names:
        factor_labels = [
            f"{name} ({'High to Low' if selected_dirs[name] == 'top' else 'Low to High'})"
            for name in selected_names
        ]
        st.success(f"Selected {len(selected_names)} factor(s): {', '.join(factor_labels)}")
    else:
        st.warning("Please select at least one factor to run the analysis")
        
    st.write("---")

    return selected_names, selected_dirs