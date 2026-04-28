"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/about.py
PURPOSE: Informational UI component detailing factor definitions and methodology.
VERSION: 2.1.0
"""

import streamlit as st

def render_about_tab() -> None:
    """
    Renders the informational 'About' section of the application.
    
    This component provides the investment thesis for each factor, explains the 
    underlying return methodology (including survivorship bias handling), 
    and outlines the operational workflow for the backtesting engine.
    """
    st.header("About Factor-Lake Portfolio Analysis")

    st.markdown("""
    ### What this app does

    Factor-Lake helps you test stock-selection ideas by building portfolios from financial factors
    and comparing how they perform over time.

    ### Quick start for new users

    1. Choose a date range and universe.
    2. Select one or more factors.
    3. Pick top/bottom portfolio rules.
    4. Run the analysis and review returns, holdings, and charts.

    ### Factor definitions

    1.  **ROE (9/30 Data):** Profit generated per dollar of shareholder equity.
    2.  **ROA (9/30 Data):** Profit generated from total assets.
    3.  **12-Mo Momentum %:** Price strength over the last 12 months.
    4.  **6-Mo Momentum %:** Price strength over the last 6 months.
    5.  **1-Mo Momentum %:** Price strength over the last 1 month.
    6.  **Price to Book (9/30):** Price relative to accounting book value.
    7.  **Next FY Earnings/P:** Expected next-year earnings yield.
    8.  **1-Yr Price Vol %:** One-year price volatility (stability/risk measure).
    9.  **Accruals/Assets:** Accounting accruals scaled by assets (earnings quality signal).
    10. **ROA %:** Standardized return on assets measure.
    11. **1-Yr Asset Growth %:** Growth in total assets over one year.
    12. **1-Yr CapEX Growth %:** Growth in capital spending over one year.
    13. **Book/Price:** Book value relative to market price.

    ### How performance is measured

    Results use next-year total return (including dividends). If a stock delists, we can choose between three options:
    1. **hold cash** (0% return)
    2. **Invest in money market** (risk-free return)
    3. **Distribute to portfolio** (proportional to other holdings)



    ### Student User Guide

    For a step-by-step class workflow, see the
    [Factor Lake User Guide (Docs Site)](https://cornell-sysen-5900.github.io/Factor-Lake/FACTOR_LAKE_USER_GUIDE/).
    If your docs deployment is unavailable, use the
    [repository version](https://github.com/cornell-sysen-5900/Factor-Lake/blob/main/DOCS/FACTOR_LAKE_USER_GUIDE.md).
    

    ---
    **Version:** 2.1.0  
    **Last Updated:** April 2026  
    **Project Repository:** [Factor-Lake GitHub](https://github.com/cuddihyd-cornell/Factor-Lake/)
    """)
