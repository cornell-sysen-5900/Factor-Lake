"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/about.py
PURPOSE: Informational UI component detailing factor definitions and methodology.
VERSION: 1.1.0
"""

import streamlit as st

def render_about_tab() -> None:
    """
    Renders the informational 'About' section of the application.
    
    This component provides the investment thesis for each factor, explains the 
    underlying return methodology (including survivorship bias handling), 
    and outlines the operational workflow for the backtesting engine.
    
    Output:
        None: Renders formatted Markdown content directly to the UI.
    """
    st.header("About Factor-Lake Portfolio Analysis")
    
    st.markdown("""
    ### What is Factor Investing?
    
    Factor investing is an empirical strategy that targets specific drivers of risk and return 
    across asset classes. By isolating these characteristics—such as value, momentum, or 
    quality—investors can construct portfolios designed to outperform traditional 
    market-capitalization-weighted indices over long horizons.

    ### Available Factor Categories
    
    * **Momentum:** Targets securities that have demonstrated strong recent performance, 
        betting on the persistence of existing trends.
    * **Value:** Identifies securities trading at a discount relative to their 
        intrinsic fundamental value (e.g., Book Value or Earnings).
    * **Profitability & Quality:** Focuses on firms with superior capital efficiency, 
        stable earnings, and robust operational margins.
    * **Growth:** Isolates companies aggressively expanding their asset bases or 
        capital expenditures to capture future market share.

    ### Factor Details & Investment Thesis
    
    1.  **ROE (9/30 Data):** *Higher is better.* Firms generating superior profit from 
        shareholder equity typically demonstrate efficient capital allocation.
    2.  **ROA (9/30 Data):** *Higher is better.* Measures how effectively a company 
        utilizes its total asset base to generate earnings.
    3.  **12-Mo Momentum %:** *Higher is better.* Captures long-term trend continuation 
        driven by investor behavior.
    4.  **6-Mo Momentum %:** *Higher is better.* Targets intermediate-term strength in 
        price action.
    5.  **1-Mo Momentum %:** *Higher is better.* Captures short-term tactical strength.
    6.  **Price to Book (9/30):** *Lower is better.* A traditional value metric; 
        we invert this to target high Book-to-Price ratios.
    7.  **Next FY Earnings/P:** *Higher is better.* An earnings yield metric that 
        indicates valuation attractiveness relative to future profits.
    8.  **1-Yr Price Vol %:** *Lower is better.* Lower volatility often indicates 
        more stable, "quality" firms.
    9.  **Accruals/Assets:** *Lower is better.* High accruals can signal poor 
        earnings quality or aggressive accounting.
    10. **ROA %:** *Higher is better.* A standardized measure of operating performance.
    11. **1-Yr Asset Growth %:** *Higher is better.* Indicates firm expansion and 
        reinvestment in the business.
    12. **1-Yr CapEX Growth %:** *Higher is better.* Suggests management's 
        confidence in future demand through capital investment.
    13. **Book/Price:** *Higher is better.* Directly aligns with the value thesis 
        by identifying "cheap" stocks relative to equity.

    ### Return Methodology
    
    Portfolio performance is derived from the **Next-Year's Return %** metric. This captures 
    total return, including dividends, over a one-year investment window. To mitigate 
    **survivorship bias**, tickers with missing return data (often due to delistings or 
    bankruptcies) are assigned a 0% return. This ensures the backtest reflects the 
    real-world impact of holding securities that exit the sample.

    ### Data Governance & Sources
    
    * **Supabase:** Primary cloud-hosted relational database for historical data.
    * **Excel Integration:** Support for local data ingestion and manual overrides.

    ---
    **Version:** 1.1.0  
    **Last Updated:** February 2026  
    **Project Repository:** [Factor-Lake GitHub](https://github.com/cuddihyd-cornell/Factor-Lake/tree/revamped_ux)
    """)