"""Explain the current WRDS workflow and access to historical saved analyses."""

import streamlit as st


def render_about_tab() -> None:
    st.header("About Factor-Lake Portfolio Analysis")
    st.markdown("""
    ### What this app does

    Factor-Lake tests stock-selection ideas using WRDS financial factors. New analyses run
    in AWS Lambda; the app receives results without loading the full market dataset.

    ### Quick start

    1. Choose years, a rebalancing frequency and a stock universe.
    2. Select factors and review their definitions and ranking directions in the factor picker.
       Colored cards explain their relationship to historical Legacy factors.
    3. Set portfolio size, weighting, trading costs and any liquidity cap. At 100% holdings,
       a factor is optional.
    4. Run the analysis, then review returns, holdings and charts in Results.
    5. Copy the saved analysis's share link or add it to Compare.

    ### Data preview

    After Prof. Steward approves sample-row display, Data Viewer can show up to five WRDS
    rows per year, filtered by sector, industry group, industry or sub-industry. The small
    preview comes from Lambda and has no download control. It is for inspecting fields,
    not a representative sample for research.

    ### How performance is measured

    WRDS analyses use monthly total returns and the chosen rebalancing schedule. CRSP
    delisting returns are included, followed by the risk-free return until the next rebalance.
    Results retain the settings and methodology used when the analysis was computed.

    ### Historical Legacy analyses

    Existing Legacy analysis IDs and share links still open saved results in Results and
    Compare. New Legacy runs are disabled. Historical settings can be exported for reference;
    starting a new strategy requires WRDS factors and settings.

    ### Student User Guide

    See the [Factor Lake User Guide](https://cornell-sysen-5900.github.io/Factor-Lake/FACTOR_LAKE_USER_GUIDE/)
    or its [repository version](https://github.com/cornell-sysen-5900/Factor-Lake/blob/main/DOCS/FACTOR_LAKE_USER_GUIDE.md).
    """)
