"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/results_view.py
PURPOSE: Visualization component for backtest metrics, charts, and cohort analysis.
VERSION: 1.1.0
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

def render_results_tab(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Acts as the primary entry point for the Performance Results interface.
    
    This function orchestrates the layout of the entire results tab, coordinating 
    metadata display, KPI metrics, time-series growth visualizations, and 
    interactive cohort analysis. It ensures that all UI elements are synchronized 
    with the current backtest session data.
    
    Input:
        results (Dict[str, Any]): The comprehensive output dictionary from the 
                                 backtest engine containing returns and risk metrics.
        user_settings (Dict[str, Any]): Global configuration parameters used to 
                                        standardize labels and calculations.
    Output:
        None: Renders multiple UI components directly to the Streamlit application.
    """
    if not results:
        st.warning("No backtest results found. Please run the analysis first.")
        return

    # 1. Summary Metrics Row
    _render_summary_metrics(results, user_settings)
    
    st.divider()

    # 2. Growth Visualization and Performance Tables
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        _render_growth_chart(results, user_settings)
        
    with col_stats:
        _render_performance_tables(results)

    st.divider()

    # 3. Interactive Cohort/Decile Analysis
    _render_cohort_analysis(results, user_settings)


def _render_summary_metrics(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Calculates and displays high-level portfolio performance indicators.
    """
    final_val = results.get('portfolio_values', [0])[-1]
    total_ret = ((final_val / user_settings['initial_aum']) - 1) * 100
    cagr = results.get('cagr', 0) * 100
    alpha = results.get('alpha', 0) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Portfolio Value", f"${final_val:,.2f}")
    m2.metric("Total Return", f"{total_ret:.2f}%")
    m3.metric("CAGR", f"{cagr:.2f}%")
    m4.metric("Alpha vs Benchmark", f"{alpha:.2f}%")


from Visualizations.portfolio_growth_plot import plot_portfolio_growth

def _render_growth_chart(results: dict, user_settings: dict) -> None:
    """
    Orchestrates the growth visualization within the Streamlit Results tab.
    
    This UI-level function extracts the necessary time-series data from the 
    application's session state and passes it to the specialized visualization 
    module. It acts as a bridge between the data store and the plotting engine.

    Args:
        results (dict): The dictionary containing 'years', 'portfolio_values', 
                        and 'benchmark_values' keys.
        user_settings (dict): Configuration dictionary used for chart labeling.

    Returns:
        None: Renders the plot directly to the active Streamlit column.
    """
    st.subheader("Growth of Portfolio")
    
    # Extract data from the results dictionary
    years = results.get('years', [])
    port_vals = results.get('portfolio_values', [])
    bench_vals = results.get('benchmark_values', [])

    # Validation gate to prevent empty chart rendering
    if not years or not port_vals:
        st.error("Visualization Error: The backtest engine returned empty datasets.")
        return

    # Call the externalized plotting utility
    fig = plot_portfolio_growth(years, port_vals, bench_vals)
    
    # Display the figure in the Streamlit UI
    st.pyplot(fig)


def _render_performance_tables(results: Dict[str, Any]) -> None:
    """
    Tabulates granular backtest data and advanced risk-adjusted return statistics.
    """
    st.subheader("Risk Analytics")
    
    stats = {
        "Sharpe Ratio": f"{results.get('sharpe', 0):.2f}",
        "Max Drawdown": f"{results.get('max_drawdown', 0)*100:.2f}%",
        "Beta": f"{results.get('beta', 0):.2f}",
        "Volatility": f"{results.get('volatility', 0)*100:.2f}%"
    }
    
    for label, val in stats.items():
        st.write(f"**{label}:** {val}")

    # CSV Download
    df_returns = pd.DataFrame({
        'Year': results.get('years', []),
        'Return': results.get('yearly_returns', [])
    })
    csv = df_returns.to_csv(index=False).encode('utf-8')
    st.download_button("Download Yearly Returns", data=csv, file_name="returns.csv", mime="text/csv")


def _render_cohort_analysis(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Facilitates interactive performance exploration through decile-based analysis.
    """
    st.subheader("Factor Cohort Analysis")
    st.write("Compare the performance of the Top decile vs. Bottom decile for the selected strategy.")
    
    # Implementation of cohort specific plotting/logic
    # (Placeholder for the specific logic block we moved from main)
    st.info("Cohort analysis visualization is calculated based on current factor tilts.")