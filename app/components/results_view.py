"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/results_view.py
PURPOSE: Visualization component for backtest metrics, charts, and risk attribution.
VERSION: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from Visualizations.portfolio_growth_plot import plot_portfolio_growth

def render_results_tab(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Primary entry point for the Performance Results interface.
    
    Orchestrates the layout of the results tab, coordinating KPI metrics, 
    time-series growth visualizations, and institutional risk analytics.
    """
    if not results:
        st.warning("No backtest results found. Please execute the analysis in the Analysis tab.")
        return

    # 1. High-Level Performance KPIs
    _render_summary_metrics(results, user_settings)
    
    st.divider()

    # 2. Time-Series Growth and Risk Analytics
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        _render_growth_chart(results, user_settings)
        
    with col_stats:
        _render_risk_analytics(results)

    st.divider()

    # 3. Yearly Comparison and Data Export
    _render_performance_tables(results)

def _render_summary_metrics(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Renders absolute and relative performance metrics.
    """
    initial_aum = user_settings.get('initial_aum', 1000000)
    final_val = results.get('final_value', 0)
    total_ret = ((final_val / initial_aum) - 1) * 100
    
    # Calculate CAGR based on the year range
    years_count = len(results.get('years', [])) - 1
    cagr = (((final_val / initial_aum) ** (1 / years_count)) - 1) * 100 if years_count > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ending Capital", f"${final_val:,.2f}")
    m2.metric("Cumulative Return", f"{total_ret:.2f}%")
    m3.metric("Annualized (CAGR)", f"{cagr:.2f}%")
    m4.metric("Win Rate vs Bench", f"{results.get('win_rate', 0)*100:.1f}%")

def _render_growth_chart(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Visualizes the growth of $1 (or initial AUM) over the investment horizon.
    """
    st.subheader("Cumulative Growth Comparison")
    
    years = results.get('years', [])
    port_vals = results.get('portfolio_values', [])
    
    # Reconstruct benchmark values for visualization if not explicitly provided
    # Standardizing to the same starting AUM
    bench_rets = results.get('benchmark_returns', [])
    bench_vals = [user_settings['initial_aum']]
    for r in bench_rets:
        bench_vals.append(bench_vals[-1] * (1 + (r / 100)))

    if not years or not port_vals:
        st.error("Data Integrity Error: Dataset is insufficient for time-series plotting.")
        return

    fig = plot_portfolio_growth(years, port_vals, bench_vals)
    st.pyplot(fig)

def _render_risk_analytics(results: Dict[str, Any]) -> None:
    """
    Displays institutional risk-adjusted return statistics.
    """
    st.subheader("Risk-Adjusted Metrics")
    
    # Portfolio vs Benchmark Comparison
    stats = {
        "Sharpe Ratio (Port)": f"{results.get('sharpe_portfolio', 0):.2f}",
        "Sharpe Ratio (Bench)": f"{results.get('sharpe_benchmark', 0):.2f}",
        "Information Ratio": f"{results.get('information_ratio', 0):.2f}",
        "Portfolio Beta": f"{results.get('portfolio_beta', 0):.2f}",
        "Max Drawdown (Port)": f"{results.get('max_drawdown_portfolio', 0)*100:.2f}%",
        "Max Drawdown (Bench)": f"{results.get('max_drawdown_benchmark', 0)*100:.2f}%"
    }
    
    for label, val in stats.items():
        st.write(f"**{label}:** {val}")

def _render_performance_tables(results: Dict[str, Any]) -> None:
    """
    Tabulates yearly performance and provides data export capabilities.
    """
    st.subheader("Yearly Performance Audit")
    
    comparisons = results.get('yearly_comparisons', [])
    if comparisons:
        df_comp = pd.DataFrame(comparisons)
        
        # Format for display
        df_display = df_comp.copy()
        df_display['portfolio_return'] = df_display['portfolio_return'].map("{:.2f}%".format)
        df_display['benchmark_return'] = df_display['benchmark_return'].map("{:.2f}%".format)
        
        st.table(df_display)

        # Export Logic
        csv = df_comp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Audit Trail (CSV)",
            data=csv,
            file_name="factor_backtest_results.csv",
            mime="text/csv"
        )