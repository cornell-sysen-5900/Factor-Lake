"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/results_tab.py
PURPOSE: Visualization of results with strict adherence to original metrics and formatting.
VERSION: 3.6.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from Visualizations.portfolio_growth_plot import plot_portfolio_growth
from Visualizations.top_bottom_portfolio_plot import plot_top_bottom_percent
from src import backtest_engine 

def render_results_tab(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    if not results:
        st.warning("No backtest results found. Please execute the analysis.")
        return

    # 1. Header Information
    st.header("Portfolio Performance Results")
    _render_header_captions(user_settings)
    
    # 2. Performance Summary
    st.subheader("Performance Summary")
    _render_summary_metrics(results, user_settings)
    st.divider()

    # 3. Main Growth Plot
    st.subheader("Portfolio Growth Over Time")
    _render_growth_plot(results, user_settings)
    st.divider()

    # 4. Year-by-Year Performance (Scrollable)
    st.subheader("Year-by-Year Performance")
    _render_year_by_year_table(results)
    st.divider()

    # 5. Top vs Bottom Cohort Analysis
    _render_cohort_analysis_section(results, user_settings)
    st.divider()

    # 6. Advanced Backtest Statistics (Restored Win Rates)
    st.header("Advanced Backtest Statistics")
    _render_advanced_stats_grid(results)
    st.divider()

    # 7. Yearly Win/Loss Summary
    st.header("Yearly Win/Loss Summary")
    _render_win_loss_ledger(results)
    st.divider()

    # 8. Export
    _render_export_utility(results)

def _render_header_captions(settings):
    # Same consolidated lookup used in cohort section
    selected_factors = (
        st.session_state.get('selected_factor_names') or 
        st.session_state.get('selected_factors') or 
        []
    )
    saved_dirs = st.session_state.get('factor_directions', {})
    
    factors_str = ", ".join([f"{f} ({saved_dirs.get(f, 'top')})" for f in selected_factors])
    weighting_str = "Market Cap Weighted" if settings.get('use_market_cap_weight') else "Equal Weighted"
    
    # If it's still empty, it means the keys aren't being set during selection
    if not selected_factors:
        st.caption("**Factors:** No factors selected in Parameters.")
    else:
        st.caption(f"**Factors:** {factors_str}")
        
    st.caption(f"**Weighting:** {weighting_str}")
    
def _render_summary_metrics(res, settings):
    col1, col2, col3, col4 = st.columns(4)
    initial_aum = settings.get('initial_aum', 1000000.0)
    
    # Use portfolio_values directly for consistency
    port_vals = res.get('portfolio_values', [initial_aum])
    final_value = port_vals[-1]
    
    # CAGR Logic
    n_years = len(res.get('yearly_returns', []))
    cagr = (((final_value / initial_aum) ** (1 / n_years)) - 1) if n_years > 0 else 0
    
    # Correct Outperformance logic using wealth indices
    bench_rets = res.get('benchmark_returns', [])
    # Reconstruct benchmark wealth index to ensure apples-to-apples comparison
    bench_wealth = [initial_aum]
    for r in bench_rets:
        bench_wealth.append(bench_wealth[-1] * (1 + (r / 100.0)))
    
    benchmark_final = bench_wealth[-1]
    cum_outperformance = ((final_value / benchmark_final) - 1) if benchmark_final != 0 else 0
    
    col1.metric("Final Portfolio Value", f"${final_value:,.2f}")
    col2.metric("Total Return", f"{((final_value/initial_aum)-1):.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Cumulative Outperformance", f"{cum_outperformance:.2%}")

def _render_year_by_year_table(res):
    years = res.get('years', [])
    p_rets = res.get('yearly_returns', [])
    b_rets = res.get('benchmark_returns', [])
    
    df = pd.DataFrame({
        "Year": [int(y) for y in years[:len(p_rets)]],
        "Portfolio Return": [f"{r:.2f}%" for r in p_rets],
        "Benchmark Return": [f"{r:.2f}%" for r in b_rets],
        "Excess Return": [f"{(p - b):.2f}%" for p, b in zip(p_rets, b_rets)]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

def _render_cohort_analysis_section(results, user_settings):
    st.subheader("Top vs Bottom Cohort Analysis")
    st.write("This visualization compares the performance of the **top N%** and **bottom N%** of stocks against the benchmark.")
    
    with st.expander("Run Cohort Comparison", expanded=False):
        col_slider, _ = st.columns([2, 2])
        with col_slider:
            cohort_pct = st.slider("Select Cohort Percentage", 1, 100, 10, key="cohort_slider_input")
        
        if st.button("Generate Comparison", type="primary", key="btn_generate_cohort"):
            # FIX: Define the 'factors' variable here before passing it to the engine
            factors = st.session_state.get('selected_factor_names', [])
            f_dirs = st.session_state.get('factor_directions', {})

            if not factors:
                st.error("No factors found. Please run the Portfolio Analysis first.")
                return

            with st.spinner("Calculating cohorts..."):
                # Now 'factors' exists and contains the internal column names
                res_top, res_bot = backtest_engine.run_cohort_comparison(
                    data=st.session_state.rdata,
                    selected_factors=factors, 
                    factor_directions=f_dirs,
                    cohort_pct=cohort_pct,
                    user_settings=user_settings
                )

                # --- Metrics Rendering ---
                m1, m2, m3, m4 = st.columns(4)
                
                # Extract values for metrics
                top_final = res_top.get('portfolio_values', [0])[-1]
                top_cagr = res_top.get('cagr', 0)
                bot_final = res_bot.get('portfolio_values', [0])[-1]
                bot_cagr = res_bot.get('cagr', 0)

                m1.metric(f"Top {cohort_pct}% Value", f"${top_final:,.2f}")
                m2.metric(f"Top {cohort_pct}% CAGR", f"{top_cagr:.2%}")
                m3.metric(f"Bottom {cohort_pct}% Value", f"${bot_final:,.2f}")
                m4.metric(f"Bottom {cohort_pct}% CAGR", f"{bot_cagr:.2%}")

                # --- Plotting ---
                fig = plot_top_bottom_percent(
                    years=results['years'],
                    percent=cohort_pct,
                    show_bottom=True,
                    benchmark_returns=results.get('benchmark_returns'),
                    initial_investment=user_settings.get('initial_aum', 1000000.0),
                    baseline_portfolio_values=results['portfolio_values'],
                    precomputed_top=res_top,
                    precomputed_bot=res_bot
                )
                st.pyplot(fig)
                
def _render_advanced_stats_grid(res):
    # Row 1: Drawdowns
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Drawdown (Portfolio)", f"{res.get('max_drawdown_portfolio', 0):.2%}")
    c2.metric("Max Drawdown (Benchmark)", f"{res.get('max_drawdown_benchmark', 0):.2%}")
    c3.metric("Max Drawdown (Growth)", f"{res.get('max_drawdown_growth', 0):.2%}")
    c4.metric("Max Drawdown (Value)", f"{res.get('max_drawdown_value', 0):.2%}")

    # Row 2: Sharpe
    c1, c2, c3, c4 = st.columns(4)
    p_sharpe = res.get('sharpe_portfolio', 0)
    b_sharpe = res.get('sharpe_benchmark', 0)
    c1.metric("Sharpe (Portfolio)", f"{p_sharpe:.4f}", f"{p_sharpe - b_sharpe:.4f} vs bench")
    c2.metric("Sharpe (Benchmark)", f"{b_sharpe:.4f}")
    c3.metric("Sharpe (Growth)", f"{res.get('sharpe_growth', 0):.4f}")
    c4.metric("Sharpe (Value)", f"{res.get('sharpe_value', 0):.4f}")

    # Row 3: Raw Vol
    st.write("**Volatility (Raw)**")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Portfolio", f"{res.get('vol_raw_portfolio', 0):.2%}")
    v2.metric("Benchmark", f"{res.get('vol_raw_benchmark', 0):.2%}")
    v3.metric("Growth", f"{res.get('vol_raw_growth', 0):.2%}")
    v4.metric("Value", f"{res.get('vol_raw_value', 0):.2%}")

    # Row 4: Excess Vol
    st.write("**Volatility (Excess)**")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Portfolio", f"{res.get('vol_excess_portfolio', 0):.2%}")
    e2.metric("Benchmark", f"{res.get('vol_excess_benchmark', 0):.2%}")
    e3.metric("Growth", f"{res.get('vol_excess_growth', 0):.2%}")
    e4.metric("Value", f"{res.get('vol_excess_value', 0):.2%}")

    # Row 5: Beta
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Beta vs R2000", f"{res.get('portfolio_beta', 0):.3f}")
    b2.metric("Beta vs Growth", f"{res.get('portfolio_beta_growth', 0):.3f}")
    b3.metric("Beta vs Value", f"{res.get('portfolio_beta_value', 0):.3f}")
    b4.write(f"**RF Rate Source**\n\n{res.get('risk_free_rate_source', 'FRED')}")

    # Row 6: Win Rates (Restored)
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Yearly Win Rate (vs R2000)", f"{res.get('win_rate', 0):.2%}")
    w2.metric("Yearly Win Rate (vs Growth)", f"{res.get('win_rate_growth', 0):.2%}")
    w3.metric("Yearly Win Rate (vs Value)", f"{res.get('win_rate_value', 0):.2%}")
    w4.empty()

    # Row 7: IR
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Information Ratio", f"{res.get('information_ratio', 0):.4f}")
    i2.metric("IR (vs Growth)", f"{res.get('information_ratio_growth', 0):.4f}")
    i3.metric("IR (vs Value)", f"{res.get('information_ratio_value', 0):.4f}")
    i4.empty()

def _render_win_loss_ledger(res):
    if 'yearly_comparisons' in res:
        comps = res['yearly_comparisons']
        df = pd.DataFrame({
            'Year': [int(c.get('year')) for c in comps],
            'Portfolio Ret': [f"{c.get('p_ret', 0):.2f}%" for c in comps],
            'Benchmark Ret': [f"{c.get('b_ret', 0):.2f}%" for c in comps],
            'vs Benchmark': ['✅' if c.get('b_win') else '❌' for c in comps],
            'vs Growth': ['✅' if c.get('g_win') else '❌' for c in comps],
            'vs Value': ['✅' if c.get('v_win') else '❌' for c in comps],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

def _render_growth_plot(res, settings):
    # Use the persistent session state keys
    factors = st.session_state.get('selected_factor_names', [])
    
    initial = settings.get('initial_aum', 1000000.0)
    def build_wealth(rets):
        if not rets: return None
        v = [initial]
        for r in rets: v.append(v[-1] * (1 + (r/100.0)))
        return v
        
    fig = plot_portfolio_growth(
        years=res.get('years', []),
        port_vals=res.get('portfolio_values', []),
        bench_vals=build_wealth(res.get('benchmark_returns')),
        val_vals=build_wealth(res.get('value_benchmark_returns')),
        gro_vals=build_wealth(res.get('growth_benchmark_returns')),
        factor_names=factors # This will now be populated
    )
    st.pyplot(fig)

def _render_export_utility(res):
    export_df = pd.DataFrame({
        'Year': res.get('years', []),
        'Portfolio Value': res.get('portfolio_values', [])
    })
    st.download_button("Download Performance (CSV)", export_df.to_csv(index=False), "results.csv", "text/csv")