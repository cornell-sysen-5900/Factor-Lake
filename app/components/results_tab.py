"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/results_tab.py
PURPOSE: Visualization of results with strict adherence to original metrics and formatting.
VERSION: 3.6.1
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from Visualizations.portfolio_growth_plot import plot_portfolio_growth
from Visualizations.top_bottom_portfolio_plot import plot_top_bottom_percent
from src import backtest_engine 

def render_results_tab(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Orchestrates the rendering of the results tab within the Streamlit interface.

    This function serves as the primary view controller for backtest outputs, 
    organizing metrics, growth charts, and advanced statistical analysis into 
    a cohesive reporting dashboard.
    """
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
    st.subheader("Ranked Stocks (Best to Worst)")
    _render_ranked_stocks_table(results)
    st.divider()

    # 4. Main Growth Plot
    st.subheader("Portfolio Growth Over Time")
    _render_growth_plot(results, user_settings)
    st.divider()

    # 5. Year-by-Year Performance
    st.subheader("Year-by-Year Performance")
    _render_year_by_year_table(results)
    st.divider()

    # 6. Top vs Bottom Cohort Analysis
    _render_cohort_analysis_section(results, user_settings)
    st.divider()

    # 7. Advanced Backtest Statistics
    st.header("Advanced Backtest Statistics")
    _render_advanced_stats_grid(results)
    st.divider()

    # 8. Yearly Win/Loss Summary
    st.header("Yearly Win/Loss Summary")
    _render_win_loss_ledger(results)
    st.divider()


def _render_ranked_stocks_table(res: Dict[str, Any]) -> None:
    """
    Displays the selected top cohort ranked by composite multi-factor score.
    """
    ranked_rows = res.get('ranked_stocks', [])
    ranking_year = res.get('ranking_year')

    if ranking_year is not None:
        st.caption(f"Ranking year: {int(ranking_year)}")

    if not ranked_rows:
        st.info("No ranked stocks are available for the selected factors and date range.")
        return

    ranked_df = pd.DataFrame(ranked_rows)

    if 'Composite Score' in ranked_df.columns:
        ranked_df['Composite Score'] = pd.to_numeric(
            ranked_df['Composite Score'], errors='coerce'
        ).round(4)

    if 'Next-Year Return %' in ranked_df.columns:
        ranked_df['Next-Year Return %'] = pd.to_numeric(
            ranked_df['Next-Year Return %'], errors='coerce'
        ).round(2)

    st.dataframe(ranked_df, use_container_width=True, hide_index=True)

def _render_year_by_year_table(res: Dict[str, Any]) -> None:
    """
    Generates the primary performance table with standardized financial headers.
    
    The table includes longitudinal portfolio values and relative returns against 
    core market benchmarks (Russell 2000, Growth, and Value).
    """
    years = res.get('years', [])
    port_vals = res.get('portfolio_values', [])
    p_rets = res.get('yearly_returns', [])
    b_rets = res.get('benchmark_returns', [])
    g_rets = res.get('growth_benchmark_returns', [])
    v_rets = res.get('value_benchmark_returns', [])

    # Align data lengths for consistent reporting
    min_len = len(p_rets)
    
    df = pd.DataFrame({
        "Year": [int(y) for y in years[:min_len]],
        "Portfolio Value": [f"${v:,.2f}" for v in port_vals[1:min_len+1]],
        "YoY Return": [f"{r:.2f}%" for r in p_rets],
        "Benchmark Returns": [f"{r:.2f}%" for r in b_rets[:min_len]],
        "Growth Returns": [f"{r:.2f}%" for r in g_rets[:min_len]],
        "Value Returns": [f"{r:.2f}%" for r in v_rets[:min_len]]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

def _render_win_loss_ledger(res: Dict[str, Any]) -> None:
    """
    Displays a comprehensive win/loss ledger using boolean performance indicators.

    This ledger compares portfolio returns against multiple indices on a yearly 
    basis, highlighting alpha generation through visual cues.
    """
    if 'yearly_comparisons' in res:
        comps = res['yearly_comparisons']
        df = pd.DataFrame({
            'Year': [int(c.get('year')) for c in comps],
            'Portfolio Return': [f"{c.get('p_ret', 0):.2f}%" for c in comps],
            'Benchmark Return': [f"{c.get('b_ret', 0):.2f}%" for c in comps],
            'Outperformed Benchmark': ['✅' if c.get('b_win') else '❌' for c in comps],
            'Growth Index Return': [f"{c.get('g_ret', 0):.2f}%" for c in comps],
            'Outperformed Growth': ['✅' if c.get('g_win') else '❌' for c in comps],
            'Value Index Return': [f"{c.get('v_ret', 0):.2f}%" for c in comps],
            'Outperformed Value': ['✅' if c.get('v_win') else '❌' for c in comps],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

def _render_advanced_stats_grid(res: Dict[str, Any]) -> None:
    """
    Renders a multi-row grid containing advanced risk and efficiency metrics.
    """
    # Row 1: Drawdowns
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


def _render_header_captions(settings: Dict[str, Any]) -> None:
    """
    Renders identifying metadata regarding the specific backtest configuration.
    """
    selected_factors = st.session_state.get('selected_factor_names') or st.session_state.get('selected_factors', [])
    saved_dirs = st.session_state.get('factor_directions', {})
    factors_str = ", ".join([f"{f} ({saved_dirs.get(f, 'top')})" for f in selected_factors])
    weighting_str = "Market Cap Weighted" if settings.get('use_market_cap_weight') else "Equal Weighted"
    
    st.caption(f"**Factors:** {factors_str}")
    st.caption(f"**Weighting:** {weighting_str}")

def _render_summary_metrics(res: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """
    Calculates and displays core summary metrics for the reporting period.
    """
    col1, col2, col3, col4 = st.columns(4)
    initial_aum = settings.get('initial_aum', 1000000.0)
    final_value = res.get('final_value', initial_aum)
    
    n_years = len(res.get('yearly_returns', []))
    cagr = (((final_value / initial_aum) ** (1 / n_years)) - 1) if n_years > 0 else 0
    
    bench_rets = res.get('benchmark_returns', [])
    benchmark_final = initial_aum * np.prod([1 + (r / 100.0) for r in bench_rets])
    cum_outperformance = ((final_value / benchmark_final) - 1)
    
    col1.metric("Final Portfolio Value", f"${final_value:,.2f}")
    col2.metric("Total Return", f"{((final_value/initial_aum)-1):.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Cumulative Outperformance", f"{cum_outperformance:.2%}")

def _render_cohort_analysis_section(results: Dict[str, Any], user_settings: Dict[str, Any]) -> None:
    """
    Executes and visualizes a comparison between top and bottom factor cohorts.
    """
    st.subheader("Top vs Bottom Cohort Analysis")
    with st.expander("Run Cohort Comparison", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cohort_pct = st.slider("Cohort Percentage", 1, 50, 10)
        
        if st.button("Generate Comparison", type="primary"):
            factors = st.session_state.get('selected_factor_names') or st.session_state.get('selected_factors', [])
            res_top, res_bot = backtest_engine.run_cohort_comparison(
                data=st.session_state.rdata,
                selected_factors=factors,
                factor_directions=st.session_state.get('factor_directions', {}),
                cohort_pct=cohort_pct,
                user_settings=user_settings
            )

            fig = plot_top_bottom_percent(
                years=results['years'],
                percent=cohort_pct,
                show_bottom=True,
                benchmark_returns=results.get('benchmark_returns'),
                initial_investment=user_settings['initial_aum'],
                baseline_portfolio_values=results['portfolio_values'],
                precomputed_top=res_top,
                precomputed_bot=res_bot
            )
            st.pyplot(fig)

def _render_growth_plot(res: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """
    Visualizes the growth of $1 (or initial AUM) across the portfolio and benchmarks.
    """
    initial = settings.get('initial_aum', 1000000.0)
    def build_wealth(rets):
        if not rets: return None
        v = [initial]
        for r in rets: v.append(v[-1] * (1 + (r/100.0)))
        return v
    
    factors = st.session_state.get('selected_factor_names') or st.session_state.get('selected_factors', [])
    fig = plot_portfolio_growth(
        years=res.get('years', []),
        port_vals=res.get('portfolio_values', []),
        bench_vals=build_wealth(res.get('benchmark_returns')),
        val_vals=build_wealth(res.get('value_benchmark_returns')),
        gro_vals=build_wealth(res.get('growth_benchmark_returns')),
        factor_names=factors
    )
    st.pyplot(fig)
