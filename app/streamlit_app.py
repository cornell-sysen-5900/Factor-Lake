"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/streamlit_app.py
PURPOSE: Main application entry point orchestrating UI components and analysis logic.
VERSION: 3.3.0
"""

import streamlit as st
import app.streamlit_utils as utils
import app.streamlit_css as css
import app.streamlit_config as config

import components.sidebar as sidebar
from components.factor_selection import render_factor_selection
from app.components.results_tab import render_results_tab
from components.about import render_about_tab

# Global Page Configuration
st.set_page_config(
    page_title="Factor-Lake Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Initialization
utils.initialize_environment()
utils.initialize_session_state()
css.inject_styles()

def main():
    """
    Orchestrates the primary application flow, including security authentication,
    parameter configuration, and multi-tab rendering.
    """
    if not utils.check_password(st.secrets):
        st.stop()
        
    st.markdown('<div class="main-header">Factor-Lake Portfolio Analysis</div>', unsafe_allow_html=True)
    
    # Render Global Sidebar and capture configuration
    user_settings = sidebar.render_sidebar(config.SECTOR_OPTIONS)
    
    # Define primary application layout
    tab_analysis, tab_results, tab_about = st.tabs(["Analysis", "Results", "About"])

    with tab_analysis:
        # Configuration phase: Capture factors and directions
        selected_factor_names, factor_directions = render_factor_selection()
        st.session_state['selected_factor_names'] = selected_factor_names
        st.session_state['factor_directions'] = factor_directions
        
        # Step 1: Data Acquisition
        if st.button("Load Market Data", type="primary", use_container_width=True):
            with st.spinner("Accessing cloud database..."):
                utils.load_and_process_data(user_settings)

        # Step 2: Execution
        if st.session_state.get('data_loaded', False):
            if st.button("Run Portfolio Analysis", type="primary", use_container_width=True):
                with st.spinner("Executing backtest simulation..."):
                    utils.run_backtest_logic(
                        user_settings, 
                        selected_factor_names, 
                        factor_directions
                    )
                
                if st.session_state.get('results') is not None:
                    st.success("Analysis complete. Review performance in the Results tab.")
                    # Trigger automatic tab switch to Results via JavaScript injection
                    st.markdown("""
                        <script>
                        window.parent.document.querySelectorAll('button[data-baseweb="tab"]')[1].click();
                        </script>
                        """, unsafe_allow_html=True)

<<<<<<< HEAD
    with tab_results:
        # Performance Visualization phase
        if st.session_state.get('results') is not None:
            render_results_tab(st.session_state.results, user_settings)
=======
                            # 3. Calculate Top Cohort (Selection)
                            res_top = rebalance_portfolio(
                                data=st.session_state.rdata,
                                factors=factor_objects,
                                start_year=analysis_years[0],
                                end_year=analysis_years[-1],
                                initial_aum=st.session_state.initial_aum,
                                verbosity=0,
                                restrict_fossil_fuels=st.session_state.restrict_ff,
                                top_pct=cohort_pct,
                                which='top',
                                factor_directions=original_dirs,
                                use_market_cap_weight=st.session_state.use_cap_weight
                            )

                            # 4. Calculate Bottom Cohort (Inverse)
                            res_bot = None
                            if show_bottom_cohort:
                                res_bot = rebalance_portfolio(
                                    data=st.session_state.rdata,
                                    factors=factor_objects,
                                    start_year=analysis_years[0],
                                    end_year=analysis_years[-1],
                                    initial_aum=st.session_state.initial_aum,
                                    verbosity=0,
                                    restrict_fossil_fuels=st.session_state.restrict_ff,
                                    top_pct=cohort_pct,
                                    which='bottom',
                                    factor_directions=flipped_dirs,
                                    use_market_cap_weight=st.session_state.use_cap_weight
                                )

                            # 5. Helper to extract metric data
                            def get_stats(res):
                                if not res or 'portfolio_values' not in res:
                                    return None
                                vals = list(res['portfolio_values'])
                                if len(vals) < 2: return None
                                start, end = vals[0], vals[-1]
                                years_n = max(1, len(analysis_years) - 1)
                                cagr = (((end / start) ** (1 / years_n)) - 1) * 100
                                return {'end': end, 'cagr': cagr}

                            top_s = get_stats(res_top)
                            bot_s = get_stats(res_bot)

                            # 6. Display Metrics
                            m_col1, m_col2 = st.columns(2)
                            with m_col1:
                                if top_s:
                                    st.metric(f"Top {cohort_pct}% (Selected) Value", f"${top_s['end']:,.2f}")
                                    st.metric(f"Top {cohort_pct}% CAGR", f"{top_s['cagr']:.2f}%")
                            with m_col2:
                                if bot_s:
                                    st.metric(f"Bottom {cohort_pct}% (Inverse) Value", f"${bot_s['end']:,.2f}")
                                    st.metric(f"Bottom {cohort_pct}% CAGR", f"{bot_s['cagr']:.2f}%")

                            # 7. Generate Plot using precomputed results
                            fig_cohort = plot_top_bottom_percent(
                                rdata=st.session_state.rdata,
                                factors=factor_objects,
                                years=analysis_years,
                                percent=cohort_pct,
                                show_bottom=show_bottom_cohort,
                                benchmark_returns=results.get('benchmark_returns'),
                                growth_returns=results.get('growth_benchmark_returns'),
                                value_returns=results.get('value_benchmark_returns'),
                                benchmark_label='Russell 2000',
                                initial_investment=st.session_state.initial_aum,
                                baseline_portfolio_values=results.get('portfolio_values'),
                                precomputed_top=res_top,
                                precomputed_bot=res_bot
                            )

                            if fig_cohort:
                                st.pyplot(fig_cohort)
                                st.success("Analysis synchronized and completed successfully.")

                        except Exception as e:
                            st.error(f"Error generating cohort analysis: {str(e)}")
                            if verbosity_level >= 2:
                                st.exception(e)
            
            st.divider()
            
            # Year-by-year performance table
            st.subheader("Year-by-Year Performance")
            
            perf_data = {
                'Year': results['years'],
                'Portfolio Value': [f"${v:,.2f}" for v in results['portfolio_values']],
            }
            
            # Calculate year-over-year returns
            if len(results['portfolio_values']) > 1:
                yoy_returns = ['-']
                for i in range(1, len(results['portfolio_values'])):
                    ret = ((results['portfolio_values'][i] / results['portfolio_values'][i-1]) - 1) * 100
                    yoy_returns.append(f"{ret:.2f}%")
                perf_data['YoY Return'] = yoy_returns
            
            if 'benchmark_returns' in results and results['benchmark_returns']:
                # Benchmark returns are already in percentage format (like 34.62)
                perf_data['Benchmark Returns'] = ['-'] + [f"{r:.2f}%" for r in results['benchmark_returns']]

            if 'growth_benchmark_returns' in results and results['growth_benchmark_returns']:
                perf_data['Growth Returns'] = ['-'] + [f"{r:.2f}%" for r in results['growth_benchmark_returns']]

            if 'value_benchmark_returns' in results and results['value_benchmark_returns']:
                perf_data['Value Returns'] = ['-'] + [f"{r:.2f}%" for r in results['value_benchmark_returns']]
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Advanced Backtest Statistics
            if 'sharpe_portfolio' in results and 'max_drawdown_portfolio' in results:
                st.subheader("Advanced Backtest Statistics")
                
                # Display key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Max Drawdown (Portfolio)", 
                             f"{results['max_drawdown_portfolio']*100:.2f}%",
                             delta=None,
                             help="Largest peak-to-trough decline in portfolio value")
                
                with col2:
                    st.metric("Max Drawdown (Benchmark)", 
                             f"{results['max_drawdown_benchmark']*100:.2f}%",
                             delta=None,
                             help="Largest peak-to-trough decline in Russell 2000")

                with col3:
                    if 'max_drawdown_growth' in results:
                        st.metric("Max Drawdown (Growth Index)", 
                                 f"{results['max_drawdown_growth']*100:.2f}%",
                                 delta=None,
                                 help="Largest peak-to-trough decline in Russell 2000 Growth")

                with col4:
                    if 'max_drawdown_value' in results:
                        st.metric("Max Drawdown (Value Index)", 
                                 f"{results['max_drawdown_value']*100:.2f}%",
                                 delta=None,
                                 help="Largest peak-to-trough decline in Russell 2000 Value")
                
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    sharpe_delta = results['sharpe_portfolio'] - results['sharpe_benchmark']
                    st.metric("Sharpe Ratio (Portfolio)", 
                             f"{results['sharpe_portfolio']:.4f}",
                             delta=f"{sharpe_delta:+.4f} vs benchmark",
                             help="Risk-adjusted return (return per unit of volatility)")
                
                with col2:
                    st.metric("Sharpe Ratio (Benchmark)", 
                             f"{results['sharpe_benchmark']:.4f}",
                             delta=None,
                             help="Risk-adjusted return for Russell 2000")

                with col3:
                    if 'sharpe_growth' in results:
                        st.metric("Sharpe Ratio (Growth Index)", 
                                 f"{results['sharpe_growth']:.4f}",
                                 delta=None,
                                 help="Risk-adjusted return for Russell 2000 Growth")

                with col4:
                    if 'sharpe_value' in results:
                        st.metric("Sharpe Ratio (Value Index)", 
                                 f"{results['sharpe_value']:.4f}",
                                 delta=None,
                                 help="Risk-adjusted return for Russell 2000 Value")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if 'vol_raw_portfolio' in results:
                        st.metric("Volatility (Raw Returns, Portfolio)",
                                 f"{results['vol_raw_portfolio']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of full-sample portfolio returns")

                with col2:
                    if 'vol_raw_benchmark' in results:
                        st.metric("Volatility (Raw Returns, Benchmark)",
                                 f"{results['vol_raw_benchmark']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of full-sample Russell 2000 returns")

                with col3:
                    if 'vol_raw_growth' in results:
                        st.metric("Volatility (Raw Returns, Growth Index)",
                                 f"{results['vol_raw_growth']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of full-sample Russell 2000 Growth returns")

                with col4:
                    if 'vol_raw_value' in results:
                        st.metric("Volatility (Raw Returns, Value Index)",
                                 f"{results['vol_raw_value']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of full-sample Russell 2000 Value returns")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if 'vol_excess_portfolio' in results:
                        st.metric("Volatility (Excess Returns, Portfolio)",
                                 f"{results['vol_excess_portfolio']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of portfolio excess returns")

                with col2:
                    if 'vol_excess_benchmark' in results:
                        st.metric("Volatility (Excess Returns, Benchmark)",
                                 f"{results['vol_excess_benchmark']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of Russell 2000 excess returns")

                with col3:
                    if 'vol_excess_growth' in results:
                        st.metric("Volatility (Excess Returns, Growth Index)",
                                 f"{results['vol_excess_growth']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of Russell 2000 Growth excess returns")

                with col4:
                    if 'vol_excess_value' in results:
                        st.metric("Volatility (Excess Returns, Value Index)",
                                 f"{results['vol_excess_value']*100:.2f}%",
                                 delta=None,
                                 help="Sample standard deviation of Russell 2000 Value excess returns")
                
                # Win rate and information ratio
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if results.get('portfolio_beta') is not None:
                        st.metric(
                            "Portfolio Beta vs Russell 2000",
                            f"{results['portfolio_beta']:.3f}",
                            help=(
                                "Sensitivity of portfolio returns to the Russell 2000. "
                                "Beta = 1 moves with the market. "
                                "> 1 is more volatile. < 1 is more defensive."
                            )
                        )

                with col2:
                    if results.get('portfolio_beta_growth') is not None:
                        st.metric(
                            "Beta (Portfolio vs Growth Index)",
                            f"{results['portfolio_beta_growth']:.3f}",
                            help=(
                                "Sensitivity of portfolio returns to the Russell 2000 Growth index. "
                                "Beta = 1 moves with the index. "
                                "> 1 is more volatile. < 1 is more defensive."
                            )
                        )

                with col3:
                    if results.get('portfolio_beta_value') is not None:
                        st.metric(
                            "Beta (Portfolio vs Value Index)",
                            f"{results['portfolio_beta_value']:.3f}",
                            help=(
                                "Sensitivity of portfolio returns to the Russell 2000 Value index. "
                                "Beta = 1 moves with the index. "
                                "> 1 is more volatile. < 1 is more defensive."
                            )
                        )

                with col4:
                    st.metric("Risk-Free Rate Source", 
                             results.get('risk_free_rate_source', 'N/A'),
                             delta=None,
                             help="Data source for risk-free rate calculations")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Yearly Win Rate (Portfolio vs Russell 2000)", 
                             f"{results['win_rate']*100:.2f}%",
                             delta=None,
                             help="Percentage of years portfolio outperformed benchmark")
                
                with col2:
                    if 'win_rate_growth' in results:
                        st.metric("Yearly Win Rate (Portfolio vs Growth Index)",
                                 f"{results['win_rate_growth']*100:.2f}%",
                                 delta=None,
                                 help="Percentage of years portfolio outperformed Russell 2000 Growth")

                with col3:
                    if 'win_rate_value' in results:
                        st.metric("Yearly Win Rate (Portfolio vs Value Index)",
                                 f"{results['win_rate_value']*100:.2f}%",
                                 delta=None,
                                 help="Percentage of years portfolio outperformed Russell 2000 Value")

                with col4:
                    if 'information_ratio' in results and results['information_ratio'] is not None:
                        st.metric("Information Ratio", 
                                 f"{results['information_ratio']:.4f}",
                                 delta=None,
                                 help="Active return per unit of active risk")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if 'information_ratio_growth' in results and results['information_ratio_growth'] is not None:
                        st.metric("Information Ratio (Portfolio vs Growth Index)",
                                 f"{results['information_ratio_growth']:.4f}",
                                 delta=None,
                                 help="Active return per unit of active risk versus Russell 2000 Growth")

                with col2:
                    if 'information_ratio_value' in results and results['information_ratio_value'] is not None:
                        st.metric("Information Ratio (Portfolio vs Value Index)",
                                 f"{results['information_ratio_value']:.4f}",
                                 delta=None,
                                 help="Active return per unit of active risk versus Russell 2000 Value")
                        
                st.divider()    
                
                # Yearly win/loss comparison table
                if 'yearly_comparisons' in results:
                    st.subheader("Yearly Win/Loss vs Benchmarks")
                    
                    comparison_data = {
                        'Year': [comp['year'] for comp in results['yearly_comparisons']],
                        'Portfolio Return': [f"{comp['portfolio_return']:.2f}%" for comp in results['yearly_comparisons']],
                        'Benchmark Return': [f"{comp['benchmark_return']:.2f}%" for comp in results['yearly_comparisons']],
                        # Use a green check for wins and a red cross for losses instead of Yes/No
                        'Outperformed Benchmark': ['✅' if comp['win'] else '❌' for comp in results['yearly_comparisons']],
                        'Growth Index Return': [f"{comp['growth_return']:.2f}%" for comp in results['yearly_comparisons']],
                        'Outperformed vs Growth': ['✅' if comp['growth_win'] else '❌' for comp in results['yearly_comparisons']],
                        'Value Index Return': [f"{comp['value_return']:.2f}%" for comp in results['yearly_comparisons']],
                        'Outperformed vs Value': ['✅' if comp['value_win'] else '❌' for comp in results['yearly_comparisons']]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                st.divider()
            
            # Download results
            st.subheader("Download Results")
            
            csv = perf_df.to_csv(index=False)
            st.download_button(
                label="Download Performance Data (CSV)",
                data=csv,
                file_name=f"portfolio_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
>>>>>>> main
        else:
            st.info("Performance metrics will appear here after a successful analysis run.")
    
    with tab_about:
        # Documentation phase
        render_about_tab()

if __name__ == "__main__":
    main()