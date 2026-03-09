"""
Streamlit Application for Factor-Lake Portfolio Analysis
Interactive web interface for running factor-based portfolio backtests
"""
import sys
import os

# Load environment variables from .env file (check app/ dir, then project root)
from pathlib import Path
for _candidate in [Path(__file__).parent / '.env', Path(__file__).resolve().parent.parent / '.env']:
    if _candidate.exists():
        with open(_candidate) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key, value)
        break

# Add src directory to path by searching parent directories until a `src` folder
# is found. This makes the app runnable from different working directories
def _ensure_src_on_path():
    p = Path(__file__).resolve().parent
    # Look up to 6 parent levels for a sibling 'src' directory
    for _ in range(6):
        candidate = p / 'src'
        if candidate.is_dir():
            sys.path.insert(0, str(p))
            return
        p = p.parent
    # Fallback: use the original one-level-up heuristic
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'src')))

_ensure_src_on_path()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def check_password():
    """Returns `True` if the user had the correct password."""
    
    try:
        configured_password = st.secrets["password"]
    except (FileNotFoundError, KeyError):
        configured_password = os.environ.get("ADMIN_PASSWORD")

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if configured_password is not None and st.session_state["password"] == configured_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if configured_password is None:
        st.info("Admin password is not configured. Set 'password' in Streamlit Secrets (or ADMIN_PASSWORD env) to enable access.")
        return False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact your administrator for access*")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        # Password correct
        return True

# Import project modules
from src.market_object import load_data
from src.calculate_holdings import rebalance_portfolio
from src.factor_function import (
    Momentum6m, Momentum12m, Momentum1m, ROE, ROA, 
    P2B, NextFYrEarns, OneYrPriceVol,
    AccrualsAssets, ROAPercentage, OneYrAssetGrowth, OneYrCapEXGrowth, BookPrice
)
# Import from Visualizations (not in src)
from Visualizations.portfolio_growth_plot import plot_portfolio_growth
from Visualizations.top_bottom_portfolio_plot import plot_top_bottom_percent

# Page configuration
st.set_page_config(
    page_title="Factor-Lake Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Extensive styling options
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Alert boxes */
    .stAlert {
        margin-top: 1rem;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar (configuration panel) - grey accent */
    [data-testid="stSidebar"] {
        background-color: #DEDFE0; /* grey */
        border-left: 4px solid #3BD2E3; /* lake accent */
        padding-left: 8px;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #1f77b4;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
                if st.session_state["password"] == configured_password:
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
            if configured_password is None:
                st.info("Admin password is not configured. Set 'password' in Streamlit Secrets (or ADMIN_PASSWORD env) to enable access.")
                return False
    .stCheckbox {
        padding: 5px 0;
    }
                    "Enter Password", type="password", on_change=password_entered, key="password"
    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 8px;
        font-weight: 600;
    }
    
                    "Enter Password", type="password", on_change=password_entered, key="password"
    .element-container .stSuccess {
                st.error("Password incorrect")
        border-left: 4px solid #28a745;
    }
    
    .element-container .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    /* Custom card styling */
    .custom-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Hide Streamlit branding (optional) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive design adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'rdata' not in st.session_state:
    st.session_state.rdata = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Factor mapping
FACTOR_MAP = {
    'ROE using 9/30 Data': ROE,
    'ROA using 9/30 Data': ROA,
    '12-Mo Momentum %': Momentum12m,
    '6-Mo Momentum %': Momentum6m,
    '1-Mo Momentum %': Momentum1m,
    'Price to Book Using 9/30 Data': P2B,
    'Next FY Earns/P': NextFYrEarns,
    '1-Yr Price Vol %': OneYrPriceVol,
    'Accruals/Assets': AccrualsAssets,
    'ROA %': ROAPercentage,
    '1-Yr Asset Growth %': OneYrAssetGrowth,
    '1-Yr CapEX Growth %': OneYrCapEXGrowth,
    'Book/Price': BookPrice
}

# Sector options
SECTOR_OPTIONS = [
    'Consumer',
    'Technology',
    'Financials',
    'Industrials',
    'Healthcare'
]

def main():
    # Check password first
    if not check_password():
        st.stop()  # Stop execution if password is incorrect
    
    # Header
    st.markdown('<div class="main-header">Factor-Lake Portfolio Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Factor-Lake Portfolio Analysis tool! This application allows you to:
    - Select investment factors for portfolio construction
    - Apply fossil fuel restrictions and sector filters
    - Backtest portfolio performance from 2002-2023
    - Visualize portfolio growth vs. Russell 2000 benchmark
    """)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Configuration")
        # Data Source
        st.subheader("Data Source")
        st.caption("Data source: Supabase (cloud)")
        use_supabase = True
        
        # Data is loaded at daily granularity by default for accuracy
        # Yearly (legacy) option preserved for backward compatibility with 2002-2023 data
        data_frequency = st.radio(
            "Data Frequency:",
            options=["Daily", "Yearly (Legacy)"],
            index=0,
            help="Daily uses point-in-time daily prices for accurate compounding. Yearly (Legacy) uses the original annual factor data (2002-2023)."
        )
        # Map display label back to internal value
        if data_frequency == "Yearly (Legacy)":
            data_frequency = "Yearly"

        # Rebalance frequency options depend on data frequency
        if data_frequency == "Daily":
            rebalance_frequency = st.radio(
                "Rebalance Frequency:",
                options=["Daily", "Monthly", "Quarterly", "Yearly"],
                index=1,  # Default Monthly
                help="How often to reconstruct the portfolio. Between rebalances, existing holdings compound daily returns."
            )
        else:
            rebalance_frequency = "Yearly"

        st.write("---")
        # Fossil Fuel Restriction
        st.subheader("ESG Filters")
        restrict_fossil_fuels = st.checkbox(
            "Restrict Fossil Fuel Companies",
            value=False,
            help="Exclude oil, gas, coal, and fossil energy companies"
        )
        st.write("---")
        # Portfolio Weighting Method
        st.subheader("Portfolio Weighting")
        weighting_options = ["Equal Weight", "Market Cap Weight", "Volatility Weighted"]

        weighting_method = st.radio(
            "Select weighting method:",
            options=weighting_options,
            index=0,
            help="Equal: Each stock gets equal dollar investment. Market Cap: Weight by market capitalization. Volatility: Weight inversely by volatility to hit a target (only available for Monthly and Daily data)."
        )
        use_market_cap_weight = (weighting_method == "Market Cap Weight")
        
        if use_market_cap_weight:
            st.info("📊 Market Cap Weighting: Stocks will be weighted by their market capitalization, similar to the Russell 2000 index.")
            
        target_volatility = None
        volatility_metric = None
        if weighting_method == "Volatility Weighted":
            st.info("📉 Inverse-Volatility Weighting (Risk Parity): Stocks are weighted inversely proportional to their volatility. Lower-vol stocks get larger allocations.")
            volatility_metric = st.selectbox(
                "Volatility Metric",
                options=["vol_gk_1m", "vol_gk_3m", "vol_gk_6m", "vol_gk_12m", "vol_close_1m", "vol_close_3m", "vol_close_6m", "vol_close_12m"],
                index=0,
                help="Which metric to use to measure stock volatility (GK = Garman-Klass, Close = standard closing returns)."
            )
        
        st.write("---")
        # Sector Selection
        st.subheader("Sector Selection")
        sector_filter_enabled = st.checkbox("Enable Sector Filter", value=False)
        selected_sectors = []
        if sector_filter_enabled:
            selected_sectors = st.multiselect(
                "Select sectors to include:",
                options=SECTOR_OPTIONS,
                default=SECTOR_OPTIONS,
                help="Choose which sectors to include in the analysis"
            )
        st.write("---")
        # Date Range
        st.subheader("Analysis Period")
        col1, col2 = st.columns(2)
        # Use number_input with steppers (+ / -) and integer formatting
        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=2002,
                max_value=2023,
                value=2002,
                step=1,
                format="%d",
                key="start_year_input",
                help="Select the first year for analysis (2002-2023)"
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=2002,
                max_value=2024,
                value=2024,
                step=1,
                format="%d",
                key="end_year_input",
                help="Select the last year for analysis (2002-2023)"
            )

        # Ensure start_year <= end_year; auto-correct end year if needed
        try:
            if int(st.session_state.get('start_year_input', 2002)) > int(st.session_state.get('end_year_input', 2023)):
                st.warning('Start Year was greater than End Year — adjusting End Year to match Start Year')
                st.session_state['end_year_input'] = int(st.session_state['start_year_input'])
        except Exception:
            pass
        st.write("---")
        # Initial Investment
        st.subheader("Initial Investment")
        initial_aum = st.number_input(
            "Initial AUM ($)",
            min_value=0.0,
            max_value=1000000000.0,
            value=1000000.0,
            step=100.0,
            format="%.0f",
            help="Starting portfolio value in dollars"
        )
        st.write("---")
        # Verbosity
        st.subheader("Output Detail")
        verbosity_level = st.select_slider(
            "Verbosity Level",
            options=[0, 1, 2, 3],
            value=1,
            format_func=lambda x: ["Silent", "Basic", "Detailed", "Debug"][x],
            help="Control the amount of detail in output logs"
        )
        show_loading = st.checkbox("Show data loading progress", value=True)

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Analysis", "Results", "About"])
    
    with tab1:
        st.header("Factor Selection")
        st.write("Select one or more factors for your portfolio strategy:")
        
        # Helper: render a factor checkbox with an inline direction toggle
        def factor_category(factors_config):
            selections = {}
            directions = {}
            for name, key in factors_config:
                fcol1, fcol2 = st.columns([3, 2])
                with fcol1:
                    checked = st.checkbox(name, key=key)
                with fcol2:
                    if checked:
                        is_bottom = st.toggle(
                            "Low to High", key=f"{key}_dir", value=False,
                            help="OFF = High to Low (positive factor tilt), ON = Low to High (negative factor tilt)"
                        )
                    else:
                        is_bottom = False
                selections[name] = checked
                directions[name] = 'bottom' if is_bottom else 'top'
            return selections, directions

        # Create columns for factor selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Momentum Factors")
            momentum_factors, momentum_dirs = factor_category([
                ('12-Mo Momentum %', '12m'),
                ('6-Mo Momentum %', '6m'),
                ('1-Mo Momentum %', '1m'),
            ])
            st.subheader("Profitability Factors")
            profitability_factors, profitability_dirs = factor_category([
                ('ROE using 9/30 Data', 'roe'),
                ('ROA using 9/30 Data', 'roa'),
                ('ROA %', 'roa_pct'),
            ])
            st.subheader("Growth Factors")
            growth_factors, growth_dirs = factor_category([
                ('1-Yr Asset Growth %', 'asset_growth'),
                ('1-Yr CapEX Growth %', 'capex_growth'),
            ])
        with col2:
            st.subheader("Value Factors")
            value_factors, value_dirs = factor_category([
                ('Price to Book Using 9/30 Data', 'ptb'),
                ('Book/Price', 'btp'),
                ('Next FY Earns/P', 'fey'),
            ])
            st.subheader("Quality Factors")
            quality_factors, quality_dirs = factor_category([
                ('Accruals/Assets', 'accruals'),
                ('1-Yr Price Vol %', 'vol'),
            ])

        all_factor_selections = {
            **momentum_factors, **profitability_factors, **growth_factors,
            **value_factors, **quality_factors
        }
        all_factor_directions = {
            **momentum_dirs, **profitability_dirs, **growth_dirs,
            **value_dirs, **quality_dirs
        }
        
        selected_factor_names = [name for name, selected in all_factor_selections.items() if selected]
        factor_directions = {name: all_factor_directions[name] for name in selected_factor_names}
        
        st.write("---")
        
        # Display selected factors with their decile direction
        if selected_factor_names:
            factor_labels = [
                f"{name} ({'High to Low' if factor_directions[name] == 'top' else 'Low to High'})"
                for name in selected_factor_names
            ]
            st.success(f"Selected {len(selected_factor_names)} factor(s): {', '.join(factor_labels)}")
        else:
            st.warning("Please select at least one factor to run the analysis")
        
        st.write("---")
        
        # Load Data Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Load Data", use_container_width=True, type="primary"):
                # Reset session state so each click does a fresh load
                st.session_state.data_loaded = False
                st.session_state.rdata = None
                st.session_state.results = None
                if True:
                    try:
                        sectors_to_use = selected_sectors if sector_filter_enabled else None

                        @st.cache_data(show_spinner="Loading daily market data...", ttl=3600)
                        def cached_load_data(_restrict_ff, _data_freq, _sectors, _start, _end):
                            return load_data(
                                restrict_fossil_fuels=_restrict_ff,
                                use_supabase=True,
                                data_frequency=_data_freq,
                                show_loading_progress=True,
                                sectors=_sectors,
                                start_year=_start,
                                end_year=_end
                            )
                        
                        sectors_key = tuple(sorted(sectors_to_use)) if sectors_to_use else None
                        rdata = cached_load_data(
                            restrict_fossil_fuels,
                            data_frequency,
                            sectors_key,
                            int(start_year),
                            int(end_year)
                        )

                        if rdata is None or rdata.empty:
                            st.error(f"No {data_frequency} data found in the database between {start_year} and {end_year}. Please expand your Analysis Period.")
                            st.stop()

                        # Data preprocessing
                        if 'Ticker' not in rdata.columns and 'Ticker-Region' in rdata.columns:
                            rdata['Ticker'] = rdata['Ticker-Region'].dropna().apply(
                                lambda x: str(x).split('-')[0].strip()
                            )
                        if 'Year' not in rdata.columns:
                            if 'Date' in rdata.columns:
                                rdata['Year'] = pd.to_datetime(rdata['Date']).dt.year
                            elif 'date' in rdata.columns:
                                rdata['Year'] = pd.to_datetime(rdata['date']).dt.year

                        # If the user selected an analysis period, filter the loaded data to that range
                        try:
                            rdata = rdata[(rdata['Year'] >= int(start_year)) & (rdata['Year'] <= int(end_year))]
                        except Exception:
                            st.warning('Unable to filter loaded data by selected years; using full dataset instead.')

                        # Keep only relevant columns
                        cols_to_keep = ['Ticker', 'Year', 'Next_Year_Return']
                        
                        if 'Period' in rdata.columns:
                            cols_to_keep.append('Period')
                        
                        if 'Ending Price' in rdata.columns:
                            cols_to_keep.append('Ending Price')
                        elif 'Ending_Price' in rdata.columns:
                            rdata['Ending Price'] = rdata['Ending_Price']
                            cols_to_keep.append('Ending Price')
                        
                        if 'Market Capitalization' in rdata.columns:
                            cols_to_keep.append('Market Capitalization')
                        elif 'Market_Capitalization' in rdata.columns:
                            rdata['Market Capitalization'] = rdata['Market_Capitalization']
                            cols_to_keep.append('Market Capitalization')

                        if weighting_method == "Volatility Weighted" and volatility_metric in rdata.columns:
                            cols_to_keep.append(volatility_metric)
                            
                        for factor in FACTOR_MAP.keys():
                            if factor in rdata.columns:
                                cols_to_keep.append(factor)

                        rdata = rdata[cols_to_keep]

                        st.session_state.rdata = rdata
                        st.session_state.data_loaded = True

                        st.success(f"Data loaded successfully! {len(rdata)} records from {rdata['Year'].min()} to {rdata['Year'].max()}")

                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                        st.exception(e)
                
                
        # Show data preview
        if st.session_state.data_loaded:
            display_df = st.session_state.rdata
            with st.expander("Data Preview"):
                st.dataframe(display_df.head(100), use_container_width=True)
                st.write(f"**Shape:** {display_df.shape[0]} rows × {display_df.shape[1]} columns")
                st.write(f"**Years:** {sorted(display_df['Year'].unique())}")
                st.write(f"**Unique Tickers:** {display_df['Ticker'].nunique()}")
                        
        
        st.write("---")
        
        # Run Analysis Button
        if st.session_state.data_loaded:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Run Portfolio Analysis", use_container_width=True, type="primary", disabled=len(selected_factor_names) == 0):
                    if not selected_factor_names:
                        st.error("Please select at least one factor")
                    else:
                        with st.spinner("Running portfolio backtest..."):
                            try:
                                # Create factor objects
                                factor_objects = [FACTOR_MAP[name]() for name in selected_factor_names]
                                
                                # Run rebalancing
                                results = rebalance_portfolio(
                                    st.session_state.rdata,
                                    factor_objects,
                                    start_year=int(start_year),
                                    end_year=int(end_year),
                                    initial_aum=initial_aum,
                                    verbosity=verbosity_level,
                                    restrict_fossil_fuels=restrict_fossil_fuels,
                                    use_market_cap_weight=use_market_cap_weight,
                                    factor_directions=factor_directions,
                                    weighting_method=weighting_method,
                                    target_volatility=target_volatility,
                                    volatility_metric=volatility_metric,
                                    data_frequency=data_frequency,
                                    rebalance_frequency=rebalance_frequency
                                )
                                
                                st.session_state.results = results
                                st.session_state.selected_factors = selected_factor_names
                                st.session_state.factor_directions = factor_directions
                                st.session_state.restrict_ff = restrict_fossil_fuels
                                st.session_state.initial_aum = initial_aum
                                st.session_state.use_cap_weight = use_market_cap_weight
                                st.session_state.weighting_method = weighting_method
                                st.session_state.target_volatility = target_volatility
                                st.session_state.volatility_metric = volatility_metric
                                st.session_state.data_frequency = data_frequency
                                st.session_state.rebalance_frequency = rebalance_frequency
                                
                                st.success("Analysis complete! Check the Results tab.")
                            
                            except Exception as e:
                                st.error(f"Error running analysis: {str(e)}")
                                st.exception(e)
    
    with tab2:
        st.header("Portfolio Performance Results")
        
        if st.session_state.results is not None:
            results = st.session_state.results
            
            # Display configuration info
            col1, col2 = st.columns([3, 1])
            with col1:
                saved_dirs = st.session_state.get('factor_directions', {})
                factor_captions = [
                    f"{f} ({'High to Low' if saved_dirs.get(f, 'High to Low') == 'High to Low' else 'Low to High'})"
                    for f in st.session_state.selected_factors
                ]
                st.caption(f"**Factors:** {', '.join(factor_captions)}")
            with col2:
                weighting_label = st.session_state.get('weighting_method', 'Equal Weight')
                if weighting_label == "Volatility Weighted":
                    tgt = st.session_state.get('target_volatility', 15.0)
                    met = st.session_state.get('volatility_metric', '')
                    weighting_label += f" ({tgt}% target using {met})"
                st.caption(f"**Weighting:** {weighting_label}")
            
            st.divider()
            
            # Summary metrics
            st.subheader("Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                final_value = results['portfolio_values'][-1]
                st.metric("Final Portfolio Value", f"${final_value:,.2f}")
            
            with col2:
                total_return = ((final_value / st.session_state.initial_aum) - 1) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            
            with col3:
                n_periods = len(results['years'])
                data_freq = st.session_state.get('data_frequency', 'Yearly')
                if data_freq == 'Monthly':
                    n_years = n_periods / 12.0
                elif data_freq == 'Daily':
                    n_years = n_periods / 252.0
                else:
                    n_years = float(n_periods)
                n_years = max(n_years, 1.0 / 252.0)  # prevent division by zero
                cagr = (((final_value / st.session_state.initial_aum) ** (1 / n_years)) - 1) * 100
                st.metric("CAGR", f"{cagr:.2f}%")
            
            with col4:
                if 'benchmark_returns' in results and results['benchmark_returns']:
                    # Convert percentages to decimals before calculating
                    benchmark_final = st.session_state.initial_aum * np.prod([1 + r/100 for r in results['benchmark_returns']])
                    alpha = ((final_value / benchmark_final) - 1) * 100
                    st.metric("Cumulative Outperformance", f"{alpha:.2f}%")
                else:
                    st.metric("Rebalances", str(len(results['years'])))
            
            st.divider()
            
            # Portfolio growth chart
            st.subheader("Portfolio Growth Over Time")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            years = results['years']
            portfolio_values = results['portfolio_values']
            initial_aum = st.session_state.initial_aum
            is_sub_annual = results.get('data_frequency', 'Yearly') in ['Monthly', 'Daily']
            
            if is_sub_annual:
                # Use sequential integer x-axis, map to period labels
                x_vals = list(range(len(years)))
                ax.plot(x_vals, portfolio_values, marker='o', linewidth=2, markersize=3, label='Portfolio', color='#1f77b4')
                # Show every Nth label to avoid overlap
                n_labels = min(12, len(years))
                step = max(1, len(years) // n_labels)
                tick_positions = list(range(0, len(years), step))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(years[i]) for i in tick_positions], rotation=45, ha='right', fontsize=8)
                ax.set_xlabel('Period', fontsize=12)
            else:
                ax.plot(years, portfolio_values, marker='o', linewidth=2, markersize=6, label='Portfolio', color='#1f77b4')
                ax.set_xlabel('Year', fontsize=12)
            
            # Benchmark overlays
            if 'benchmark_returns' in results and results['benchmark_returns']:
                benchmark_values = [initial_aum]
                for ret in results['benchmark_returns']:
                    benchmark_values.append(benchmark_values[-1] * (1 + ret / 100))
                
                if is_sub_annual:
                    # Map yearly benchmark values to x-positions
                    year_end_positions = []
                    for yr_idx in range(len(benchmark_values)):
                        if yr_idx == 0:
                            # Initial value maps to the very first period
                            year_end_positions.append(0)
                        else:
                            # Value at end of year N maps to December of that year
                            yr = start_year + (yr_idx - 1)
                            dec_label = f"{yr}-12"
                            if dec_label in years:
                                year_end_positions.append(years.index(dec_label))
                            else:
                                yr_periods = [i for i, p in enumerate(years) if str(p).startswith(str(yr))]
                                if yr_periods:
                                    year_end_positions.append(yr_periods[-1])
                                else:
                                    year_end_positions.append(None)
                    # Filter out Nones and plot
                    valid = [(pos, val) for pos, val in zip(year_end_positions, benchmark_values) if pos is not None]
                    if valid:
                        bx, bv = zip(*valid)
                        ax.plot(list(bx), list(bv), marker='s', linewidth=2, markersize=5,
                               label='Russell 2000', linestyle='--', alpha=0.7, color='#ff7f0e')
                else:
                    ax.plot(years, benchmark_values, marker='s', linewidth=2, markersize=4, 
                           label='Russell 2000', linestyle='--', alpha=0.7, color='#ff7f0e')

            try:
                from src.benchmarks import get_benchmark_list
                val_rets = get_benchmark_list(3, start_year + 1, (start_year + len(results.get('benchmark_returns', []))) + 1)
                gro_rets = get_benchmark_list(2, start_year + 1, (start_year + len(results.get('benchmark_returns', []))) + 1)

                indices_to_plot = [
                    ('Value Index', val_rets, 'g', '^'),
                    ('Growth Index', gro_rets, 'orange', 'D')
                ]

                for label, rets, color, mkr in indices_to_plot:
                    if rets:
                        vals = [initial_aum]
                        for r in rets:
                            vals.append(vals[-1] * (1 + float(r) / 100))
                        if is_sub_annual:
                            # Same approach: map to year-end x-positions
                            idx_positions = []
                            for yr_idx in range(len(vals)):
                                if yr_idx == 0:
                                    idx_positions.append(0)
                                else:
                                    yr = start_year + (yr_idx - 1)
                                    dec_label = f"{yr}-12"
                                    if dec_label in years:
                                        idx_positions.append(years.index(dec_label))
                                    else:
                                        yr_periods = [i for i, p in enumerate(years) if str(p).startswith(str(yr))]
                                        if yr_periods:
                                            idx_positions.append(yr_periods[-1])
                                        else:
                                            idx_positions.append(None)
                            valid = [(pos, val) for pos, val in zip(idx_positions, vals) if pos is not None]
                            if valid:
                                ix, iv = zip(*valid)
                                ax.plot(list(ix), list(iv), marker=mkr, linestyle=':', linewidth=1.5,
                                       alpha=0.6, label=label, color=color)
                        else:
                            common_len = min(len(years), len(vals))
                            ax.plot(years[:common_len], vals[:common_len], 
                                   marker=mkr, linestyle=':', linewidth=1.5, 
                                   alpha=0.6, label=label, color=color)
            except Exception as e:
                st.error(f"Error loading extra benchmarks: {e}")
            
            ax.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax.set_title(f'Portfolio Growth: {", ".join(st.session_state.selected_factors)}', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            st.pyplot(fig)
            st.divider()
            # Top/Bottom Cohort Analysis
            st.subheader("Top vs Bottom Cohort Analysis")
            
            with st.expander("View Top/Bottom Portfolio Performance", expanded=False):
                st.markdown("""
                This visualization compares the performance of the **top N%** (your selection) and **bottom N%** (the inverse) of stocks against the benchmark.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    cohort_pct = st.slider(
                        "Select Cohort Percentage",
                        min_value=1,
                        max_value=100,
                        value=10,
                        step=1,
                        help="Percentage of stocks to include in top/bottom cohorts"
                    )
                with col2:
                    show_bottom_cohort = st.checkbox("Show Bottom Cohort", value=True)
                
                if st.button("Generate Top/Bottom Analysis", key="top_bottom_btn"):
                    with st.spinner("Generating cohort analysis..."):
                        try:
                            # 1. Setup factors and timeframe
                            factor_objects = [FACTOR_MAP[name]() for name in st.session_state.selected_factors]
                            analysis_years = results['years']
                            
                            # 2. Define inverse directions for the bottom cohort
                            original_dirs = st.session_state.factor_directions or {}
                            flipped_dirs = {
                                k: ('bottom' if v == 'top' else 'top') 
                                for k, v in original_dirs.items()
                            }

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
                                weighting_method=st.session_state.get('weighting_method', 'Equal Weight'),
                                target_volatility=st.session_state.get('target_volatility'),
                                volatility_metric=st.session_state.get('volatility_metric')
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
                                    weighting_method=st.session_state.get('weighting_method', 'Equal Weight'),
                                    target_volatility=st.session_state.get('target_volatility'),
                                    volatility_metric=st.session_state.get('volatility_metric')
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
            
            # Period-by-period performance table
            is_sub_annual_table = results.get('data_frequency', 'Yearly') in ['Monthly', 'Daily']
            period_label = 'Period' if is_sub_annual_table else 'Year'
            st.subheader(f"{period_label}-by-{period_label} Performance")
            
            perf_data = {
                period_label: [str(y) for y in results['years']],
                'Portfolio Value': [f"${v:,.2f}" for v in results['portfolio_values']],
            }
            
            # Calculate year-over-year returns
            if len(results['portfolio_values']) > 1:
                yoy_returns = ['-']
                for i in range(1, len(results['portfolio_values'])):
                    prev = results['portfolio_values'][i-1]
                    if prev != 0:
                        ret = ((results['portfolio_values'][i] / prev) - 1) * 100
                        yoy_returns.append(f"{ret:.2f}%")
                    else:
                        yoy_returns.append("N/A")
                perf_data['YoY Return'] = yoy_returns
            
            if not is_sub_annual_table:
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
            
        else:
            st.info("Run an analysis from the Analysis tab to see results here")
    
   
    with tab3:
        st.header("About Factor-Lake Portfolio Analysis")
        
        st.markdown("""
        ### What is Factor Investing?
            
        Factor investing is a strategy that targets specific drivers of returns across asset classes. 
        This tool allows you to backtest various factor strategies on historical market data.
            
        ### Available Factors
            
        **Momentum Factors:**
        - Stocks that have performed well in the past tend to continue performing well
            
        **Value Factors:**
        - Stocks trading at low prices relative to their fundamentals
            
        **Profitability Factors:**
        - Companies with strong profit margins and returns
            
        **Growth Factors:**
        - Companies with strong growth in assets and capital expenditure
            
        **Quality Factors:**
        - Companies with stable earnings and low volatility
            
        ### Factor Details & Investment Thesis
        
        **1. ROE using 9/30 Data** *(Higher is better)*  
        Firms with higher return on equity generate more profit from shareholder capital, indicating efficient capital allocation and stronger profitability prospects.
        
        **2. ROA using 9/30 Data** *(Higher is better)*  
        Return on assets measures how efficiently a company uses its assets to generate earnings; higher ROA typically signals better operational efficiency.
        
        **3. 12-Mo Momentum %** *(Higher is better)*  
        Stocks that have performed well over the past 12 months tend to continue to outperform in the near-term due to persistent investor behavior and trend continuation.
        
        **4. 6-Mo Momentum %** *(Higher is better)*  
        Stocks with strong 6-month performance often continue upward in the short-term; this captures intermediate-term momentum.
        
        **5. 1-Mo Momentum %** *(Higher is better)*  
        One-month momentum captures very short-term trend continuation; higher recent returns indicate near-term strength.
        
        **6. Price to Book Using 9/30 Data** *(Lower is better - factor inverted)*  
        A lower price-to-book (P/B) implies the stock is cheaper relative to its book value; economically we expect higher book-to-price (inverse of P/B) to indicate value.
        
        **7. Next FY Earns/P** *(Higher is better)*  
        Earnings yield (next fiscal year earnings / price) indicates how cheaply the market prices future earnings; higher values suggest more attractive valuation.
        
        **8. 1-Yr Price Vol %** *(Lower is better - factor inverted)*  
        Higher trailing 1-year price volatility may indicate higher risk or mispricing; lower volatility is preferable for risk-averse portfolios.
        
        **9. Accruals/Assets** *(Lower is better - factor inverted)*  
        High accruals relative to assets can indicate lower earnings quality; lower accrual ratios are generally preferable.
        
        **10. ROA %** *(Higher is better)*  
        Return on assets (percentage) measures profitability relative to asset base; higher ROA suggests better operating performance.
        
        **11. 1-Yr Asset Growth %** *(Higher is better)*  
        Higher asset growth can signal expansion and investment opportunities; higher is treated as more attractive for growth-oriented factors.
        
        **12. 1-Yr CapEX Growth %** *(Higher is better)*  
        Rising capital expenditures can indicate investment in future growth; higher CapEx growth is often treated as positive for growth strategies.
        
        **13. Book/Price** *(Higher is better)*  
        Book-to-price is the inverse of price-to-book and aligns directly with the value thesis: higher book/price means cheaper relative to book value and thus more attractive.
            
        ### How to Use
            
        1. **Configure** your data source and filters in the sidebar
        2. **Select** one or more factors in the Analysis tab
        3. **Load** the market data
        4. **Run** the portfolio analysis
        5. **View** results and download performance data
        
        ### Return Methodology
        
        The portfolio's performance is derived from the **Next-Year's Return %** column within the dataset. This metric captures the total percentage return, including dividends, from the initial data capture date through a one-year investment window. To ensure the backtest remains robust against survivorship bias, we account for tickers with missing return data—often the result of delisting due to mergers, acquisitions, or bankruptcy—by assigning them a return of 0%. This approach ensures that every selected security impacts the final AUM, providing a realistic representation of historical performance.
        
        ### Data Sources
            
        - **Supabase**: Cloud-hosted database with historical market data
        - **Excel**: Load data from a local Excel file
            
        ### Disclaimer
            
        This tool is for educational and research purposes only. Past performance does not guarantee future results.
        Always consult with a qualified financial advisor before making investment decisions.
            
        ### Resources
            
        - [Project Repository](https://github.com/cuddihyd-cornell/Factor-Lake/tree/revamped_ux)
            
        ---
            
        **Version:** 1.1.0  
        **Last Updated:** Feb 2026
        """)

if __name__ == "__main__":
    main()
