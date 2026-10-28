"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/sidebar.py
PURPOSE: Configuration interface for global backtest parameters and ESG filters.
VERSION: 2.7.0
"""

import streamlit as st
import streamlit_config as config
from typing import Callable, Dict, Any, List, Optional

from src.wrds_calendar import FIRST_YEAR as WRDS_FIRST_YEAR, LAST_YEAR as WRDS_LAST_YEAR, describe_schedule
from src.wrds_factors import WRDS_SECTORS

# Widget key of the Legacy / WRDS switch
DATA_MODE_KEY = "data_mode_input"
WRDS_NEEDS_LAMBDA = "WRDS mode runs in AWS Lambda, which is not configured for this app."


def _render_data_mode(backtest_lambda: str, on_change: Optional[Callable[[], None]]) -> bool:
    """
    The Legacy / WRDS switch. WRDS runs only in AWS Lambda, so without a configured
    Lambda the switch stays on Legacy and is disabled. Returns True for WRDS.
    """
    if not backtest_lambda and st.session_state.get(DATA_MODE_KEY) == 'WRDS':
        st.session_state[DATA_MODE_KEY] = 'Legacy'
    choice = st.radio(
        "Data source",
        options=config.DATA_MODE_OPTIONS,
        key=DATA_MODE_KEY,
        horizontal=True,
        label_visibility="collapsed",
        disabled=not backtest_lambda,
        on_change=on_change,
        help="Legacy: annual rebalancing on the 9/30 data, as before. WRDS: point-in-time WRDS data "
             "with semi-annual, quarterly or monthly rebalancing, run in AWS Lambda. "
             "Switching clears the selected factors."
    )
    if not backtest_lambda:
        st.caption(WRDS_NEEDS_LAMBDA)
    return choice == 'WRDS'


def render_sidebar(sector_options: List[str], data_source_label: str,
                   backtest_lambda: str = '',
                   on_data_mode_change: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
    """
    Constructs the application sidebar to capture user configuration settings.

    This interface manages global constraints, including date ranges, initial
    capitalization, sector exposures, and weighting methodologies.

    backtest_lambda is the configured backtest Lambda's name ('' for none); when
    set, a switch chooses whether backtests run there or in the app, and the WRDS
    data source can be chosen. on_data_mode_change runs when the data source changes.

    Returns:
        Dict[str, Any]: Standardized dictionary of user-defined parameters. Legacy
            settings are exactly as before WRDS mode (no data_mode key); WRDS settings
            have data_mode 'wrds' and frequency, and no delisting strategy.
    """
    with st.sidebar:
        st.header("Configuration")

        # Data Source Information
        st.subheader("Data Source")
        st.caption(f"Primary Source: {data_source_label}")
        wrds = _render_data_mode(backtest_lambda, on_data_mode_change)

        # Where backtests run
        use_lambda = False
        if wrds:
            use_lambda = True
            st.caption(f"WRDS backtests run in AWS Lambda: `{backtest_lambda}`")
        elif backtest_lambda:
            use_lambda = st.toggle(
                "Run backtests in AWS Lambda",
                value=True,
                key="use_lambda_input",
                help="On: the app sends your settings to AWS Lambda, which reads the market data "
                     "from S3, runs the backtest and caches the result. Off: the app loads the data "
                     "and runs the backtest itself."
            )
            st.caption(f"Lambda function: `{backtest_lambda}`" if use_lambda
                       else "Backtests run in this app (load the market data first).")

        st.divider()
        
        # ESG & Ethical Filtering
        st.subheader("ESG Filters")
        restrict_fossil_fuels = st.checkbox(
            "Restrict Fossil Fuel Companies",
            value=False,
            help="Exclude coal, oil, gas and petroleum companies."
        )
        
        st.divider()
        
        # Allocation Methodology
        st.subheader("Portfolio Weighting")
        weighting_method = st.radio(
            "Select weighting method:",
            options=["Equal Weight", "Market Cap Weight"],
            index=0,
            help="Equal: same dollars per stock. Market Cap: sized by company value."
        )
        use_market_cap_weight = (weighting_method == "Market Cap Weight")
        
        if use_market_cap_weight:
            st.info("Portfolio will be weighted by market capitalization, aligning with the Russell 2000 methodology.")

        st.divider()

        # Factor Combination
        st.subheader("Factor Combination")
        combination_method = st.radio(
            "Combine factors",
            options=["Mixing", "Integrating"],
            index=0,
            help=(
                "Mixing: each factor buys its own top stocks with an equal share of the money; "
                "a stock picked by two factors is held twice. "
                "Integrating: stocks are ranked by their average score across all factors and "
                "the top ones are bought once each; stocks missing any factor are left out."
            ),
            key="factor_combination_input"
        )
        factor_combination = combination_method.lower()

        st.divider()

        if wrds:
            # Rebalancing (WRDS returns include delisting returns, so there is no delisting option)
            st.subheader("Rebalancing")
            frequency_label = st.radio(
                "Rebalancing frequency",
                options=list(config.WRDS_FREQUENCY_OPTIONS),
                index=list(config.WRDS_FREQUENCY_OPTIONS).index(config.WRDS_DEFAULT_FREQUENCY),
                key="wrds_frequency_input",
                help="How often the portfolio is rebuilt, at month ends: Semi-annual (Jun, Dec), "
                     "Quarterly (Mar, Jun, Sep, Dec) or Monthly. Returns include CRSP delisting "
                     "returns, so there is no delisting option."
            )
            frequency = config.WRDS_FREQUENCY_OPTIONS[frequency_label]
        else:
            # Delisting Strategy
            st.subheader("Delisting Strategy")
            delisting_strategy = st.radio(
                "How to treat delisted positions:",
                options=["Zero Return", "Hold Cash", "Reinvest"],
                index=0,
                help="Zero Return: earns 0%. Hold Cash: earns the risk free rate. Reinvest: spread across remaining stocks."
            )
            delisting_map = {"Zero Return": "zero_return", "Hold Cash": "hold_cash", "Reinvest": "reinvest"}
            delisting_strategy_key = delisting_map[delisting_strategy]

        st.divider()

        # Sector Exposure Configuration
        st.subheader("Sector Selection")
        sector_filter_enabled = st.checkbox(
            "Enable Sector Filter",
            value=False,
            help="Limit the portfolio to chosen sectors."
        )
        selected_sectors = []
        if sector_filter_enabled:
            if wrds:
                selected_sectors = st.multiselect(
                    "Include following sectors:",
                    options=list(WRDS_SECTORS),
                    default=list(WRDS_SECTORS),
                    key="wrds_sectors_input",
                    help="Only stocks in these GICS sectors can be held. Sectors use today's GICS "
                         "codes, not historical ones, and stocks without a sector are left out "
                         "while this filter is on."
                )
            else:
                selected_sectors = st.multiselect(
                    "Include following sectors:",
                    options=sector_options,
                    default=sector_options,
                    help="Only stocks in these sectors can be held."
                )

        st.divider()

        # Analysis Temporal Range
        st.subheader("Analysis Period")
        if wrds:
            start_year, end_year = _render_wrds_period(frequency)
        else:
            start_year, end_year = _render_legacy_period()

        st.divider()

        # Capitalization
        st.subheader("Initial Investment")
        initial_aum = st.number_input(
            "Initial AUM ($)",
            min_value=0.0,
            max_value=1000000000.0,
            value=1000.0,
            step=100.0,
            format="%.0f",
            help="Starting portfolio value."
        )

    if wrds:
        return {
            "data_mode": "wrds",
            "frequency": frequency,
            "restrict_fossil_fuels": restrict_fossil_fuels,
            "use_market_cap_weight": use_market_cap_weight,
            "factor_combination": factor_combination,
            "selected_sectors": selected_sectors if sector_filter_enabled else None,
            "sector_filter_enabled": sector_filter_enabled,
            "start_year": start_year,
            "end_year": end_year,
            "initial_aum": initial_aum,
            "use_lambda": use_lambda
        }
    return {
        "restrict_fossil_fuels": restrict_fossil_fuels,
        "use_market_cap_weight": use_market_cap_weight,
        "factor_combination": factor_combination,
        "selected_sectors": selected_sectors if sector_filter_enabled else None,
        "sector_filter_enabled": sector_filter_enabled,
        "start_year": start_year,
        "end_year": end_year,
        "initial_aum": initial_aum,
        "delisting_strategy": delisting_strategy_key,
        "use_lambda": use_lambda
    }


def _render_wrds_period(frequency: str):
    """WRDS start and end years, limited to the data, with the rebalances they give."""
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start Year",
            min_value=WRDS_FIRST_YEAR,
            max_value=WRDS_LAST_YEAR,
            value=WRDS_FIRST_YEAR,
            step=1,
            format="%d",
            help="First calendar year of the backtest. The WRDS data starts in April 2008.",
            key="wrds_start_year_input"
        )
    with col2:
        end_year = st.number_input(
            "End Year",
            min_value=WRDS_FIRST_YEAR,
            max_value=WRDS_LAST_YEAR,
            value=WRDS_LAST_YEAR,
            step=1,
            format="%d",
            help="Last calendar year of the backtest. The WRDS data ends in September 2026.",
            key="wrds_end_year_input"
        )

    if start_year > end_year:
        st.warning('Start Year exceeded End Year. Synchronizing dates...')
        end_year = start_year
    try:
        st.caption(describe_schedule(frequency, start_year, end_year).capitalize() + ".")
    except ValueError as e:
        st.warning(str(e))
    return start_year, end_year


def _render_legacy_period():
    """Legacy start and end years, exactly as before WRDS mode."""
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start Year",
            min_value=2002,
            max_value=2023,
            value=2002,
            step=1,
            format="%d",
            help="First year of the backtest. Earliest is 2008.",
            key="start_year_input"
        )
    with col2:
        end_year = st.number_input(
            "End Year",
            min_value=2002,
            max_value=2024,
            value=2024,
            step=1,
            format="%d",
            help="Last year of the backtest. Latest is 2024.",
            key="end_year_input"
        )

    # Logic to ensure chronological consistency
    if start_year > end_year:
        st.warning('Start Year exceeded End Year. Synchronizing dates...')
        end_year = start_year
    return start_year, end_year