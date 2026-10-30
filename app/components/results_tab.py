"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/results_tab.py
PURPOSE: Visualization of results with strict adherence to original metrics and formatting.
VERSION: 3.8.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, List, Optional

from Visualizations.portfolio_growth_plot import plot_portfolio_growth
from Visualizations.top_bottom_portfolio_plot import plot_top_bottom_percent
from app.streamlit_config import MAX_SAVED_RUNS
from app.streamlit_utils import (describe_compute, get_backtest_lambda_name, list_all_saved_analyses,
                                 open_saved_analysis, run_cohort)
from src.saved_analyses import SavedAnalysisError
from app.saved_runs import remove_saved_run
from app.holdings_table import (
    FACTOR_COMBINATION_LABELS, build_holdings_display_table, default_holdings_year,
    holdings_number_formats, holdings_period_return, holdings_table_to_csv
)

DIRECTION_LABELS = {'top': 'Higher is better', 'bottom': 'Lower is better'}
# Column headers of the Year-by-Year and Win/Loss tables; WRDS runs pass their own
YEAR_BY_YEAR_COLUMNS = ["Year", "Portfolio Value", "YoY Return", "Benchmark Returns", "Growth Returns",
                        "Value Returns"]
WIN_LOSS_COLUMNS = ['Year', 'Portfolio Return', 'Benchmark Return', 'Outperformed Benchmark',
                    'Growth Index Return', 'Outperformed Growth', 'Value Index Return', 'Outperformed Value']
RUN_TABS_KEY = "saved_run_tabs"
# Copy of the open run tab that survives the Results tab being hidden
OPEN_RUN_TAB_KEY = "saved_run_tabs_open"


def _build_wealth_series(returns: Optional[List[float]], initial: float) -> Optional[List[float]]:
    """
    Converts a yearly return series (in percent) into a cumulative wealth series.
    """
    if not returns:
        return None

    values = [initial]
    for r in returns:
        values.append(values[-1] * (1 + (r / 100.0)))
    return values

def render_saved_runs(saved_runs: List[Dict[str, Any]]) -> None:
    """
    Shows each saved backtest run in its own tab, newest first.

    The tabs are stateful (key + on_change="rerun"), so only the open tab is
    rendered. When a run is added or removed the tab labels change, Streamlit
    treats the tab bar as a new widget and opens the first tab: the newest run.
    The open run is kept while the Results tab is hidden (see below).
    """
    st.header("Portfolio Performance Results")
    st.caption(
        f"Each analysis run is saved as its own tab (newest first). Only the last "
        f"{MAX_SAVED_RUNS} runs are kept, and they are cleared when you refresh the page."
    )
    labels = [run['label'] for run in saved_runs]
    # Only the open main tab is rendered, and Streamlit drops the state of widgets that
    # are not rendered, so restore the open run after the Results tab was hidden. Only for
    # the same runs: after a run is added or removed the newest opens, as before.
    remembered = st.session_state.get(OPEN_RUN_TAB_KEY)
    if (RUN_TABS_KEY not in st.session_state and remembered
            and remembered['labels'] == labels and remembered['open'] in labels):
        st.session_state[RUN_TABS_KEY] = remembered['open']
    tabs = st.tabs(
        labels,
        key=RUN_TABS_KEY,
        on_change="rerun",
    )
    open_label = next((label for tab, label in zip(tabs, labels) if tab.open), labels[0])
    st.session_state[OPEN_RUN_TAB_KEY] = {'labels': labels, 'open': open_label}
    for tab, run in zip(tabs, saved_runs):
        if not tab.open:
            continue
        with tab:
            st.button(
                "Remove this run",
                key=f"remove_run_{run['id']}",
                on_click=_remove_run,
                args=(run['id'],),
            )
            # A run that fails to render must not break the other tabs or its Remove button
            try:
                render_results_tab(run)
            except Exception as e:
                st.error(f"Could not display {run['label']}: {str(e)}")


def _remove_run(run_id: int) -> None:
    """on_click callback: drops one saved run before the script reruns."""
    st.session_state.saved_runs = remove_saved_run(st.session_state.saved_runs, run_id)


def render_results_tab(run: Dict[str, Any]) -> None:
    """
    Renders one saved backtest run.

    Everything shown comes from the run snapshot, never from the live sidebar
    or factor selections, so a saved tab does not change after it is created.
    """
    results = run.get('results')
    if not results:
        st.warning("No backtest results found. Please execute the analysis.")
        return
    if run.get('data_mode') == 'wrds':
        from components.wrds_results import render_wrds_results
        render_wrds_results(run)
        return
    settings = run['settings']

    # 1. Header Information
    _render_header_captions(run)
    
    # 2. Performance Summary
    st.subheader("Performance Summary")
    _render_summary_metrics(results, settings)
    total_delisted = results.get('total_delisted_positions', 0)
    if total_delisted > 0:
        st.caption(f"**Delisted positions encountered:** {total_delisted}")
    st.divider()

    # 3. Holdings the backtest bought each year
    st.subheader("Holdings and Ranks")
    _render_holdings_table(run)
    st.divider()

    # 4. Main Growth Plot
    st.subheader("Portfolio Growth Over Time")
    _render_growth_plot(results, settings, run['factor_labels'])
    st.divider()

    # 5. Year-by-Year Performance
    st.subheader("Year-by-Year Performance")
    _render_year_by_year_table(results)
    st.divider()

    # 6. Top vs Bottom Cohort Analysis
    _render_cohort_analysis_section(run)
    st.divider()

    # 7. Advanced Backtest Statistics
    st.header("Advanced Backtest Statistics")
    _render_advanced_stats_grid(results)
    st.divider()

    # 8. Yearly Win/Loss Summary
    st.header("Yearly Win/Loss Summary")
    _render_win_loss_ledger(results)
    st.divider()


def _render_holdings_table(run: Dict[str, Any]) -> None:
    """
    Displays every stock the backtest held for the rebalance year chosen in
    the year selector (from results['holdings_by_year']), with a CSV download.
    """
    results = run['results']
    holdings_by_year = results.get('holdings_by_year') or {}
    years = sorted(holdings_by_year)

    if not years:
        st.info("No holdings are available for the selected factors and date range.")
        return

    year = _pick_rebalance(run, "Rebalance year", years)
    rows = holdings_by_year[year]
    _render_holdings_caption(results, len(rows), f"{year} to {year + 1}", holdings_period_return(results, year))
    if not rows:
        st.info("The backtest held no stocks this year.")
        return

    holdings_df = build_holdings_display_table(rows, run['factor_labels'])
    _render_holdings_dataframe(run, holdings_df, holdings_number_formats(run['factor_labels']),
                               f"holdings_{year}.csv")


def _pick_rebalance(run: Dict[str, Any], label: str, options: List[Any]) -> Any:
    """The rebalance year (or date, for WRDS runs) picker, newest by default."""
    # Only the open run tab is rendered, and Streamlit drops the state of widgets
    # that are not rendered, so keep a copy to restore the year after a tab switch.
    saved_key = f"ranked_year_saved_{run['id']}"
    saved_year = st.session_state.get(saved_key)
    year = st.selectbox(
        label,
        options=options,
        index=options.index(saved_year if saved_year in options else default_holdings_year(options)),
        key=f"ranked_year_{run['id']}",
    )
    st.session_state[saved_key] = year
    return year


def _render_holdings_caption(results: Dict[str, Any], count: int, period: str,
                             period_return: Optional[float]) -> None:
    """Method, number of holdings and the portfolio's return over the holding period."""
    method = FACTOR_COMBINATION_LABELS.get(results.get('factor_combination', 'mixing'), 'Mixing')
    return_str = f"{period_return:.2f}%" if period_return is not None else "n/a"
    st.caption(
        f"**Method:** {method} | **Holdings:** {count} | "
        f"**Portfolio return ({period}):** {return_str}"
    )


def _render_holdings_dataframe(run: Dict[str, Any], holdings_df: pd.DataFrame, number_formats: Dict[str, str],
                               file_name: str, csv_key: str = "ranked_csv") -> None:
    """A holdings table, rounded on screen only, with its CSV download."""
    # Round on screen only; the data (and the CSV) keep full precision
    column_config = {
        col: st.column_config.NumberColumn(format=fmt)
        for col, fmt in number_formats.items()
        if col in holdings_df.columns
    }
    st.dataframe(holdings_df, width="stretch", hide_index=True, column_config=column_config)

    st.download_button(
        "Download CSV",
        data=holdings_table_to_csv(holdings_df),
        file_name=file_name,
        mime="text/csv",
        key=f"{csv_key}_{run['id']}",
        on_click="ignore",
    )

def _render_year_by_year_table(res: Dict[str, Any], periods: Optional[List[Any]] = None,
                               columns: List[str] = YEAR_BY_YEAR_COLUMNS) -> None:
    """
    Generates the primary performance table with standardized financial headers.
    
    The table includes longitudinal portfolio values and relative returns against 
    core market benchmarks (Russell 2000, Growth, and Value). WRDS runs pass their
    periods' labels and their own column headers.
    """
    years = res.get('years', [])
    port_vals = res.get('portfolio_values', [])
    p_rets = res.get('yearly_returns', [])
    b_rets = res.get('benchmark_returns', [])
    g_rets = res.get('growth_benchmark_returns', [])
    v_rets = res.get('value_benchmark_returns', [])

    # Align data lengths for consistent reporting
    min_len = len(p_rets)
    
    df = pd.DataFrame(dict(zip(columns, [
        [int(y) for y in years[:min_len]] if periods is None else periods,
        [f"${v:,.2f}" for v in port_vals[1:min_len+1]],
        [f"{r:.2f}%" for r in p_rets],
        [f"{r:.2f}%" for r in b_rets[:min_len]],
        [f"{r:.2f}%" for r in g_rets[:min_len]],
        [f"{r:.2f}%" for r in v_rets[:min_len]]
    ])))
    st.dataframe(df, width="stretch", hide_index=True)

def _render_win_loss_ledger(res: Dict[str, Any], columns: List[str] = WIN_LOSS_COLUMNS,
                            period: Callable[[Dict[str, Any]], Any] = lambda c: int(c.get('year'))) -> None:
    """
    Displays a comprehensive win/loss ledger using boolean performance indicators.

    This ledger compares portfolio returns against multiple indices on a yearly
    basis, highlighting alpha generation through visual cues. WRDS runs pass their
    own column headers and label each row by its holding period.
    """
    if 'yearly_comparisons' in res:
        comps = res['yearly_comparisons']
        df = pd.DataFrame(dict(zip(columns, [
            [period(c) for c in comps],
            [f"{c.get('p_ret', 0):.2f}%" for c in comps],
            [f"{c.get('b_ret', 0):.2f}%" for c in comps],
            ['✅' if c.get('b_win') else '❌' for c in comps],
            [f"{c.get('g_ret', 0):.2f}%" for c in comps],
            ['✅' if c.get('g_win') else '❌' for c in comps],
            [f"{c.get('v_ret', 0):.2f}%" for c in comps],
            ['✅' if c.get('v_win') else '❌' for c in comps],
        ])))
        st.dataframe(df, width="stretch", hide_index=True)

def _render_advanced_stats_grid(res: Dict[str, Any], index: str = "R2000", period: str = "Yearly",
                                rf_label: Optional[str] = None) -> None:
    """
    Renders a multi-row grid containing advanced risk and efficiency metrics.

    WRDS runs name their benchmark (index="IWM") and their period ("Quarterly"), and
    give a short risk-free label (rf_label) whose tooltip holds the full description.
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
    b1.metric(f"Beta vs {index}", f"{res.get('portfolio_beta', 0):.3f}")
    b2.metric("Beta vs Growth", f"{res.get('portfolio_beta_growth', 0):.3f}")
    b3.metric("Beta vs Value", f"{res.get('portfolio_beta_value', 0):.3f}")
    if rf_label:
        # A short label, with the full description in the tooltip
        b4.markdown(f"**RF Rate Source**\n\n{rf_label}", help=res.get('risk_free_rate_source'))
    else:
        b4.write(f"**RF Rate Source**\n\n{res.get('risk_free_rate_source', 'FRED')}")

    # Row 6: Win Rates (Restored)
    w1, w2, w3, w4 = st.columns(4)
    w1.metric(f"{period} Win Rate (vs {index})", f"{res.get('win_rate', 0):.2%}")
    w2.metric(f"{period} Win Rate (vs Growth)", f"{res.get('win_rate_growth', 0):.2%}")
    w3.metric(f"{period} Win Rate (vs Value)", f"{res.get('win_rate_value', 0):.2%}")
    w4.empty()

    # Row 7: IR
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Information Ratio", f"{res.get('information_ratio', 0):.4f}")
    i2.metric("IR (vs Growth)", f"{res.get('information_ratio_growth', 0):.4f}")
    i3.metric("IR (vs Value)", f"{res.get('information_ratio_value', 0):.4f}")
    i4.empty()


def _render_header_captions(run: Dict[str, Any], data_source: Optional[str] = None,
                            delisting: Optional[str] = None) -> None:
    """
    Renders the settings this run was created with (from its snapshot).

    WRDS runs add a data source caption and describe their delisting treatment.
    """
    settings = run['settings']
    directions = run['factor_directions']
    factors_str = ", ".join([
        f"{label} ({DIRECTION_LABELS.get(directions.get(col, 'top'), 'Higher is better')})"
        for label, col in zip(run['factor_labels'], run['factors'])
    ])
    weighting_str = "Market Cap Weighted" if settings.get('use_market_cap_weight') else "Equal Weighted"
    delisting_labels = {
        "zero_return": "Zero Return",
        "hold_cash": "Hold Cash (risk-free rate)",
        "reinvest": "Reinvest (pro-rata)"
    }
    delisting_str = delisting or delisting_labels.get(
        settings.get('delisting_strategy', 'zero_return'), 'Zero Return'
    )

    st.caption(f"**Factors:** {factors_str}")
    st.caption(
        f"**Period:** {int(settings['start_year'])}-{int(settings['end_year'])} | "
        f"**Initial AUM:** ${settings['initial_aum']:,.0f}"
    )
    if data_source:
        st.caption(data_source)
    universe = run.get('universe') or {}
    if universe:
        sectors = universe.get('selected_sectors') or []
        sectors_str = ", ".join(sectors) if sectors else "All sectors"
        fossil_str = "excluded" if universe.get('restrict_fossil_fuels') else "included"
        st.caption(f"**Universe:** {sectors_str} | Fossil fuels {fossil_str}")
    combination = settings.get('factor_combination', 'mixing')
    combination_descriptions = {
        'mixing': "Mixing (one sleeve per factor)",
        'integrating': "Integrating (top stocks by composite score)",
    }
    st.caption(f"**Factor Combination:** {combination_descriptions.get(combination, combination_descriptions['mixing'])}")
    st.caption(f"**Weighting:** {weighting_str}")
    st.caption(f"**Delisting Strategy:** {delisting_str}")
    compute = run.get('compute')
    if compute:
        how = describe_compute(compute)
        st.caption(f"**Computed:** {how[:1].upper() + how[1:]}")
    _render_analysis_id(compute)


def _render_analysis_id(compute: Optional[Dict[str, Any]]) -> None:
    """The run's analysis ID, with a copy button (st.code's), to reopen or share it."""
    analysis_id = (compute or {}).get('hash')
    if analysis_id:
        st.caption("**Analysis ID:** copy it (button on the right) to open this analysis again "
                   "or share it, under Open a saved analysis.")
        st.code(analysis_id, language=None)
    elif get_backtest_lambda_name():
        st.caption("**Analysis ID:** none. This run was computed in the app, so it was not saved. "
                   "Turn on **Run backtests in AWS Lambda** to get an ID you can share.")


def render_open_saved_analysis(has_runs: bool) -> None:
    """
    The "Open a saved analysis" box: loads a saved analysis by its ID as a new Results
    tab (newest first, so it opens), or shows why it cannot be opened.
    """
    with st.expander("Open a saved analysis", expanded=not has_runs, key="open_analysis_expander"):
        st.caption("Paste an analysis ID from your history or from a colleague to see that analysis "
                   "with its original results, factors and settings.")
        analysis_id = st.text_input("Analysis ID", key="open_analysis_id",
                                    placeholder="64-character analysis ID")
        if st.button("Open analysis", key="open_analysis_btn"):
            try:
                with st.spinner("Loading the saved analysis from S3..."):
                    run = open_saved_analysis(analysis_id)
                st.success(f"Opened as \"{run['label']}\".")
            except SavedAnalysisError as e:
                st.error(str(e))


def render_analysis_history(history: List[Dict[str, Any]]) -> None:
    """This session's analyses, newest first, with their IDs (only this session's: nothing is listed from S3)."""
    if not history:
        return
    with st.expander(f"This session's analyses ({len(history)})", key="analysis_history_expander"):
        st.caption("Only analyses run or opened in this browser session. Copy an ID to reopen or share it.")
        for entry in reversed(history):
            st.markdown(f"**{entry['label']}** · {entry['how']} at {entry['time']}")
            _caption_summary(entry)
            if entry.get('analysis_id'):
                st.code(entry['analysis_id'], language=None)
            else:
                st.caption("Not saved (computed in this app), so it has no ID.")


# How many rows each list of saved analyses shows
LISTED_ANALYSES = 20
# Message shown at the top of Results after opening an analysis from a list
OPENED_MESSAGE_KEY = "opened_analysis_message"


def render_earlier_analyses(entries: Optional[List[Dict[str, Any]]]) -> None:
    """
    Analyses from earlier sessions in this browser (see app/analysis_history.py), newest
    first, each with its ID and an Open button. Hidden until there is one.
    """
    if not entries:
        return
    with st.expander(f"Earlier analyses on this browser ({len(entries)})", key="earlier_analyses_expander"):
        st.caption("Analyses you ran or opened in earlier sessions on this browser. They are kept in this "
                   "browser only, so they don't appear on another computer or after clearing site data.")
        for entry in entries[:LISTED_ANALYSES]:
            when = (entry.get('saved_at') or '').replace('T', ' ').rstrip('Z')
            st.markdown(f"**{entry.get('label', 'Analysis')}** · {entry.get('how', 'saved')} on {when} UTC")
            _caption_summary(entry)
            _render_listed_analysis(entry['analysis_id'], 'earlier')
        _caption_hidden_rows(len(entries))


def render_all_saved_analyses() -> None:
    """
    Everyone's saved analyses in S3, newest first, loaded only while the expander is open
    (it reruns the app when opened or closed).
    """
    expander = st.expander("All saved analyses", key="all_analyses_expander", on_change="rerun")
    with expander:
        if not expander.open:
            return
        st.caption("Every analysis saved by anyone using this app, newest first. Saved analyses are "
                   "deleted after 90 days.")
        try:
            with st.spinner("Listing saved analyses in S3..."):
                listing = list_all_saved_analyses(LISTED_ANALYSES)
        except SavedAnalysisError as e:
            st.error(str(e))
            return
        if not listing['entries']:
            st.info("No saved analyses yet. Run one with AWS Lambda backtests on.")
            return
        for entry in listing['entries']:
            summary = entry['summary'] or "Details not recorded (saved before listings had summaries)"
            st.markdown(f"**{summary}** · saved {entry['saved_at']:%Y-%m-%d %H:%M} UTC")
            _render_listed_analysis(entry['analysis_id'], 'all')
        if listing['total'] > len(listing['entries']):
            st.caption(f"Showing the newest {len(listing['entries'])}. Older ones can still be opened by their ID.")


def _render_listed_analysis(analysis_id: str, list_name: str) -> None:
    """One listed analysis: its ID (with a copy button) and an Open button."""
    id_col, button_col = st.columns([5, 1], vertical_alignment="center")
    with id_col:
        st.code(analysis_id, language=None)
    with button_col:
        if st.button("Open", key=f"open_{list_name}_{analysis_id}"):
            try:
                run = open_saved_analysis(analysis_id)
            except SavedAnalysisError as e:
                st.error(str(e))
                return
            # Rerun so the lists above show the analysis as this session's, with the message on top
            st.session_state[OPENED_MESSAGE_KEY] = f"Opened as \"{run['label']}\"."
            st.rerun()


def _caption_summary(entry: Dict[str, Any]) -> None:
    """A WRDS entry's summary (e.g. "WRDS, Quarterly | ROE (higher), ..."); Legacy entries have none."""
    if entry.get('summary'):
        st.caption(entry['summary'])


def _caption_hidden_rows(total: int) -> None:
    if total > LISTED_ANALYSES:
        st.caption(f"Showing the newest {LISTED_ANALYSES} of {total}.")

def _render_summary_metrics(res: Dict[str, Any], settings: Dict[str, Any],
                            outperformance_help: Optional[str] = None) -> None:
    """
    Calculates and displays core summary metrics for the reporting period.

    The CAGR counts years as periods / periods_per_year (WRDS runs have 2, 4 or 12).
    """
    col1, col2, col3, col4 = st.columns(4)
    initial_aum = settings.get('initial_aum', 1000000.0)
    final_value = res.get('final_value', initial_aum)
    
    n_years = len(res.get('yearly_returns', [])) / res.get('periods_per_year', 1)
    cagr = (((final_value / initial_aum) ** (1 / n_years)) - 1) if n_years > 0 else 0
    
    bench_rets = res.get('benchmark_returns', [])
    benchmark_final = initial_aum * np.prod([1 + (r / 100.0) for r in bench_rets])
    cum_outperformance = ((final_value / benchmark_final) - 1)
    
    col1.metric("Final Portfolio Value", f"${final_value:,.2f}")
    col2.metric("Total Return", f"{((final_value/initial_aum)-1):.2%}")
    col3.metric("CAGR", f"{cagr:.2%}")
    col4.metric("Cumulative Outperformance", f"{cum_outperformance:.2%}", help=outperformance_help)

def _render_cohort_analysis_section(run: Dict[str, Any], n_years: Optional[float] = None,
                                    chart: Optional[Dict[str, Any]] = None) -> None:
    """
    Runs and shows a comparison between the top and bottom factor cohorts.

    WRDS runs pass their number of years for the CAGR and their chart options
    (benchmark values and labels, a date axis).

    Uses the run's own data, factor columns, directions and settings (in AWS
    Lambda for Lambda runs, see run_cohort). The
    result is stored on the run, so it is still shown after reruns and tab
    switches. Widget keys include the run id so every run has its own controls.
    """
    results = run['results']
    settings = run['settings']
    run_id = run['id']
    st.subheader("Top vs Bottom Cohort Analysis")
    # The key makes the browser remember that the user opened the expander. Without
    # one it follows `expanded`, which is still False in the run that computes the
    # comparison, so the result appeared inside a collapsed expander.
    with st.expander("Run Cohort Comparison", expanded=run.get('cohort') is not None,
                     key=f"cohort_expander_{run_id}"):
        col1, col2 = st.columns(2)
        with col1:
            # Keep a copy so the chosen percentage survives the tab being hidden
            # (Streamlit drops the state of widgets that are not rendered)
            saved_pct_key = f"cohort_pct_saved_{run_id}"
            default_pct = run['cohort']['pct'] if run.get('cohort') else 10
            cohort_pct = st.slider(
                "Cohort Percentage", 1, 50,
                value=st.session_state.get(saved_pct_key, default_pct),
                key=f"cohort_pct_{run_id}",
            )
            st.session_state[saved_pct_key] = cohort_pct

        if st.button("Generate Comparison", type="primary", key=f"cohort_btn_{run_id}"):
            spinner = "Running the comparison in AWS Lambda..." if run.get('data') is None else "Running the comparison..."
            try:
                with st.spinner(spinner):
                    res_top, res_bot, compute = run_cohort(run, cohort_pct)
                run['cohort'] = {'pct': cohort_pct, 'top': res_top, 'bottom': res_bot, 'compute': compute}
            except Exception as e:
                st.error(f"Cohort Comparison Error: {str(e)}")

        cohort = run.get('cohort')
        if not cohort:
            return
        res_top, res_bot = cohort['top'], cohort['bottom']
        if cohort.get('compute'):
            st.caption(f"Comparison {describe_compute(cohort['compute'])}.")

        # --- THE 4 STATS BLOCK ---
        st.write(f"### Cohort Performance Summary (top vs bottom {cohort['pct']}%)")
        init_aum = settings['initial_aum']
        num_years = len(results['years']) if n_years is None else n_years

        stats_df = pd.DataFrame({
            "Metric": ["Total Return (%)", "Final Value ($)", "CAGR (%)"],
            "Top Cohort": [
                f"{((res_top[-1] / init_aum) - 1) * 100:.2f}%",
                f"${res_top[-1]:,.0f}",
                f"{(((res_top[-1] / init_aum) ** (1 / num_years)) - 1) * 100:.2f}%"
            ],
            "Bottom Cohort": [
                f"{((res_bot[-1] / init_aum) - 1) * 100:.2f}%",
                f"${res_bot[-1]:,.0f}",
                f"{(((res_bot[-1] / init_aum) ** (1 / num_years)) - 1) * 100:.2f}%"
            ]
        })
        st.table(stats_df)
        # -------------------------

        benchmarks = dict(
            benchmark_returns=results.get('benchmark_returns'),
            value_returns=_build_wealth_series(results.get('value_benchmark_returns'), init_aum),
            growth_returns=_build_wealth_series(results.get('growth_benchmark_returns'), init_aum)
        )
        fig = plot_top_bottom_percent(
            years=results['years'],
            percent=cohort['pct'],
            show_bottom=True,
            initial_investment=init_aum,
            baseline_portfolio_values=results['portfolio_values'],
            precomputed_top=res_top,
            precomputed_bot=res_bot,
            **{**benchmarks, **(chart or {})}
        )
        st.plotly_chart(fig, width="stretch")

def _render_growth_plot(res: Dict[str, Any], settings: Dict[str, Any], factor_labels: List[str],
                        **chart: Any) -> None:
    """
    Visualizes the growth of $1 (or initial AUM) across the portfolio and benchmarks.

    WRDS runs pass chart options (benchmark names, a date axis) to plot_portfolio_growth.
    """
    initial = settings.get('initial_aum', 1000000.0)
    
    fig = plot_portfolio_growth(
        years=res.get('years', []),
        port_vals=res.get('portfolio_values', []),
        bench_vals=_build_wealth_series(res.get('benchmark_returns'), initial),
        val_vals=_build_wealth_series(res.get('value_benchmark_returns'), initial),
        gro_vals=_build_wealth_series(res.get('growth_benchmark_returns'), initial),
        factor_names=factor_labels,
        **chart
    )
    st.plotly_chart(fig, width="stretch")
