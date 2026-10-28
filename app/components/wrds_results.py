"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/wrds_results.py
PURPOSE: Results of a WRDS run, shown like Legacy's: the same sections, order, titles, widgets,
         tables and charts, drawn by Legacy's own render functions (results_tab.py) with the
         WRDS result passed in the shape they expect. WRDS adds the period returns chart,
         the calendar-year view and the ranked stocks of a rebalance date.
VERSION: 3.0.0

Everything comes from the run's snapshot (and, for holdings, the run's holdings file in
S3, read one date at a time), never from the live sidebar. See DOCS/WRDS_MODE_DESIGN.md,
section 8.
"""

from typing import Any, Dict

import streamlit as st

from components.results_tab import (_build_wealth_series, _pick_rebalance, _render_advanced_stats_grid,
                                    _render_cohort_analysis_section, _render_growth_plot,
                                    _render_header_captions, _render_holdings_caption,
                                    _render_holdings_dataframe, _render_summary_metrics,
                                    _render_win_loss_ledger, _render_year_by_year_table)
from app.streamlit_utils import describe_compute, load_wrds_holdings, run_wrds_ranked
from app.wrds_tables import (FREQUENCY_LABELS, benchmark_label, calendar_year_results, frequency_caption,
                             holdings_display_table, period_by_period_columns, period_label,
                             ranked_display_table, result_notes, short_benchmark, table_number_formats,
                             win_loss_columns)
from Visualizations.wrds_plots import plot_wrds_period_returns

WRDS_DELISTING = "CRSP delisting returns included, then the risk-free rate until the next rebalance"


def render_wrds_results(run: Dict[str, Any]) -> None:
    """One WRDS run, section by section as Legacy's render_results_tab."""
    results, settings = run['results'], run['settings']
    frequency = FREQUENCY_LABELS.get(results['frequency'], results['frequency'])
    names = {role: benchmark_label(results, role) for role in ('benchmark', 'value', 'growth')}
    dates = results['rebalance_dates']

    # 1. Header Information
    _render_header_captions(
        run,
        data_source=f"**Data Source:** WRDS | **Rebalancing:** {frequency} "
                    f"({len(dates)} rebalances, {dates[0]} to {dates[-1]})",
        delisting=WRDS_DELISTING,
    )

    # 2. Performance Summary
    st.subheader("Performance Summary")
    _render_summary_metrics(results, settings,
                            outperformance_help=f"Against the same initial AUM invested in {names['benchmark']}.")
    total_delisted = results.get('total_delisted_positions', 0)
    if total_delisted > 0:
        st.caption(f"**Delisted positions encountered:** {total_delisted}")
    for note in result_notes(results):
        st.caption(note)
    st.divider()

    # 3. Holdings bought at each rebalance date
    st.subheader("Holdings and Ranks")
    _render_holdings(run)
    st.divider()

    # 4. Growth plot, then each period's return
    st.subheader("Portfolio Growth Over Time")
    _render_growth_plot(results, settings, run['factor_labels'], names=names, dates=True)
    st.plotly_chart(plot_wrds_period_returns(results['period_end_dates'], results['yearly_returns'],
                                             results['benchmark_returns'], names['benchmark'], frequency),
                    width="stretch")
    st.divider()

    # 5. Period-by-Period Performance
    st.subheader("Period-by-Period Performance")
    _render_period_table(run, frequency)
    st.divider()

    # 6. Top vs Bottom Cohort Analysis, at the run's frequency on its dates
    initial = float(settings['initial_aum'])
    _render_cohort_analysis_section(
        run,
        n_years=len(results['yearly_returns']) / int(results['periods_per_year']),
        chart=dict(benchmark_returns=_build_wealth_series(results['benchmark_returns'], initial),
                   benchmark_label=names['benchmark'], growth_label=names['growth'],
                   value_label=names['value'], dates=True),
    )
    st.divider()

    # 7. Advanced Backtest Statistics
    st.header("Advanced Backtest Statistics")
    st.caption(frequency_caption(results))
    _render_advanced_stats_grid(results, index=short_benchmark(results, 'benchmark'), period=frequency)
    st.divider()

    # 8. Win/Loss Summary, one row per period
    st.header(f"{frequency} Win/Loss Summary")
    _render_win_loss_ledger(results, columns=win_loss_columns(results),
                            period=lambda c: period_label(c['period_start'], c['period_end']))
    st.divider()


def _labels(run: Dict[str, Any]) -> Dict[str, str]:
    """Factor column -> display name."""
    return dict(zip(run['factors'], run['factor_labels']))


def _render_holdings(run: Dict[str, Any]) -> None:
    """
    The stocks held from the picked rebalance date (only that date is read from S3), as
    Legacy's holdings table, then every stock ranked on that date.
    """
    results, labels = run['results'], _labels(run)
    dates = list(results['rebalance_dates'])
    date = _pick_rebalance(run, "Rebalance date", dates)
    i = dates.index(date)
    period, period_return = period_label(date, results['period_end_dates'][i]), results['yearly_returns'][i]
    holdings = results.get('holdings') or {}
    if date not in holdings.get('dates', []):
        _render_holdings_caption(results, 0, period, period_return)
        st.info("The backtest held no stocks on this date; the money earned the risk-free rate.")
    else:
        try:
            raw = load_wrds_holdings(holdings['file'], holdings['dates'].index(date), date)
        except Exception as e:
            st.error(f"Could not read the holdings for {date}: {e}")
            raw = None
        if raw is not None:
            table = holdings_display_table(raw, run['factors'], labels)
            _render_holdings_caption(results, len(table), period, period_return)
            _render_holdings_dataframe(run, table, table_number_formats(table, run['factors'], labels),
                                       f"holdings_{date}.csv")

    with st.expander("All ranked stocks on this date", expanded=False, key=f"wrds_ranked_expander_{run['id']}"):
        st.caption("Every stock the strategy could buy on this date, ranked by the composite score "
                   "(the mean of the factors' z-scores). Computed in AWS Lambda.")
        ranked_key = f"wrds_ranked_{run['id']}"
        if st.button("Rank every stock on this date", key=f"wrds_rank_btn_{run['id']}"):
            try:
                with st.spinner("Ranking the stocks in AWS Lambda..."):
                    rows, info = run_wrds_ranked(run, date[:7])
                st.session_state[ranked_key] = {'date': date, 'rows': rows, 'compute': info}
            except Exception as e:
                st.error(f"Ranked List Error: {e}")
        ranked = st.session_state.get(ranked_key)
        if ranked and ranked['date'] == date:
            table = ranked_display_table(ranked['rows'], run['factors'], labels, results['factor_combination'])
            st.caption(f"**Ranked stocks:** {len(table)} | **Held:** {int(table['Held'].sum())} | "
                       f"Ranking {describe_compute(ranked['compute'])}.")
            _render_holdings_dataframe(run, table, table_number_formats(table, run['factors'], labels),
                                       f"ranked_{date}.csv", csv_key="wrds_ranked_csv")


def _render_period_table(run: Dict[str, Any], frequency: str) -> None:
    """Legacy's Year-by-Year table with one row per period, or per calendar year."""
    results = run['results']
    view = st.segmented_control("View", options=["By period", "By calendar year"], default="By period",
                                key=f"wrds_period_view_{run['id']}", label_visibility="collapsed")
    if view == "By calendar year":
        years, by_year = calendar_year_results(results)
        st.caption(f"Each calendar year's {frequency.lower()} periods compounded. The first year starts at "
                   f"{results['years'][0]} and the last ends at {results['years'][-1]}, so they can be partial.")
        _render_year_by_year_table(by_year, periods=years,
                                   columns=period_by_period_columns(results, 'Year', 'YoY Return'))
    else:
        n = len(results['yearly_returns'])
        periods = [period_label(s, e) for s, e in zip(results['rebalance_dates'][:n], results['period_end_dates'][:n])]
        _render_year_by_year_table(results, periods=periods, columns=period_by_period_columns(results))
