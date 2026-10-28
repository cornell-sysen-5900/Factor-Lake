"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/wrds_results.py
PURPOSE: Results of a WRDS run, laid out like Legacy's: summary, holdings and ranked stocks per
         rebalance date, charts on a date axis, period table, cohort and statistics.
VERSION: 2.0.0

Everything comes from the run's snapshot (and, for holdings, the run's holdings file in
S3, read one date at a time), never from the live sidebar. See DOCS/WRDS_MODE_DESIGN.md,
section 8.
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.holdings_table import FACTOR_COMBINATION_LABELS, holdings_table_to_csv
from app.streamlit_utils import describe_compute, load_wrds_holdings, run_cohort, run_wrds_ranked
from app.wrds_tables import (FREQUENCY_LABELS, benchmark_label, calendar_year_table, drawdowns,
                             frequency_caption, holdings_display_table, holdings_period_caption, metrics_table,
                             period_table, ranked_display_table, result_notes, short_benchmark,
                             table_number_formats, wealth_series)
from Visualizations.wrds_plots import (plot_wrds_cohort, plot_wrds_drawdown, plot_wrds_growth,
                                       plot_wrds_period_returns)

DIRECTION_LABELS = {'top': 'Higher is better', 'bottom': 'Lower is better'}


def render_wrds_results(run: Dict[str, Any]) -> None:
    """One WRDS run, in Legacy's order."""
    results = run['results']
    _render_captions(run)

    st.subheader("Performance Summary")
    _render_summary(run)
    for note in result_notes(results):
        st.caption(note)
    st.divider()

    st.subheader("Holdings and Ranks")
    _render_holdings(run)
    st.divider()

    st.subheader("Portfolio Growth Over Time")
    _render_charts(run)
    st.divider()

    st.subheader("Period-by-Period Performance")
    _render_period_table(run)
    st.divider()

    _render_cohort(run)
    st.divider()

    st.header("Advanced Backtest Statistics")
    st.caption(frequency_caption(results))
    st.dataframe(metrics_table(results), width="stretch", hide_index=True)
    st.divider()


def _labels(run: Dict[str, Any]) -> Dict[str, str]:
    """Factor column -> display name."""
    return dict(zip(run['factors'], run['factor_labels']))


def _render_captions(run: Dict[str, Any]) -> None:
    """The settings the run was created with, like Legacy's header."""
    from components.results_tab import _render_analysis_id
    results, settings = run['results'], run['settings']
    factors = ", ".join(f"{label} ({DIRECTION_LABELS[run['factor_directions'][col]]})"
                        for label, col in zip(run['factor_labels'], run['factors']))
    st.caption(f"**Factors:** {factors}")
    frequency = FREQUENCY_LABELS.get(settings['frequency'], settings['frequency'])
    st.caption(f"**Data Source:** WRDS | **Rebalancing:** {frequency} ({len(results['rebalance_dates'])} rebalances)")
    st.caption(f"**Period:** {results['years'][0]} to {results['years'][-1]} "
               f"(years {int(settings['start_year'])}-{int(settings['end_year'])}) | "
               f"**Initial AUM:** ${float(settings['initial_aum']):,.0f}")
    universe = run.get('universe') or {}
    sectors = ", ".join(universe.get('selected_sectors') or []) or "All sectors"
    st.caption(f"**Universe:** {sectors} (GICS) | Fossil fuels "
               f"{'excluded' if universe.get('restrict_fossil_fuels') else 'included'}")
    combination = {'mixing': "Mixing (one sleeve per factor)",
                   'integrating': "Integrating (top stocks by composite score)"}[settings['factor_combination']]
    st.caption(f"**Factor Combination:** {combination}")
    st.caption(f"**Weighting:** {'Market Cap Weighted' if settings.get('use_market_cap_weight') else 'Equal Weighted'}")
    st.caption("**Delisting:** CRSP delisting returns included; the money then earns the risk-free rate "
               "until the next rebalance")
    compute = run.get('compute')
    if compute:
        how = describe_compute(compute)
        st.caption(f"**Computed:** {how[:1].upper() + how[1:]}")
    _render_analysis_id(compute)


def _render_summary(run: Dict[str, Any]) -> None:
    results = run['results']
    initial = float(run['settings']['initial_aum'])
    final = results['final_value']
    benchmark_final = wealth_series(results['benchmark_returns'], initial)[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Portfolio Value", f"${final:,.2f}")
    c2.metric("Total Return", f"{final / initial - 1:.2%}")
    c3.metric("CAGR", f"{results['cagr']:.2%}")
    c4.metric(f"Cumulative Outperformance vs {short_benchmark(results, 'benchmark')}",
              f"{final / benchmark_final - 1:.2%}")


def _pick_date(run: Dict[str, Any]) -> str:
    """The rebalance date picker (newest first by default), remembered per run across tab switches."""
    dates = list(run['results']['rebalance_dates'])[::-1]
    saved_key = f"wrds_date_saved_{run['id']}"
    saved = st.session_state.get(saved_key)
    date = st.selectbox("Rebalance date", options=dates, index=dates.index(saved) if saved in dates else 0,
                        key=f"wrds_date_{run['id']}")
    st.session_state[saved_key] = date
    return date


def _render_holdings(run: Dict[str, Any]) -> None:
    """The stocks held from the picked rebalance date (only that date is read from S3), then every stock ranked."""
    results, labels = run['results'], _labels(run)
    date = _pick_date(run)
    holdings = results.get('holdings') or {}
    method = FACTOR_COMBINATION_LABELS.get(results.get('factor_combination', 'mixing'), 'Mixing')
    period = holdings_period_caption(results, date) or ''
    if date not in holdings.get('dates', []):
        st.caption(f"**Method:** {method} | **Holdings:** 0 | **Portfolio return {period}**")
        st.info("Nothing could be bought on this date; the money earned the risk-free rate.")
    else:
        try:
            raw = load_wrds_holdings(holdings['file'], holdings['dates'].index(date), date)
        except Exception as e:
            st.error(f"Could not read the holdings for {date}: {e}")
            raw = None
        if raw is not None:
            table = holdings_display_table(raw, run['factors'], labels)
            st.caption(f"**Method:** {method} | **Holdings:** {len(table)} | **Portfolio return {period}**")
            _show_table(table, run, f"holdings_{date}.csv", f"wrds_holdings_csv_{run['id']}")

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
                       f"Result {describe_compute(ranked['compute'])}.")
            _show_table(table, run, f"ranked_{date}.csv", f"wrds_ranked_csv_{run['id']}")


def _show_table(table: pd.DataFrame, run: Dict[str, Any], file_name: str, key: str) -> None:
    """A table rounded on screen only (the CSV keeps full precision), with its CSV download."""
    column_config = {col: st.column_config.NumberColumn(format=fmt)
                     for col, fmt in table_number_formats(table, run['factors'], _labels(run)).items()}
    st.dataframe(table, width="stretch", hide_index=True, column_config=column_config)
    st.download_button("Download CSV", data=holdings_table_to_csv(table), file_name=file_name, mime="text/csv",
                       key=key, on_click="ignore")


def _render_charts(run: Dict[str, Any]) -> None:
    results = run['results']
    initial = float(run['settings']['initial_aum'])
    labels = {role: benchmark_label(results, role) for role in ('benchmark', 'value', 'growth')}
    benchmarks = {'benchmark': wealth_series(results['benchmark_returns'], initial),
                  'value': wealth_series(results['value_benchmark_returns'], initial),
                  'growth': wealth_series(results['growth_benchmark_returns'], initial)}
    st.plotly_chart(plot_wrds_growth(results['years'], results['portfolio_values'], benchmarks, labels,
                                     run['factor_labels']), width="stretch")
    frequency = FREQUENCY_LABELS.get(results['frequency'], results['frequency'])
    st.plotly_chart(plot_wrds_period_returns(results['period_end_dates'], results['yearly_returns'],
                                             results['benchmark_returns'], labels['benchmark'], frequency),
                    width="stretch")
    st.plotly_chart(plot_wrds_drawdown(results['years'], drawdowns(results['portfolio_values']),
                                       drawdowns(benchmarks['benchmark']), labels['benchmark']), width="stretch")


def _render_period_table(run: Dict[str, Any]) -> None:
    results = run['results']
    view = st.segmented_control("View", options=["By period", "By calendar year"], default="By period",
                                key=f"wrds_period_view_{run['id']}", label_visibility="collapsed")
    if view == "By calendar year":
        st.caption("Each calendar year's periods compounded; the first and last years can be partial.")
        st.dataframe(calendar_year_table(results), width="stretch", hide_index=True)
    else:
        st.dataframe(period_table(results), width="stretch", hide_index=True)


def _render_cohort(run: Dict[str, Any]) -> None:
    """Top vs bottom cohort at the run's frequency: the strategy at a cohort percentage and its exact opposite."""
    results, settings, run_id = run['results'], run['settings'], run['id']
    st.subheader("Top vs Bottom Cohort Analysis")
    with st.expander("Run Cohort Comparison", expanded=run.get('cohort') is not None,
                     key=f"wrds_cohort_expander_{run_id}"):
        saved_pct_key = f"wrds_cohort_pct_saved_{run_id}"
        default_pct = run['cohort']['pct'] if run.get('cohort') else 10
        pct = st.slider("Cohort Percentage", 1, 50, value=st.session_state.get(saved_pct_key, default_pct),
                        key=f"wrds_cohort_pct_{run_id}")
        st.session_state[saved_pct_key] = pct
        if st.button("Generate Comparison", type="primary", key=f"wrds_cohort_btn_{run_id}"):
            try:
                with st.spinner("Running the comparison in AWS Lambda..."):
                    top, bottom, info = run_cohort(run, pct)
                run['cohort'] = {'pct': pct, 'top': top, 'bottom': bottom, 'compute': info}
            except Exception as e:
                st.error(f"Cohort Comparison Error: {e}")

        cohort = run.get('cohort')
        if not cohort:
            return
        st.caption(f"Comparison {describe_compute(cohort['compute'])}.")
        initial = float(settings['initial_aum'])
        years = len(results['yearly_returns']) / int(results['periods_per_year'])

        def stats(values):
            return [f"{(values[-1] / initial - 1) * 100:.2f}%", f"${values[-1]:,.0f}",
                    f"{((values[-1] / initial) ** (1 / years) - 1) * 100:.2f}%"]
        st.write(f"### Cohort Performance Summary (top vs bottom {cohort['pct']}%)")
        st.table(pd.DataFrame({"Metric": ["Total Return (%)", "Final Value ($)", "CAGR (%)"],
                               "Top Cohort": stats(cohort['top']), "Bottom Cohort": stats(cohort['bottom'])}))
        st.plotly_chart(plot_wrds_cohort(results['years'], cohort['pct'], cohort['top'], cohort['bottom'],
                                         results['portfolio_values'],
                                         wealth_series(results['benchmark_returns'], initial),
                                         benchmark_label(results, 'benchmark'), initial), width="stretch")
