"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/wrds_results.py
PURPOSE: Results of a WRDS run: summary, holdings and ranked stocks per rebalance date, cohort.
VERSION: 1.0.0

A first version (stage 4 of DOCS/WRDS_MODE_DESIGN.md): the run's dates, total return
and CAGR, its holdings for one rebalance date (read from S3 one date at a time), the
ranked list of every stock on that date, and the cohort comparison. Stage 5 adds the
plots and tables Legacy runs show.
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.streamlit_utils import describe_compute, load_wrds_holdings, run_cohort, run_wrds_ranked

DIRECTION_LABELS = {'top': 'Higher is better', 'bottom': 'Lower is better'}
FREQUENCY_LABELS = {'semiannual': 'Semi-annual', 'quarterly': 'Quarterly', 'monthly': 'Monthly'}


def render_wrds_results(run: Dict[str, Any]) -> None:
    """One WRDS run, from its saved snapshot."""
    from components.results_tab import _render_analysis_id
    results, settings = run['results'], run['settings']
    _render_captions(run)
    _render_analysis_id(run.get('compute'))

    st.subheader("Performance Summary")
    initial = float(settings['initial_aum'])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Portfolio Value", f"${results['final_value']:,.2f}")
    c2.metric("Total Return", f"{results['final_value'] / initial - 1:.2%}")
    c3.metric("CAGR", f"{results['cagr']:.2%}")
    c4.metric("Rebalances", len(results['rebalance_dates']))
    notes = []
    if results.get('total_delisted_positions'):
        notes.append(f"delisted positions: {results['total_delisted_positions']}")
    if results.get('missing_return_stock_months'):
        notes.append(f"stock-months without a return (counted as 0%): {results['missing_return_stock_months']}")
    if results.get('rf_filled_months'):
        notes.append(f"risk-free months filled from the 3-month T-bill: {', '.join(results['rf_filled_months'])}")
    if notes:
        st.caption("; ".join(notes).capitalize() + ".")
    st.divider()

    _render_holdings(run)
    st.divider()
    _render_cohort(run)


def _render_captions(run: Dict[str, Any]) -> None:
    results, settings = run['results'], run['settings']
    factors = ", ".join(f"{label} ({DIRECTION_LABELS[run['factor_directions'][col]]})"
                        for label, col in zip(run['factor_labels'], run['factors']))
    st.caption(f"**Data source:** WRDS | **Factors:** {factors}")
    st.caption(f"**Rebalancing:** {FREQUENCY_LABELS.get(settings['frequency'], settings['frequency'])} | "
               f"**Period:** {results['years'][0]} to {results['years'][-1]} | "
               f"**Initial AUM:** ${float(settings['initial_aum']):,.0f}")
    universe = run.get('universe') or {}
    sectors = ", ".join(universe.get('selected_sectors') or []) or "All sectors"
    st.caption(f"**Universe:** {sectors} | Fossil fuels "
               f"{'excluded' if universe.get('restrict_fossil_fuels') else 'included'} | "
               f"**Combination:** {settings['factor_combination'].capitalize()} | "
               f"**Weighting:** {'Market Cap' if settings.get('use_market_cap_weight') else 'Equal'} | "
               f"Delisting returns included (CRSP)")
    if run.get('compute'):
        how = describe_compute(run['compute'])
        st.caption(f"**Computed:** {how[:1].upper() + how[1:]}")


def _render_holdings(run: Dict[str, Any]) -> None:
    """Holdings for one rebalance date, and the ranked list of every stock on that date."""
    results = run['results']
    st.subheader("Holdings and Ranks")
    dates = list(results['rebalance_dates'])
    saved_key = f"wrds_date_saved_{run['id']}"
    saved = st.session_state.get(saved_key)
    date = st.selectbox("Rebalance date", options=dates[::-1],
                        index=dates[::-1].index(saved) if saved in dates else 0, key=f"wrds_date_{run['id']}")
    st.session_state[saved_key] = date

    holdings = results.get('holdings') or {}
    if date not in holdings.get('dates', []):
        st.info("Nothing was bought on this date; the money earned the risk-free rate.")
    else:
        row_group = holdings['dates'].index(date)
        try:
            table = load_wrds_holdings(holdings['file'], row_group, date)
            st.caption(f"**Holdings:** {len(table)} (read from S3 for this date only)")
            st.dataframe(table, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Could not read the holdings: {e}")

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
        st.caption(f"**Ranked stocks:** {len(ranked['rows'])}, {describe_compute(ranked['compute'])}.")
        st.dataframe(pd.DataFrame(ranked['rows']), width="stretch", hide_index=True)


def _render_cohort(run: Dict[str, Any]) -> None:
    """Top vs bottom cohort: the strategy at a cohort percentage and its exact opposite."""
    st.subheader("Top vs Bottom Cohort")
    pct = st.slider("Cohort Percentage", 1, 50, value=(run.get('cohort') or {}).get('pct', 10),
                    key=f"wrds_cohort_pct_{run['id']}")
    if st.button("Generate Comparison", key=f"wrds_cohort_btn_{run['id']}"):
        try:
            with st.spinner("Running the comparison in AWS Lambda..."):
                top, bottom, info = run_cohort(run, pct)
            run['cohort'] = {'pct': pct, 'top': top, 'bottom': bottom, 'compute': info}
        except Exception as e:
            st.error(f"Cohort Comparison Error: {e}")
    cohort = run.get('cohort')
    if cohort:
        initial = float(run['settings']['initial_aum'])
        st.caption(f"Comparison {describe_compute(cohort['compute'])}.")
        st.table(pd.DataFrame({
            'Cohort': [f"Top {cohort['pct']}%", f"Bottom {cohort['pct']}%"],
            'Final Value ($)': [f"{cohort['top'][-1]:,.2f}", f"{cohort['bottom'][-1]:,.2f}"],
            'Total Return': [f"{cohort['top'][-1] / initial - 1:.2%}", f"{cohort['bottom'][-1] / initial - 1:.2%}"],
        }))
