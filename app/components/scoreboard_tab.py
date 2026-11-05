"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/scoreboard_tab.py
PURPOSE: The Scoreboard tab: everyone's saved analyses ranked by a metric (Sharpe by
         default), with filters, and Open / Add to comparison for the analysis picked.
VERSION: 1.0.0

The scoreboard only reads S3 metadata (each analysis's scores, saved with it by the
backtest Lambda): it downloads no result and runs no backtest. The ranking and filters
come from app/scoreboard.py.
"""

from typing import Any, List

import streamlit as st

from app.compare_state import FULL_MESSAGE, OPEN_COMPARE_KEY, add_to_comparison, ids_in_comparison
from app.scoreboard import (
    DEFAULT_RANK,
    METRICS,
    TOP_ROWS,
    distinct_entries,
    filter_entries,
    filter_options,
    rank_entries,
    row_label,
    scoreboard_table,
)
from app.streamlit_utils import list_scored_analyses, open_saved_analysis
from components.results_tab import OPENED_MESSAGE_KEY
from src.saved_analyses import SavedAnalysisError

# Set by Open: open the Results tab on the next run (the analysis is its newest run tab)
OPEN_RESULTS_KEY = "scoreboard_open_results"
# One-time warning for the scoreboard (e.g. the comparison is full)
MESSAGE_KEY = "scoreboard_message"
RANK_KEY = "scoreboard_rank_by"
SOURCES_KEY = "scoreboard_sources"
COMBINATIONS_KEY = "scoreboard_combinations"
FACTORS_KEY = "scoreboard_factors"
YEARS_KEY = "scoreboard_years"
SHOW_ALL_KEY = "scoreboard_show_all"
PICK_KEY = "scoreboard_pick"
# Copies of the widgets' values: only the open main tab is drawn, and Streamlit drops the
# state of widgets that are not drawn, so they are restored from these
SAVED_KEY = "scoreboard_saved"


def _restore(key: str, default: Any, choices: Any = None) -> None:
    """Puts a widget's saved value back before it is drawn (dropping choices that no longer exist)."""
    if key in st.session_state:
        return
    value = st.session_state.get(SAVED_KEY, {}).get(key, default)
    if choices is not None and isinstance(value, list):
        value = [v for v in value if v in choices]
    st.session_state[key] = value


def _save(key: str) -> None:
    st.session_state.setdefault(SAVED_KEY, {})[key] = st.session_state[key]


def _refresh() -> None:
    """on_click of Refresh: read the scores from S3 again."""
    list_scored_analyses.clear()


def _add(analysis_id: str) -> None:
    """on_click of Add to comparison: adds the ID and opens the Compare tab, unless every box is full."""
    if add_to_comparison(analysis_id) == 'full':
        st.session_state[MESSAGE_KEY] = FULL_MESSAGE
    else:
        st.session_state[OPEN_COMPARE_KEY] = True


def render_scoreboard_tab() -> None:
    st.header("Backtest scoreboard")
    st.caption("Every analysis saved by anyone using this app, ranked by the metric you choose. The metrics "
               "are the ones its Results show, saved with it, so nothing is downloaded or rerun. Saved analyses "
               "are deleted after 90 days.")
    if MESSAGE_KEY in st.session_state:
        st.warning(st.session_state.pop(MESSAGE_KEY))
    try:
        with st.spinner("Reading the saved analyses' scores from S3..."):
            listing = list_scored_analyses()
    except SavedAnalysisError as e:
        st.error(str(e))
        return
    # Analyses saved before scores were recorded have nothing to rank or filter by
    entries = [e for e in listing['entries'] if e.get('scores')]
    unscored = len(listing['entries']) - len(entries)
    entries, repeats = distinct_entries(entries)
    if not entries:
        st.info("No scored analyses yet. Run one with AWS Lambda backtests on, and it appears here.")
        _caption_unscored(unscored)
        return

    options = filter_options(entries)
    rank_col, source_col, combination_col, refresh_col = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
    _restore(RANK_KEY, DEFAULT_RANK)
    rank_by = rank_col.selectbox("Rank by", [m.name for m in METRICS], key=RANK_KEY,
                                 help="Best first: highest, except lowest volatility. Max drawdown is a loss, "
                                      "so the smallest loss is best.")
    _restore(SOURCES_KEY, [], options['sources'])
    sources = source_col.multiselect("Data source", options['sources'], key=SOURCES_KEY, placeholder="All")
    _restore(COMBINATIONS_KEY, [], options['combinations'])
    combinations = combination_col.multiselect("Factor combination", options['combinations'], key=COMBINATIONS_KEY,
                                               placeholder="All")
    refresh_col.button("Refresh", key="scoreboard_refresh", on_click=_refresh, width="stretch",
                       help="Read the scores from S3 again (they are reused for a minute).")
    factor_col, years_col = st.columns([4, 3], vertical_alignment="bottom")
    _restore(FACTORS_KEY, [], options['factors'])
    factors = factor_col.multiselect("Uses every factor", options['factors'], key=FACTORS_KEY, placeholder="Any")
    years = None
    if options['years'] and options['years'][0] < options['years'][1]:
        low, high = options['years']
        # The saved range, kept within the years listed now (newer analyses can widen them, expiry narrows them)
        start, end = (st.session_state.get(YEARS_KEY) or st.session_state.get(SAVED_KEY, {}).get(YEARS_KEY)
                      or (low, high))
        start, end = min(max(start, low), high), max(min(end, high), low)
        st.session_state[YEARS_KEY] = (start, end) if start <= end else (low, high)
        chosen = years_col.slider("Period within", low, high, key=YEARS_KEY,
                                  help="Only analyses whose whole period lies within these years.")
        _save(YEARS_KEY)
        years = None if tuple(chosen) == (low, high) else tuple(chosen)
    for key in (RANK_KEY, SOURCES_KEY, COMBINATIONS_KEY, FACTORS_KEY):
        _save(key)

    ranked = rank_entries(filter_entries(entries, sources, combinations, factors, years), rank_by)
    if not ranked:
        st.info("No saved analysis matches these filters.")
        return
    _render_table(ranked, rank_by)
    if repeats:
        st.caption(f"{repeats} more {'run' if repeats == 1 else 'runs'} of listed strategies (the same settings and "
                   f"returns, run again after an update or with another initial amount) are folded into their "
                   f"newest run.")
    _caption_unscored(unscored)
    _render_picker(ranked)


def _caption_unscored(count: int) -> None:
    if count:
        one = count == 1
        st.caption(f"{count} {'analysis' if one else 'analyses'} saved before the scoreboard {'is' if one else 'are'} "
                   f"not ranked: no scores were recorded for {'it' if one else 'them'}. "
                   f"{'It is' if one else 'They are'} still listed under All saved analyses in Results.")


def _render_table(ranked: List[dict], rank_by: str) -> None:
    shown = ranked
    if len(ranked) > TOP_ROWS:
        _restore(SHOW_ALL_KEY, False)
        if not st.toggle(f"Show all {len(ranked)}", key=SHOW_ALL_KEY):
            shown = ranked[:TOP_ROWS]
        _save(SHOW_ALL_KEY)
    # The ranked column's header says which way is best (▼: highest first, ▲: lowest first)
    formats = {m.name: st.column_config.NumberColumn(
        m.name + ((' ▼' if m.best == 'higher' else ' ▲') if m.name == rank_by else ''),
        format="dollar" if m.key == 'final_value' else "%.2f%%" if m.percent else "%.4f")
        for m in METRICS}
    st.dataframe(scoreboard_table(shown), hide_index=True, width="stretch", column_config={
        'Rank': st.column_config.NumberColumn('Rank', format="%d", width="small"),
        'Analysis': st.column_config.TextColumn('Analysis', width="large"),
        **formats,
    }, key="scoreboard_table")
    notes = [f"Showing {len(shown)} of {len(ranked)}." if len(shown) < len(ranked) else f"{len(ranked)} analyses."]
    notes.append("Analyses that tie (as shown) share a rank; ties are listed newest first. "
                 "Outperformance and win rate are against each analysis's own benchmark (Russell 2000, or IWM for "
                 "WRDS). Periods and initial amounts differ: compare final values only at the same initial amount, "
                 "and filter by period for a like-for-like ranking.")
    st.caption(" ".join(notes))


def _render_picker(ranked: List[dict]) -> None:
    """Pick an analysis: its full ID (copy button), Open and Add to comparison."""
    ids = [e['analysis_id'] for e in ranked]
    labels = {e['analysis_id']: row_label(e) for e in ranked}
    if st.session_state.get(PICK_KEY) not in labels:
        st.session_state[PICK_KEY] = ids[0]
    st.subheader("Open an analysis")
    picked = st.selectbox("Analysis", ids, format_func=labels.get, key=PICK_KEY)
    id_col, open_col, compare_col = st.columns([5, 1, 1], vertical_alignment="center")
    id_col.code(picked, language=None)
    with open_col:
        if st.button("Open", key="scoreboard_open", type="primary", width="stretch",
                     help="Open it in the Results tab, with its results, factors and settings."):
            try:
                with st.spinner("Loading the saved analysis from S3..."):
                    run = open_saved_analysis(picked)
            except SavedAnalysisError as e:
                st.error(str(e))
                return
            st.session_state[OPENED_MESSAGE_KEY] = f"Opened \"{run['label']}\" from the scoreboard."
            st.session_state[OPEN_RESULTS_KEY] = True
            st.rerun()
    with compare_col:
        if picked in ids_in_comparison():
            st.button("In comparison", key="scoreboard_compare", disabled=True, width="stretch")
        else:
            st.button("Compare", key="scoreboard_compare", on_click=_add, args=(picked,), width="stretch",
                      help="Add it to the Compare tab and open it.")
