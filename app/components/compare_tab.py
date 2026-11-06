"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/compare_tab.py
PURPOSE: The Compare tab: 2-4 saved analyses (Legacy, WRDS or both) side by side, one
         column each, with the best value of each row highlighted, the rows that differ
         marked, a Sort by menu and one growth chart. The Results tab's history lists are
         shown below the ID boxes, each entry with an "Add to comparison" button.
VERSION: 1.1.0

Comparing only reads saved results from S3 (src/saved_analyses): it runs no backtest and
loads no market data. Each result is read once and cached by ID for the session, so adding,
removing or sorting never reads one again. The table and chart come from app/compare.py,
the boxes' state from app/compare_state.py.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.compare import (
    GROWTH_BASE,
    MAX_COMPARED,
    MIN_COMPARED,
    SORT_OPTIONS,
    benchmark_lines,
    chart_series,
    column_titles,
    compact_analysis,
    comparison_table,
    short_id,
    sort_order,
)
from app.compare_state import (
    FULL_MESSAGE,
    IDS_KEY,
    MESSAGE_KEY,
    add_slot,
    add_to_comparison,
    box_text,
    id_key,
    ids_in_comparison,
    remove_slot,
    slots,
)
from app.streamlit_utils import load_compared_analysis
from components.results_tab import render_all_saved_analyses, render_analysis_history, render_earlier_analyses
from src.saved_analyses import SavedAnalysisError, normalize_analysis_id
from Visualizations.compare_growth_plot import COMPARE_COLORS, plot_compare_growth

SORT_KEY, SORT_SAVED_KEY = "compare_sort", "compare_sort_saved"
DIFF_KEY, DIFF_SAVED_KEY = "compare_differences_only", "compare_differences_only_saved"
# Loaded results by analysis ID ({'analysis': ...} or {'error': ...}), most recently used
# last; the IDs in the boxes are always kept, and at most CACHE_LIMIT in all
CACHE_KEY = "compare_cache"
CACHE_LIMIT = 12
# Key prefix of the history lists' widgets (the Results tab's have none)
LIST_KEY_PREFIX = "compare_"

BEST_STYLE = "background-color: #d3f3dc; color: #0b5d1e; font-weight: 600"
GROUP_STYLE = "background-color: #f0f2f6; font-weight: 700"


def short_error(error: SavedAnalysisError) -> str:
    """A short message for an ID that cannot be compared."""
    if error.reason == 'not_an_id':
        return "Not an analysis ID: an ID is 64 characters of 0-9 and a-f."
    if error.reason == 'not_found':
        return "No saved analysis has this ID: it is mistyped, or expired (analyses are kept 90 days)."
    if error.reason == 'kind':
        what = {'cohort': 'a cohort comparison', 'ranked': 'a ranked stock list'}.get(error.kind, 'another job')
        return f"This ID is {what}, not an analysis."
    if error.reason == 'not_shared':
        return "Saved before analyses could be shared, so it has no settings to compare."
    return str(error)


def _cached(analysis_id: str) -> Dict[str, Any]:
    """
    An analysis from the session's cache, read from S3 only the first time. A failed read
    is cached too (a mistyped or expired ID stays so), except an S3 error, which is retried.
    """
    cache: Dict[str, Dict[str, Any]] = st.session_state.setdefault(CACHE_KEY, {})
    if analysis_id in cache:
        cache[analysis_id] = cache.pop(analysis_id)            # most recently used last
        return cache[analysis_id]
    try:
        entry = {'analysis': compact_analysis(load_compared_analysis(analysis_id))}
    except SavedAnalysisError as e:
        entry = {'error': short_error(e)}
        if e.reason == 'unreadable':
            return entry
    cache[analysis_id] = entry
    return entry


def _trim_cache(keep: set) -> None:
    cache = st.session_state.get(CACHE_KEY, {})
    for analysis_id in [i for i in cache if i not in keep][:max(0, len(cache) - CACHE_LIMIT)]:
        del cache[analysis_id]


def resolve_boxes(box_slots: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Each box with text: {'id', 'analysis'} or {'id', 'error'} (a short message), from the
    cache or read from S3. Empty boxes are skipped; a bad, duplicate, expired, cohort or
    ranked ID gets an error.
    """
    entries: Dict[int, Dict[str, Any]] = {}
    seen: Dict[str, int] = {}
    for number, slot in enumerate(box_slots, start=1):
        text = box_text(slot).strip()
        if not text:
            continue
        try:
            analysis_id = normalize_analysis_id(text)
        except SavedAnalysisError as e:
            entries[slot] = {'id': None, 'error': short_error(e)}
            continue
        if analysis_id in seen:
            entries[slot] = {'id': analysis_id, 'error': f"Same analysis as box {seen[analysis_id]}."}
            continue
        seen[analysis_id] = number
        entries[slot] = {'id': analysis_id, **_cached(analysis_id)}
    _trim_cache(set(seen))
    return entries


def _add_from_list(analysis_id: str) -> None:
    """on_click of a list's Add to comparison."""
    if add_to_comparison(analysis_id) == 'full':
        st.session_state[MESSAGE_KEY] = FULL_MESSAGE


def _list_button(analysis_id: str, list_name: str, entry: Dict[str, Any]) -> None:
    """
    A history list entry's button: Add to comparison, or Added when it is compared. The lists
    hold analyses only (as in Results); a pasted cohort or ranked ID gets its box's error.
    """
    key = f"compare_add_{list_name}_{analysis_id}"
    if analysis_id in ids_in_comparison():
        st.button("Added", key=key, disabled=True, help="This analysis is already in the comparison.")
    else:
        st.button("Add to comparison", key=key, on_click=_add_from_list, args=(analysis_id,))


def _restore(widget_key: str, saved_key: str, default: Any) -> None:
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(saved_key, default)


def render_compare_tab(session_entries: List[Dict[str, Any]],
                       earlier_entries: Optional[List[Dict[str, Any]]]) -> None:
    """
    The Compare tab. session_entries and earlier_entries are this session's and earlier
    sessions' analyses, listed as in the Results tab below the boxes.
    """
    st.header("Compare analyses")
    st.caption("Enter 2 to 4 saved analysis IDs (Legacy, WRDS or both): paste an ID in a box, or add one from "
               "your analyses below or from a run in Results. The comparison appears as soon as 2 are valid. "
               "Comparing only reads the saved results: no backtest runs.")
    if MESSAGE_KEY in st.session_state:
        st.warning(st.session_state.pop(MESSAGE_KEY))
    box_slots = slots()
    with st.spinner("Loading the saved analyses from S3..."):
        entries = resolve_boxes(box_slots)
    saved_ids = st.session_state.setdefault(IDS_KEY, {})

    for number, (col, slot) in enumerate(zip(st.columns(len(box_slots)), box_slots), start=1):
        with col:
            st.markdown(f"**Analysis {number}**")
            if id_key(slot) not in st.session_state and slot in saved_ids:
                st.session_state[id_key(slot)] = saved_ids[slot]
            # Boxes beyond the first 2 have a remove button beside the ID, so every box lines up
            id_col, remove_col = (st.columns([6, 1], vertical_alignment="bottom") if number > MIN_COMPARED
                                  else (st.container(), None))
            saved_ids[slot] = id_col.text_input("Analysis ID", key=id_key(slot),
                                                placeholder="Paste a 64-character analysis ID")
            if remove_col is not None:
                remove_col.button("✕", key=f"compare_remove_{slot}", on_click=remove_slot, args=(slot,),
                                  help="Remove this analysis", type="tertiary")
            entry = entries.get(slot)
            if entry and entry.get('error'):
                st.error(entry['error'])

    st.button("+ Add analysis", key="compare_add", on_click=add_slot, disabled=len(box_slots) >= MAX_COMPARED)

    # Your analyses, as in the Results tab, each with Add to comparison
    render_analysis_history(session_entries, button=_list_button, key_prefix=LIST_KEY_PREFIX)
    render_earlier_analyses(earlier_entries, button=_list_button, key_prefix=LIST_KEY_PREFIX)
    render_all_saved_analyses(button=_list_button, key_prefix=LIST_KEY_PREFIX)

    analyses = [entries[s]['analysis'] for s in box_slots if 'analysis' in entries.get(s, {})]
    if len(analyses) < MIN_COMPARED:
        if entries:
            st.info(f"Enter at least {MIN_COMPARED} valid analysis IDs to compare.")
        return
    st.divider()
    _render_comparison(analyses)


def _render_comparison(analyses: List[Dict[str, Any]]) -> None:
    """The column headers, table and chart for the loaded analyses (no S3 reads)."""
    sort_col, diff_col = st.columns([1, 1], vertical_alignment="bottom")
    _restore(SORT_KEY, SORT_SAVED_KEY, 'Entered order')
    with sort_col:
        option = st.selectbox("Sort by", options=list(SORT_OPTIONS), key=SORT_KEY,
                              help="Best first (lowest for volatility, drawdown losses and cost drag); "
                                   "n/a last.")
    st.session_state[SORT_SAVED_KEY] = option
    _restore(DIFF_KEY, DIFF_SAVED_KEY, False)
    with diff_col:
        differences_only = st.toggle("Show only differences", key=DIFF_KEY)
    st.session_state[DIFF_SAVED_KEY] = differences_only

    order = sort_order(analyses, option)
    _render_headers(analyses, order)
    _render_table(analyses, order, differences_only)
    st.caption("Green: the best value in the row; analyses that tie for best are all green, and nothing is when "
               "every analysis ties or for n/a. ≠: the values in the row differ. "
               "Volatility, Sharpe and information ratio are annualized, so they compare across frequencies as "
               "they are. Best and worst period and turnover per rebalance are each analysis's own periods "
               "(years, half-years, quarters or months), so they are only highlighted when every analysis has "
               "the same frequency. Final value and total cost paid are in dollars, so they are only highlighted "
               "when every analysis has the same initial AUM.")

    st.subheader("Portfolio Growth Over Time")
    titles = column_titles(analyses)
    series, scaled = chart_series(analyses)
    st.plotly_chart(plot_compare_growth(series, titles, benchmark_lines(series, order), order=order,
                                        growth_of=GROWTH_BASE if scaled else None),
                    width="stretch", key="compare_growth_chart")
    notes = ["Each analysis after its trading costs, in its own color. Each benchmark is drawn once, grey and "
             "dotted, from the first analysis in the current order that uses it. "
             "Legacy's yearly values are placed on September 30 (it rebalances on the 9/30 data)."]
    if scaled:
        notes.append(f"The analyses start with different initial AUMs, so every line (each portfolio and its "
                     f"benchmark) is scaled to start at ${GROWTH_BASE:,.0f}.")
    st.caption(" ".join(notes))


def _render_headers(analyses: List[Dict[str, Any]], order: List[int]) -> None:
    """One header per column, in the sorted order: its color, label, frequency and full ID (copy button)."""
    for col, i in zip(st.columns(len(order)), order):
        a = analyses[i]
        with col:
            st.markdown(f'<span style="color:{COMPARE_COLORS[i]}">●</span> **{a["label"]}**',
                        unsafe_allow_html=True)
            st.caption(f"{'WRDS' if a['mode'] == 'wrds' else 'Legacy'} · {a['frequency']} · "
                       f"ID {short_id(a['id'])}")
            st.code(a['id'], language=None)


def _style(table: pd.DataFrame):
    best, groups = table.attrs['best'], table.attrs['groups']

    def cells(column: pd.Series) -> List[str]:
        return [GROUP_STYLE if name in groups else BEST_STYLE if column.name in best.get(name, ()) else ''
                for name in column.index]
    return table.style.apply(cells, axis=0)


def _render_table(analyses: List[Dict[str, Any]], order: List[int], differences_only: bool) -> None:
    table = comparison_table(analyses, order, differences_only)
    if table is None:
        st.info("Every row is the same in these analyses.")
        return
    st.table(_style(table))
