"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/compare_tab.py
PURPOSE: The Compare tab: 2-4 saved analyses (Legacy, WRDS or both) side by side, one
         column each, with the best value of each row highlighted, the rows that differ
         marked, a Sort by menu and one growth chart.
VERSION: 1.0.0

Comparing only reads saved results from S3 (src/saved_analyses): it runs no backtest and
loads no market data. The table and chart come from app/compare.py.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.analysis_history import merge_entries
from app.compare import (MAX_COMPARED, MIN_COMPARED, SORT_OPTIONS, column_titles, compact_analysis,
                         comparison_table, growth_series, short_id, sort_order)
from app.streamlit_utils import load_compared_analysis
from src.saved_analyses import SavedAnalysisError, normalize_analysis_id
from Visualizations.compare_growth_plot import COMPARE_COLORS, plot_compare_growth

# The ID boxes (a list of box numbers, in order) and the next box number
SLOTS_KEY = "compare_slots"
NEXT_SLOT_KEY = "compare_next_slot"
# Copies of the widget values. Only the open main tab is drawn, and Streamlit drops the
# state of widgets that are not drawn, so the IDs, sort and switch are restored from these.
IDS_KEY = "compare_ids"
SORT_KEY, SORT_SAVED_KEY = "compare_sort", "compare_sort_saved"
DIFF_KEY, DIFF_SAVED_KEY = "compare_differences_only", "compare_differences_only_saved"
# What the last Compare click loaded: {'slots': [...], 'entries': {slot: {'id', 'analysis' or 'error'}}}
LOADED_KEY = "compare_loaded"

BEST_STYLE = "background-color: #d3f3dc; color: #0b5d1e; font-weight: 600"
GROUP_STYLE = "background-color: #f0f2f6; font-weight: 700"


def _id_key(slot: int) -> str:
    return f"compare_id_{slot}"


def _slots() -> List[int]:
    if SLOTS_KEY not in st.session_state:
        st.session_state[SLOTS_KEY] = list(range(MIN_COMPARED))
        st.session_state[NEXT_SLOT_KEY] = MIN_COMPARED
    return st.session_state[SLOTS_KEY]


def _add_slot() -> None:
    """on_click: one more ID box (up to MAX_COMPARED)."""
    slots = _slots()
    if len(slots) < MAX_COMPARED:
        st.session_state[SLOTS_KEY] = slots + [st.session_state[NEXT_SLOT_KEY]]
        st.session_state[NEXT_SLOT_KEY] += 1


def _remove_slot(slot: int) -> None:
    """on_click: removes an ID box, its ID and its column."""
    st.session_state[SLOTS_KEY] = [s for s in _slots() if s != slot]
    st.session_state.get(IDS_KEY, {}).pop(slot, None)
    st.session_state.pop(_id_key(slot), None)
    loaded = st.session_state.get(LOADED_KEY)
    if loaded:
        loaded['entries'].pop(slot, None)
        loaded['slots'] = [s for s in loaded['slots'] if s != slot]


def _picked(slot: int) -> None:
    """on_change of a box's picker: puts the picked analysis's ID in the box."""
    pick_key = f"compare_pick_{slot}"
    picked = st.session_state.get(pick_key)
    if picked:
        st.session_state[_id_key(slot)] = picked
        st.session_state.setdefault(IDS_KEY, {})[slot] = picked
    st.session_state[pick_key] = None


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


def load_comparison(slots: List[int], texts: Dict[int, str]) -> Dict[str, Any]:
    """
    Reads the saved analyses in the boxes from S3 (nothing else is loaded or run). Empty
    boxes are skipped; a bad, expired, duplicate, cohort or ranked ID gets a short error.
    """
    entries: Dict[int, Dict[str, Any]] = {}
    seen: Dict[str, int] = {}
    for number, slot in enumerate(slots, start=1):
        text = (texts.get(slot) or '').strip()
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
        try:
            entries[slot] = {'id': analysis_id,
                             'analysis': compact_analysis(load_compared_analysis(analysis_id))}
        except SavedAnalysisError as e:
            entries[slot] = {'id': analysis_id, 'error': short_error(e)}
    return {'slots': list(slots), 'entries': entries}


def _compare_clicked() -> None:
    """on_click of Compare: reads the analyses in the boxes from S3."""
    st.session_state[LOADED_KEY] = load_comparison(_slots(), st.session_state.get(IDS_KEY, {}))


def _restore(widget_key: str, saved_key: str, default: Any) -> None:
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(saved_key, default)


def render_compare_tab(session_entries: List[Dict[str, Any]],
                       earlier_entries: Optional[List[Dict[str, Any]]]) -> None:
    """
    The Compare tab. session_entries and earlier_entries are this session's and earlier
    sessions' analyses (as listed in the Results tab), offered in each box's picker.
    """
    st.header("Compare analyses")
    st.caption("Enter 2 to 4 saved analysis IDs (Legacy, WRDS or both) and click Compare to see them side "
               "by side. Paste an ID, or pick one of your analyses. Comparing only reads the saved results: "
               "no backtest runs.")
    slots = _slots()
    picks = merge_entries(session_entries, earlier_entries)
    labels = {e['analysis_id']: f"{e.get('label', 'Analysis')} · {short_id(e['analysis_id'])}" for e in picks}
    loaded = st.session_state.get(LOADED_KEY) or {'slots': [], 'entries': {}}
    saved_ids = st.session_state.setdefault(IDS_KEY, {})

    for number, (col, slot) in enumerate(zip(st.columns(len(slots)), slots), start=1):
        with col:
            title, remove = st.columns([4, 1], vertical_alignment="center")
            title.markdown(f"**Analysis {number}**")
            if number > MIN_COMPARED:
                remove.button("✕", key=f"compare_remove_{slot}", on_click=_remove_slot, args=(slot,),
                              help="Remove this analysis")
            if _id_key(slot) not in st.session_state and slot in saved_ids:
                st.session_state[_id_key(slot)] = saved_ids[slot]
            saved_ids[slot] = st.text_input("Analysis ID", key=_id_key(slot),
                                            placeholder="Paste a 64-character analysis ID")
            st.selectbox("Or pick one of your analyses", options=list(labels), index=None,
                         format_func=lambda i: labels.get(i, i), key=f"compare_pick_{slot}",
                         on_change=_picked, args=(slot,), disabled=not labels,
                         placeholder="Your analyses" if labels else "No analyses in this browser yet")
            entry = loaded['entries'].get(slot)
            if entry and entry.get('error'):
                st.error(entry['error'])

    add, run = st.columns([1, 1])
    add.button("+ Add analysis", key="compare_add", on_click=_add_slot, disabled=len(slots) >= MAX_COMPARED,
               width="stretch")
    # Loads before the script reruns, so the boxes show their errors in the same run
    run.button("Compare", type="primary", key="compare_btn", width="stretch", on_click=_compare_clicked)

    analyses = [loaded['entries'][s]['analysis'] for s in loaded['slots']
                if 'analysis' in loaded['entries'].get(s, {})]
    if not loaded['entries']:
        return
    if len(analyses) < MIN_COMPARED:
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
    st.caption("Green: the best value in the row (none on a tie or n/a). ≠: the values in the row differ. "
               "Volatility, Sharpe and information ratio are annualized, so they compare across frequencies as "
               "they are; best and worst period are each analysis's own periods (years, half-years, quarters "
               "or months).")

    st.subheader("Portfolio Growth Over Time")
    titles = column_titles(analyses)
    st.plotly_chart(plot_compare_growth([growth_series(a) for a in analyses], titles,
                                        [short_id(a['id']) for a in analyses], order=order),
                    width="stretch", key="compare_growth_chart")
    notes = ["Each analysis after its trading costs, with its own benchmark dashed in the same color. "
             "Legacy's yearly values are placed on September 30 (it rebalances on the 9/30 data)."]
    if len({float(a['job']['initial_aum']) for a in analyses}) > 1:
        notes.append("The analyses start with different initial AUMs, so the lines start at different values.")
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
        return [GROUP_STYLE if name in groups else BEST_STYLE if best.get(name) == column.name else ''
                for name in column.index]
    return table.style.apply(cells, axis=0)


def _render_table(analyses: List[Dict[str, Any]], order: List[int], differences_only: bool) -> None:
    table = comparison_table(analyses, order, differences_only)
    if table is None:
        st.info("Every row is the same in these analyses.")
        return
    st.table(_style(table))
