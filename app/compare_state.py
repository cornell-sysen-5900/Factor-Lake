"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/compare_state.py
PURPOSE: The comparison's ID boxes, kept in the session: which boxes there are, the ID in
         each, and adding an ID to the first empty box. Shared by the Compare tab and the
         Results tab's "Add to comparison" button.
VERSION: 1.0.0
"""

from typing import List, Set

import streamlit as st

from app.compare import MAX_COMPARED, MIN_COMPARED
from src.saved_analyses import SavedAnalysisError, normalize_analysis_id

# The ID boxes (a list of box numbers, in order) and the next box number
SLOTS_KEY = "compare_slots"
NEXT_SLOT_KEY = "compare_next_slot"
# Copy of each box's text (box number -> text). Only the open main tab is drawn, and
# Streamlit drops the state of widgets that are not drawn, so the boxes are restored from it.
IDS_KEY = "compare_ids"
# Set by "Add to comparison" in Results: open the Compare tab on the next run
OPEN_COMPARE_KEY = "compare_open_tab"
# One-time message for the Compare tab (e.g. the comparison is full)
MESSAGE_KEY = "compare_message"

FULL_MESSAGE = f"The comparison already has {MAX_COMPARED} analyses. Remove one to add another."


def id_key(slot: int) -> str:
    """Widget key of a box's ID input."""
    return f"compare_id_{slot}"


def slots() -> List[int]:
    """The boxes, in order (2 to start with)."""
    if SLOTS_KEY not in st.session_state:
        st.session_state[SLOTS_KEY] = list(range(MIN_COMPARED))
        st.session_state[NEXT_SLOT_KEY] = MIN_COMPARED
    return st.session_state[SLOTS_KEY]


def box_text(slot: int) -> str:
    """A box's current text: its input's value, or the saved copy when the input is not drawn."""
    value = st.session_state.get(id_key(slot))
    if value is None:
        value = st.session_state.get(IDS_KEY, {}).get(slot)
    return value or ''


def set_box(slot: int, text: str) -> None:
    """Puts text in a box (call from a callback, before the box is drawn)."""
    st.session_state.setdefault(IDS_KEY, {})[slot] = text
    st.session_state[id_key(slot)] = text


def add_slot() -> None:
    """on_click: one more box (up to MAX_COMPARED)."""
    current = slots()
    if len(current) < MAX_COMPARED:
        st.session_state[SLOTS_KEY] = current + [st.session_state[NEXT_SLOT_KEY]]
        st.session_state[NEXT_SLOT_KEY] += 1


def remove_slot(slot: int) -> None:
    """on_click: removes a box and its ID."""
    st.session_state[SLOTS_KEY] = [s for s in slots() if s != slot]
    st.session_state.get(IDS_KEY, {}).pop(slot, None)
    st.session_state.pop(id_key(slot), None)


def ids_in_comparison() -> Set[str]:
    """The analysis IDs in the boxes (text that is not an ID is left out)."""
    found = set()
    for slot in slots():
        try:
            found.add(normalize_analysis_id(box_text(slot)))
        except SavedAnalysisError:
            pass
    return found


def add_to_comparison(analysis_id: str) -> str:
    """
    Puts an analysis ID in the first empty box, adding a box when every box has text and
    there are fewer than MAX_COMPARED. Call from a callback. Returns 'added', 'duplicate'
    (already in a box) or 'full' (MAX_COMPARED boxes, none empty).
    """
    analysis_id = normalize_analysis_id(analysis_id)
    if analysis_id in ids_in_comparison():
        return 'duplicate'
    empty = next((s for s in slots() if not box_text(s).strip()), None)
    if empty is None:
        if len(slots()) >= MAX_COMPARED:
            return 'full'
        add_slot()
        empty = slots()[-1]
    set_box(empty, analysis_id)
    return 'added'
