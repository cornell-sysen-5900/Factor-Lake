"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/saved_panel.py
PURPOSE: The Results tab's Saved analyses panel: opening saved analyses (by ID or from the
         lists), beside the runs so they stay in view.
VERSION: 1.3.0

With runs, the panel floats over the right edge of the page with only its "Saved analyses"
tab (the handle) showing. With a mouse it slides in while the pointer is over it (or
something in it has focus); on a touch screen, tapping the handle opens and closes it. That
is CSS only (the handle is the label of a hidden checkbox), so showing or hiding the panel
does not rerun the app or move the runs. The runs keep a right margin the handle's width,
so the handle never covers them. Without runs there is nothing to cover, so the panel's
contents fill the tab.
"""
from typing import Any, Dict, List

import streamlit as st

from components.results_tab import (
    render_all_saved_analyses,
    render_analysis_history,
    render_earlier_analyses,
    render_open_saved_analysis,
)

# The panel's container key: Streamlit gives the container the class st-key-<key>
PANEL_KEY = "saved_analyses_panel"
# The hidden checkbox that keeps the panel open on touch screens (its label is the handle)
TOGGLE_ID = "saved-analyses-panel-open"
# Width of the handle left showing at the right edge while the panel is hidden
HANDLE_WIDTH = "2.25rem"

PANEL = f".st-key-{PANEL_KEY}"
PANEL_CSS = f"""
<style>
{PANEL} {{
    position: fixed;
    top: 7rem;
    right: 0;
    z-index: 100;
    width: min(28rem, 90vw);
    max-height: calc(100vh - 9rem);
    overflow-y: auto;
    padding: 1rem 1rem 1rem calc({HANDLE_WIDTH} + 1rem);
    background-color: #ffffff;
    border: 1px solid #DEDFE0;
    border-right: none;
    border-radius: 0.75rem 0 0 0.75rem;
    box-shadow: -4px 4px 16px rgba(0, 0, 0, 0.12);
    transform: translateX(calc(100% - {HANDLE_WIDTH}));
    transition: transform 0.2s ease;
}}
/* With a mouse: open while the pointer is over the panel or something in it has focus */
@media (hover: hover) {{
    {PANEL}:hover,
    {PANEL}:focus-within {{
        transform: translateX(0);
    }}
}}
/* On a touch screen: tapping the handle (the checkbox's label) opens and closes it */
@media (hover: none) {{
    {PANEL}:has(#{TOGGLE_ID}:checked) {{
        transform: translateX(0);
    }}
}}
/* The checkbox and its element take no room in the panel */
{PANEL} #{TOGGLE_ID} {{
    display: none;
}}
{PANEL} .stElementContainer:has(#{TOGGLE_ID}) {{
    position: static;
    height: 0;
    margin-bottom: -1rem;
}}
{PANEL} .saved-panel-handle {{
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: {HANDLE_WIDTH};
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    font-weight: 600;
    font-size: 0.9rem;
    color: #31333F;
    background-color: #F0F2F6;
    border-radius: 0 0.75rem 0.75rem 0;
    cursor: pointer;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
}}
/* Keep the runs clear of the handle: a right margin its width, besides the page's padding */
.stMainBlockContainer:has({PANEL}) {{
    border-right: {HANDLE_WIDTH} solid transparent;
}}
</style>
"""
HANDLE_HTML = (f'<input type="checkbox" id="{TOGGLE_ID}">'
               f'<label for="{TOGGLE_ID}" class="saved-panel-handle">Saved analyses</label>')


def render_saved_panel(history: List[Dict[str, Any]], earlier: List[Dict[str, Any]], has_runs: bool) -> None:
    """The panel's contents: open by ID and the analysis lists."""
    render_open_saved_analysis(has_runs=has_runs)
    render_analysis_history(history)
    render_earlier_analyses(earlier)
    render_all_saved_analyses()


def render_hover_panel(history: List[Dict[str, Any]], earlier: List[Dict[str, Any]]) -> None:
    """The panel floating at the right edge (see the module docstring)."""
    st.html(PANEL_CSS)
    with st.container(key=PANEL_KEY):
        st.html(HANDLE_HTML)
        render_saved_panel(history, earlier, has_runs=True)
