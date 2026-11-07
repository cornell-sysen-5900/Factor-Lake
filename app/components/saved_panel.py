"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/saved_panel.py
PURPOSE: The Results tab's Saved analyses panel: opening saved analyses (by ID or from the
         lists) and importing a settings file, beside the runs so they stay in view.
VERSION: 1.1.0

With runs, the panel floats over the right edge of the page with only its "Saved analyses"
tab showing, and slides in while the pointer is over it (or something in it has focus). That
is CSS only, so showing or hiding it does not rerun the app or move the runs. Without runs
there is nothing to cover, so the panel's contents fill the tab.
"""
from typing import Any, Dict, List

import streamlit as st

from components.results_tab import (
    render_all_saved_analyses,
    render_analysis_history,
    render_earlier_analyses,
    render_open_saved_analysis,
)
from components.strategy_settings import render_import_settings

# The panel's container key: Streamlit gives the container the class st-key-<key>
PANEL_KEY = "saved_analyses_panel"
# Width of the tab left showing at the right edge while the panel is hidden
HANDLE_WIDTH = "2.25rem"

PANEL_CSS = f"""
<style>
.st-key-{PANEL_KEY} {{
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
.st-key-{PANEL_KEY}:hover,
.st-key-{PANEL_KEY}:focus-within {{
    transform: translateX(0);
}}
.st-key-{PANEL_KEY}::before {{
    content: "Saved analyses";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: {HANDLE_WIDTH};
    display: flex;
    align-items: center;
    justify-content: center;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    font-weight: 600;
    font-size: 0.9rem;
    color: #31333F;
    background-color: #F0F2F6;
    border-radius: 0 0.75rem 0.75rem 0;
    cursor: pointer;
}}
</style>
"""


def render_saved_panel(history: List[Dict[str, Any]], earlier: List[Dict[str, Any]], has_runs: bool) -> None:
    """The panel's contents: open by ID, the analysis lists, and Import settings from file."""
    render_open_saved_analysis(has_runs=has_runs)
    render_analysis_history(history)
    render_earlier_analyses(earlier)
    render_all_saved_analyses()
    render_import_settings()


def render_hover_panel(history: List[Dict[str, Any]], earlier: List[Dict[str, Any]]) -> None:
    """The panel floating at the right edge, shown on hover (see the module docstring)."""
    st.html(PANEL_CSS)
    with st.container(key=PANEL_KEY):
        render_saved_panel(history, earlier, has_runs=True)
