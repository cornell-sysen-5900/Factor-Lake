"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/analysis_history.py
PURPOSE: Remembers the user's analysis IDs in their browser, across sessions.
VERSION: 1.0.0

The app has no logins, so "my earlier analyses" are kept where the user is: in the
browser's localStorage. Each run adds its analysis ID there, and the Results tab
lists the ones from earlier sessions, so a forgotten ID can still be found.
"""

from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

# localStorage key, and how many analyses a browser keeps (newest first)
BROWSER_STORAGE_KEY = 'factor-lake-analyses'
MAX_BROWSER_ENTRIES = 200
# Session state copy of the browser's list (last reported by the browser)
BROWSER_ANALYSES_KEY = 'browser_analyses'

# Reads the browser's list, adds `data.add` (this session's analyses), keeps the newest
# MAX entries and writes it back. It reports the list to Python only once per page load:
# Python already knows this session's analyses, and every report reruns the app, which
# would clear one-time messages such as "Analysis complete". Mirrors merge_entries.
_JS = """
export default function(component) {
    const { data, setStateValue } = component;
    let stored = [];
    try {
        const parsed = JSON.parse(window.localStorage.getItem(data.storage_key) || '[]');
        if (Array.isArray(parsed)) stored = parsed;
    } catch (e) { stored = []; }
    const byId = new Map();
    for (const entry of stored.concat(data.add || [])) {
        if (!entry || !entry.analysis_id) continue;
        const old = byId.get(entry.analysis_id);
        if (!old || (entry.saved_at || '') > (old.saved_at || '')) byId.set(entry.analysis_id, entry);
    }
    const merged = Array.from(byId.values())
        .sort((a, b) => (b.saved_at || '').localeCompare(a.saved_at || ''))
        .slice(0, data.max_entries);
    try { window.localStorage.setItem(data.storage_key, JSON.stringify(merged)); } catch (e) { /* blocked */ }
    if (!window.__factorLakeAnalysesReported) {
        window.__factorLakeAnalysesReported = true;
        setStateValue('stored', merged);
    }
}
"""

# (component registry, mount function): the component is registered once per Streamlit
# runtime, since each runtime (e.g. each AppTest) has its own registry
_registration = None


def _browser_store():
    """The browser-storage component's mount function, registered with the current runtime."""
    global _registration
    registry = st.components.v2.get_bidi_component_manager()
    if _registration is None or _registration[0] is not registry:
        _registration = (registry, st.components.v2.component("factor_lake_analysis_store", js=_JS))
    return _registration[1]


def merge_entries(*lists: Optional[Iterable[Dict[str, Any]]],
                  limit: int = MAX_BROWSER_ENTRIES) -> List[Dict[str, Any]]:
    """
    Analyses from several lists, one per ID (the most recently saved), newest first,
    at most `limit`. Entries without an analysis ID (runs computed in the app) are left out.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for entries in lists:
        for entry in entries or []:
            analysis_id = (entry or {}).get('analysis_id')
            if not analysis_id:
                continue
            old = by_id.get(analysis_id)
            if old is None or (entry.get('saved_at') or '') > (old.get('saved_at') or ''):
                by_id[analysis_id] = dict(entry)
    return sorted(by_id.values(), key=lambda e: e.get('saved_at') or '', reverse=True)[:limit]


def earlier_analyses(browser_entries: Optional[Iterable[Dict[str, Any]]],
                     session_entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The browser's analyses that are not from this session, newest first."""
    this_session = {e.get('analysis_id') for e in session_entries}
    return [e for e in merge_entries(browser_entries) if e['analysis_id'] not in this_session]


def sync_browser_analyses(session_entries: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Adds this session's analyses to the browser's list and returns that list (None until
    the browser has reported it once). Call on every run so new analyses are saved.
    """
    result = _browser_store()(
        data={'storage_key': BROWSER_STORAGE_KEY, 'max_entries': MAX_BROWSER_ENTRIES,
              'add': [e for e in session_entries if e.get('analysis_id')]},
        key='factor_lake_analysis_store',
        on_stored_change=lambda: None,
    )
    stored = getattr(result, 'stored', None)
    if stored is not None:
        st.session_state[BROWSER_ANALYSES_KEY] = stored
    return st.session_state.get(BROWSER_ANALYSES_KEY)
