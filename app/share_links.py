"""Saved-analysis URLs and a browser clipboard button. Opening never starts a backtest."""

from urllib.parse import urlencode, urlsplit, urlunsplit

import streamlit as st

from src.saved_analyses import SavedAnalysisError, normalize_analysis_id

QUERY_KEY = 'analysis'
REQUEST_KEY = 'share_link_request'
ERROR_KEY = 'share_link_error'
DEFAULT_APP_URL = 'https://cornellfactorlake.streamlit.app/'


def build_share_url(base_url: str, analysis_id: str) -> str:
    """Keep the current deployment's path, with only the saved ID in the query."""
    analysis_id = normalize_analysis_id(analysis_id)
    parts = urlsplit(base_url)
    if parts.scheme not in ('http', 'https') or not parts.netloc or parts.username or parts.password:
        raise ValueError('The app URL must be an HTTP(S) URL without credentials.')
    return urlunsplit((parts.scheme, parts.netloc, parts.path or '/',
                       urlencode({QUERY_KEY: analysis_id}), ''))


def consume_share_link(query, state, opener):
    """Open each incoming query once per session; return (new run, readable error).

    Keeping the URL lets a page refresh or a copied address reopen the same result.
    The marker prevents tab changes and widget reruns from reopening it. Removing
    the query resets the marker, so a later navigation to that link works again.
    """
    if QUERY_KEY not in query:
        state.pop(REQUEST_KEY, None)
        state.pop(ERROR_KEY, None)
        return None, None
    values = tuple(query.get_all(QUERY_KEY))
    if state.get(REQUEST_KEY) == values:
        return None, state.get(ERROR_KEY)
    state[REQUEST_KEY] = values
    state.pop(ERROR_KEY, None)
    try:
        if len(values) != 1 or not values[0].strip():
            raise SavedAnalysisError('This share link needs exactly one analysis ID.')
        analysis_id = normalize_analysis_id(values[0])
        return opener(analysis_id), None
    except SavedAnalysisError as exc:
        message = f'Could not open the shared analysis. {exc}'
    except Exception:
        # A missing S3 configuration or temporarily unavailable store must not stop
        # the rest of the app, or expose credentials and service error details.
        message = ('Could not open the shared analysis because saved results are temporarily unavailable. '
                   'You can use the app normally, or reload this link to retry.')
    state[ERROR_KEY] = message
    return None, message


_HTML = '''
<button type="button" title="Copy a link that opens this saved result">Copy share link</button>
<span role="status" aria-live="polite"></span>
<input aria-label="Share link" readonly hidden />
'''
_CSS = '''
button { font: inherit; color: var(--st-text-color); background: var(--st-background-color);
         border: 1px solid var(--st-border-color, #a0a0a0); border-radius: .5rem;
         padding: .45rem .75rem; cursor: pointer; min-height: 2.5rem; }
button:hover { border-color: var(--st-primary-color); }
button:focus-visible { outline: 2px solid var(--st-primary-color); outline-offset: 2px; }
span { display: block; font-size: .8rem; }
input { width: 100%; box-sizing: border-box; font: inherit; margin-top: .4rem; }
'''
_JS = '''
export default function(component) {
    const { data, parentElement } = component;
    const button = parentElement.querySelector('button');
    const status = parentElement.querySelector('[role="status"]');
    const input = parentElement.querySelector('input');
    input.value = data.url;
    const copy = async () => {
        try {
            await navigator.clipboard.writeText(data.url);
            status.textContent = 'Link copied';
            input.hidden = true;
        } catch (error) {
            input.hidden = false;
            input.focus();
            input.select();
            status.textContent = 'Select and copy the link below.';
        }
    };
    button.addEventListener('click', copy);
    return () => button.removeEventListener('click', copy);
}
'''
_registration = None


def render_copy_share_link(analysis_id: str) -> None:
    """Copy in the click handler (clipboard permission requires a user gesture)."""
    global _registration
    registry = st.components.v2.get_bidi_component_manager()
    if _registration is None or _registration[0] is not registry:
        _registration = (registry, st.components.v2.component(
            'factor_lake_share_link', html=_HTML, css=_CSS, js=_JS))
    url = build_share_url(st.context.url or DEFAULT_APP_URL, analysis_id)
    _registration[1](data={'url': url}, key=f'share_link_{analysis_id}')
