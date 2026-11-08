"""
Strategy settings files and run names.

- Import settings from file: on the Analysis tab, below the factors. It fills the inputs
  (sidebar and factors) and says so in its box; it never runs a backtest.
- Each run's actions, beside Remove this run: Rename (a session-only name and note),
  Export settings (JSON) and Use these settings (loads the run's inputs into the Analysis tab).

Inputs can only be changed before their widgets are drawn, so an import or Use these
settings is queued (PENDING_SETTINGS_KEY) and applied by apply_pending_settings at the top
of the script, before the sidebar.
"""
import re
from typing import Any, Callable, Dict

import streamlit as st

from app import strategy_config as configs
from app.saved_runs import named_run_label
from components.data_viewer import clear_viewer_data
from components.factor_selection import reset_factor_selection
from components.sidebar import clear_gics_inputs

# Settings waiting to be applied before the sidebar is drawn: (source, payload, run_id), where
# source is 'file' (payload: the file's bytes) or 'run' (payload: the run's settings document)
PENDING_SETTINGS_KEY = 'strategy_pending_settings'
# The file_id of the last uploaded file queued, so a file is imported once, not on every rerun
IMPORTED_FILE_KEY = 'strategy_imported_file'
# The message shown at the top of the Analysis tab after a run's settings were loaded into it
ANALYSIS_MESSAGE_KEY = 'strategy_import_message'
# The outcome of importing a file, shown in the import box: imported, or why it could not be
IMPORT_SUCCESS_KEY = 'strategy_import_success'
IMPORT_ERROR_KEY = 'strategy_import_error'
# Bumped after each import, so the file uploader is new (empty) and the next file can be the same
IMPORT_VERSION_KEY = 'strategy_upload_version'
# An imported file's name and note, given to the next run (then cleared)
NEXT_NAME_KEY = 'strategy_name'
NEXT_NOTE_KEY = 'strategy_note'


def apply_import(raw, lambda_available):
    """Validate completely before changing inputs. Never execute a backtest on import."""
    document = configs.loads(raw)
    _apply_document(document, lambda_available)
    st.session_state[NEXT_NAME_KEY] = document['name']
    st.session_state[NEXT_NOTE_KEY] = document['note']
    return document


def _apply_document(document, lambda_available):
    if document['data_mode'] == 'wrds' and not lambda_available:
        raise ValueError('These settings need WRDS mode. Configure the backtest Lambda before loading them.')
    updates = configs.widget_values(document)
    reset_factor_selection()
    st.session_state.pop('selected_sectors_input', None)
    clear_gics_inputs()
    st.session_state.update(updates)
    st.session_state.update(data_loaded=False, rdata=None, loaded_filters=None)
    clear_viewer_data()


def apply_pending_settings(lambda_available) -> bool:
    """
    Applies queued settings (an imported file or a run's Use these settings). Call before the
    sidebar is drawn. True when a run's settings were loaded, so the caller opens the Analysis
    tab (an import is made there, so it stays).
    """
    pending = st.session_state.pop(PENDING_SETTINGS_KEY, None)
    if pending is None:
        return False
    source, payload, run_id = pending
    try:
        if source == 'file':
            document = apply_import(payload, lambda_available)
        else:
            document = payload
            _apply_document(document, lambda_available)
            clear_next_run_name()
    except ValueError as exc:
        st.session_state[IMPORT_ERROR_KEY if source == 'file' else _run_error_key(run_id)] = str(exc)
        return False
    if source == 'file':
        message = 'Settings imported. Review them, then run the analysis.'
        if document['name']:
            message += f' The next run will be named "{document["name"]}".'
    else:
        message = f'Run {run_id}\'s settings are loaded. Change what you like, then run the analysis.'
    if document['data_mode'] == 'legacy':
        message += ' For local backtests, click Load Market Data first.'
    st.session_state[IMPORT_SUCCESS_KEY if source == 'file' else ANALYSIS_MESSAGE_KEY] = message
    return source == 'run'


def clear_next_run_name():
    """Forget an imported file's name and note (Clear Parameters, or other settings loaded)."""
    st.session_state.pop(NEXT_NAME_KEY, None)
    st.session_state.pop(NEXT_NOTE_KEY, None)


def render_analysis_notice():
    """At the top of the Analysis tab: the message after a run's settings were loaded into it."""
    message = st.session_state.pop(ANALYSIS_MESSAGE_KEY, None)
    if message:
        st.success(message)


def render_next_run_name():
    """Above the run buttons: the name an imported file gives the next run, if any."""
    name = st.session_state.get(NEXT_NAME_KEY, '').strip()
    if name:
        st.caption(f':material/label: The next run will be named "{name}" (from the imported settings file). '
                   'Clear Parameters forgets it.')


def render_import_settings():
    """The Import settings from file box, on the Analysis tab below the factors."""
    success = st.session_state.pop(IMPORT_SUCCESS_KEY, None)
    error = st.session_state.pop(IMPORT_ERROR_KEY, None)
    with st.expander('Import settings from file', expanded=bool(success or error)):
        st.caption('Load a settings file (JSON, from a run\'s Export settings) into these inputs. '
                   'It replaces the current inputs and runs nothing.')
        version = st.session_state.get(IMPORT_VERSION_KEY, 0)
        uploaded = st.file_uploader('Settings file', type=['json'], key=f'strategy_upload_{version}',
                                    help='A settings file exported from this app. Maximum 64 KB.')
        if uploaded is not None and uploaded.file_id != st.session_state.get(IMPORTED_FILE_KEY):
            # Applied at the top of the next rerun, before the inputs are drawn; the new
            # uploader key empties the box, so the same file can be imported again later
            st.session_state[IMPORTED_FILE_KEY] = uploaded.file_id
            st.session_state[PENDING_SETTINGS_KEY] = ('file', uploaded.getvalue(), None)
            st.session_state[IMPORT_VERSION_KEY] = version + 1
            st.rerun()
        if success:
            st.success(success)
        if error:
            st.error(error)


def annotate_new_run(run):
    """Called only after successful computation; annotations never enter engine/cache inputs."""
    run['default_label'] = run['label']
    run['name'] = st.session_state.pop(NEXT_NAME_KEY, '').strip()
    run['note'] = st.session_state.pop(NEXT_NOTE_KEY, '')
    if run['name']:
        run['label'] = named_run_label(run['id'], run['name'], run['default_label'])


def _run_error_key(run_id):
    return f'run_settings_error_{run_id}'


def _settings_document(run) -> Dict[str, Any]:
    """The run's settings file. ValueError when the current controls cannot restore them."""
    return configs.build(run['settings'], run['factors'], run['factor_directions'],
                         run.get('name', ''), run.get('note', ''),
                         mode=run.get('data_mode', run['settings'].get('data_mode', 'legacy')))


def _save_annotation(run_id, keep_open: Callable[[str], None]):
    for run in st.session_state.get('saved_runs', []):
        if run['id'] == run_id:
            run.setdefault('default_label', run['label'])
            old_label = run['label']
            run['name'] = st.session_state[f'run_name_{run_id}'].strip()
            run['note'] = st.session_state[f'run_note_{run_id}']
            run['label'] = (named_run_label(run_id, run['name'], run['default_label']) if run['name']
                            else run['default_label'])
            # This session's analyses lists the run by its label too
            for entry in st.session_state.get('analysis_history', []):
                if entry['label'] == old_label:
                    entry['label'] = run['label']
            keep_open(run['label'])
            break


def _use_settings(run_id, document):
    st.session_state[PENDING_SETTINGS_KEY] = ('run', document, run_id)


def render_run_actions(run, remove: Callable[[], None], keep_open: Callable[[str], None]):
    """
    The run's actions in one row: Remove this run (drawn by `remove`), Rename, Export settings
    (JSON) and Use these settings. Then the run's note and why its settings cannot be loaded.
    """
    run_id = run['id']
    try:
        document, unavailable = _settings_document(run), None
    except ValueError as exc:
        document, unavailable = None, f'Not available: {exc}'
    with st.container(horizontal=True):
        remove()
        with st.popover('Rename', help='Name this run and add a note (kept for this session).'):
            for field in ('name', 'note'):
                key = f'run_{field}_{run_id}'
                if key not in st.session_state:
                    st.session_state[key] = run.get(field, '')
            with st.form(f'run_annotation_{run_id}', border=False):
                st.text_input('Name', key=f'run_name_{run_id}', max_chars=100,
                              placeholder='Leave empty for the default name')
                st.text_area('Note', key=f'run_note_{run_id}', max_chars=1000)
                st.form_submit_button('Apply', on_click=_save_annotation, args=(run_id, keep_open))
            st.caption('Names and notes are kept for this session and included in exported settings files.')
        if document is None:
            st.button('Export settings (JSON)', key=f'run_export_{run_id}', disabled=True, help=unavailable)
        else:
            filename = re.sub(r'[^a-zA-Z0-9_-]+', '_', document['name']).strip('_')[:80] or f'run_{run_id}'
            st.download_button('Export settings (JSON)', configs.dumps(document), file_name=f'{filename}.json',
                               mime='application/json', key=f'run_export_{run_id}', on_click='ignore',
                               help='Save this run\'s inputs (no data or results) to reuse or share them.')
        st.button('Use these settings', key=f'run_use_settings_{run_id}', on_click=_use_settings,
                  args=(run_id, document), disabled=document is None,
                  help=unavailable or 'Load this run\'s inputs into the Analysis tab, to change and run again.')
    if run.get('note'):
        st.text(run['note'])
    error = st.session_state.pop(_run_error_key(run_id), None)
    if error:
        st.error(error)
