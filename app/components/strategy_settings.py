"""Strategy import/export controls and session-only run annotations."""
import re
import streamlit as st
from app import strategy_config as configs
from components.factor_selection import reset_factor_selection
from components.data_viewer import clear_viewer_data


def apply_import(raw, lambda_available):
    """Validate completely before changing inputs. Never execute a backtest on import."""
    document = configs.loads(raw)
    if document['data_mode'] == 'wrds' and not lambda_available:
        raise ValueError('This file needs WRDS mode. Configure the backtest Lambda before importing it.')
    updates = configs.widget_values(document)
    reset_factor_selection()
    st.session_state.pop('selected_sectors_input', None)
    st.session_state.pop('wrds_sectors_input', None)
    st.session_state.update(updates)
    st.session_state.update(data_loaded=False, rdata=None, loaded_filters=None)
    clear_viewer_data()


def _import_uploaded(raw, lambda_available):
    if raw is None:
        return
    try:
        apply_import(raw, lambda_available)
    except ValueError as exc:
        st.session_state['strategy_import_message'] = ('error', str(exc))
    else:
        st.session_state['strategy_import_message'] = (
            'success', 'Settings imported. Review your inputs, then run the analysis. '
            'For local backtests, click Load Market Data first.')


def _remember_draft():
    for field in ('name', 'note'):
        st.session_state[f'strategy_{field}'] = st.session_state.get(f'strategy_{field}_input', '')


def _download(document, key, label):
    filename = re.sub(r'[^a-zA-Z0-9_-]+', '_', document['name']).strip('_')[:80] or 'strategy'
    st.download_button(label, configs.dumps(document), file_name=f'{filename}.json',
                       mime='application/json', key=key, on_click='ignore')


def render_strategy_settings(settings, factor_names, directions, lambda_available):
    with st.expander('Save or import strategy settings'):
        st.caption('Save inputs as JSON to reuse or share a strategy. Files contain no market data, '
                   'results, or credentials. Matching results also require the same data and calculation logic.')
        for field in ('name', 'note'):
            key = f'strategy_{field}_input'
            if key not in st.session_state:
                st.session_state[key] = st.session_state.get(f'strategy_{field}', '')
        st.text_input('Backtest name (optional)', key='strategy_name_input', max_chars=100, on_change=_remember_draft)
        st.text_area('Note (optional)', key='strategy_note_input', max_chars=1000, on_change=_remember_draft)
        metadata = configs.catalog(settings.get('data_mode', 'legacy'))
        columns = [metadata[n]['column'] for n in factor_names]
        engine_dirs = {metadata[n]['column']: directions[n] for n in factor_names}
        try:
            document = configs.build(settings, columns, engine_dirs,
                                     st.session_state['strategy_name_input'], st.session_state['strategy_note_input'])
        except ValueError as exc:
            st.caption(f'Export unavailable: {exc}')
        else:
            _download(document, 'strategy_export', 'Export settings (JSON)')
        uploaded = st.file_uploader('Import strategy JSON', type=['json'], key='strategy_upload',
                         help='Choose a file, then click Apply imported settings. Maximum 64 KB.')
        st.button('Apply imported settings', key='strategy_apply', on_click=_import_uploaded,
                  args=(uploaded.getvalue() if uploaded is not None else None, lambda_available),
                  disabled=uploaded is None)
        message = st.session_state.pop('strategy_import_message', None)
        if message:
            getattr(st, message[0])(message[1])


def annotate_new_run(run):
    """Called only after successful computation; annotations never enter engine/cache inputs."""
    run['default_label'] = run['label']
    run['name'] = st.session_state.get('strategy_name', '').strip()
    run['note'] = st.session_state.get('strategy_note', '')
    if run['name']:
        run['label'] = f"Run {run['id']}: {run['name']}"


def _save_annotation(run_id):
    for run in st.session_state.get('saved_runs', []):
        if run['id'] == run_id:
            run.setdefault('default_label', run['label'])
            run['name'] = st.session_state[f'run_name_{run_id}'].strip()
            run['note'] = st.session_state[f'run_note_{run_id}']
            run['label'] = f"Run {run_id}: {run['name']}" if run['name'] else run['default_label']
            break


def render_run_settings(run):
    run_id = run['id']
    if run.get('note'):
        st.text(run['note'])
    with st.expander('Name, note and settings file'):
        for field in ('name', 'note'):
            key = f'run_{field}_{run_id}'
            if key not in st.session_state:
                st.session_state[key] = run.get(field, '')
        with st.form(f'run_annotation_{run_id}'):
            st.text_input('Backtest name', key=f'run_name_{run_id}', max_chars=100)
            st.text_area('Note', key=f'run_note_{run_id}', max_chars=1000)
            st.form_submit_button('Save name and note', on_click=_save_annotation, args=(run_id,))
        st.caption('Names and notes are kept for this session and included in exported settings files.')
        try:
            document = configs.build(run['settings'], run['factors'], run['factor_directions'],
                                     run.get('name', ''), run.get('note', ''),
                                     mode=run.get('data_mode', run['settings'].get('data_mode', 'legacy')))
        except ValueError as exc:
            st.caption(f'Export unavailable: {exc}')
        else:
            _download(document, f'run_export_{run_id}', 'Export this run’s settings (JSON)')
