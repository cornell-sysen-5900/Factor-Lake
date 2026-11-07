"""Inputs-only JSON contract: both modes, validation, and real Streamlit workflows."""
import json
from copy import deepcopy
from io import BytesIO

import pytest

from app import strategy_config as cfg
from app.streamlit_config import WRDS_FACTOR_METADATA
from tests.unit.test_saved_runs import (  # noqa: F401
    _add_factor,
    _assert_clean,
    _click,
    _click_main_tab,
    _open_tab,
    at,
)
from tests.unit.test_wrds_mode_switch import LEGACY_DEFAULT_SETTINGS

# The open main tab (MAIN_TABS_KEY in app/streamlit_app.py)
MAIN_TABS_KEY = 'main_tabs'


def document(mode='legacy'):
    settings = dict(LEGACY_DEFAULT_SETTINGS, trading_cost_bps=25)
    if mode == 'wrds':
        settings.pop('delisting_strategy')
        settings.update(data_mode='wrds', frequency='monthly', start_year=2010, end_year=2025)
    columns = [next(iter(cfg.catalog(mode).values()))['column']]
    return cfg.build(settings, columns, {columns[0]: 'bottom'}, '策略 A', 'Test hypothesis', mode)


@pytest.mark.parametrize('mode', ['legacy', 'wrds'])
def test_json_round_trip_and_explicit_export_allowlist(mode):
    doc = document(mode)
    assert cfg.loads(cfg.dumps(doc).encode()) == doc
    settings = dict(doc['settings'], data_mode=mode, SUPABASE_KEY='secret', data=[123], results={'x': 1},
                    use_lambda=True)
    exported = cfg.dumps(cfg.build(settings, [doc['factors'][0]['column']], {}, mode=mode))
    for excluded in ('SUPABASE_KEY', 'secret', 'results', 'use_lambda'):
        assert excluded not in exported
    assert doc['name'] == '策略 A'


@pytest.mark.parametrize('field,value', [
    ('schema_version', 2), ('schema_version', True), ('data_mode', 'unknown'),
    ('name', 2), ('name', '\ud800'), ('name', 'x' * 101), ('note', 'x' * 1001),
    ('factors', []), ('factors', [{'column': 'unknown', 'direction': 'top'}]),
    ('factors', [{'column': 'ROA', 'direction': 'wrong'}]), ('settings', []),
])
def test_invalid_document(field, value):
    doc = document()
    doc[field] = value
    with pytest.raises(ValueError):
        cfg.loads(json.dumps(doc))


def test_legacy_year_limits_come_from_config():
    from app.streamlit_config import LEGACY_FIRST_YEAR, LEGACY_LAST_START_YEAR, LEGACY_LAST_YEAR
    ok = document()
    ok['settings'].update(start_year=LEGACY_LAST_START_YEAR, end_year=LEGACY_LAST_YEAR)
    cfg.loads(json.dumps(ok))
    for start, end in [(LEGACY_FIRST_YEAR - 1, LEGACY_LAST_YEAR), (LEGACY_LAST_START_YEAR + 1, LEGACY_LAST_YEAR),
                       (LEGACY_FIRST_YEAR, LEGACY_LAST_YEAR + 1)]:
        bad = document()
        bad['settings'].update(start_year=start, end_year=end)
        with pytest.raises(ValueError):
            cfg.loads(json.dumps(bad))


@pytest.mark.parametrize('field,value', [
    ('start_year', 2001), ('start_year', True), ('end_year', 2001), ('end_year', 2025),
    ('initial_aum', float('nan')), ('initial_aum', float('inf')), ('initial_aum', 0),
    ('initial_aum', 10**400), ('initial_aum', True), ('initial_aum', '1000'),
    ('trading_cost_bps', -1), ('trading_cost_bps', 2.5), ('factor_combination', 'other'),
    ('restrict_fossil_fuels', 'false'), ('delisting_strategy', 'other'),
    ('selected_sectors', ['Technology']),
])
def test_invalid_settings(field, value):
    doc = document()
    doc['settings'][field] = value
    with pytest.raises(ValueError):
        cfg.loads(json.dumps(doc))


@pytest.mark.parametrize('raw', [b'not json', b'\xff', b'{}', b'[]', b' ' * (cfg.MAX_FILE_BYTES + 1),
                              b'{"name":"a","name":"b"}', b'[' * 2000])
def test_bad_files(raw):
    with pytest.raises(ValueError):
        cfg.loads(raw)


def test_unknown_fields_duplicate_factors_and_sectors():
    for change in (lambda d: d.update(secret='x'),
                   lambda d: d['settings'].update(unknown=1),
                   lambda d: d['factors'].append(deepcopy(d['factors'][0])),
                   lambda d: d['settings'].update(sector_filter_enabled=True, selected_sectors=['Not a sector']),
                   lambda d: d['settings'].update(sector_filter_enabled=True, selected_sectors=['Technology'] * 2)):
        doc = document()
        change(doc)
        with pytest.raises(ValueError):
            cfg.loads(json.dumps(doc))


class _UploadedFile(BytesIO):
    """What st.file_uploader returns: the bytes, and an ID that is new for every upload."""
    uploads = 0

    def __init__(self, raw):
        super().__init__(raw if isinstance(raw, bytes) else raw.encode())
        _UploadedFile.uploads += 1
        self.file_id = f'test-upload-{_UploadedFile.uploads}'


def upload(at, monkeypatch, raw):
    # AppTest cannot upload a file. Supply its bytes at that boundary only; exercise the
    # Analysis tab's import box, its rerun, the widgets and the backtest below.
    import streamlit as st
    file = _UploadedFile(raw)
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **k: file)
    _click_main_tab(at, 'Analysis')


def _next_run_name_shown(at, name):
    return any(f'The next run will be named "{name}"' in c.value for c in at.caption)


def test_import_export_run_rename_and_clear(at, monkeypatch):
    from src.backtest_jobs import results_match
    doc = document()
    doc['factors'] = [{'column': 'ROA', 'direction': 'bottom'}]
    doc['settings'].update(start_year=2010, end_year=2020, initial_aum=2500.0,
                           sector_filter_enabled=True, selected_sectors=['Technology'],
                           use_market_cap_weight=True, factor_combination='integrating', delisting_strategy='hold_cash')
    raw_data = at.session_state['raw_data']
    upload(at, monkeypatch, cfg.dumps(doc))
    _assert_clean(at)
    assert at.session_state['raw_data'] is raw_data
    assert not at.session_state['data_loaded']
    assert at.session_state['saved_runs'] == []
    assert at.radio(key='weighting_method_input').value == 'Market Cap Weight'
    assert at.radio(key='factor_combination_input').value == 'Integrating'
    assert at.radio(key='delisting_strategy_input').value == 'Hold Cash'
    assert at.number_input(key='trading_cost_bps_input').value == 25
    assert at.number_input(key='start_year_input').value == 2010
    assert at.number_input(key='end_year_input').value == 2020
    assert at.number_input(key='initial_aum_input').value == 2500.0
    assert at.multiselect(key='selected_sectors_input').value == ['Technology']
    assert at.button_group(key='factor_dir_roa_pct').value == 'Lower is better'
    # The import stays on the Analysis tab and says so in its box, with the name the next run will get
    assert at.session_state[MAIN_TABS_KEY] == 'Analysis'
    import_box = next(e for e in at.expander if e.label == 'Import settings from file')
    assert any(m.value.startswith('Settings imported.') for m in import_box.success)
    assert at.session_state['strategy_upload_version'] == 1         # the box is emptied
    assert _next_run_name_shown(at, '策略 A')
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    run = at.session_state['saved_runs'][0]
    results = deepcopy(run['results'])
    settings = deepcopy(run['settings'])
    assert run['label'] == 'Run 1: 策略 A (ROA %, 25 bps)'          # the name, then the default details
    assert cfg.build(run['settings'], run['factors'], run['factor_directions'], run['name'], run['note']) == doc
    at.text_input(key='run_name_1').set_value('Revised name')
    at.text_area(key='run_note_1').set_value('New note')
    _click(at, 'Apply')
    _assert_clean(at)
    assert run['label'] == 'Run 1: Revised name (ROA %, 25 bps)'
    assert run['settings'] == settings
    assert results_match(run['results'], results)
    _click_main_tab(at, 'Analysis')
    _click(at, 'Clear Parameters')
    assert run['name'] == 'Revised name' and run['note'] == 'New note'
    upload(at, monkeypatch, cfg.dumps(doc))
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    assert results_match(at.session_state['saved_runs'][0]['results'], results)
    assert at.session_state['saved_runs'][1] is run


def test_invalid_import_leaves_inputs_and_loaded_data_untouched(at, monkeypatch):
    _add_factor(at, 'ROA %')
    _click(at, 'Load Market Data')
    before = at.session_state['rdata']
    selected = deepcopy(at.session_state['factor_builder_selected'])
    for raw in ('{', cfg.dumps(document('wrds'))):
        upload(at, monkeypatch, raw)
        assert not at.exception
        assert at.error
        assert at.session_state['rdata'] is before
        assert at.session_state['factor_builder_selected'] == selected
        assert at.session_state['data_loaded']


def test_wrds_import_switches_mode_and_restores_frequency(at, monkeypatch):
    import streamlit_utils as utils
    monkeypatch.setattr(utils, 'get_backtest_lambda_name', lambda: 'fake-lambda')
    doc = document('wrds')
    upload(at, monkeypatch, cfg.dumps(doc))
    _assert_clean(at)
    assert at.radio(key='data_mode_input').value == 'WRDS'
    assert at.radio(key='wrds_frequency_input').value == 'Monthly'
    assert at.number_input(key='wrds_start_year_input').value == 2010
    assert at.number_input(key='wrds_end_year_input').value == 2025
    label = next(iter(WRDS_FACTOR_METADATA))
    assert at.session_state['factor_builder_selected'] == {label: 'bottom'}
    assert at.session_state['saved_runs'] == []
    upload(at, monkeypatch, cfg.dumps(document()))
    _assert_clean(at)
    assert at.radio(key='data_mode_input').value == 'Legacy'

# Use an in-process Lambda and synthetic S3 to verify WRDS without live credentials.
from tests.unit.test_wrds_saved import manager, new_session, s3  # noqa: E402,F401


def test_wrds_round_trip_keeps_cache_identity_and_results(new_session, monkeypatch):
    from src.backtest_jobs import results_match
    at = new_session()
    doc = document('wrds')
    doc['settings'].update(start_year=2010, end_year=2012, frequency='quarterly')
    doc['factors'] = [{'column': 'f_book_to_price', 'direction': 'top'}]
    upload(at, monkeypatch, cfg.dumps(doc))
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    first = at.session_state['saved_runs'][0]
    assert first['name'] == doc['name'] and first['note'] == doc['note']
    exported = cfg.build(first['settings'], first['factors'], first['factor_directions'], first['name'], first['note'])
    assert exported == doc
    _click_main_tab(at, 'Analysis')
    _click(at, 'Clear Parameters')
    doc['name'] = 'Same calculation, another name'
    upload(at, monkeypatch, cfg.dumps(doc))
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    second = at.session_state['saved_runs'][0]
    assert second['compute']['hash'] == first['compute']['hash']
    assert second['compute']['cached']
    assert results_match(second['results'], first['results'])


def test_imported_name_names_only_the_next_run_and_clear_forgets_it(at, monkeypatch):
    doc = document()
    doc['factors'] = [{'column': 'ROA', 'direction': 'top'}]
    upload(at, monkeypatch, cfg.dumps(doc))
    _click_main_tab(at, 'About')
    _click_main_tab(at, 'Analysis')
    assert _next_run_name_shown(at, '策略 A')                 # kept while other tabs are open
    _click(at, 'Clear Parameters')
    assert not _next_run_name_shown(at, '策略 A')
    upload(at, monkeypatch, cfg.dumps(doc))
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    first = at.session_state['saved_runs'][0]
    assert first['label'] == 'Run 1: 策略 A (ROA %, 25 bps)' and first['note'] == doc['note']
    _click_main_tab(at, 'Analysis')
    assert not _next_run_name_shown(at, '策略 A')             # given to that run only
    _click(at, 'Run Portfolio Analysis')
    second = at.session_state['saved_runs'][0]
    assert second['label'] == second['default_label'] and second['name'] == '' and second['note'] == ''


def test_rename_keeps_the_run_open_and_renames_it_in_the_history(at):
    _add_factor(at, 'ROA %')
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _click_main_tab(at, 'Analysis')
    _click(at, 'Run Portfolio Analysis')
    older = at.session_state['saved_runs'][1]
    _open_tab(at, older['label'])
    at.text_input(key=f"run_name_{older['id']}").set_value('Value hypothesis')
    at.text_area(key=f"run_note_{older['id']}").set_value('Lower ROA test')
    _click(at, 'Apply')
    _assert_clean(at)
    assert older['label'] == f"Run {older['id']}: Value hypothesis (ROA %)"
    assert at.session_state['saved_run_tabs'] == older['label']          # still the open run
    assert older['label'] in [e['label'] for e in at.session_state['analysis_history']]
    assert any(t.value == 'Lower ROA test' for t in at.text)
    at.text_input(key=f"run_name_{older['id']}").set_value('')
    _click(at, 'Apply')
    _assert_clean(at)
    assert older['label'] == older['default_label']
    assert older['note'] == 'Lower ROA test'


def test_use_these_settings_loads_the_run_into_the_analysis_tab(at):
    _add_factor(at, 'ROA %')
    at.button_group(key='factor_dir_roa_pct').set_value('Lower is better').run()
    next(r for r in at.sidebar.radio if r.key == 'weighting_method_input').set_value('Market Cap Weight').run()
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    run = at.session_state['saved_runs'][0]
    _click_main_tab(at, 'Analysis')
    _click(at, 'Clear Parameters')
    assert at.session_state['factor_builder_selected'] == {}
    _click_main_tab(at, 'Results')
    at.button(key=f"run_use_settings_{run['id']}").click().run()
    _assert_clean(at)
    assert at.session_state[MAIN_TABS_KEY] == 'Analysis'
    assert any(m.value.startswith(f"Run {run['id']}'s settings are loaded.") for m in at.success)
    assert at.session_state['factor_builder_selected'] == {'ROA %': 'bottom'}
    assert at.radio(key='weighting_method_input').value == 'Market Cap Weight'
    assert not at.session_state['data_loaded']                # load data for the new inputs
    assert at.session_state['saved_runs'] == [run]            # nothing was run


def test_saved_analyses_panel_floats_over_the_runs(at):
    from components.saved_panel import PANEL_CSS, PANEL_KEY
    from tests.unit.results_snapshot import main_area
    _add_factor(at, 'ROA %')
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    render = main_area(at)
    # The panel holds the saved-analysis boxes, and shows on hover with CSS only (no toggle
    # widget, no rerun). Import settings from file is on the Analysis tab, not here.
    assert render['panel']['expander'] == ['Open a saved analysis', "This session's analyses (1)",
                                           'All saved analyses']
    assert 'Open a saved analysis' not in render['expander']
    assert not at.toggle
    assert f'.st-key-{PANEL_KEY}:hover' in PANEL_CSS and 'position: fixed' in PANEL_CSS
