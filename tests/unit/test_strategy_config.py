"""Inputs-only JSON contract: both modes, validation, and real Streamlit workflows."""
from copy import deepcopy
from io import BytesIO
import json
import pytest
from app import strategy_config as cfg
from app.streamlit_config import WRDS_FACTOR_METADATA
from tests.unit.test_wrds_mode_switch import LEGACY_DEFAULT_SETTINGS
from tests.unit.test_saved_runs import at, _add_factor, _click, _click_main_tab, _assert_clean  # noqa: F401


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
    settings = dict(doc['settings'], data_mode=mode, SUPABASE_KEY='secret', data=[123], results={'x': 1}, use_lambda=True)
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


def upload(at, monkeypatch, raw):
    # AppTest cannot upload a file. Supply its bytes at that boundary only; exercise
    # the actual Apply button callback, widgets, reruns and backtest below.
    import streamlit as st
    file = BytesIO(raw if isinstance(raw, bytes) else raw.encode())
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **k: file)
    at.run()
    at.button(key='strategy_apply').click().run()


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
    assert at.text_input(key='strategy_name_input').value == '策略 A'
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    run = at.session_state['saved_runs'][0]
    results = deepcopy(run['results'])
    settings = deepcopy(run['settings'])
    assert run['label'] == 'Run 1: 策略 A'
    assert cfg.build(run['settings'], run['factors'], run['factor_directions'], run['name'], run['note']) == doc
    at.text_input(key='run_name_1').set_value('Revised name')
    at.text_area(key='run_note_1').set_value('New note')
    _click(at, 'Save name and note')
    _assert_clean(at)
    assert run['label'] == 'Run 1: Revised name'
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
from tests.unit.test_wrds_saved import new_session, manager, s3  # noqa: E402,F401


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


def test_draft_name_persists_across_tabs_and_blank_restores_default(at):
    at.text_input(key='strategy_name_input').set_value('Value hypothesis').run()
    at.text_area(key='strategy_note_input').set_value('Lower ROA test').run()
    _click_main_tab(at, 'About')
    _click_main_tab(at, 'Analysis')
    assert at.text_input(key='strategy_name_input').value == 'Value hypothesis'
    _add_factor(at, 'ROA %')
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    run = at.session_state['saved_runs'][0]
    assert run['name'] == 'Value hypothesis'
    at.text_input(key='run_name_1').set_value('')
    _click(at, 'Save name and note')
    _assert_clean(at)
    assert run['label'] == run['default_label']
    assert run['note'] == 'Lower ROA test'
