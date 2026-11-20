"""Current production UI: WRDS-only runs and a bounded, gated, non-downloadable preview."""
from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.strategy_config import dumps
from src.wrds_preview import APPROVAL_ENV, build_index, select_preview
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab, browser_like_widget_states
from tests.unit.test_strategy_config import document
from tests.unit.test_wrds_preview import panel

PREVIEW_STATE_KEY = 'wrds_preview'

APP = Path(__file__).resolve().parents[2] / 'app' / 'streamlit_app.py'


@pytest.fixture
def app_factory(monkeypatch):
    monkeypatch.syspath_prepend(str(APP.parent))
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', 'preview-dev')
    monkeypatch.setenv(APPROVAL_ENV, 'true')
    browser_like_widget_states(monkeypatch)
    index = build_index([panel()])
    class Gateway:
        def __init__(self):
            self.calls = []
        def preview(self, request):
            self.calls.append(deepcopy(request))
            return {'status': 'ok', 'preview': select_preview(index, request)}
    gateway = Gateway()
    monkeypatch.setattr('app.streamlit_utils.get_lambda_client', lambda _: gateway)
    def start():
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        at.gateway = gateway
        _assert_clean(at)
        return at
    return start


def test_wrds_is_the_only_new_run_mode_even_without_lambda(app_factory, monkeypatch):
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', '')
    at = app_factory()
    assert not any(r.key == 'data_mode_input' for r in at.radio)
    assert 'Load Market Data' not in [b.label for b in at.button]
    assert not at.sidebar.toggle
    assert at.session_state['data_mode_input'] == 'WRDS'
    at.button(key='factor_add_w_roa').click().run()
    assert at.button(key='run_wrds_analysis').disabled
    assert not at.session_state['saved_runs']


def test_stale_legacy_inputs_cannot_reactivate_legacy(app_factory):
    at = app_factory()
    at.session_state['data_mode_input'] = 'Legacy'
    at.session_state['factor_builder_selected'] = {'ROA %': 'top'}
    at.session_state['raw_data'] = panel()
    at.session_state['rdata'] = panel()
    at.run()
    _assert_clean(at)
    assert at.session_state['data_mode_input'] == 'WRDS'
    assert at.session_state['factor_builder_selected'] == {}
    assert not any(b.key == 'factor_add_roa_pct' for b in at.button)
    assert at.session_state['raw_data'] is None and at.session_state['rdata'] is None


def test_legacy_import_and_queued_use_settings_are_rejected_without_mutating_inputs(app_factory):
    at = app_factory()
    at.button(key='factor_add_w_roa').click().run()
    before = deepcopy(at.session_state['factor_builder_selected'])
    for source, payload in [('file', dumps(document())), ('run', document())]:
        at.session_state['strategy_pending_settings'] = (source, payload, 1)
        at.run()
        assert not at.exception
        assert at.session_state['data_mode_input'] == 'WRDS'
        assert at.session_state['factor_builder_selected'] == before
        assert at.session_state['saved_runs'] == []
        if source == 'file':
            assert any('WRDS only' in e.value for e in at.error)
        else:
            assert 'WRDS only' in at.session_state['run_settings_error_1']


def test_disabled_preview_never_calls_lambda(app_factory, monkeypatch):
    monkeypatch.setenv(APPROVAL_ENV, 'false')
    at = app_factory()
    _click_main_tab(at, 'Data Viewer')
    assert any('approval' in i.value for i in at.info)
    assert at.gateway.calls == []
    assert not at.get('download_button')
    assert 'Load preview' not in [b.label for b in at.button]


def test_preview_has_all_gics_levels_no_download_and_escaped_html(app_factory):
    at = app_factory()
    _click_main_tab(at, 'Data Viewer')
    assert [m.label for m in at.multiselect if (m.key or '').startswith('preview_gics_')] == [
        'Sector', 'Industry group', 'Industry', 'Sub-industry']
    at.slider(key='preview_years').set_value((2010, 2012)).run()
    at.button(key='preview_load').click().run()
    _assert_clean(at)
    holder = at.session_state[PREVIEW_STATE_KEY]
    assert len(holder['preview']['rows']) == 15
    html = at.get('html')[0].proto.body
    assert '&lt;script&gt;' in html and '<script>' not in html
    assert not at.get('download_button') and not at.dataframe
    assert at.session_state.filtered_state.get('raw_data') is None
    assert at.session_state['rdata'] is None


def test_filters_clear_previous_samples_and_filter_before_sampling(app_factory):
    at = app_factory()
    _click_main_tab(at, 'Data Viewer')
    at.slider(key='preview_years').set_value((2010, 2012)).run()
    at.button(key='preview_load').click().run()
    at.multiselect(key='preview_gics_sector').set_value(['Financials']).run()
    assert PREVIEW_STATE_KEY not in at.session_state
    assert 'Banks' in at.multiselect(key='preview_gics_industry_group').options
    assert 'Energy' not in at.multiselect(key='preview_gics_industry_group').options
    at.multiselect(key='preview_gics_industry_group').set_value(['Banks']).run()
    at.button(key='preview_load').click().run()
    assert at.gateway.calls[-1]['gics_filter'] == {'level': 'industry_group', 'codes': [4010]}
    assert all(r['gsubind'] == 40101010 for r in at.session_state[PREVIEW_STATE_KEY]['preview']['rows'])
    at.multiselect(key='preview_gics_sector').set_value(['Energy']).run()
    assert at.multiselect(key='preview_gics_industry_group').value == []
    assert PREVIEW_STATE_KEY not in at.session_state


@pytest.mark.parametrize('tab', ['Analysis', 'Results', 'Compare', 'About'])
def test_leaving_preview_releases_rows(app_factory, tab):
    at = app_factory()
    _click_main_tab(at, 'Data Viewer')
    at.button(key='preview_load').click().run()
    assert PREVIEW_STATE_KEY in at.session_state
    _click_main_tab(at, tab)
    assert PREVIEW_STATE_KEY not in at.session_state


def test_repeated_loads_replace_one_bounded_response_and_clear_removes_it(app_factory):
    at = app_factory()
    _click_main_tab(at, 'Data Viewer')
    for years in [(2010, 2010), (2011, 2012), (2010, 2012)] * 3:
        at.slider(key='preview_years').set_value(years).run()
        at.button(key='preview_load').click().run()
        holder = at.session_state[PREVIEW_STATE_KEY]
        assert len(holder['preview']['rows']) == 5 * (years[1] - years[0] + 1)
        assert set(holder) == {'request', 'function', 'preview'}
    at.button(key='preview_clear').click().run()
    assert PREVIEW_STATE_KEY not in at.session_state
    assert len(at.gateway.calls) == 9


def test_empty_results_and_failures_are_readable_and_clear_stale_rows(app_factory, monkeypatch):
    at = app_factory()
    _click_main_tab(at, 'Data Viewer')
    at.slider(key='preview_years').set_value((2020, 2021)).run()
    at.button(key='preview_load').click().run()
    assert any('No WRDS rows match' in i.value for i in at.info)
    def fail(_):
        raise RuntimeError('secret internal details')
    monkeypatch.setattr(at.gateway, 'preview', fail)
    at.button(key='preview_load').click().run()
    assert PREVIEW_STATE_KEY not in at.session_state
    assert any('could not be loaded' in e.value and 'secret' not in e.value for e in at.error)
