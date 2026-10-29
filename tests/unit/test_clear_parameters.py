"""Clear and rerun through the real Streamlit UI, using synthetic data and fake AWS."""
from copy import deepcopy

import pandas as pd
import pytest

from app.streamlit_config import FACTOR_METADATA, SECTOR_OPTIONS
from src.backtest_jobs import results_match
from tests.unit.test_backtest_lambda import lambda_app, manager, s3  # noqa: F401 (fixtures)
from tests.unit.test_saved_runs import (
    _add_factor, _assert_clean, _click, _click_main_tab, _open_tab, at,
)  # noqa: F401 (fixture)


def _clear(at):
    at.button(key='clear_parameters').click().run()
    _assert_clean(at)


def test_clear_restores_every_analysis_input_and_factor_direction(at):
    at.checkbox(key='restrict_fossil_fuels_input').check().run()
    at.radio(key='weighting_method_input').set_value('Market Cap Weight').run()
    at.radio(key='factor_combination_input').set_value('Integrating').run()
    at.radio(key='delisting_strategy_input').set_value('Reinvest').run()
    at.checkbox(key='sector_filter_enabled_input').check().run()
    at.multiselect(key='selected_sectors_input').set_value(['Technology']).run()
    at.number_input(key='start_year_input').set_value(2010).run()
    at.number_input(key='end_year_input').set_value(2020).run()
    at.number_input(key='initial_aum_input').set_value(5000.0).run()
    for name, meta in FACTOR_METADATA.items():
        _add_factor(at, name)
        opposite = 'Lower is better' if meta['higher_is_better'] else 'Higher is better'
        at.button_group(key=f"factor_dir_{meta['key']}").set_value(opposite).run()

    _clear(at)

    # The server must tell the browser to replace the old visible values, not
    # just return defaults internally (AppTest has no actual browser frontend).
    for widget in list(at.sidebar.checkbox) + list(at.sidebar.radio) + list(at.sidebar.number_input):
        assert widget.proto.set_value, f'{widget.label} did not send its reset to the browser'
    assert not at.checkbox(key='restrict_fossil_fuels_input').value
    assert at.radio(key='weighting_method_input').value == 'Equal Weight'
    assert at.radio(key='factor_combination_input').value == 'Mixing'
    assert at.radio(key='delisting_strategy_input').value == 'Zero Return'
    assert not at.checkbox(key='sector_filter_enabled_input').value
    assert not at.multiselect
    assert at.number_input(key='start_year_input').value == 2002
    assert at.number_input(key='end_year_input').value == 2024
    assert at.number_input(key='initial_aum_input').value == 1000.0
    assert at.session_state['factor_builder_selected'] == {}
    assert not at.button_group
    assert len([b for b in at.button if (b.key or '').startswith('factor_add_')]) == 13
    assert next(b for b in at.button if b.label == 'Load Market Data').disabled

    # Old sector and direction choices must not reappear when the controls return.
    at.checkbox(key='sector_filter_enabled_input').check().run()
    assert at.multiselect(key='selected_sectors_input').value == SECTOR_OPTIONS
    for name, meta in FACTOR_METADATA.items():
        _add_factor(at, name)
        assert at.button_group(key=f"factor_dir_{meta['key']}").value == (
            'Higher is better' if meta['higher_is_better'] else 'Lower is better')
    _assert_clean(at)


def test_clear_empty_inputs_is_repeatable_and_does_not_run_or_download(at, monkeypatch):
    import streamlit_utils as utils

    def unexpected(*args, **kwargs):
        pytest.fail('Clearing parameters must not load data or run a backtest')

    monkeypatch.setattr(utils, 'load_and_process_data', unexpected)
    monkeypatch.setattr(utils, 'run_backtest_logic', unexpected)
    at.session_state['unrelated_state'] = {'keep': True}
    raw = at.session_state['raw_data']
    for _ in range(2):
        _clear(at)
        assert at.session_state['saved_runs'] == []
        assert at.session_state['analysis_history'] == []
        assert at.session_state['next_run_id'] == 1
        assert at.session_state['raw_data'] is raw
        assert at.session_state['unrelated_state'] == {'keep': True}


def test_local_clear_and_rerun_uses_new_universe_and_keeps_old_results(at, monkeypatch):
    import streamlit_utils as utils

    raw = at.session_state['raw_data']
    raw.loc[raw['Ticker'] >= 'T15', 'Scotts_Sector_5'] = 'Healthcare'
    original_raw = raw.copy(deep=True)

    def unexpected_download(*args, **kwargs):
        pytest.fail('The cached source data must be reused')

    monkeypatch.setattr(utils, 'load_shared_universe', unexpected_download)
    _add_factor(at, 'ROE using 9/30 Data')
    at.checkbox(key='sector_filter_enabled_input').check().run()
    at.multiselect(key='selected_sectors_input').set_value(['Technology']).run()
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    first = at.session_state['saved_runs'][0]
    original_first = deepcopy(first)
    history = deepcopy(at.session_state['analysis_history'])

    _click_main_tab(at, 'Analysis')
    _clear(at)

    assert at.session_state['saved_runs'][0] is first
    assert at.session_state['analysis_history'] == history
    assert at.session_state['next_run_id'] == 2
    assert at.session_state['raw_data'] is raw
    assert not at.session_state['data_loaded']
    assert at.session_state['rdata'] is None
    assert at.session_state['loaded_filters'] is None
    assert not any(b.label == 'Run Portfolio Analysis' for b in at.button)
    _click_main_tab(at, 'Data Viewer')
    assert any('Load market data' in i.value for i in at.info)
    _click_main_tab(at, 'Results')
    _open_tab(at, first['label'])
    assert at.metric  # Old results still render after clearing.
    _click_main_tab(at, 'Analysis')

    _add_factor(at, 'ROA %')
    at.button_group(key='factor_dir_roa_pct').set_value('Lower is better').run()
    at.checkbox(key='sector_filter_enabled_input').check().run()
    at.multiselect(key='selected_sectors_input').set_value(['Healthcare']).run()
    at.number_input(key='start_year_input').set_value(2010).run()
    at.number_input(key='end_year_input').set_value(2015).run()
    at.number_input(key='initial_aum_input').set_value(2500.0).run()
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)

    second, kept_first = at.session_state['saved_runs']
    assert [second['id'], kept_first['id']] == [2, 1]
    assert second['factor_labels'] == ['ROA %']
    assert second['factor_directions'] == {'ROA': 'bottom'}
    assert second['settings']['initial_aum'] == 2500.0
    assert second['settings']['start_year'] == 2010
    assert second['settings']['end_year'] == 2015
    assert set(second['data']['Scotts_Sector_5']) == {'Healthcare'}
    assert set(second['data']['Year']) == set(range(2010, 2016))
    assert second['universe']['selected_sectors'] == ['Healthcare']
    assert not results_match(second['results'], first['results'])
    assert first['settings'] == original_first['settings']
    assert first['factor_directions'] == original_first['factor_directions']
    assert results_match(first['results'], original_first['results'])
    pd.testing.assert_frame_equal(first['data'], original_first['data'])
    pd.testing.assert_frame_equal(raw, original_raw)
    assert len(at.session_state['analysis_history']) == 2


@pytest.mark.parametrize('use_lambda', [True, False])
def test_clear_keeps_execution_preference(lambda_app, use_lambda):
    at = lambda_app
    at.toggle(key='use_lambda_input').set_value(use_lambda).run()
    _add_factor(at, 'ROE using 9/30 Data')
    _clear(at)
    assert at.toggle(key='use_lambda_input').value is use_lambda
    assert at.fake_lambda.invocations == 0
    assert at.session_state['raw_data'] is None
    if use_lambda:
        assert next(b for b in at.button if b.label == 'Run Portfolio Analysis').disabled
    else:
        assert not any(b.label == 'Run Portfolio Analysis' for b in at.button)


def test_lambda_clear_and_rerun_uses_new_parameters_without_loading_data(lambda_app):
    at = lambda_app
    _add_factor(at, 'ROE using 9/30 Data')
    _click(at, 'Run Portfolio Analysis')
    first = at.session_state['saved_runs'][0]
    original_result = deepcopy(first['results'])
    _click_main_tab(at, 'Analysis')
    _clear(at)
    assert at.fake_lambda.invocations == 1

    _add_factor(at, 'ROA %')
    at.number_input(key='initial_aum_input').set_value(3000.0).run()
    at.number_input(key='start_year_input').set_value(2010).run()
    at.radio(key='factor_combination_input').set_value('Integrating').run()
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    second, kept_first = at.session_state['saved_runs']
    assert at.fake_lambda.invocations == 2
    assert at.session_state['raw_data'] is None
    assert second['data'] is None
    assert second['settings']['initial_aum'] == 3000.0
    assert second['settings']['start_year'] == 2010
    assert second['settings']['factor_combination'] == 'integrating'
    assert second['factor_labels'] == ['ROA %']
    assert second['compute']['hash'] != first['compute']['hash']
    assert not second['compute']['cached']
    assert kept_first is first
    assert results_match(first['results'], original_result)
    assert len(at.session_state['analysis_history']) == 2
