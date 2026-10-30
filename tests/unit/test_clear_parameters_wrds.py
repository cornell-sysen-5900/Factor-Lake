"""Clear Parameters in WRDS mode: the WRDS inputs return to their defaults and the data source is kept."""
from src.wrds_calendar import FIRST_YEAR, LAST_YEAR
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab
from tests.unit.test_wrds_saved import _run_wrds, manager, new_session, s3  # noqa: F401 (fixtures)


def test_clear_in_wrds_mode_restores_its_inputs_and_keeps_the_data_source(new_session):
    at = new_session()
    at.sidebar.radio(key='data_mode_input').set_value('WRDS').run()
    at.radio(key='wrds_frequency_input').set_value('Monthly').run()
    at.number_input(key='wrds_start_year_input').set_value(2010).run()
    at.number_input(key='wrds_end_year_input').set_value(2012).run()
    at.checkbox(key='sector_filter_enabled_input').check().run()
    at.multiselect(key='wrds_sectors_input').set_value(['Consumer Staples']).run()
    at.radio(key='factor_combination_input').set_value('Integrating').run()
    at.number_input(key='initial_aum_input').set_value(5000.0).run()
    for key in ('w_roe', 'w_mcap'):
        at.button(key=f'factor_add_{key}').click().run()
    at.button_group(key='factor_dir_w_roe').set_value('Lower is better').run()

    at.button(key='clear_parameters').click().run()
    _assert_clean(at)

    assert at.radio(key='data_mode_input').value == 'WRDS'                 # the data source is kept
    for widget in list(at.sidebar.checkbox) + list(at.sidebar.radio) + list(at.sidebar.number_input):
        if widget.key != 'data_mode_input':
            assert widget.proto.set_value, f'{widget.label} did not send its reset to the browser'
    assert at.radio(key='wrds_frequency_input').value == 'Quarterly'
    assert at.number_input(key='wrds_start_year_input').value == FIRST_YEAR
    assert at.number_input(key='wrds_end_year_input').value == LAST_YEAR
    assert at.radio(key='factor_combination_input').value == 'Mixing'
    assert not at.checkbox(key='sector_filter_enabled_input').value and not at.multiselect
    assert at.number_input(key='initial_aum_input').value == 1000.0
    assert at.session_state['factor_builder_selected'] == {} and not at.button_group
    assert any(i.value.startswith('Parameters cleared. Choose new inputs, then run the analysis.') for i in at.info)

    # Old sector and direction choices do not come back, and Legacy shows its defaults
    at.checkbox(key='sector_filter_enabled_input').check().run()
    assert len(at.multiselect(key='wrds_sectors_input').value) == 11
    at.button(key='factor_add_w_roe').click().run()
    assert at.button_group(key='factor_dir_w_roe').value == 'Higher is better'
    at.sidebar.radio(key='data_mode_input').set_value('Legacy').run()
    _assert_clean(at)
    assert at.radio(key='delisting_strategy_input').value == 'Zero Return'
    assert at.number_input(key='start_year_input').value == 2002


def test_a_wrds_run_after_clearing_uses_the_new_inputs(new_session):
    at = new_session()
    first = _run_wrds(at, 'w_roe')
    _click_main_tab(at, 'Analysis')
    at.button(key='clear_parameters').click().run()
    at.number_input(key='wrds_start_year_input').set_value(2010).run()
    at.number_input(key='wrds_end_year_input').set_value(2012).run()
    at.radio(key='wrds_frequency_input').set_value('Semi-annual').run()
    at.button(key='factor_add_w_roa').click().run()
    at.button(key='run_wrds_analysis').click().run()
    _assert_clean(at)
    second = at.session_state['saved_runs'][0]
    assert second['label'] == 'Run 2: WRDS, Semi-annual' and second['factor_labels'] == ['ROA']
    assert at.session_state['saved_runs'][1] is first                     # the earlier run is kept
