"""
The WRDS GICS filter in Results, Holdings, Compare and strategy JSON files, and older
WRDS analyses and files (with selected_sectors only) that must still open.
"""
import json

import pandas as pd
import pytest

from app import compare
from app import strategy_config as cfg
from app.wrds_tables import GICS_HEADERS, holdings_display_table
from src.backtest_service import BacktestService
from src.wrds_engine import HOLDINGS_GICS_COLUMNS
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab, _open_tab
from tests.unit.test_strategy_config import document
from tests.unit.test_wrds_saved import manager, new_session, s3  # noqa: F401 (fixtures)

INDUSTRY = {'level': 'industry', 'codes': [101020, 451030]}


def _wrds_document(**settings):
    doc = document('wrds')
    doc['settings'].update(settings)
    return doc


# ---- Strategy JSON ------------------------------------------------------------------------------

def test_a_gics_filter_is_exported_and_imported():
    settings = dict(document('wrds')['settings'], data_mode='wrds', sector_filter_enabled=True,
                    selected_sectors=[], gics_filter=INDUSTRY)
    doc = cfg.build(settings, ['f_roe'], {'f_roe': 'top'}, mode='wrds')
    assert doc['settings']['gics_filter'] == INDUSTRY and doc['settings']['selected_sectors'] == []
    assert cfg.loads(cfg.dumps(doc).encode()) == doc
    values = cfg.widget_values(doc)
    assert values['wrds_gics_level_input'] == 'Industry'
    assert values['wrds_sectors_input'] == ['Energy', 'Information Technology']
    assert values['wrds_industry_groups_input'] == ['Energy', 'Software & Services']
    assert values['wrds_industries_input'] == ['Oil, Gas & Consumable Fuels', 'Software']
    assert 'wrds_sub_industries_input' not in values


def test_without_a_gics_filter_nothing_new_is_exported():
    settings = dict(document('wrds')['settings'], data_mode='wrds', gics_filter=None)
    assert 'gics_filter' not in cfg.build(settings, ['f_roe'], {}, mode='wrds')['settings']
    legacy = dict(document()['settings'], gics_filter=INDUSTRY)
    assert 'gics_filter' not in cfg.build(legacy, ['ROA'], {}, mode='legacy')['settings']


def test_an_old_wrds_file_with_selected_sectors_opens_at_the_sector_level():
    doc = _wrds_document(sector_filter_enabled=True, selected_sectors=['Energy', 'Utilities'])
    assert 'gics_filter' not in doc['settings']
    values = cfg.widget_values(cfg.loads(cfg.dumps(doc).encode()))
    assert values['wrds_gics_level_input'] == 'Sector'
    assert values['wrds_sectors_input'] == ['Energy', 'Utilities']


@pytest.mark.parametrize('settings, message', [
    ({'sector_filter_enabled': True, 'selected_sectors': [], 'gics_filter': {'level': 'industry', 'codes': [9]}},
     'Invalid GICS filter'),
    ({'sector_filter_enabled': True, 'selected_sectors': [], 'gics_filter': {'level': 'country', 'codes': [10]}},
     'Invalid GICS filter'),
    ({'sector_filter_enabled': True, 'selected_sectors': ['Energy'], 'gics_filter': INDUSTRY}, 'empty selected'),
    ({'sector_filter_enabled': False, 'selected_sectors': None, 'gics_filter': INDUSTRY}, 'sector filtering on'),
])
def test_invalid_gics_files_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        cfg.loads(json.dumps(_wrds_document(**settings)))


def test_a_legacy_file_cannot_have_a_gics_filter():
    doc = document()
    doc['settings'].update(sector_filter_enabled=True, selected_sectors=[], gics_filter=INDUSTRY)
    with pytest.raises(ValueError, match='unsupported fields'):
        cfg.loads(json.dumps(doc))


# ---- Compare and holdings ------------------------------------------------------------------------

def test_the_compare_sectors_row_shows_the_filter():
    assert compare._sectors({'job': {'selected_sectors': [], 'gics_filter': INDUSTRY}}) == \
        'Industry: Oil, Gas & Consumable Fuels, Software'
    assert compare._sectors({'job': {'selected_sectors': ['Energy']}}) == 'Energy'          # an older job
    assert compare._sectors({'job': {'selected_sectors': [],
                                     'gics_filter': {'level': 'sector', 'codes': [10]}}}) == 'Energy'


def _holdings(with_gics):
    holdings = pd.DataFrame({'rank': [1, 2], 'ticker': ['A', 'B'], 'name': ['Alpha', 'Beta'],
                             'sector': ['Financials', 'Energy'], 'score': [1.0, 0.5], 'f_roe': [0.1, 0.2],
                             'weight': [0.5, 0.5], 'shares': [1.0, 2.0], 'buy_price': [10.0, 5.0],
                             'dollar_value': [10.0, 10.0], 'period_return': [0.1, 0.0],
                             'contribution': [0.05, 0.0], 'delisted': [False, False], 'permno': [1, 2],
                             'missing_return_months': [0, 0]})
    if with_gics:     # the first stock's sector comes from its SIC code: no GICS names
        holdings['gics_sector'] = pd.Categorical([None, 'Energy'])
        holdings['gics_industry_group'] = pd.Categorical([None, 'Energy'])
        holdings['gics_industry'] = pd.Categorical([None, 'Oil, Gas & Consumable Fuels'])
        holdings['gics_sub_industry'] = pd.Categorical([None, 'Oil & Gas Exploration & Production'])
    return holdings


def test_the_holdings_table_shows_the_four_gics_names_after_the_sector():
    table = holdings_display_table(_holdings(True), ['f_roe'], {'f_roe': 'ROE'})
    columns = list(table.columns)
    assert columns[3:8] == ['Sector'] + list(GICS_HEADERS.values())
    assert list(table['GICS Sub-Industry']) == ['', 'Oil & Gas Exploration & Production']
    assert set(GICS_HEADERS) == set(HOLDINGS_GICS_COLUMNS.values())


def test_holdings_saved_before_the_gics_names_still_show():
    table = holdings_display_table(_holdings(False), ['f_roe'], {'f_roe': 'ROE'})
    assert not set(GICS_HEADERS.values()) & set(table.columns)


# ---- The app --------------------------------------------------------------------------------------

def _universe_captions(at):
    return [c.value for c in at.caption if c.value.startswith('**Universe:**')]


def test_a_run_with_a_gics_filter_shows_and_saves_it(new_session):
    at = new_session()
    _click_main_tab(at, 'Analysis')
    at.sidebar.radio(key='data_mode_input').set_value('WRDS').run()
    at.sidebar.number_input(key='wrds_start_year_input').set_value(2010).run()
    at.sidebar.number_input(key='wrds_end_year_input').set_value(2012).run()
    at.button(key='factor_add_w_roe').click().run()
    at.checkbox(key='sector_filter_enabled_input').check().run()
    at.radio(key='wrds_gics_level_input').set_value('Industry').run()
    at.multiselect(key='wrds_industries_input').set_value(['Software']).run()
    _click_main_tab(at, 'Analysis')
    at.button(key='run_wrds_analysis').click().run()
    _assert_clean(at)

    run = at.session_state['saved_runs'][0]
    assert run['settings']['gics_filter'] == {'level': 'industry', 'codes': [451030]}
    assert '**Universe:** Industry: Software | Fossil fuels included' in _universe_captions(at)
    holdings = next(d.value for d in at.dataframe if 'GICS Industry' in d.value.columns)
    assert set(holdings['GICS Industry']) == {'Software'}
    exported = cfg.build(run['settings'], run['factors'], run['factor_directions'], mode='wrds')
    assert exported['settings']['gics_filter'] == {'level': 'industry', 'codes': [451030]}

    # The ID opens in another session with the same filter
    theirs = new_session()
    _click_main_tab(theirs, 'Results')
    theirs.text_input(key='open_analysis_id').set_value(run['compute']['hash']).run()
    theirs.button(key='open_analysis_btn').click().run()
    _assert_clean(theirs)
    opened = theirs.session_state['saved_runs'][0]
    assert opened['settings']['gics_filter'] == run['settings']['gics_filter']
    _open_tab(theirs, opened['label'])
    assert '**Universe:** Industry: Software | Fossil fuels included' in _universe_captions(theirs)


def test_an_older_wrds_analysis_with_selected_sectors_still_opens(new_session, manager):
    old_job = {'kind': 'backtest', 'data_mode': 'wrds', 'wrds_job_version': 1, 'factors': ['f_roe'],
               'factor_directions': {'f_roe': 'top'}, 'factor_labels': {'f_roe': 'ROE'},
               'frequency': 'quarterly', 'start_year': 2010, 'end_year': 2012, 'initial_aum': 1000.0,
               'top_pct': 10.0, 'use_market_cap_weight': False, 'factor_combination': 'mixing',
               'restrict_fossil_fuels': False, 'selected_sectors': ['Energy'], 'cohort_pct': None,
               'rebalance_date': None}
    analysis_id = BacktestService(manager).run(old_job)['hash']
    at = new_session()
    _click_main_tab(at, 'Results')
    at.text_input(key='open_analysis_id').set_value(analysis_id).run()
    at.button(key='open_analysis_btn').click().run()
    _assert_clean(at)
    opened = at.session_state['saved_runs'][0]
    assert 'gics_filter' not in opened['settings'] and opened['settings']['selected_sectors'] == ['Energy']
    _open_tab(at, opened['label'])
    assert '**Universe:** Energy | Fossil fuels included' in _universe_captions(at)
    # Its settings still export as an older file would, and import at the Sector level
    doc = cfg.build(opened['settings'], opened['factors'], opened['factor_directions'], mode='wrds')
    assert cfg.widget_values(doc)['wrds_sectors_input'] == ['Energy']
