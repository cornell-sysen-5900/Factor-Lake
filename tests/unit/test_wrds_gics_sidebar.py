"""
The WRDS sidebar's GICS filter (app/components/sidebar.py): a level picker, then one list per
level that only offers what is inside the choices above it, and the job it sends.
"""
from app.streamlit_config import WRDS_FACTOR_METADATA
from src.wrds_factors import WRDS_SECTORS
from tests.unit.test_wrds_mode_switch import (  # noqa: F401 (lambda_at is a fixture)
    _add,
    _assert_clean,
    _button,
    _captions,
    _sidebar_radio_labels,
    _switch,
    lambda_at,
)

LEVEL = 'wrds_gics_level_input'
LISTS = ('wrds_sectors_input', 'wrds_industry_groups_input', 'wrds_industries_input', 'wrds_sub_industries_input')


def _enable(at):
    _switch(at, 'WRDS')
    next(c for c in at.sidebar.checkbox if c.label == 'Enable Sector Filter').check().run()
    _assert_clean(at)


def _lists(at):
    return [m.key for m in at.sidebar.multiselect]


def _run(at):
    _add(at, WRDS_FACTOR_METADATA, 'Book/Price')
    _button(at, 'Run Portfolio Analysis').click().run()
    return at.fake_lambda.jobs[-1]


def test_the_level_picker_starts_on_sector_with_every_sector(lambda_at):
    at = lambda_at
    _enable(at)
    level = at.sidebar.radio(key=LEVEL)
    assert level.label == 'Filter level' and level.value == 'Sector'
    assert level.options == ['Sector', 'Industry group', 'Industry', 'Sub-industry']
    assert 'current_header' in level.help                       # said once, on the level picker
    assert _sidebar_radio_labels(at)[-1] == 'Filter level'
    assert _lists(at) == ['wrds_sectors_input']
    sectors = at.sidebar.multiselect(key='wrds_sectors_input')
    assert sectors.options == list(WRDS_SECTORS) and sectors.value == list(WRDS_SECTORS)
    assert 'current_header' not in sectors.help

    job = _run(at)                                              # all 11: Unclassified stocks are left out
    assert job['gics_filter'] == {'level': 'sector', 'codes': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]}
    assert job['selected_sectors'] == []


def test_each_list_narrows_to_the_choices_above(lambda_at):
    at = lambda_at
    _enable(at)
    at.sidebar.radio(key=LEVEL).set_value('Sub-industry').run()
    assert _lists(at) == list(LISTS)
    assert len(at.sidebar.multiselect(key='wrds_industry_groups_input').options) == 27   # all while none chosen
    at.sidebar.multiselect(key='wrds_sectors_input').set_value(['Financials']).run()
    groups = at.sidebar.multiselect(key='wrds_industry_groups_input')
    assert groups.options == ['Banks', 'Financial Services', 'Insurance', 'Real Estate (discontinued 2016)']
    assert groups.value == [] and groups.placeholder == 'All industry groups'
    groups.set_value(['Banks']).run()
    assert at.sidebar.multiselect(key='wrds_industries_input').options == [
        'Banks', 'Thrifts & Mortgage Finance (discontinued 2023)']
    # No industry chosen: the sub-industries of every bank industry
    subs = at.sidebar.multiselect(key='wrds_sub_industries_input')
    assert subs.options == ['Diversified Banks', 'Regional Banks', 'Thrifts & Mortgage Finance (discontinued 2023)']
    assert subs.placeholder == 'Choose sub-industries'
    assert 'No sub-industries chosen: the filter is off.' in _captions(at)
    subs.set_value(['Regional Banks']).run()
    _assert_clean(at)

    job = _run(at)
    assert job['gics_filter'] == {'level': 'sub_industry', 'codes': [40101015]}


def test_a_choice_outside_a_narrower_list_above_is_dropped(lambda_at):
    at = lambda_at
    _enable(at)
    at.sidebar.radio(key=LEVEL).set_value('Industry').run()
    at.sidebar.multiselect(key='wrds_industries_input').set_value(['Insurance', 'Software']).run()
    at.sidebar.multiselect(key='wrds_sectors_input').set_value(['Information Technology']).run()
    _assert_clean(at)
    assert at.sidebar.multiselect(key='wrds_industries_input').value == ['Software']
    assert _run(at)['gics_filter'] == {'level': 'industry', 'codes': [451030]}


def test_no_choice_at_the_level_sends_no_filter(lambda_at):
    at = lambda_at
    _enable(at)
    at.sidebar.radio(key=LEVEL).set_value('Industry group').run()
    job = _run(at)
    assert 'gics_filter' not in job and job['selected_sectors'] == []


def test_switching_back_to_legacy_shows_the_legacy_sector_list(lambda_at):
    at = lambda_at
    _enable(at)
    at.sidebar.radio(key=LEVEL).set_value('Industry').run()
    _switch(at, 'Legacy')
    assert _lists(at) == ['selected_sectors_input']
    assert 'Filter level' not in _sidebar_radio_labels(at)
