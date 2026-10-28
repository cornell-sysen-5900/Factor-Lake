"""
Tests for the Legacy / WRDS data source switch (stage 2 of DOCS/WRDS_MODE_DESIGN.md):
the switch, the factor reset, which controls each mode shows, and that Legacy's
settings and jobs are exactly as before WRDS mode.

AppTest drives app/streamlit_app.py with the synthetic universe of test_saved_runs.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.streamlit_config import FACTOR_METADATA, WRDS_FACTOR_GROUPS, WRDS_FACTOR_METADATA
from src.backtest_jobs import JOB_FIELDS, build_job, run_job
from src.wrds_factors import WRDS_SECTORS
from tests.unit.test_saved_runs import _synthetic_universe

APP_DIR = Path(__file__).resolve().parents[2] / 'app'
LAMBDA_NAME = 'factor-lake-backtest'
LOWER = 'Lower is better'
HIGHER = 'Higher is better'

# The sidebar's Legacy settings on main, before WRDS mode (local backtests)
LEGACY_DEFAULT_SETTINGS = {
    'restrict_fossil_fuels': False,
    'use_market_cap_weight': False,
    'factor_combination': 'mixing',
    'selected_sectors': None,
    'sector_filter_enabled': False,
    'start_year': 2002,
    'end_year': 2024,
    'initial_aum': 1000.0,
    'delisting_strategy': 'zero_return',
    'use_lambda': False,
}


def _app(monkeypatch, backtest_lambda=''):
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', backtest_lambda)
    # `streamlit run` puts app/ on sys.path; AppTest does not.
    monkeypatch.syspath_prepend(str(APP_DIR))
    at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    at.session_state['raw_data'] = _synthetic_universe()
    at.run()
    return at


@pytest.fixture
def at(monkeypatch):
    """The app without a backtest Lambda."""
    return _app(monkeypatch)


class FakeLambdaClient:
    """Stands in for BacktestLambdaClient: records each job and runs Legacy jobs in-process."""

    def __init__(self):
        self.jobs = []

    def run(self, job):
        self.jobs.append(job)
        info = {'hash': 'a' * 64, 'cached': False, 'computed_at': None, 'compute_seconds': 0.1,
                'size_bytes': 1, 'function_name': LAMBDA_NAME, 'delivery': 'inline',
                'round_trip_seconds': 0.1}
        return run_job(_synthetic_universe(), job), info


@pytest.fixture
def lambda_at(monkeypatch):
    """The app with a backtest Lambda configured (served by FakeLambdaClient)."""
    import streamlit as st
    st.cache_resource.clear()          # get_lambda_client keeps one client per function name
    client = FakeLambdaClient()

    def init(self, function_name=None, region=None, client=None):
        self.function_name = function_name
    monkeypatch.setattr('src.lambda_client.BacktestLambdaClient.__init__', init)
    monkeypatch.setattr('src.lambda_client.BacktestLambdaClient.run', lambda self, job: client.run(job))
    at = _app(monkeypatch, LAMBDA_NAME)
    at.fake_lambda = client
    yield at
    st.cache_resource.clear()


def _assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _mode(at):
    return at.sidebar.radio(key='data_mode_input')


def _switch(at, mode):
    _mode(at).set_value(mode).run()
    _assert_clean(at)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _labels(at):
    return [b.label for b in at.button]


def _add(at, metadata, *names):
    for name in names:
        at.button(key=f"factor_add_{metadata[name]['key']}").click().run()


def _sidebar_radio_labels(at):
    return [r.label for r in at.sidebar.radio]


def _number_inputs(at):
    return {n.key: n for n in at.sidebar.number_input}


def _captions(at):
    return [c.value for c in at.sidebar.caption]


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_legacy_is_the_default_and_wrds_needs_lambda(at):
    _assert_clean(at)
    assert _mode(at).value == 'Legacy' and _mode(at).options == ['Legacy', 'WRDS']
    assert _mode(at).disabled
    assert 'WRDS mode runs in AWS Lambda, which is not configured for this app.' in _captions(at)


def test_with_lambda_the_switch_is_enabled_and_starts_on_legacy(lambda_at):
    _assert_clean(lambda_at)
    assert _mode(lambda_at).value == 'Legacy' and not _mode(lambda_at).disabled
    assert 'WRDS mode runs in AWS Lambda, which is not configured for this app.' not in _captions(lambda_at)


def test_switching_resets_the_factor_selection(lambda_at):
    at = lambda_at
    _add(at, FACTOR_METADATA, 'ROA %', '1-Yr Price Vol %')
    at.button_group(key=f"factor_dir_{FACTOR_METADATA['ROA %']['key']}").set_value(LOWER).run()
    assert list(at.session_state['factor_builder_selected']) == ['ROA %', '1-Yr Price Vol %']

    _switch(at, 'WRDS')
    assert at.session_state['factor_builder_selected'] == {}
    assert 'No factors added yet. Click one above.' in [c.value for c in at.caption]
    assert not any(k.startswith('factor_dir_') for k in at.session_state.filtered_state)

    _add(at, WRDS_FACTOR_METADATA, 'Book/Price')
    assert at.session_state['factor_builder_selected'] == {'Book/Price': 'top'}
    _switch(at, 'Legacy')
    assert at.session_state['factor_builder_selected'] == {}

    # Legacy factors come back with their defaults, not the directions chosen before the switch
    _add(at, FACTOR_METADATA, 'ROA %')
    assert at.button_group(key=f"factor_dir_{FACTOR_METADATA['ROA %']['key']}").value == HIGHER


# ---------------------------------------------------------------------------
# Controls of each mode
# ---------------------------------------------------------------------------

def test_legacy_controls_are_as_on_main(lambda_at):
    at = lambda_at
    assert _sidebar_radio_labels(at) == ['Data source', 'Select weighting method:', 'Combine factors',
                                         'How to treat delisted positions:']
    assert [t.key for t in at.sidebar.toggle] == ['use_lambda_input']
    years = _number_inputs(at)
    assert (years['start_year_input'].min, years['start_year_input'].max, years['start_year_input'].value) \
        == (2002, 2023, 2002)
    assert (years['end_year_input'].min, years['end_year_input'].max, years['end_year_input'].value) \
        == (2002, 2024, 2024)
    assert 'wrds_start_year_input' not in years
    assert 'Load Market Data' in _labels(at)
    assert {f"factor_add_{m['key']}" for m in FACTOR_METADATA.values()} <= {b.key for b in at.button}
    assert not any((b.key or '').startswith('factor_add_w_') for b in at.button)


def test_wrds_controls(lambda_at):
    at = lambda_at
    _switch(at, 'WRDS')

    # Frequency instead of the delisting option, and no Lambda switch (WRDS always runs in Lambda)
    assert _sidebar_radio_labels(at) == ['Data source', 'Select weighting method:', 'Combine factors',
                                         'Rebalancing frequency']
    frequency = at.sidebar.radio(key='wrds_frequency_input')
    assert frequency.options == ['Semi-annual', 'Quarterly', 'Monthly'] and frequency.value == 'Quarterly'
    assert not at.sidebar.toggle
    assert f"WRDS backtests run in AWS Lambda: `{LAMBDA_NAME}`" in _captions(at)

    # Years limited to the data, with the rebalances they give
    years = _number_inputs(at)
    assert (years['wrds_start_year_input'].min, years['wrds_start_year_input'].max,
            years['wrds_start_year_input'].value) == (2008, 2026, 2008)
    assert (years['wrds_end_year_input'].min, years['wrds_end_year_input'].max,
            years['wrds_end_year_input'].value) == (2008, 2026, 2026)
    assert 'start_year_input' not in years
    assert '73 quarterly rebalances, 2008-06 to 2026-06; holds to 2026-09.' in _captions(at)

    at.sidebar.radio(key='wrds_frequency_input').set_value('Monthly').run()
    assert '221 monthly rebalances, 2008-04 to 2026-08; holds to 2026-09.' in _captions(at)

    # The 11 GICS sectors, without Unclassified
    next(c for c in at.sidebar.checkbox if c.label == 'Enable Sector Filter').check().run()
    sectors = at.sidebar.multiselect(key='wrds_sectors_input')
    assert sectors.options == list(WRDS_SECTORS) and sectors.value == list(WRDS_SECTORS)

    # Nothing to load: WRDS runs in Lambda
    assert 'Load Market Data' not in _labels(at)
    _assert_clean(at)


def test_wrds_factor_list_grouped_with_names_tooltips_and_defaults(lambda_at):
    at = lambda_at
    _switch(at, 'WRDS')
    titles = [m.value for m in at.markdown if m.value in {f"**{g}**" for g in WRDS_FACTOR_GROUPS}]
    assert titles == [f"**{g}**" for g in WRDS_FACTOR_GROUPS]
    for name, meta in WRDS_FACTOR_METADATA.items():
        button = at.button(key=f"factor_add_{meta['key']}")
        assert button.label == name and button.help == meta['tooltip']
    add_keys = [b.key for b in at.button if (b.key or '').startswith('factor_add_')]
    assert add_keys and all(k.startswith('factor_add_w_') for k in add_keys)   # no Legacy factors

    _add(at, WRDS_FACTOR_METADATA, 'Book/Price', '1-Month Return (Reversal)', 'Market Cap')
    _assert_clean(at)
    directions = {n: at.button_group(key=f"factor_dir_{WRDS_FACTOR_METADATA[n]['key']}").value
                  for n in ('Book/Price', '1-Month Return (Reversal)', 'Market Cap')}
    assert directions == {'Book/Price': HIGHER, '1-Month Return (Reversal)': LOWER, 'Market Cap': LOWER}


def test_wrds_run_shows_a_note_and_sends_no_job(lambda_at):
    at = lambda_at
    _switch(at, 'WRDS')
    assert _button(at, 'Run Portfolio Analysis').disabled          # no factor yet
    _add(at, WRDS_FACTOR_METADATA, 'Book/Price')
    _button(at, 'Run Portfolio Analysis').click().run()

    _assert_clean(at)
    assert any('coming in the next stage' in i.value for i in at.info)
    assert at.fake_lambda.jobs == []
    assert at.session_state['saved_runs'] == []


def test_without_lambda_a_stale_wrds_choice_falls_back_to_legacy(monkeypatch):
    monkeypatch.syspath_prepend(str(APP_DIR))
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', '')
    at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    at.session_state['data_mode_input'] = 'WRDS'
    at.run()
    _assert_clean(at)
    assert _mode(at).value == 'Legacy'
    assert 'How to treat delisted positions:' in _sidebar_radio_labels(at)


# ---------------------------------------------------------------------------
# Legacy settings and jobs unchanged
# ---------------------------------------------------------------------------

def test_legacy_settings_are_exactly_as_on_main(at):
    _add(at, FACTOR_METADATA, 'ROE using 9/30 Data')
    _button(at, 'Load Market Data').click().run()
    _button(at, 'Run Portfolio Analysis').click().run()
    _assert_clean(at)
    assert at.session_state['saved_runs'][0]['settings'] == LEGACY_DEFAULT_SETTINGS


def test_legacy_lambda_job_is_exactly_as_on_main_even_after_a_wrds_visit(lambda_at, monkeypatch):
    # One session runs Legacy straight away, a second one visits WRDS first
    for visit_wrds in (False, True):
        at = lambda_at if not visit_wrds else _app(monkeypatch, LAMBDA_NAME)
        if visit_wrds:
            _switch(at, 'WRDS')
            _switch(at, 'Legacy')
        _add(at, FACTOR_METADATA, 'ROE using 9/30 Data')
        _button(at, 'Run Portfolio Analysis').click().run()
        _assert_clean(at)
        assert at.session_state['saved_runs'][0]['settings'] == dict(LEGACY_DEFAULT_SETTINGS, use_lambda=True)

    first, second = lambda_at.fake_lambda.jobs
    expected = build_job('backtest', ['ROE_using_9-30_Data'], {'ROE_using_9-30_Data': 'top'},
                         dict(LEGACY_DEFAULT_SETTINGS, use_lambda=True),
                         factor_labels={'ROE_using_9-30_Data': 'ROE using 9/30 Data'})
    assert first == second == expected
    assert tuple(first) == JOB_FIELDS and 'data_mode' not in first
