"""
Where a WRDS run's Hold choice is shown and saved: the Results tab's Universe caption, the
Compare tab's Settings rows, strategy JSON files (export and import, older files at 10%) and
opened analyses. The calculation preview's drift test is in test_calculation_preview.py.
"""
import json

import pytest
from streamlit.testing.v1 import AppTest

from app import strategy_config as cfg
from app.calculation_preview import build_wrds_calculation_preview
from app.compare import comparison_rows
from src.wrds_jobs import build_wrds_job
from tests.unit.test_compare import _legacy, _wrds
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab
from tests.unit.test_strategy_config import document, upload
from tests.unit.test_wrds_hold_sidebar import COUNT, HOLD, PCT, _run, _start
from tests.unit.test_wrds_hold_sidebar import _wrds as _wrds_mode
from tests.unit.test_wrds_results import s3  # noqa: F401 (fixture)
from tests.unit.test_wrds_saved import APP_DIR, manager, new_session  # noqa: F401 (fixtures)
from tests.unit.wrds_fixtures import WRDS_SETTINGS


def _universe_caption(at):
    return next(c.value for c in at.main.caption if c.value.startswith('**Universe:**'))


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def test_compare_settings_show_how_many_stocks_each_analysis_holds():
    analyses = [_legacy(), _wrds(), _wrds('c' * 64, job={'top_pct': None, 'top_n': 20}),
                _wrds('d' * 64, job={'top_pct': 25.0}),
                _wrds('e' * 64, job={'top_pct': 100.0, 'factors': [], 'factor_directions': {}, 'factor_labels': {}})]
    rows = {r['name']: r['values'] for r in comparison_rows(analyses)}
    names = [r['name'] for r in comparison_rows(analyses)]
    assert names.index('Hold') == names.index('Weighting') + 1
    assert rows['Hold'] == ['Top 10%', 'Top 10%', 'Top 20 stocks', 'Top 25%', 'Every stock in the filter (100%)']
    assert rows['Factors'][-1] == 'None'


# ---------------------------------------------------------------------------
# Strategy JSON
# ---------------------------------------------------------------------------

def _wrds_document(**settings):
    doc = document('wrds')
    doc['settings'].update(settings)
    return doc


@pytest.mark.parametrize('hold, widgets', [
    ({}, ('Top X%', 10, 50)),
    ({'top_pct': 25}, ('Top X%', 25, 50)),
    ({'top_n': 20}, ('Top N stocks', 10, 20)),
    ({'top_pct': 100}, ('Top X%', 100, 50)),
])
def test_a_file_restores_the_hold_choice(hold, widgets):
    doc = cfg.loads(json.dumps(_wrds_document(**hold)))
    values = cfg.widget_values(doc)
    assert (values[HOLD], values[PCT], values[COUNT]) == widgets


def test_an_older_file_without_the_choice_opens_at_10_percent():
    doc = document('wrds')
    assert 'top_pct' not in doc['settings'] and 'top_n' not in doc['settings']
    assert cfg.widget_values(cfg.loads(json.dumps(doc)))[HOLD] == 'Top X%'
    assert cfg.widget_values(cfg.loads(json.dumps(doc)))[PCT] == 10


def test_export_saves_the_choice_only_when_it_is_not_10_percent():
    base = dict(document('wrds')['settings'], data_mode='wrds')
    column = [document('wrds')['factors'][0]['column']]
    assert 'top_pct' not in cfg.build(dict(base, top_pct=10.0), column, {})['settings']
    assert cfg.build(dict(base, top_pct=25.0), column, {})['settings']['top_pct'] == 25
    assert cfg.build(dict(base, top_n=20), column, {})['settings']['top_n'] == 20
    # At 100% a run may have no factors
    whole = cfg.build(dict(base, top_pct=100.0), [], {})
    assert whole['factors'] == [] and cfg.loads(cfg.dumps(whole)) == whole


@pytest.mark.parametrize('settings, message', [
    ({'top_pct': 25, 'top_n': 20}, 'not both'),
    ({'top_pct': 12.5}, 'whole percent'),
    ({'top_pct': 0}, 'whole percent'),
    ({'top_pct': 101}, 'whole percent'),
    ({'top_pct': True}, 'whole percent'),
    ({'top_n': 0}, 'at least 1'),
    ({'top_n': 2.5}, 'at least 1'),
    ({'top_n': '20'}, 'at least 1'),
])
def test_bad_choices_in_a_file_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        cfg.loads(json.dumps(_wrds_document(**settings)))


def test_a_file_needs_a_factor_unless_it_holds_100_percent():
    for settings in ({}, {'top_pct': 50}, {'top_n': 20}):
        doc = dict(_wrds_document(**settings), factors=[])
        with pytest.raises(ValueError, match='at least one supported factor'):
            cfg.loads(json.dumps(doc))
    legacy = dict(document(), factors=[])
    legacy['settings'] = dict(legacy['settings'], top_pct=100)
    with pytest.raises(ValueError):                          # Legacy has no Hold choice
        cfg.loads(json.dumps(legacy))


def test_import_sets_the_sidebar_and_an_older_file_resets_it(new_session, monkeypatch):
    at = _start(new_session)
    upload(at, monkeypatch, cfg.dumps(_wrds_document(top_n=20)))
    _assert_clean(at)
    assert at.sidebar.radio(key=HOLD).value == 'Top N stocks'
    assert at.sidebar.number_input(key=COUNT).value == 20
    upload(at, monkeypatch, cfg.dumps(document('wrds')))
    _assert_clean(at)
    assert at.sidebar.radio(key=HOLD).value == 'Top X%'
    assert at.sidebar.number_input(key=PCT).value == 10


@pytest.mark.parametrize('hold', [{'top_n': 7}, {'top_pct': 100, 'no_factors': True}])
def test_a_run_exports_and_imports_to_the_same_job(new_session, monkeypatch, hold):
    at = _start(new_session)
    hold = dict(hold)
    no_factors = hold.pop('no_factors', False)
    doc = _wrds_document(start_year=2010, end_year=2012, frequency='quarterly', **hold)
    doc['factors'] = [] if no_factors else [{'column': 'f_book_to_price', 'direction': 'top'}]
    upload(at, monkeypatch, cfg.dumps(doc))
    first_job, first = _run(at)
    assert {k: first_job.get(k) for k in ('top_pct', 'top_n')} == (
        {'top_pct': None, 'top_n': 7} if 'top_n' in hold else {'top_pct': 100.0, 'top_n': None})
    exported = cfg.build(first['settings'], first['factors'], first['factor_directions'], first['name'], first['note'])
    assert exported == doc

    _click_main_tab(at, 'Analysis')
    at.button(key='clear_parameters').click().run()
    upload(at, monkeypatch, cfg.dumps(exported))
    second_job, second = _run(at)
    assert second_job == first_job and second['compute']['hash'] == first['compute']['hash']


# ---------------------------------------------------------------------------
# Results: the Universe caption, for new and opened runs
# ---------------------------------------------------------------------------

def test_the_universe_caption_says_how_many_stocks_are_held(new_session, monkeypatch):
    at = _start(new_session)
    _wrds_mode(at, 'w_roe')
    at.sidebar.radio(key=HOLD).set_value('Top N stocks').run()
    at.sidebar.number_input(key=COUNT).set_value(7).run()
    _, run = _run(at)
    _click_main_tab(at, 'Results')
    _assert_clean(at)
    assert _universe_caption(at) == '**Universe:** All sectors | Fossil fuels included | Holds: Top 7 stocks'

    # Opened by its ID in another session: the same settings and caption
    opened = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    opened.query_params['analysis'] = run['compute']['hash']
    opened.run()
    _assert_clean(opened)
    assert opened.session_state['saved_runs'][0]['settings']['top_n'] == 7
    assert _universe_caption(opened).endswith('| Holds: Top 7 stocks')


def test_a_default_and_a_no_factor_run_describe_themselves(new_session):
    at = _start(new_session)
    _wrds_mode(at, 'w_roe')
    _run(at)
    _click_main_tab(at, 'Results')
    assert _universe_caption(at).endswith('| Holds: Top 10%')

    _click_main_tab(at, 'Analysis')
    at.button(key='clear_parameters').click().run()   # removes the factor
    _wrds_mode(at)
    at.sidebar.number_input(key=PCT).set_value(100).run()
    _run(at)
    _click_main_tab(at, 'Results')
    _assert_clean(at)
    assert _universe_caption(at).endswith('| Holds: Every stock in the filter (100%)')
    assert '**Factors:** None (every stock in the filter is held)' in [c.value for c in at.main.caption]


# ---------------------------------------------------------------------------
# Calculation preview at 100% (the percent and count rules: test_calculation_preview.py)
# ---------------------------------------------------------------------------

def test_the_preview_at_100_percent_holds_the_filter_and_scores_nothing_without_factors():
    whole = build_wrds_calculation_preview(build_wrds_job('backtest', [], {}, dict(WRDS_SETTINGS, top_pct=100.0)))
    assert 'Top 100%: H(t) = U(t), every stock in the filter' in whole['formula']
    assert 'floor(' not in whole['formula'] and 'min(' not in whole['formula']
    titles = [title for title, _ in whole['steps']]
    assert 'Select stocks' in titles and 'Normalize each selected factor' not in titles
    with_factor = build_wrds_calculation_preview(build_wrds_job(
        'backtest', ['f_roe'], {'f_roe': 'top'}, dict(WRDS_SETTINGS, top_pct=100.0)))
    assert 'Normalize each selected factor' in [title for title, _ in with_factor['steps']]
    assert 'Top 100%: H(t) = U(t)' in with_factor['formula']
