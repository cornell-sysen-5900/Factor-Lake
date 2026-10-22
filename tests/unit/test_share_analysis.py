"""
Tests for saving an analysis and sharing it by its ID (story: "Save An Analysis and
share with colleagues").

Analyses run through the in-process Lambda (see test_backtest_lambda.py) against a
fake S3, and a second AppTest plays the colleague who opens the shared ID.

Definition of done covered here:
- Entering an ID shows the same results as the original run.
- Invalid or expired IDs show a clear error.
"""
import io
import json

import pytest
from botocore.exceptions import ClientError
from streamlit.testing.v1 import AppTest

from src.backtest_jobs import decode_saved, encode_result, results_match
from src.backtest_service import BacktestService, result_file
from src.saved_analyses import SavedAnalysisError, load_saved_analysis, normalize_analysis_id
from tests.unit.test_backtest_lambda import (APP_DIR, BUCKET, PREFIX, FakeLambda, _add, _client_init_with,
                                             _job, _run, manager, s3)  # noqa: F401 (fixtures)
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab, _open_tab, browser_like_widget_states

MISSING_ID = 'f' * 64


# ---------------------------------------------------------------------------
# What is saved, and loading it
# ---------------------------------------------------------------------------

def test_saved_analysis_keeps_its_job(manager, s3):
    response = BacktestService(manager).run(_job())
    job, result = decode_saved(s3.objects[f'{PREFIX}/{result_file(response["hash"])}'][0])
    assert job == _job()
    assert result == response['result']                      # the Lambda still returns just the result


def test_results_saved_before_jobs_were_kept_still_decode():
    old = encode_result({'final_value': 1.0})
    assert decode_saved(old) == (None, {'final_value': 1.0})


def test_load_gives_the_original_job_and_result(manager):
    response = BacktestService(manager).run(_job(delisting_strategy='reinvest'))
    saved = load_saved_analysis(f"  `{response['hash'].upper()}`\n", manager)   # pasted with extra characters
    assert saved['id'] == response['hash']
    assert saved['job'] == _job(delisting_strategy='reinvest')
    assert results_match(saved['result'], response['result'])
    assert all(isinstance(year, int) for year in saved['result']['holdings_by_year'])
    assert saved['computed_at'] == response['computed_at']


@pytest.mark.parametrize('text, message', [
    ('', 'Paste an analysis ID'),
    ('d6cf4640e6c5', 'is not an analysis ID'),                 # the short form is not enough
    ('g' * 64, 'is not an analysis ID'),
    ('https://cornellfactorlake.streamlit.app/', 'is not an analysis ID'),
])
def test_text_that_is_not_an_id_is_explained(text, message):
    with pytest.raises(SavedAnalysisError, match=message):
        normalize_analysis_id(text)


@pytest.mark.parametrize('code', ['NoSuchKey', 'AccessDenied'])   # AccessDenied: keys without s3:ListBucket
def test_missing_or_expired_id_is_explained(manager, s3, code):
    s3.can_list = code == 'NoSuchKey'
    with pytest.raises(SavedAnalysisError, match=r'No saved analysis has the ID ffffffff…ffff\. .*deleted after 90 days'):
        load_saved_analysis(MISSING_ID, manager)


def test_result_saved_before_sharing_is_explained(manager, s3):
    s3.put_object(BUCKET, f'{PREFIX}/{result_file(MISSING_ID)}', encode_result({'final_value': 1.0}))
    with pytest.raises(SavedAnalysisError, match='saved before analyses could be shared'):
        load_saved_analysis(MISSING_ID, manager)


def test_cohort_ids_are_not_opened_as_analyses(manager):
    response = BacktestService(manager).run(_job('cohort', cohort_pct=20))
    with pytest.raises(SavedAnalysisError, match='cohort comparison'):
        load_saved_analysis(response['hash'], manager)


def test_other_s3_errors_are_reported(manager, s3, monkeypatch):
    def slow_down(**kwargs):
        raise ClientError({'Error': {'Code': 'SlowDown'}}, 'GetObject')
    monkeypatch.setattr(s3, 'get_object', slow_down)
    with pytest.raises(SavedAnalysisError, match='Could not read the saved analysis'):
        load_saved_analysis(MISSING_ID, manager)


# ---------------------------------------------------------------------------
# The app: run, copy the ID, open it in another session
# ---------------------------------------------------------------------------

@pytest.fixture
def new_session(monkeypatch, manager):
    """Starts app sessions (browser tabs) that share the in-process Lambda and the fake S3."""
    import streamlit as st
    st.cache_resource.clear()
    fake_lambda = FakeLambda(BacktestService(manager))
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', 'factor-lake-backtest')
    monkeypatch.setattr('src.lambda_client.BacktestLambdaClient.__init__', _client_init_with(fake_lambda))
    monkeypatch.setattr('app.streamlit_utils.get_results_store', lambda: manager)
    monkeypatch.syspath_prepend(str(APP_DIR))
    browser_like_widget_states(monkeypatch)

    def start():
        at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
        at.run()
        at.fake_lambda = fake_lambda
        return at
    yield start
    st.cache_resource.clear()


def _open(at, text):
    _click_main_tab(at, 'Results')
    at.text_input(key='open_analysis_id').set_value(text).run()
    at.button(key='open_analysis_btn').click().run()


def _shown_ids(at):
    return [c.value for c in at.code]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_shared_id_shows_the_same_analysis_in_a_colleagues_session(new_session):
    mine = new_session()
    _add(mine, 'ROE using 9/30 Data')
    _add(mine, 'ROA %')
    next(r for r in mine.sidebar.radio if r.key == 'factor_combination_input').set_value('Integrating').run()
    original = _run(mine)
    analysis_id = original['compute']['hash']
    assert analysis_id in _shown_ids(mine)                     # shown with a copy button (st.code)
    my_metrics = (_metric(mine, 'Final Portfolio Value'), _metric(mine, 'CAGR'))

    colleague = new_session()                                  # another browser, nothing run there
    _open(colleague, analysis_id)
    _assert_clean(colleague)
    opened = colleague.session_state['saved_runs'][0]

    assert opened['label'] == 'Run 1: ROE using 9/30 Data +1 (opened)'
    assert results_match(opened['results'], original['results'])
    assert (opened['factors'], opened['factor_directions'], opened['factor_labels']) == \
           (original['factors'], original['factor_directions'], original['factor_labels'])
    assert opened['settings']['factor_combination'] == 'integrating'
    assert (_metric(colleague, 'Final Portfolio Value'), _metric(colleague, 'CAGR')) == my_metrics
    assert any(f'Opened as "{opened["label"]}"' in s.value for s in colleague.success)
    assert any('Opened from a saved analysis' in c.value for c in colleague.caption)
    assert analysis_id in _shown_ids(colleague)
    assert colleague.fake_lambda.invocations == 1              # opening did not rerun the backtest


@pytest.mark.parametrize('text, message', [
    ('not-an-id', 'is not an analysis ID'),
    (MISSING_ID, 'deleted after 90 days'),
])
def test_invalid_or_expired_id_shows_a_clear_error_in_the_app(new_session, text, message):
    at = new_session()
    _open(at, text)
    assert not at.exception, [e.value for e in at.exception]
    assert any(message in e.value for e in at.error)
    assert at.session_state['saved_runs'] == []
    assert at.session_state['analysis_history'] == []


def test_history_lists_only_this_sessions_runs(new_session):
    mine = new_session()
    _add(mine, 'ROE using 9/30 Data')
    runs = [_run(mine) for _ in range(6)]                      # more than the 5 Results tabs kept
    _click_main_tab(mine, 'Results')

    history = mine.session_state['analysis_history']
    assert [h['label'] for h in history] == [r['label'] for r in runs]
    assert len(mine.session_state['saved_runs']) == 5
    assert {h['analysis_id'] for h in history} == {runs[0]['compute']['hash']}   # same inputs, same ID
    assert any(e.label.startswith("This session's analyses (6)") for e in mine.expander)

    assert new_session().session_state['analysis_history'] == []   # another session sees none of them


def test_local_runs_are_in_the_history_without_an_id(new_session):
    at = new_session()
    _add(at, 'ROE using 9/30 Data')
    at.toggle(key='use_lambda_input').set_value(False).run()
    at.session_state['raw_data'] = at.fake_lambda.service.manager.fetch_all_data()
    _click_main_tab(at, 'Analysis')
    next(b for b in at.button if b.label == 'Load Market Data').click().run()
    run = _run(at)
    assert run['compute'] is None
    assert at.session_state['analysis_history'][-1]['analysis_id'] is None
    assert any('This run was computed in the app, so it was not saved' in c.value for c in at.caption)


def test_cohort_comparison_works_on_an_opened_analysis(new_session):
    mine = new_session()
    _add(mine, 'ROE using 9/30 Data')
    analysis_id = _run(mine)['compute']['hash']

    colleague = new_session()
    _open(colleague, analysis_id)
    run = colleague.session_state['saved_runs'][0]
    _open_tab(colleague, run['label'])
    colleague.button(key=f"cohort_btn_{run['id']}").click().run()
    _assert_clean(colleague)
    cohort = colleague.session_state['saved_runs'][0]['cohort']
    assert cohort['compute'] is not None and len(cohort['top']) == len(run['results']['portfolio_values'])
