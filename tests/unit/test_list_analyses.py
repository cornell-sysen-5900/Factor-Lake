"""
Tests for listing existing analysis IDs (story: "List existing hashes").

Two lists help a user find an ID they forgot to copy:
- "Earlier analyses on this browser": IDs remembered in the browser's localStorage
  across sessions. The JavaScript only reads and writes storage; FakeBrowser plays it
  here with the same merge rule (merge_entries).
- "All saved analyses": everyone's saved analyses, listed from S3.
"""
import pytest
from streamlit.testing.v1 import AppTest

from src.backtest_service import BacktestService, result_file, summarize_job
from src.saved_analyses import SavedAnalysisError, list_saved_analyses
from tests.unit.test_backtest_lambda import (APP_DIR, BUCKET, PREFIX, FakeLambda, _add, _client_init_with,
                                             _job, _run, manager, s3)  # noqa: F401 (fixtures)
from tests.unit.test_saved_runs import _click_main_tab, browser_like_widget_states

ID_A, ID_B, ID_C = 'a' * 64, 'b' * 64, 'c' * 64


def _entry(analysis_id, saved_at, label='Run 1: ROE using 9/30 Data'):
    return {'analysis_id': analysis_id, 'saved_at': saved_at, 'label': label, 'how': 'computed in Lambda'}


# ---------------------------------------------------------------------------
# Remembering IDs in the browser
# ---------------------------------------------------------------------------

def test_merge_keeps_one_entry_per_id_newest_first():
    from app.analysis_history import merge_entries
    stored = [_entry(ID_A, '2026-10-20T10:00:00Z'), _entry(ID_B, '2026-10-21T10:00:00Z')]
    session = [_entry(ID_A, '2026-10-29T09:00:00Z', 'Run 4: ROA %'), {'analysis_id': None, 'label': 'local run'}]
    merged = merge_entries(stored, session)
    assert [e['analysis_id'] for e in merged] == [ID_A, ID_B]           # newest first, no local run
    assert merged[0]['label'] == 'Run 4: ROA %'                          # the newer copy wins
    assert merge_entries(stored, limit=1) == [stored[1]]
    assert merge_entries(None, []) == []


def test_earlier_analyses_leave_out_this_sessions():
    from app.analysis_history import earlier_analyses
    browser = [_entry(ID_A, '2026-10-29T09:00:00Z'), _entry(ID_B, '2026-10-20T10:00:00Z')]
    assert [e['analysis_id'] for e in earlier_analyses(browser, [_entry(ID_A, '2026-10-29T09:00:00Z')])] == [ID_B]
    assert earlier_analyses(None, []) == []


# ---------------------------------------------------------------------------
# Listing everyone's saved analyses from S3
# ---------------------------------------------------------------------------

def test_listing_is_newest_first_with_summaries_and_without_cohorts(manager, s3):
    service = BacktestService(manager)
    first = service.run(_job())['hash']
    service.run(_job('cohort', cohort_pct=20))                            # not an analysis
    second = service.run(_job(factor_combination='integrating'))['hash']
    s3.put_object(BUCKET, f'{PREFIX}/{result_file(ID_C)}', b'{}')          # saved before summaries existed
    s3.put_object(BUCKET, f'{PREFIX}/backtest-results/notes.txt', b'x')   # not a result

    listing = list_saved_analyses(manager)
    assert [e['analysis_id'] for e in listing['entries']] == [ID_C, second, first]
    assert listing['total'] == 4                                           # cohorts are counted, not listed
    assert listing['entries'][0]['summary'] is None
    assert listing['entries'][1]['summary'] == summarize_job(_job(factor_combination='integrating'))
    assert listing['entries'][1]['saved_at'] > listing['entries'][2]['saved_at']


def test_listing_reads_every_page_and_respects_the_limit(manager, s3):
    s3.page_size = 2
    for i in range(5):
        s3.put_object(BUCKET, f'{PREFIX}/{result_file(f"{i:x}" * 64)}', b'{}')
    assert len(list_saved_analyses(manager)['entries']) == 5
    assert [e['analysis_id'][0] for e in list_saved_analyses(manager, limit=2)['entries']] == ['4', '3']


def test_listing_without_permission_is_explained(manager, s3):
    s3.can_list = False
    with pytest.raises(SavedAnalysisError, match='may not list saved analyses'):
        list_saved_analyses(manager)


def test_summary_describes_the_job_in_ascii():
    job = _job(initial_aum=2500)
    summary = summarize_job(job)
    assert summary == ("ROE using 9/30 Data (higher), ROA (lower), 6-Mo Momentum % (higher) "
                       "| Mixing | 2004-2020 | $2,500")
    assert summary.isascii()


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

class FakeBrowser:
    """localStorage of one browser, shared by the app sessions opened in it (like the real JS)."""

    def __init__(self):
        self.storage = []

    def mount(self, data, key, on_stored_change):
        from app.analysis_history import merge_entries
        self.storage = merge_entries(self.storage, data['add'], limit=data['max_entries'])
        return type('Result', (), {'stored': list(self.storage)})()


@pytest.fixture
def browser():
    return FakeBrowser()


@pytest.fixture
def new_session(monkeypatch, manager, browser):
    """Starts app sessions on one browser, sharing the in-process Lambda and the fake S3."""
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    fake_lambda = FakeLambda(BacktestService(manager))
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', 'factor-lake-backtest')
    monkeypatch.setattr('src.lambda_client.BacktestLambdaClient.__init__', _client_init_with(fake_lambda))
    monkeypatch.setattr('app.streamlit_utils.get_results_store', lambda: manager)
    monkeypatch.syspath_prepend(str(APP_DIR))
    monkeypatch.setattr('analysis_history._browser_store', lambda: browser.mount)
    browser_like_widget_states(monkeypatch)

    def start():
        at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
        at.run()
        return at
    yield start
    st.cache_resource.clear()
    st.cache_data.clear()


def _expander(at, prefix):
    return next((e for e in at.expander if e.label.startswith(prefix)), None)


def test_ids_from_earlier_sessions_are_listed_and_open(new_session, browser):
    first = new_session()
    _add(first, 'ROE using 9/30 Data')
    run = _run(first)
    assert [e['analysis_id'] for e in browser.storage] == [run['compute']['hash']]   # saved in the browser

    later = new_session()                                   # next day, same browser: nothing in this session
    _click_main_tab(later, 'Results')
    earlier = _expander(later, 'Earlier analyses on this browser')
    assert earlier.label == 'Earlier analyses on this browser (1)'
    assert run['compute']['hash'] in [c.value for c in later.code]
    later.button(key=f"open_earlier_{run['compute']['hash']}").click().run()

    opened = later.session_state['saved_runs'][0]
    assert opened['label'] == 'Run 1: ROE using 9/30 Data (opened)'
    assert any(f'Opened as "{opened["label"]}"' in s.value for s in later.success)
    assert opened['results']['final_value'] == run['results']['final_value']
    assert _expander(later, 'Earlier analyses on this browser') is None   # now it is this session's
    assert _expander(later, "This session's analyses").label == "This session's analyses (1)"


def test_local_runs_are_not_remembered(new_session, browser, manager):
    at = new_session()
    _add(at, 'ROE using 9/30 Data')
    at.toggle(key='use_lambda_input').set_value(False).run()
    at.session_state['raw_data'] = manager.fetch_all_data()
    _click_main_tab(at, 'Analysis')
    next(b for b in at.button if b.label == 'Load Market Data').click().run()
    assert _run(at)['compute'] is None
    assert at.session_state['analysis_history'][-1]['analysis_id'] is None
    assert browser.storage == []                            # nothing to find later without an ID


def test_all_saved_analyses_lists_everyones_and_opens_them(new_session, manager):
    colleague_id = BacktestService(manager).run(_job(initial_aum=5000))['hash']   # someone else's run
    at = new_session()
    _click_main_tab(at, 'Results')
    assert _expander(at, 'All saved analyses').label == 'All saved analyses'
    assert colleague_id not in [c.value for c in at.code]                         # loaded only when opened

    at.session_state['all_analyses_expander'] = True
    at.run()
    assert colleague_id in [c.value for c in at.code]
    assert any(summarize_job(_job(initial_aum=5000)) in m.value for m in at.markdown)
    at.session_state['all_analyses_expander'] = True       # the browser keeps sending its open state
    at.button(key=f'open_all_{colleague_id}').click().run()
    assert at.session_state['saved_runs'][0]['settings']['initial_aum'] == 5000


def test_all_saved_analyses_explains_a_listing_error(new_session, s3):
    s3.can_list = False
    at = new_session()
    _click_main_tab(at, 'Results')
    at.session_state['all_analyses_expander'] = True
    at.run()
    assert any('may not list saved analyses' in e.value for e in at.error)
