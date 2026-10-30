"""
WRDS analyses in the lists of earlier and saved analyses (PR #69): "All saved analyses"
lists WRDS backtests with a summary and skips their holdings files, cohort comparisons
and ranked lists; the session and browser lists show the same summary under a WRDS run;
and a WRDS analysis opened from either list shows its full results and holdings.
Legacy entries are covered, unchanged, by test_list_analyses.py.
"""
import pytest
from streamlit.testing.v1 import AppTest

from src.backtest_service import BacktestService, summarize_job
from src.saved_analyses import list_saved_analyses
from src.wrds_jobs import build_wrds_job
from src.wrds_service import summarize_wrds_job
from tests.unit.test_backtest_lambda import APP_DIR, FakeLambda, _client_init_with, _job
from tests.unit.test_list_analyses import FakeBrowser, _expander
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab, browser_like_widget_states
from tests.unit.test_wrds_saved import OLD_IDS, _run_wrds, manager, s3  # noqa: F401 (fixtures)
from tests.unit.wrds_fixtures import WRDS_SETTINGS

LABELS = {'f_roe': 'ROE', 'f_roa': 'ROA'}


def _wrds_job(kind='backtest', **extra):
    return build_wrds_job(kind, list(LABELS), {f: 'top' for f in LABELS}, WRDS_SETTINGS,
                          factor_labels=LABELS, **extra)


def test_summary_names_the_data_source_frequency_and_factors():
    summary = summarize_wrds_job(_wrds_job())
    assert summary == "WRDS, Quarterly | ROE (higher), ROA (higher) | Mixing | 2010-2012 | $1,000"
    assert summary.isascii()
    assert summary.endswith(summarize_job(_wrds_job()))          # Legacy's summary after the frequency


def test_listing_shows_wrds_backtests_and_skips_their_other_files(manager, s3):
    service = BacktestService(manager)
    wrds_id = service.run(_wrds_job())['hash']
    service.run(_wrds_job('ranked', rebalance_date='2011-06'))      # not an analysis
    service.run(_wrds_job('cohort', cohort_pct=20))                  # not an analysis
    legacy_id = service.run(_job())['hash']
    assert f'factor-lake-data/backtest-results/{wrds_id}.holdings.parquet' in s3.objects

    listing = list_saved_analyses(manager)
    listed = [e['analysis_id'] for e in listing['entries']]
    assert listed[:2] == [legacy_id, wrds_id]                        # newest first
    assert set(listed[2:]) == set(OLD_IDS.values())                  # main's files, saved before summaries
    assert listing['total'] == 4 + len(OLD_IDS)                      # results only, no holdings files
    by_id = {e['analysis_id']: e['summary'] for e in listing['entries']}
    assert by_id[wrds_id] == summarize_wrds_job(_wrds_job())
    assert by_id[legacy_id] == summarize_job(_job())                 # Legacy's summary as on main


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

@pytest.fixture
def browser():
    return FakeBrowser()


@pytest.fixture
def new_session(monkeypatch, manager, browser):
    """App sessions on one browser, sharing an in-process Lambda and the fake S3 with WRDS data."""
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


def _shows_full_results_and_holdings(at, analysis_id):
    """The newest tab is the opened WRDS analysis, with Legacy's sections and a date's holdings."""
    _assert_clean(at)
    run = at.session_state['saved_runs'][0]
    assert run['data_mode'] == 'wrds' and run['compute']['hash'] == analysis_id
    assert run['label'] == 'Run 1: WRDS, Quarterly (opened)'
    assert [s.value for s in at.main.subheader] == ['Performance Summary', 'Holdings and Ranks',
                                                     'Portfolio Growth Over Time', 'Period-by-Period Performance',
                                                     'Top vs Bottom Cohort Analysis']
    picker = at.selectbox(key=f"ranked_year_{run['id']}")
    assert picker.label == 'Rebalance date' and picker.value == run['results']['rebalance_dates'][-1]
    holdings = at.main.dataframe[0].value
    assert len(holdings) > 0 and list(holdings.columns[-2:]) == ['PERMNO', 'Months Without Return']
    return run


def test_wrds_runs_show_their_summary_in_the_session_and_browser_lists(new_session, browser):
    first = new_session()
    run = _run_wrds(first, 'w_roe', 'w_roa')
    summary = "WRDS, Quarterly | ROE (higher), ROA (higher) | Mixing | 2010-2012 | $1,000"
    _click_main_tab(first, 'Results')
    assert summary in [c.value for c in first.caption]                  # This session's analyses
    assert [e.get('summary') for e in browser.storage] == [summary]     # kept in the browser with it

    later = new_session()                                               # same browser, a later session
    _click_main_tab(later, 'Results')
    assert _expander(later, 'Earlier analyses on this browser').label == 'Earlier analyses on this browser (1)'
    assert summary in [c.value for c in later.caption]
    later.button(key=f"open_earlier_{run['compute']['hash']}").click().run()
    opened = _shows_full_results_and_holdings(later, run['compute']['hash'])
    assert any(f'Opened as "{opened["label"]}"' in s.value for s in later.success)
    assert opened['results'] == run['results']


def test_all_saved_analyses_lists_and_opens_a_wrds_analysis(new_session, manager):
    colleague_id = BacktestService(manager).run(_wrds_job())['hash']   # someone else's WRDS run
    at = new_session()
    _click_main_tab(at, 'Results')
    at.session_state['all_analyses_expander'] = True
    at.run()
    assert any(m.value.startswith(f"**{summarize_wrds_job(_wrds_job())}** · saved ") for m in at.markdown)
    at.session_state['all_analyses_expander'] = True       # the browser keeps sending its open state
    at.button(key=f'open_all_{colleague_id}').click().run()
    _shows_full_results_and_holdings(at, colleague_id)
