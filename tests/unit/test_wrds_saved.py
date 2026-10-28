"""
Saved results and analysis IDs with WRDS mode (DOCS/WRDS_MODE_DESIGN.md, section 5.4),
and WRDS runs in the real app with an in-process Lambda and a fake S3.

- Old IDs: tests/fixtures/saved_main_*.json were written by main's code (c063100):
  a shared backtest, a cohort comparison, and a result saved before sharing existed.
- WRDS IDs: a WRDS run's ID opens in another session with the same results and holdings.
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.backtest_jobs import decode_result, decode_saved, results_match
from src.backtest_service import BacktestService, result_file
from src.s3_client import S3DataManager
from src.saved_analyses import SavedAnalysisError, load_saved_analysis
from tests.unit.test_backtest_lambda import (APP_DIR, DELISTING_FILE, UNIVERSE_FILE, FakeLambda,
                                             _client_init_with, _delisting, _parquet, _raw_universe)
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab, _open_tab, browser_like_widget_states
from tests.unit.wrds_fixtures import RangeS3, put_wrds_files, synthetic_frames

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures'
BUCKET, PREFIX = 'test-bucket', 'factor-lake-data'
OLD_IDS = {'backtest': '1' * 64, 'cohort': '2' * 64, 'pre_sharing': '3' * 64}


@pytest.fixture
def s3():
    s3 = RangeS3()
    s3.objects[f'{PREFIX}/{UNIVERSE_FILE}'] = (_parquet(_raw_universe()), {})
    s3.objects[f'{PREFIX}/{DELISTING_FILE}'] = (_parquet(_delisting()), {})
    put_wrds_files(s3, synthetic_frames())
    for kind, analysis_id in OLD_IDS.items():
        body = (FIXTURES / f'saved_main_{kind}.json').read_bytes()
        s3.objects[f'{PREFIX}/{result_file(analysis_id)}'] = (body, {'computed_at': '2026-10-20T10:00:00Z'})
    return s3


@pytest.fixture
def manager(s3):
    return S3DataManager(bucket=BUCKET, prefix=PREFIX, client=s3)


# ---------------------------------------------------------------------------
# Old IDs saved by main
# ---------------------------------------------------------------------------

def test_an_id_saved_by_main_still_opens_with_its_job_and_result(manager):
    saved = load_saved_analysis(OLD_IDS['backtest'], manager)
    job, result = decode_saved((FIXTURES / 'saved_main_backtest.json').read_bytes())
    assert saved['job'] == job and 'data_mode' not in saved['job']
    assert results_match(saved['result'], decode_result('backtest', result))
    assert saved['computed_at'] == '2026-10-20T10:00:00Z'


def test_old_cohort_and_pre_sharing_ids_get_the_same_messages_as_on_main(manager):
    with pytest.raises(SavedAnalysisError, match='This ID belongs to a cohort comparison, not an analysis.'):
        load_saved_analysis(OLD_IDS['cohort'], manager)
    with pytest.raises(SavedAnalysisError, match='saved before analyses could be shared'):
        load_saved_analysis(OLD_IDS['pre_sharing'], manager)


# ---------------------------------------------------------------------------
# The app with WRDS runs
# ---------------------------------------------------------------------------

@pytest.fixture
def new_session(monkeypatch, manager):
    """App sessions sharing an in-process Lambda and the fake S3 (as in test_share_analysis.py)."""
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
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
    st.cache_data.clear()


def _run_wrds(at, *factor_keys):
    """Runs a quarterly 2010-2012 WRDS backtest from the Analysis tab; returns the saved run."""
    _click_main_tab(at, 'Analysis')
    if at.sidebar.radio(key='data_mode_input').value != 'WRDS':
        at.sidebar.radio(key='data_mode_input').set_value('WRDS').run()
        at.sidebar.number_input(key='wrds_start_year_input').set_value(2010).run()
        at.sidebar.number_input(key='wrds_end_year_input').set_value(2012).run()
        for key in factor_keys:
            at.button(key=f'factor_add_{key}').click().run()
    _click_main_tab(at, 'Analysis')
    at.button(key='run_wrds_analysis').click().run()
    _assert_clean(at)
    return at.session_state['saved_runs'][0]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_a_wrds_run_is_computed_in_lambda_cached_and_shows_its_summary_and_holdings(new_session, s3):
    at = new_session()
    run = _run_wrds(at, 'w_roe', 'w_mom_12_1')
    assert run['data_mode'] == 'wrds' and run['data'] is None
    assert run['label'] == 'Run 1: WRDS, Quarterly'
    assert run['compute']['cached'] is False
    results = run['results']
    assert results['rebalance_dates'][0].startswith('2009-12') and len(results['rebalance_dates']) == 12
    assert any('Analysis complete' in s.value for s in at.success)

    # Summary and the newest date's holdings, read with ranged GETs
    assert '**Data Source:** WRDS | **Rebalancing:** Quarterly (12 rebalances)' in [c.value for c in at.caption]
    assert _metric(at, 'CAGR') == f"{results['cagr']:.2%}"
    newest = results['rebalance_dates'][-1]
    assert at.selectbox(key=f"wrds_date_{run['id']}").value == newest
    rows = results['holdings']['rows'][results['holdings']['dates'].index(newest)]
    assert any(c.value.startswith(f"**Method:** Mixing | **Holdings:** {rows} |") for c in at.caption)
    holdings_gets = [r for k, r in s3.gets if k.endswith('.holdings.parquet')]
    assert holdings_gets and all(r is not None for r in holdings_gets)

    # The same inputs again: a cache hit with the same ID
    again = _run_wrds(at)
    assert again['compute']['cached'] is True and again['compute']['hash'] == run['compute']['hash']
    assert again['results'] == results


def test_ranked_and_cohort_jobs_run_from_a_wrds_run(new_session):
    at = new_session()
    run = _run_wrds(at, 'w_roe')
    at.selectbox(key=f"wrds_date_{run['id']}").set_value(run['results']['rebalance_dates'][2]).run()
    at.button(key=f"wrds_rank_btn_{run['id']}").click().run()
    _assert_clean(at)
    ranked = at.session_state[f"wrds_ranked_{run['id']}"]
    assert ranked['date'] == run['results']['rebalance_dates'][2] and len(ranked['rows']) > 10
    assert ranked['compute']['cached'] is False

    at.button(key=f"wrds_cohort_btn_{run['id']}").click().run()
    _assert_clean(at)
    cohort = at.session_state['saved_runs'][0]['cohort']
    assert cohort['pct'] == 10 and len(cohort['top']) == len(run['results']['years'])
    assert at.fake_lambda.invocations == 3                  # backtest, ranked, cohort


def test_a_wrds_id_opens_in_another_session_with_the_same_results_and_holdings(new_session):
    mine = new_session()
    run = _run_wrds(mine, 'w_roe', 'w_btp')
    theirs = new_session()
    _click_main_tab(theirs, 'Results')
    theirs.text_input(key='open_analysis_id').set_value(run['compute']['hash']).run()
    theirs.button(key='open_analysis_btn').click().run()
    _assert_clean(theirs)
    opened = theirs.session_state['saved_runs'][0]
    assert opened['label'] == 'Run 1: WRDS, Quarterly (opened)'
    assert opened['data_mode'] == 'wrds' and opened['compute']['opened'] is True
    assert opened['results'] == json.loads(json.dumps(run['results']))
    assert opened['factor_directions'] == run['factor_directions']
    _open_tab(theirs, opened['label'])
    assert _metric(theirs, 'CAGR') == f"{run['results']['cagr']:.2%}"
    assert any(c.value.startswith('**Method:** Mixing | **Holdings:**') for c in theirs.caption)
    assert theirs.fake_lambda.invocations == 1               # opening never reruns it


def test_an_id_saved_by_main_opens_as_a_legacy_run_in_the_app(new_session):
    at = new_session()
    _click_main_tab(at, 'Results')
    at.text_input(key='open_analysis_id').set_value(OLD_IDS['backtest']).run()
    at.button(key='open_analysis_btn').click().run()
    _assert_clean(at)
    opened = at.session_state['saved_runs'][0]
    assert 'data_mode' not in opened and opened['settings']['delisting_strategy'] == 'hold_cash'
    assert _metric(at, 'Final Portfolio Value') == f"${opened['results']['final_value']:,.2f}"
