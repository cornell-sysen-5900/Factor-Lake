"""Direct URLs open saved results in fresh sessions, without any compute/data load."""
import pytest
from streamlit.testing.v1 import AppTest

from app.share_links import build_share_url, consume_share_link
from src.backtest_jobs import results_match
from src.saved_analyses import SavedAnalysisError
from tests.unit.test_backtest_lambda import APP_DIR
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab
from tests.unit.test_wrds_saved import (
    OLD_IDS,
    _run_wrds,
    manager,
    new_session,
    s3,
)


class Query(dict):
    def get_all(self, key):
        value = self[key]
        return value if isinstance(value, list) else [value]


def test_share_url_preserves_deployment_and_drops_unrelated_query_and_fragment():
    assert build_share_url('https://example.org/lake/?token=private#old', 'A' * 64) == (
        'https://example.org/lake/?analysis=' + 'a' * 64)
    assert build_share_url('http://localhost:8501', 'b' * 64).startswith('http://localhost:8501/?analysis=')
    with pytest.raises(ValueError):
        build_share_url('https://user:secret@example.org', 'b' * 64)
    with pytest.raises(SavedAnalysisError):
        build_share_url('https://example.org', 'bad')


def test_link_consumed_once_and_navigation_resets_marker():
    calls, state = [], {}
    def open_it(value):
        calls.append(value)
        return {'id': value}
    query = Query(analysis='a' * 64)
    assert consume_share_link(query, state, open_it)[0] == {'id': 'a' * 64}
    assert consume_share_link(query, state, open_it) == (None, None)
    consume_share_link(Query(), state, open_it)
    consume_share_link(query, state, open_it)
    assert calls == ['a' * 64] * 2


@pytest.mark.parametrize('value', ['', 'bad', ['a' * 64, 'b' * 64]])
def test_invalid_query_never_calls_loader(value):
    def forbidden(_):
        pytest.fail('Invalid ID reached the result store')
    _, error = consume_share_link(Query(analysis=value), {}, forbidden)
    assert 'Could not open the shared analysis' in error


def test_unavailable_store_does_not_leak_exception_details_or_retry_on_rerun():
    calls, state = [], {}
    def unavailable(_):
        calls.append(1)
        raise RuntimeError('secret implementation detail')
    query = Query(analysis='a' * 64)
    _, error = consume_share_link(query, state, unavailable)
    assert 'temporarily unavailable' in error and 'secret' not in error
    assert consume_share_link(query, state, unavailable) == (None, error)
    assert calls == [1]


@pytest.mark.parametrize('mode', ['legacy', 'wrds'])
def test_fresh_session_opens_link_in_results_without_running_or_loading_market_data(new_session, monkeypatch, mode):
    source = new_session()
    original = _run_wrds(source, 'w_roe') if mode == 'wrds' else None
    analysis_id = original['compute']['hash'] if original else OLD_IDS['backtest']
    previous_invokes = source.fake_lambda.invocations
    def forbidden(*args, **kwargs):
        pytest.fail('Opening a share link must not run a backtest or load market data')
    for method in ('run_backtest_logic', 'run_wrds_backtest_logic', 'load_and_process_data'):
        monkeypatch.setattr(f'app.streamlit_utils.{method}', forbidden)
    at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    at.query_params['analysis'] = analysis_id
    at.run()
    _assert_clean(at)
    assert at.session_state['main_tabs'] == 'Results'
    run = at.session_state['saved_runs'][0]
    assert run['compute']['hash'] == analysis_id and run['data'] is None
    assert at.session_state['raw_data'] is None and not at.session_state['data_loaded']
    if original:
        assert results_match(run['results'], original['results'])
    assert source.fake_lambda.invocations == previous_invokes
    # Changing tabs must neither steal focus nor append a duplicate run/history entry.
    _click_main_tab(at, 'Analysis')
    at.run()
    assert at.session_state['main_tabs'] == 'Analysis'
    assert len(at.session_state['saved_runs']) == len(at.session_state['analysis_history']) == 1


@pytest.mark.parametrize('value, message', [('', 'exactly one'), ('bad', 'not an analysis ID'),
                                          ('f' * 64, 'expired')])
def test_bad_link_leaves_normal_app_usable(new_session, value, message):
    at = new_session()
    at.query_params['analysis'] = value
    at.run()
    assert not at.exception
    assert any(message in error.value for error in at.error)
    assert at.session_state['saved_runs'] == []
    assert at.session_state['main_tabs'] == 'Analysis'
    at.button(key='factor_add_roe').click().run()
    assert not at.exception
    assert at.session_state['factor_builder_selected']['ROE using 9/30 Data'] == 'top'
