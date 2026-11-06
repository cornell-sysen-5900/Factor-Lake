"""Verify the opt-in calculation visibility demo works without cloud data."""
import pandas as pd
from app.demo_data import build_demo_universe
from app.streamlit_config import FACTOR_METADATA
from tests.unit.test_saved_runs import _add_factor, _click, _assert_clean


def test_demo_data_is_repeatable_and_covers_all_factors():
    data = build_demo_universe()
    pd.testing.assert_frame_equal(data, build_demo_universe())
    assert len(data) == 80 * 23
    assert {m['column'] for m in FACTOR_METADATA.values()} <= set(data)
    assert data['Next-Years_Return'].isna().any()
    assert (data['FactSet_Industry'] == 'Coal').any()


def test_calculation_demo_runs_locally_without_clear_rerun(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from tests.unit.test_saved_runs import browser_like_widget_states
    app_dir = Path(__file__).resolve().parents[2] / 'app'
    monkeypatch.syspath_prepend(str(app_dir))
    browser_like_widget_states(monkeypatch)
    monkeypatch.setenv('FACTOR_LAKE_DEMO', '1')
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', 'must-not-call')
    def no_cloud(*args, **kwargs):
        raise AssertionError('demo must not fetch cloud market data')
    monkeypatch.setattr('app.streamlit_utils.load_shared_universe', no_cloud)
    monkeypatch.setattr('streamlit_utils.load_shared_universe', no_cloud)
    app = AppTest.from_file(str(app_dir / 'streamlit_app.py'), default_timeout=60).run()
    assert any('DEMO MODE' in w.value for w in app.warning)
    _add_factor(app, 'ROA %')
    _add_factor(app, '12-Mo Momentum %')
    _click(app, 'Load Market Data')
    _click(app, 'Run Portfolio Analysis')
    _assert_clean(app)
    assert app.session_state['saved_runs'][0]['compute'] is None
    assert any('U = market data' in c.value for c in app.code)
    assert not any(b.label == 'Clear results and edit parameters' for b in app.button)
