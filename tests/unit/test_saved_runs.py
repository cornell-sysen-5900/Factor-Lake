"""
Tests for saving backtest runs as tabs in the Results area.

Pure tests cover app/saved_runs.py. AppTest tests drive app/streamlit_app.py
with a small synthetic universe placed in session_state['raw_data'], so no
Supabase access is needed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.saved_runs import add_saved_run, build_run_label, remove_saved_run

APP_DIR = Path(__file__).resolve().parents[2] / 'app'
RUN = 'Run Portfolio Analysis'


class TestBuildRunLabel:
    def test_single_factor(self):
        assert build_run_label(1, ['ROA %']) == 'Run 1: ROA %'

    def test_multiple_factors_shows_count_of_extra(self):
        assert build_run_label(3, ['ROA %', 'Book/Price', '6-Mo Momentum %']) == 'Run 3: ROA % +2'

    def test_no_factors(self):
        assert build_run_label(7, []) == 'Run 7'


class TestAddSavedRun:
    def test_newest_run_is_first(self):
        runs = add_saved_run([], {'id': 1}, max_runs=5)
        runs = add_saved_run(runs, {'id': 2}, max_runs=5)
        assert [r['id'] for r in runs] == [2, 1]

    def test_oldest_run_dropped_at_cap(self):
        runs = []
        for run_id in range(1, 8):
            runs = add_saved_run(runs, {'id': run_id}, max_runs=5)
        assert [r['id'] for r in runs] == [7, 6, 5, 4, 3]

    def test_does_not_mutate_input(self):
        original = [{'id': 1}]
        add_saved_run(original, {'id': 2}, max_runs=5)
        assert original == [{'id': 1}]


class TestRemoveSavedRun:
    def test_removes_only_that_run_and_keeps_order(self):
        runs = [{'id': 3}, {'id': 2}, {'id': 1}]
        assert [r['id'] for r in remove_saved_run(runs, 2)] == [3, 1]
        assert [r['id'] for r in runs] == [3, 2, 1]

    def test_unknown_id_is_a_no_op(self):
        assert remove_saved_run([{'id': 1}], 9) == [{'id': 1}]


# ---------------------------------------------------------------------------
# AppTest: the real app with a synthetic universe
# ---------------------------------------------------------------------------

def _synthetic_universe() -> pd.DataFrame:
    """30 tickers x 2002-2024 with the columns the app and engine read."""
    rng = np.random.RandomState(0)
    rows = []
    for year in range(2002, 2025):
        for i in range(30):
            rows.append({
                'Ticker': f'T{i:02d}',
                'Year': year,
                'Ending_Price': rng.uniform(10, 100),
                'Next-Years_Return': rng.normal(8, 20),
                'ROE_using_9-30_Data': rng.normal(0.1, 0.05),
                'ROA': rng.normal(0.05, 0.03),
                'Market_Capitalization': rng.uniform(1e8, 5e9),
                'Scotts_Sector_5': 'Technology',
                'FactSet_Industry': 'Software',
            })
    return pd.DataFrame(rows)


@pytest.fixture
def at(monkeypatch):
    # `streamlit run` puts app/ on sys.path; AppTest does not.
    monkeypatch.syspath_prepend(str(APP_DIR))
    app = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    app.session_state['raw_data'] = _synthetic_universe()
    app.run()
    _click(app, 'Load Market Data')
    return app


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def _assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _run_tab_labels(at):
    return [t.label for t in at.tabs if t.label.startswith('Run ')]


def _open_tab(at, label):
    # AppTest does not remember which stateful tab was selected, so set it
    # before every run that needs a tab other than the newest.
    at.session_state['saved_run_tabs'] = label


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _captions(at):
    return [c.value for c in at.caption]


def test_each_run_is_saved_in_its_own_tab(at):
    at.checkbox(key='roe').check().run()
    _click(at, RUN)
    at.checkbox(key='roa_pct').check().run()
    _click(at, RUN)

    _assert_clean(at)
    runs = at.session_state['saved_runs']
    assert [r['id'] for r in runs] == [2, 1]
    assert runs[0]['factor_labels'] == ['ROE using 9/30 Data', 'ROA %']
    assert runs[1]['factor_labels'] == ['ROE using 9/30 Data']
    assert runs[0]['data'] is runs[1]['data']  # same load is shared, not copied
    assert _run_tab_labels(at) == ['Run 2: ROE using 9/30 Data +1', 'Run 1: ROE using 9/30 Data']
    assert at.session_state['saved_run_tabs'] == 'Run 2: ROE using 9/30 Data +1'
    assert any('Saved as "Run 2: ROE using 9/30 Data +1"' in s.value for s in at.success)
    # Only the open tab is rendered.
    assert [c for c in _captions(at) if c.startswith('**Factors:**')] == [
        '**Factors:** ROE using 9/30 Data (High to Low), ROA % (High to Low)']


def test_saved_tab_ignores_later_sidebar_and_factor_changes(at):
    at.checkbox(key='roe').check().run()
    _click(at, RUN)
    before = (_metric(at, 'Total Return'), _metric(at, 'CAGR'), _captions(at))

    next(n for n in at.number_input if n.label == 'Initial AUM ($)').set_value(5000.0)
    at.toggle(key='roe_dir').set_value(True)
    at.run()

    _assert_clean(at)
    assert (_metric(at, 'Total Return'), _metric(at, 'CAGR'), _captions(at)) == before
    assert '**Period:** 2002-2024 | **Initial AUM:** $1,000' in _captions(at)


def test_cohort_comparison_uses_the_runs_own_snapshot(at):
    at.checkbox(key='roe').check().run()
    _click(at, RUN)                                   # Run 1: ROE, High to Low
    at.toggle(key='roe_dir').set_value(True).run()   # live direction now differs
    _click(at, RUN)                                   # Run 2: ROE, Low to High

    label_1 = 'Run 1: ROE using 9/30 Data'
    _open_tab(at, label_1)
    at.run()
    _open_tab(at, label_1)
    at.button(key='cohort_btn_1').click().run()

    _assert_clean(at)
    run_1 = next(r for r in at.session_state['saved_runs'] if r['id'] == 1)
    assert run_1['cohort']['pct'] == 10
    assert run_1['cohort']['top'] != run_1['cohort']['bottom']  # regression: both were 'bottom'
    assert len(at.table) == 1

    at.run()                                          # newest tab (Run 2) open
    assert len(at.table) == 0
    _open_tab(at, label_1)
    at.run()                                          # back to Run 1: result still shown
    assert len(at.table) == 1


def test_invalid_or_failed_runs_are_not_saved(at, monkeypatch):
    _click(at, RUN)                                   # no factors selected
    assert any('Select at least one factor' in w.value for w in at.warning)

    at.checkbox(key='roe').check().run()
    aum = next(n for n in at.number_input if n.label == 'Initial AUM ($)')
    aum.set_value(0.0).run()
    _click(at, RUN)
    assert any('Initial AUM must be greater' in w.value for w in at.warning)

    aum = next(n for n in at.number_input if n.label == 'Initial AUM ($)')
    aum.set_value(1000.0).run()
    at.number_input(key='start_year_input').set_value(2023)
    at.number_input(key='end_year_input').set_value(2023).run()
    _click(at, RUN)
    assert any('no yearly returns' in w.value for w in at.warning)

    def boom(*args, **kwargs):
        raise RuntimeError('engine failed')
    monkeypatch.setattr('src.backtest_engine.rebalance_portfolio', boom)
    at.number_input(key='start_year_input').set_value(2002)
    at.number_input(key='end_year_input').set_value(2024).run()
    _click(at, RUN)
    assert any('Backtest Execution Error: engine failed' in e.value for e in at.error)

    assert at.session_state['saved_runs'] == []
    assert at.session_state['next_run_id'] == 1
    assert not any('Analysis complete' in s.value for s in at.success)


def test_remove_run(at):
    at.checkbox(key='roe').check().run()
    _click(at, RUN)
    _click(at, RUN)
    assert _run_tab_labels(at) == ['Run 2: ROE using 9/30 Data', 'Run 1: ROE using 9/30 Data']

    at.button(key='remove_run_2').click().run()
    _assert_clean(at)
    assert [r['id'] for r in at.session_state['saved_runs']] == [1]
    assert _run_tab_labels(at) == ['Run 1: ROE using 9/30 Data']

    at.button(key='remove_run_1').click().run()
    _assert_clean(at)
    assert at.session_state['saved_runs'] == []
    assert any('will appear here' in i.value for i in at.info)

    _click(at, RUN)                                   # ids are never reused
    assert [r['id'] for r in at.session_state['saved_runs']] == [3]
