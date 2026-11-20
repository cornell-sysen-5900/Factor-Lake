"""
The WRDS sidebar's Hold choice (app/components/sidebar.py): Top X% or Top N stocks next to the
weighting, one number input that changes with it, the job each one sends, and runs at 100%
without factors. Runs go through the in-process Lambda on the synthetic WRDS data.
"""
import json

import pandas as pd

from app.streamlit_utils import load_wrds_holdings
from src.wrds_engine import WHOLE_FILTER_LABEL, eligible_universe
from tests.unit.test_saved_runs import _assert_clean, _click_main_tab
from tests.unit.test_wrds_results import s3  # noqa: F401 (fixture)
from tests.unit.test_wrds_saved import manager, new_session  # noqa: F401 (fixtures)

HOLD, PCT, COUNT = 'wrds_hold_input', 'wrds_top_pct_input', 'wrds_top_n_input'
# app/components/sidebar.WHOLE_FILTER_CAPTION (the app's modules import from the app folder)
WHOLE_FILTER_CAPTION = "Holds every stock in the filter, so factors are not used: you can run without any."


def _start(new_session):
    """A session whose Lambda records every job it receives (at.jobs)."""
    at = new_session()
    at.jobs, invoke = [], at.fake_lambda.invoke

    def record(FunctionName, InvocationType, Payload):
        event = json.loads(Payload)
        at.jobs.append(event.get('job', event))
        return invoke(FunctionName, InvocationType, Payload)
    at.fake_lambda.invoke = record
    return at


def _wrds(at, *factor_keys):
    """WRDS mode, quarterly 2010-2012 (the synthetic data), with these factors added."""
    at.sidebar.number_input(key='wrds_start_year_input').set_value(2010).run()
    at.sidebar.number_input(key='wrds_end_year_input').set_value(2012).run()
    for key in factor_keys:
        at.button(key=f'factor_add_{key}').click().run()
    _click_main_tab(at, 'Analysis')
    _assert_clean(at)


def _run(at):
    _click_main_tab(at, 'Analysis')
    at.button(key='run_wrds_analysis').click().run()
    _assert_clean(at)
    return next(j for j in reversed(at.jobs) if j.get('kind') == 'backtest'), at.session_state['saved_runs'][0]


def _sidebar_captions(at):
    return [c.value for c in at.sidebar.caption]


def _holdings_counts(at, run):
    """Stocks held at each rebalance date, read from the run's holdings file in S3."""
    holdings = run['results']['holdings']
    return {date: len(load_wrds_holdings(holdings['file'], i, date)) for i, date in enumerate(holdings['dates'])}


def test_hold_is_next_to_the_weighting_with_one_input_that_changes_with_it(new_session):
    at = _start(new_session)
    assert HOLD in [r.key for r in at.sidebar.radio]                     # WRDS is the default
    _wrds(at)
    radios = [r.key for r in at.sidebar.radio]
    assert radios.index(HOLD) == radios.index('weighting_method_input') + 1
    hold = at.sidebar.radio(key=HOLD)
    assert (hold.label, hold.options, hold.value) == ('Hold', ['Top X%', 'Top N stocks'], 'Top X%')
    keys = [n.key for n in at.sidebar.number_input]
    assert PCT in keys and COUNT not in keys
    pct = at.sidebar.number_input(key=PCT)
    assert (pct.label, pct.min, pct.max, pct.value) == ('Percent of stocks', 1, 100, 10)

    hold.set_value('Top N stocks').run()
    keys = [n.key for n in at.sidebar.number_input]
    assert COUNT in keys and PCT not in keys
    count = at.sidebar.number_input(key=COUNT)
    assert (count.label, count.min, count.value) == ('Number of stocks', 1, 50)
    _assert_clean(at)


def test_the_default_sends_the_same_job_as_before(new_session):
    at = _start(new_session)
    _wrds(at, 'w_roe')
    job, run = _run(at)
    assert job['top_pct'] == 10.0 and 'top_n' not in job
    assert 'top_pct' not in run['settings'] and 'top_n' not in run['settings']


def test_a_percent_and_a_count_are_sent_and_held(new_session):
    at = _start(new_session)
    _wrds(at, 'w_roe')
    at.sidebar.number_input(key=PCT).set_value(25).run()
    job, run = _run(at)
    assert job['top_pct'] == 25.0 and 'top_n' not in job and run['settings']['top_pct'] == 25.0

    at.sidebar.radio(key=HOLD).set_value('Top N stocks').run()
    at.sidebar.number_input(key=COUNT).set_value(7).run()
    job, run = _run(at)
    assert job['top_n'] == 7 and job['top_pct'] is None and run['settings']['top_n'] == 7
    assert set(_holdings_counts(at, run).values()) == {7}


def test_at_100_percent_a_run_needs_no_factor_and_holds_the_whole_filter(new_session, manager):
    at = _start(new_session)
    _wrds(at)
    assert at.button(key='run_wrds_analysis').disabled                   # 10% needs a factor
    assert WHOLE_FILTER_CAPTION not in _sidebar_captions(at)
    at.sidebar.number_input(key=PCT).set_value(100).run()
    assert WHOLE_FILTER_CAPTION in _sidebar_captions(at)
    assert not at.button(key='run_wrds_analysis').disabled
    job, run = _run(at)
    assert job['factors'] == [] and job['top_pct'] == 100.0
    assert run['factors'] == [] and run['label'] == 'Run 1: WRDS, Quarterly'

    # Every stock in the filter on every date, in a Results tab without cohorts or ranked lists
    data = at.fake_lambda.service.wrds.runner.data()
    dates = run['results']['holdings']['dates']
    panel = data.panel[data.panel['date'].isin(pd.to_datetime(dates))]
    assert _holdings_counts(at, run) == {d: n for d, n in zip(dates, eligible_universe(panel).groupby('date').size())}
    _click_main_tab(at, 'Results')
    _assert_clean(at)
    assert 'Top vs Bottom Cohort Analysis' not in [s.value for s in at.main.subheader]
    assert 'All ranked stocks on this date' not in [e.label for e in at.main.expander]
    assert any((df.value['Picked By'] == WHOLE_FILTER_LABEL).all() for df in at.main.dataframe
               if 'Picked By' in df.value)


def test_a_count_still_needs_a_factor(new_session):
    at = _start(new_session)
    _wrds(at)
    at.sidebar.radio(key=HOLD).set_value('Top N stocks').run()
    at.sidebar.number_input(key=COUNT).set_value(10_000).run()
    assert at.button(key='run_wrds_analysis').disabled
    assert WHOLE_FILTER_CAPTION not in _sidebar_captions(at)


def test_clear_parameters_restores_the_top_10_percent(new_session):
    at = _start(new_session)
    _wrds(at, 'w_roe')
    at.sidebar.radio(key=HOLD).set_value('Top N stocks').run()
    at.sidebar.number_input(key=COUNT).set_value(7).run()
    at.button(key='clear_parameters').click().run()
    _assert_clean(at)
    assert at.sidebar.radio(key=HOLD).value == 'Top X%'
    assert at.sidebar.number_input(key=PCT).value == 10
