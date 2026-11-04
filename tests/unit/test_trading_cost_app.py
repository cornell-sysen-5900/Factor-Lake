"""
Trading costs in the app: the sidebar input, a 25 bps run's views in both modes (summary,
growth chart, statistics, tables, holdings and their CSV, labels and captions), cohorts,
Clear Parameters, history and listings, and old IDs opening at 0 bps.

At 0 bps every view is main's: tests/unit/test_wrds_results.py compares the Legacy Results
tab with main's snapshot.
"""
import io

import pandas as pd
import pytest

from tests.unit.test_holdings_table import _downloaded_csv, served_files  # noqa: F401 (fixture)
from tests.unit.test_saved_runs import (_add_factor, _assert_clean, _captions, _click, _click_main_tab, _metric,
                                        _run_tab_labels, at)  # noqa: F401 (fixture)
from tests.unit.test_wrds_saved import OLD_IDS, _run_wrds, manager, new_session, s3  # noqa: F401 (fixtures)

COST_KEY = 'trading_cost_bps_input'
TILES = ('Avg Turnover per Rebalance', 'Total Trading Cost Paid', 'Annualized Cost Drag', 'CAGR Before Costs')


def _set_cost(at, bps):
    at.sidebar.number_input(key=COST_KEY).set_value(bps).run()


def _run_legacy(at, bps):
    _set_cost(at, bps)
    _click_main_tab(at, 'Analysis')
    _add_factor(at, 'ROE using 9/30 Data')
    _add_factor(at, 'ROA %')
    _click(at, 'Load Market Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    return at.session_state['saved_runs'][0]


def _record_jobs(at):
    """Records every job the session's in-process Lambda receives, in at.fake_lambda.jobs."""
    import json
    fake, invoke = at.fake_lambda, at.fake_lambda.invoke
    fake.jobs = []

    def recording(FunctionName, InvocationType, Payload):
        fake.jobs.append(json.loads(Payload))
        return invoke(FunctionName, InvocationType, Payload)
    fake.invoke = recording


def _figure_traces(at, index=0):
    import json
    return json.loads(at.get('plotly_chart')[index].proto.spec)['data']


def _metric_labels(at):
    return [m.label for m in at.metric]


def test_the_sidebar_input_has_its_default_range_step_and_tooltip(at):
    widget = at.sidebar.number_input(key=COST_KEY)
    assert widget.label == 'Trading cost (bps per trade)' and widget.value == 0
    assert (widget.min, widget.max, widget.step) == (0, 500, 5)
    assert widget.help == ("Charged on every buy and sell. Russell 2000 stocks often cost roughly 20-60 bps "
                           "per side, more for the smallest names.")


def test_a_legacy_run_at_25_bps_shows_its_costs_everywhere(at, served_files):
    run = _run_legacy(at, 25)
    results = run['results']
    assert run['settings']['trading_cost_bps'] == 25.0 and results['trading_cost_bps'] == 25.0
    assert _run_tab_labels(at) == ['Run 1: ROE using 9/30 Data +1 | 25 bps']

    captions = _captions(at)
    assert '**Trading Cost:** 25 bps per trade' in captions
    assert any(c.startswith('After trading costs of 25 bps per trade. Before costs: final value $') for c in captions)
    assert _metric(at, 'Final Portfolio Value') == f"${results['final_value']:,.2f}"     # after costs
    # The newest year's turnover and cost in the holdings caption
    assert any(c.endswith(f"**Turnover:** {results['turnover_pct'][-1]:.2f}% | "
                          f"**Trading cost:** {results['trading_cost_pct'][-1]:.2f}%") for c in captions)

    # Growth chart: a dashed "Before costs" line ending at the value before costs
    traces = {t['name']: t for t in _figure_traces(at)}
    assert traces['Before costs']['line']['dash'] == 'dash'
    assert traces['Before costs']['y'][-1] == pytest.approx(results['portfolio_values_before_costs'][-1])
    assert traces['Portfolio']['y'][-1] == pytest.approx(results['final_value'])

    # Statistics tiles
    assert all(t in _metric_labels(at) for t in TILES)
    assert _metric(at, 'Avg Turnover per Rebalance') == f"{results['average_turnover_pct']:.2f}%"
    assert _metric(at, 'Total Trading Cost Paid') == f"${results['total_trading_cost']:,.2f}"
    assert _metric(at, 'Annualized Cost Drag') == f"{results['cost_drag']:.2%}"

    # Holdings (and its CSV) with Traded %, the Year-by-Year table with Turnover % and Cost %
    holdings = at.dataframe[0].value
    assert list(holdings.columns[holdings.columns.get_loc('Weight %'):][:2]) == ['Weight %', 'Traded %']
    csv = pd.read_csv(io.BytesIO(_downloaded_csv(at, served_files, 'ranked_csv_1').content))
    assert csv['Traded %'].tolist() == pytest.approx(holdings['Traded %'].tolist())
    yearly = at.dataframe[1].value
    assert list(yearly.columns[-2:]) == ['Turnover %', 'Cost %']
    assert yearly['Turnover %'].iloc[0] == '100.00'


def test_a_legacy_run_at_0_bps_has_no_cost_views(at):
    run = _run_legacy(at, 0)
    assert 'trading_cost_bps' not in run['settings'] and 'trading_cost_bps' not in run['results']
    assert _run_tab_labels(at) == ['Run 1: ROE using 9/30 Data +1']
    assert not any('bps' in c or 'Turnover' in c for c in _captions(at))
    assert not any(t in _metric_labels(at) for t in TILES)
    assert 'Before costs' not in {t['name'] for t in _figure_traces(at)}
    assert 'Traded %' not in at.dataframe[0].value.columns
    assert 'Turnover %' not in at.dataframe[1].value.columns


def test_the_legacy_cohort_uses_the_runs_cost(at):
    run = _run_legacy(at, 50)
    at.button(key=f"cohort_btn_{run['id']}").click().run()
    _assert_clean(at)
    assert "Both cohorts pay the run's trading cost of 50 bps per trade." in _captions(at)
    from src.backtest_engine import run_cohort_comparison
    top, _ = run_cohort_comparison(run['data'], run['factors'], run['factor_directions'], 10, run['settings'])
    free, _ = run_cohort_comparison(run['data'], run['factors'], run['factor_directions'], 10,
                                    {k: v for k, v in run['settings'].items() if k != 'trading_cost_bps'})
    assert run['cohort']['top'] == top and top[-1] < free[-1]


def test_clear_parameters_resets_the_cost(at):
    _set_cost(at, 40)
    _click_main_tab(at, 'Analysis')
    at.button(key='clear_parameters').click().run()
    assert at.sidebar.number_input(key=COST_KEY).value == 0


def test_a_legacy_lambda_run_sends_the_cost_and_its_history_and_listing_show_it(new_session):
    at = new_session()
    _set_cost(at, 25)
    _click_main_tab(at, 'Analysis')
    _add_factor(at, 'ROE using 9/30 Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    run = at.session_state['saved_runs'][0]
    assert run['label'] == 'Run 1: ROE using 9/30 Data | 25 bps'
    assert at.session_state['analysis_history'][-1]['label'].endswith('| 25 bps')
    from src.saved_analyses import list_saved_analyses
    from app.streamlit_utils import get_results_store
    listing = list_saved_analyses(get_results_store())['entries']
    assert listing[0]['analysis_id'] == run['compute']['hash'] and listing[0]['summary'].endswith('| 25 bps')

    # The ID opens in another session with its cost
    theirs = new_session()
    _click_main_tab(theirs, 'Results')
    theirs.text_input(key='open_analysis_id').set_value(run['compute']['hash']).run()
    theirs.button(key='open_analysis_btn').click().run()
    _assert_clean(theirs)
    opened = theirs.session_state['saved_runs'][0]
    assert opened['label'] == 'Run 1: ROE using 9/30 Data | 25 bps (opened)'
    assert opened['settings']['trading_cost_bps'] == 25.0
    assert '**Trading Cost:** 25 bps per trade' in _captions(theirs)


def test_a_wrds_run_at_25_bps_shows_its_costs_everywhere(new_session):
    at = new_session()
    _record_jobs(at)
    _set_cost(at, 25)
    run = _run_wrds(at, 'w_roe', 'w_mom_12_1')
    results = run['results']
    assert run['label'] == 'Run 1: WRDS, Quarterly | 25 bps'
    assert at.fake_lambda.jobs[-1]['trading_cost_bps'] == 25.0
    entry = at.session_state['analysis_history'][-1]
    assert entry['summary'].startswith('WRDS, Quarterly') and entry['summary'].endswith('| 25 bps')

    captions = _captions(at)
    assert '**Trading Cost:** 25 bps per trade' in captions
    assert any(c.startswith('After trading costs of 25 bps per trade.') for c in captions)
    assert _metric(at, 'CAGR') == f"{results['cagr']:.2%}"
    assert all(t in _metric_labels(at) for t in TILES)
    assert _metric(at, 'Annualized Cost Drag') == f"{results['cost_drag']:.2%}"
    assert 'Before costs' in {t['name'] for t in _figure_traces(at)}

    holdings = at.dataframe[0].value
    assert 'Traded %' in holdings.columns
    period_table = at.dataframe[1].value
    assert list(period_table.columns[-2:]) == ['Turnover %', 'Cost %']
    assert len(period_table) == len(results['turnover_pct'])

    # Calendar years: turnover summed, cost compounded
    at.button_group(key=f"wrds_period_view_{run['id']}").set_value('By calendar year').run()
    _assert_clean(at)
    by_year = at.dataframe[1].value
    assert list(by_year.columns[-2:]) == ['Turnover %', 'Cost %']

    # Ranked lists ignore the cost (same job as at 0 bps); the cohort pays it
    at.button(key=f"wrds_rank_btn_{run['id']}").click().run()
    _assert_clean(at)
    assert 'trading_cost_bps' not in at.fake_lambda.jobs[-1]
    at.button(key=f"cohort_btn_{run['id']}").click().run()
    _assert_clean(at)
    assert at.fake_lambda.jobs[-1]['kind'] == 'cohort' and at.fake_lambda.jobs[-1]['trading_cost_bps'] == 25.0
    assert "Both cohorts pay the run's trading cost of 25 bps per trade." in _captions(at)


def test_an_id_saved_by_main_opens_at_0_bps_without_cost_views(new_session):
    at = new_session()
    _click_main_tab(at, 'Results')
    at.text_input(key='open_analysis_id').set_value(OLD_IDS['backtest']).run()
    at.button(key='open_analysis_btn').click().run()
    _assert_clean(at)
    opened = at.session_state['saved_runs'][0]
    assert 'trading_cost_bps' not in opened['settings'] and 'bps' not in opened['label']
    assert not any(t in _metric_labels(at) for t in TILES)
    assert not any('bps' in c for c in _captions(at))
    assert 'Traded %' not in at.dataframe[0].value.columns
