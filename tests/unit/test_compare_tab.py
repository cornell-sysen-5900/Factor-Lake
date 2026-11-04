"""
The Compare tab in the real app (AppTest), with an in-process Lambda and a fake S3: saved
Legacy and WRDS analyses (with and without trading costs, and one saved by main's code)
compared 2, 3 and 4 at a time; adding and removing boxes; the picker; bad, duplicate,
expired, cohort and pre-sharing IDs; sorting and the differences switch without reloads;
and the tab only reading saved results (no backtest, no market data).
"""
import json

import pytest

from app.compare import column_titles, compact_analysis, comparison_table
from src.saved_analyses import load_saved_analysis
from tests.unit.test_saved_runs import _add_factor, _assert_clean, _click, _click_main_tab
from Visualizations.compare_growth_plot import COMPARE_COLORS
from tests.unit.test_wrds_saved import OLD_IDS, _run_wrds, manager, new_session, s3  # noqa: F401 (fixtures)

COST_KEY = 'trading_cost_bps_input'


def _run_legacy(at, bps=0):
    at.sidebar.number_input(key=COST_KEY).set_value(bps).run()
    _click_main_tab(at, 'Analysis')
    _add_factor(at, 'ROE using 9/30 Data')
    _click(at, 'Run Portfolio Analysis')
    _assert_clean(at)
    return at.session_state['saved_runs'][0]['compute']['hash']


def _run_wrds_at(at, frequency, bps, *factors):
    at.sidebar.number_input(key=COST_KEY).set_value(bps).run()
    if at.sidebar.radio(key='data_mode_input').value == 'WRDS':
        at.sidebar.radio(key='wrds_frequency_input').set_value(frequency).run()
    run = _run_wrds(at, *factors)
    if run['results']['frequency'] != frequency.lower():
        at.sidebar.radio(key='wrds_frequency_input').set_value(frequency).run()
        run = _run_wrds(at)
    return run['compute']['hash']


@pytest.fixture
def saved(new_session):
    """A session that saved a Legacy annual run (0 bps), a WRDS quarterly run (25 bps) and a WRDS monthly run (0 bps)."""
    at = new_session()
    ids = {'legacy': _run_legacy(at)}
    ids['quarterly'] = _run_wrds_at(at, 'Quarterly', 25, 'w_roe')
    ids['monthly'] = _run_wrds_at(at, 'Monthly', 0)
    ids['main'] = OLD_IDS['backtest']           # Legacy, saved by main before trading costs
    return at, ids


def _open_compare(at):
    _click_main_tab(at, 'Compare')
    assert not at.exception, [e.value for e in at.exception]


def _boxes(at):
    return [t for t in at.text_input if t.key.startswith('compare_id_')]


def _compare(at, *ids):
    """Fills the boxes (adding boxes as needed) and clicks Compare."""
    _open_compare(at)
    while len(_boxes(at)) < len(ids):
        at.button(key='compare_add').click().run()
    for box, analysis_id in zip(_boxes(at), ids):
        box.set_value(analysis_id).run()
    at.button(key='compare_btn').click().run()
    assert not at.exception, [e.value for e in at.exception]


def _table(at):
    return at.table[0].value


def _expected_titles(manager, *ids):
    return column_titles([compact_analysis(load_saved_analysis(i, manager)) for i in ids])


def _row(table, name):
    return table.loc[name if name in table.index else f'{name} ≠']


# ---------------------------------------------------------------------------
# 2, 3 and 4 analyses, both modes and mixed
# ---------------------------------------------------------------------------

def test_two_three_and_four_analyses_compare_one_column_each(saved, manager, s3):
    at, ids = saved
    invocations, market_gets = at.fake_lambda.invocations, len(s3.gets)
    order = [ids['legacy'], ids['quarterly'], ids['monthly'], ids['main']]
    for n in (2, 3, 4):
        _compare(at, *order[:n])
        _assert_clean(at)
        table = _table(at)
        assert list(table.columns) == _expected_titles(manager, *order[:n])
        assert len(at.get('plotly_chart')) == 1
        traces = at.get('plotly_chart')[0].proto.spec
        assert traces.count('"legendgroup"') == 2 * n          # a portfolio and its benchmark each
    # Comparing ran no backtest and read only saved results
    assert at.fake_lambda.invocations == invocations
    assert all(key.split('/')[-2] == 'backtest-results' and key.endswith('.json') for key, _ in s3.gets[market_gets:])
    assert at.session_state['raw_data'] is None
    assert at.button(key='compare_add').proto.disabled                    # 4 boxes: no more


def test_mixed_legacy_and_wrds_values_match_their_results(saved, manager):
    at, ids = saved
    _compare(at, ids['legacy'], ids['quarterly'], ids['main'])
    _assert_clean(at)
    table = _table(at)
    legacy, quarterly, main = (load_saved_analysis(i, manager) for i in (ids['legacy'], ids['quarterly'], ids['main']))
    titles = _expected_titles(manager, ids['legacy'], ids['quarterly'], ids['main'])
    assert _row(table, 'Data source').tolist() == ['Legacy', 'WRDS', 'Legacy']
    assert _row(table, 'Frequency').tolist() == ['Annual', 'Quarterly', 'Annual']
    assert _row(table, 'Trading Cost').tolist() == ['0 bps', '25 bps', '0 bps']
    assert _row(table, 'Final Portfolio Value').tolist() == [
        f"${s['result']['final_value']:,.2f}" for s in (legacy, quarterly, main)]
    # WRDS's CAGR as its Results tab shows it
    assert _row(table, 'CAGR')[titles[1]] == f"{quarterly['result']['cagr']:.2%}"
    # Costs: the 25 bps run's, and 0 bps / n/a turnover for the others (main's was saved before costs)
    r = quarterly['result']
    assert _row(table, 'Avg Turnover per Rebalance').tolist() == ['n/a', f"{r['average_turnover_pct']:.2f}%", 'n/a']
    assert _row(table, 'Total Trading Cost Paid').tolist() == ['$0.00', f"${r['total_trading_cost']:,.2f}", '$0.00']
    assert _row(table, 'Annualized Cost Drag')[titles[1]] == f"{r['cost_drag']:.2%}"
    assert _row(table, 'Sharpe')[titles[1]] == f"{r['sharpe_portfolio']:.4f}"
    # The headers: label, frequency and the full ID with a copy button (st.code)
    codes = [c.value for c in at.code]
    assert all(i in codes for i in (ids['legacy'], ids['quarterly'], ids['main']))
    assert any(c.value.startswith('WRDS · Quarterly · ID ') for c in at.caption)


def test_the_best_value_of_a_row_is_highlighted_green(saved, manager):
    at, ids = saved
    _compare(at, ids['legacy'], ids['quarterly'])
    from app.compare import comparison_table
    from app.components.compare_tab import BEST_STYLE
    styles = at.table[0].proto.arrow_data.styler.styles
    best_rows = comparison_table([compact_analysis(load_saved_analysis(i, manager))
                                  for i in (ids['legacy'], ids['quarterly'])], [0, 1], False).attrs['best']
    # Pandas writes one CSS rule per style, listing its cells: one green cell per best value
    green = [rule for rule in styles.split('}') if BEST_STYLE.split(';')[0] in rule]
    assert best_rows and len(green) == 1 and green[0].count('#T_') == sum(map(len, best_rows.values()))


def test_mixed_initial_aums_compare_rates_and_chart_the_growth_of_1000(new_session, manager):
    at = new_session()
    at.sidebar.number_input(key='initial_aum_input').set_value(1_000_000.0).run()
    big = _run_legacy(at)
    at.sidebar.number_input(key='initial_aum_input').set_value(1000.0).run()
    small = _run_wrds_at(at, 'Quarterly', 25, 'w_roe')
    _compare(at, big, small)
    _assert_clean(at)
    table = _table(at)
    assert _row(table, 'Initial AUM').tolist() == ['$1,000,000', '$1,000']
    best = comparison_table([compact_analysis(load_saved_analysis(i, manager)) for i in (big, small)],
                            [0, 1], False).attrs['best']
    assert not {'Final Portfolio Value ≠', 'Total Trading Cost Paid ≠'} & set(best)
    assert 'CAGR ≠' in best and 'Total Return ≠' in best
    chart = json.loads(at.get('plotly_chart')[0].proto.spec)
    assert chart['layout']['title']['text'].startswith('<b>Growth of $1,000: compared analyses')
    lines = [t for t in chart['data'] if t['name'] in column_titles([compact_analysis(load_saved_analysis(i, manager))
                                                                      for i in (big, small)])]
    assert [t['y'][0] for t in lines] == pytest.approx([1000.0, 1000.0])
    assert any('scaled to start at $1,000' in c.value for c in at.caption)

    # The same AUM again: dollar values and the usual title
    at.sidebar.number_input(key='initial_aum_input').set_value(1_000_000.0).run()
    other = _run_wrds_at(at, 'Quarterly', 0)
    _compare(at, big, other)
    chart = json.loads(at.get('plotly_chart')[0].proto.spec)
    assert chart['layout']['title']['text'] == '<b>Cumulative Wealth: compared analyses</b>'
    assert not any('scaled to start at' in c.value for c in at.caption)


# ---------------------------------------------------------------------------
# Boxes and the picker
# ---------------------------------------------------------------------------

def test_boxes_start_at_two_add_up_to_four_and_can_be_removed(saved):
    at, ids = saved
    _open_compare(at)
    assert len(_boxes(at)) == 2
    assert not [b for b in at.button if b.key.startswith('compare_remove_')]     # the first 2 stay
    at.button(key='compare_add').click().run()
    at.button(key='compare_add').click().run()
    assert len(_boxes(at)) == 4 and at.button(key='compare_add').proto.disabled
    assert [b.key for b in at.button if b.key.startswith('compare_remove_')] == ['compare_remove_2',
                                                                                  'compare_remove_3']
    for box, analysis_id in zip(_boxes(at), (ids['legacy'], ids['quarterly'], ids['monthly'], ids['main'])):
        box.set_value(analysis_id).run()
    at.button(key='compare_btn').click().run()
    assert len(_table(at).columns) == 4

    # Removing the third box removes its ID and its column, and Add works again
    at.button(key='compare_remove_2').click().run()
    _assert_clean(at)
    assert [b.value for b in _boxes(at)] == [ids['legacy'], ids['quarterly'], ids['main']]
    assert len(_table(at).columns) == 3 and not at.button(key='compare_add').proto.disabled
    at.button(key='compare_add').click().run()
    assert [b.key for b in _boxes(at)][-1] == 'compare_id_4' and _boxes(at)[-1].value == ''


def test_the_picker_lists_this_sessions_analyses_and_fills_the_box(saved):
    at, ids = saved
    _open_compare(at)
    picker = at.selectbox(key='compare_pick_1')
    # This session's three runs, each named by its label and short ID
    assert sorted(o.split(' · ')[-1] for o in picker.options) == sorted(
        ids[k][:8] for k in ('legacy', 'quarterly', 'monthly'))
    assert any(o.startswith('Run 2: WRDS, Quarterly | 25 bps · ') for o in picker.options)
    picker.set_value(ids['quarterly']).run()
    _assert_clean(at)
    assert at.text_input(key='compare_id_1').value == ids['quarterly']
    assert at.selectbox(key='compare_pick_1').value is None


def test_the_ids_sort_and_switch_survive_a_tab_switch(saved):
    at, ids = saved
    _compare(at, ids['legacy'], ids['quarterly'])
    at.selectbox(key='compare_sort').set_value('Sharpe').run()
    at.toggle(key='compare_differences_only').set_value(True).run()
    _click_main_tab(at, 'Results')
    _open_compare(at)
    assert [b.value for b in _boxes(at)] == [ids['legacy'], ids['quarterly']]
    assert at.selectbox(key='compare_sort').value == 'Sharpe'
    assert at.toggle(key='compare_differences_only').value is True
    assert len(_table(at).columns) == 2


# ---------------------------------------------------------------------------
# Bad IDs
# ---------------------------------------------------------------------------

def test_bad_duplicate_expired_cohort_and_pre_sharing_ids_show_an_error_in_their_box(saved):
    at, ids = saved
    _compare(at, ids['legacy'], 'not-an-id', ids['legacy'], 'f' * 64)
    errors = [e.value for e in at.error]
    assert errors == ["Not an analysis ID: an ID is 64 characters of 0-9 and a-f.",
                      "Same analysis as box 1.",
                      "No saved analysis has this ID: it is mistyped, or expired (analyses are kept 90 days)."]
    assert not at.table and any(i.value == "Enter at least 2 valid analysis IDs to compare." for i in at.info)

    # The rest still compare when at least 2 are valid
    _boxes(at)[3].set_value(ids['quarterly']).run()
    at.button(key='compare_btn').click().run()
    assert len(at.error) == 2
    assert len(_table(at).columns) == 2

    _compare(at, ids['legacy'], OLD_IDS['cohort'], OLD_IDS['pre_sharing'], ids['monthly'])
    assert [e.value for e in at.error] == [
        "This ID is a cohort comparison, not an analysis.",
        "Saved before analyses could be shared, so it has no settings to compare."]
    assert len(_table(at).columns) == 2


def test_empty_boxes_are_skipped_and_one_analysis_is_not_enough(saved):
    at, ids = saved
    _compare(at, ids['legacy'], '')
    assert not at.error and not at.table
    assert any(i.value == "Enter at least 2 valid analysis IDs to compare." for i in at.info)


# ---------------------------------------------------------------------------
# Sort and differences
# ---------------------------------------------------------------------------

def test_sorting_reorders_the_table_and_legend_without_reloading(saved, manager, s3):
    at, ids = saved
    entered = [ids['legacy'], ids['quarterly'], ids['monthly']]
    _compare(at, *entered)
    analyses = [compact_analysis(load_saved_analysis(i, manager)) for i in entered]
    titles = column_titles(analyses)
    gets = len(s3.gets)
    from app.compare import SORT_OPTIONS, sort_order
    for option in SORT_OPTIONS:
        at.selectbox(key='compare_sort').set_value(option).run()
        _assert_clean(at)
        order = sort_order(analyses, option)
        assert list(_table(at).columns) == [titles[i] for i in order]
        traces = json.loads(at.get('plotly_chart')[0].proto.spec)['data']
        lines = {t['name']: t for t in traces if t['name'] in titles}
        # The legend follows the sort; each line keeps its color (the order entered)
        assert sorted(titles, key=lambda n: lines[n]['legendrank']) == [titles[i] for i in order]
        assert [lines[n]['line']['color'] for n in titles] == list(COMPARE_COLORS[:3])
    assert len(s3.gets) == gets                         # never reloaded


def test_show_only_differences_hides_the_rows_that_are_the_same(saved):
    at, ids = saved
    _compare(at, ids['quarterly'], ids['monthly'])
    full = _table(at)
    assert 'Initial AUM' in full.index and 'Frequency ≠' in full.index
    at.toggle(key='compare_differences_only').set_value(True).run()
    only = _table(at)
    assert 'Initial AUM' not in only.index and 'Frequency ≠' in only.index
    assert all(n.endswith(' ≠') or n in ('SETTINGS', 'PERFORMANCE', 'RISK', 'COSTS') for n in only.index)
