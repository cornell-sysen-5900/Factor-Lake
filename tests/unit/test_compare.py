"""
The Compare tab's pure helpers (app/compare.py): the rows and their values, the best value
of each row (per-metric rule, ties, n/a), the rows that differ, every sort option, the chart
series, and the table the tab shows (app/components/compare_tab.comparison_table).
"""
import pytest

from app.compare import (GROWTH_BASE, MAX_COMPARED, NA, ROWS, ROWS_BY_NAME, SORT_OPTIONS, best_index, cagr,
                         chart_series, column_titles, compact_analysis, comparison_rows, growth_series,
                         outperformance, sort_order)
from app.compare import comparison_table
from Visualizations.compare_growth_plot import COMPARE_COLORS, plot_compare_growth


def _legacy(analysis_id='a' * 64, returns=(10.0, -5.0, 20.0), bench=(5.0, 5.0, 5.0), aum=1000.0, bps=0.0,
            **result):
    values = [aum]
    for r in returns:
        values.append(values[-1] * (1 + r / 100))
    job = {'kind': 'backtest', 'factors': ['ROE'], 'factor_directions': {'ROE': 'top'},
           'factor_labels': {'ROE': 'ROE using 9/30 Data'}, 'start_year': 2002, 'end_year': 2002 + len(returns),
           'initial_aum': aum, 'benchmark_index': 1, 'top_pct': 10.0, 'use_market_cap_weight': False,
           'delisting_strategy': 'zero_return', 'factor_combination': 'mixing', 'restrict_fossil_fuels': False,
           'selected_sectors': [], 'cohort_pct': None}
    if bps:
        job['trading_cost_bps'] = bps
    res = {'final_value': values[-1], 'portfolio_values': values, 'yearly_returns': list(returns),
           'benchmark_returns': list(bench), 'years': list(range(2002, 2003 + len(returns))),
           'vol_raw_portfolio': 0.2, 'sharpe_portfolio': 0.5, 'max_drawdown_portfolio': -0.05,
           'portfolio_beta': 1.1, 'information_ratio': 0.3, 'holdings_by_year': {2002: [{'Ticker': 'X'}]}}
    res.update(result)
    return compact_analysis({'id': analysis_id, 'job': job, 'result': res, 'computed_at': None})


def _wrds(analysis_id='b' * 64, frequency='quarterly', bps=25.0, **result):
    dates = ['2010-03-31', '2010-06-30', '2010-09-30', '2010-12-31', '2011-03-31']
    returns = [2.0, -1.0, 3.0, 1.0]
    values = [1000.0]
    for r in returns:
        values.append(values[-1] * (1 + r / 100))
    job = {'kind': 'backtest', 'data_mode': 'wrds', 'factors': ['roe', 'mom_12_1'],
           'factor_directions': {'roe': 'top', 'mom_12_1': 'bottom'},
           'factor_labels': {'roe': 'ROE', 'mom_12_1': '12-1 Momentum'}, 'frequency': frequency,
           'start_year': 2010, 'end_year': 2011, 'initial_aum': 1000.0, 'top_pct': 10.0,
           'use_market_cap_weight': True, 'factor_combination': 'integrating', 'restrict_fossil_fuels': True,
           'selected_sectors': ['Energy'], 'cohort_pct': None, 'rebalance_date': None}
    res = {'final_value': values[-1], 'portfolio_values': values, 'yearly_returns': returns,
           'benchmark_returns': [1.0] * 4, 'years': dates, 'rebalance_dates': dates[:-1],
           'period_end_dates': dates[1:], 'frequency': frequency, 'periods_per_year': 4,
           'benchmark_labels': {'benchmark': 'IWM (ETF, total return)'},
           'vol_raw_portfolio': 0.1, 'sharpe_portfolio': 0.9, 'max_drawdown_portfolio': -0.01,
           'portfolio_beta': 0.9, 'information_ratio': 0.6}
    if bps:
        job['trading_cost_bps'] = bps
        res.update(trading_cost_bps=bps, cagr_before_costs=0.06, average_turnover_pct=80.0,
                   total_trading_cost=12.5, cost_drag=0.004, turnover_pct=[100, 70, 70, 80],
                   trading_cost_pct=[0.25] * 4, returns_before_costs=returns, portfolio_values_before_costs=values)
    job.update({k: v for k, v in result.pop('job', {}).items()})
    res.update(result)
    return compact_analysis({'id': analysis_id, 'job': job, 'result': res, 'computed_at': None})


def _row(analyses, name):
    return next(r for r in comparison_rows(analyses) if r['name'] == name)


# ---------------------------------------------------------------------------
# Loading, labels and values
# ---------------------------------------------------------------------------

def test_a_compared_analysis_keeps_only_what_the_tab_shows():
    a = _legacy()
    assert 'holdings_by_year' not in a['result']
    assert a['mode'] == 'legacy' and a['frequency'] == 'Annual'
    assert a['label'] == 'Legacy, Annual: ROE using 9/30 Data'
    w = _wrds()
    assert w['mode'] == 'wrds' and w['label'] == 'WRDS, Quarterly: ROE +1 | 25 bps'
    assert column_titles([a, w]) == ['Legacy, Annual: ROE using 9/30 Data · aaaaaaaa',
                                     'WRDS, Quarterly: ROE +1 | 25 bps · bbbbbbbb']


def test_settings_rows_describe_each_analysis():
    rows = {r['name']: r['values'] for r in comparison_rows([_legacy(), _wrds()])}
    assert rows['Data source'] == ['Legacy', 'WRDS']
    assert rows['Frequency'] == ['Annual', 'Quarterly']
    assert rows['Period'] == ['2002-2005', '2010-2011 (2010-03-31 to 2011-03-31)']
    assert rows['Initial AUM'] == ['$1,000', '$1,000']
    assert rows['Factors'] == ['ROE using 9/30 Data (higher)', 'ROE (higher), 12-1 Momentum (lower)']
    assert rows['Factor Combination'] == ['Mixing', 'Integrating']
    assert rows['Weighting'] == ['Equal Weighted', 'Market Cap Weighted']
    assert rows['Sectors'] == ['All sectors', 'Energy']
    assert rows['Fossil fuels'] == ['Included', 'Excluded']
    assert rows['Delisting Strategy'] == ['Zero Return', 'CRSP delisting returns, then the risk-free rate']
    assert rows['Benchmark'] == ['Russell 2000', 'IWM']
    assert rows['Trading Cost'] == ['0 bps', '25 bps']


def test_performance_values_use_the_results_tabs_formulas_and_formats():
    a, w = _legacy(), _wrds()
    rows = {r['name']: r['values'] for r in comparison_rows([a, w])}
    final = 1000 * 1.10 * 0.95 * 1.20
    assert rows['Final Portfolio Value'][0] == f"${final:,.2f}"
    assert rows['Total Return'][0] == f"{final / 1000 - 1:.2%}"
    assert rows['CAGR'][0] == f"{(final / 1000) ** (1 / 3) - 1:.2%}"
    # WRDS: 4 quarters = 1 year
    w_final = w['result']['final_value']
    assert cagr(w) == pytest.approx(w_final / 1000 - 1)
    assert outperformance(a) == pytest.approx(final / (1000 * 1.05 ** 3) - 1)
    assert rows['Best Period'] == ['20.00% (2004)', '3.00% (2010-09-30 to 2010-12-31)']
    assert rows['Worst Period'] == ['-5.00% (2003)', '-1.00% (2010-06-30 to 2010-09-30)']
    assert rows['Volatility (Raw)'] == ['20.00%', '10.00%']
    assert rows['Sharpe'] == ['0.5000', '0.9000']
    assert rows['Max Drawdown'] == ['-5.00%', '-1.00%']
    assert rows['Beta vs Benchmark'] == ['1.100', '0.900']
    assert rows['Information Ratio'] == ['0.3000', '0.6000']


def test_runs_without_trading_costs_show_0_bps_and_n_a_turnover():
    rows = {r['name']: r['values'] for r in comparison_rows([_legacy(), _wrds()])}
    assert rows['Trading Cost'][0] == '0 bps'
    assert rows['CAGR Before Costs'][0] == rows['CAGR'][0]       # nothing was charged
    assert rows['Avg Turnover per Rebalance'] == [NA, '80.00%']
    assert rows['Total Trading Cost Paid'] == ['$0.00', '$12.50']
    assert rows['Annualized Cost Drag'] == ['0.00%', '0.40%']
    assert rows['CAGR Before Costs'][1] == '6.00%'


# ---------------------------------------------------------------------------
# Best values
# ---------------------------------------------------------------------------

def test_the_best_value_follows_each_metrics_rule():
    a, w = _legacy(), _wrds()
    best = {r['name']: r['best'] for r in comparison_rows([a, w])}
    # Higher is better
    assert best['Final Portfolio Value'] == 0 and best['Total Return'] == 0 and best['CAGR'] == 0
    assert best['Sharpe'] == 1 and best['Information Ratio'] == 1
    # Lower is better (a drawdown is negative, so the smallest loss is the highest value)
    assert best['Volatility (Raw)'] == 1 and best['Max Drawdown'] == 1
    assert best['Total Trading Cost Paid'] == 0 and best['Annualized Cost Drag'] == 0
    # Never: settings, beta, and turnover with one n/a (one number left)
    assert all(best[r.name] is None for r in ROWS if r.group == 'Settings')
    assert best['Beta vs Benchmark'] is None
    assert best['Avg Turnover per Rebalance'] is None
    # A best year and a best quarter do not compare: per-period rows need one frequency
    assert best['Best Period'] is None and best['Worst Period'] is None


def test_per_period_rows_have_a_best_value_within_one_frequency():
    q1 = _wrds('b' * 64)
    q2 = _wrds('c' * 64, yearly_returns=[4.0, -3.0, 1.0, 1.0], average_turnover_pct=60.0)
    best = {r['name']: r['best'] for r in comparison_rows([q1, q2])}
    assert best['Best Period'] == 1 and best['Worst Period'] == 0     # -1% is the smaller loss
    assert best['Avg Turnover per Rebalance'] == 1
    # n/a does not count: a monthly run without turnover leaves the two quarterly runs comparable
    monthly = _wrds('d' * 64, frequency='monthly', bps=0.0)
    best = {r['name']: r['best'] for r in comparison_rows([q1, monthly, q2])}
    assert best['Avg Turnover per Rebalance'] == 2
    assert best['Best Period'] is None
    # Two Legacy runs (both annual) compare too
    a, b = _legacy('a' * 64), _legacy('c' * 64, returns=(10.0, -8.0, 30.0))
    best = {r['name']: r['best'] for r in comparison_rows([a, b])}
    assert best['Best Period'] == 1 and best['Worst Period'] == 0


@pytest.mark.parametrize('values, rule, expected', [
    ([1.0, 2.0, 3.0], 'higher', 2),
    ([1.0, 2.0, 3.0], 'lower', 0),
    ([3.0, None, 1.0], 'higher', 0),        # n/a is skipped
    ([None, None, 1.0], 'higher', None),    # one number: nothing to compare
    ([2.0, 2.0, 1.0], 'higher', None),      # tie for best
    ([2.0, 1.0, 1.0], 'higher', 0),         # a tie that is not for best
    ([0.123401, 0.12340, 0.1], 'higher', None),   # same as shown (12.34%) is a tie
    ([float('nan'), 1.0, 2.0], 'lower', 1),
])
def test_best_index_rules(values, rule, expected):
    row = ROWS_BY_NAME['CAGR'] if rule == 'higher' else ROWS_BY_NAME['Volatility (Raw)']
    assert best_index(row, values) == expected


def test_best_period_compares_returns_not_their_period_names():
    row = ROWS_BY_NAME['Best Period']
    assert best_index(row, [(0.1, '2009'), (0.2, '2003'), None]) == 1


def _costs(total, drag=0.001):
    return dict(trading_cost_bps=10.0, cagr_before_costs=0.1, average_turnover_pct=100.0, total_trading_cost=total,
                cost_drag=drag, turnover_pct=[100] * 3, trading_cost_pct=[0.1] * 3,
                portfolio_values_before_costs=[1.0] * 4)


def test_dollar_rows_are_highlighted_only_when_every_initial_aum_is_the_same():
    small = _legacy('a' * 64, returns=(10.0, 10.0, 10.0), aum=1000.0, bps=10.0, **_costs(3.0, 0.001))
    big = _legacy('b' * 64, returns=(5.0, 5.0, 5.0), aum=1_000_000.0, bps=10.0, **_costs(2500.0, 0.002))
    best = {r['name']: r['best'] for r in comparison_rows([small, big])}
    # $1,157,625 beats $1,331 and $3 of costs beats $2,500 only because of the AUMs: no highlight
    assert best['Final Portfolio Value'] is None and best['Total Trading Cost Paid'] is None
    # Rates are always compared
    assert best['Total Return'] == 0 and best['CAGR'] == 0 and best['Annualized Cost Drag'] == 0
    assert best['Cumulative Outperformance'] == 0
    # The same AUM: dollars compare again
    same = _legacy('c' * 64, returns=(5.0, 5.0, 5.0), aum=1000.0, bps=10.0, **_costs(2.5, 0.002))
    best = {r['name']: r['best'] for r in comparison_rows([small, same])}
    assert best['Final Portfolio Value'] == 0 and best['Total Trading Cost Paid'] == 1
    # Every compared analysis counts, even one whose value is n/a
    third = _legacy('d' * 64, aum=50_000.0, final_value=None)
    assert {r['name']: r['best'] for r in comparison_rows([small, same, third])}['Final Portfolio Value'] is None


@pytest.mark.parametrize('aums, expected', [
    ([1000.0, 1000.0], 0), ([1000.0, 1000.004], 0), ([1000.0, 1001.0], None), (None, 0)])
def test_best_index_with_initial_aums(aums, expected):
    assert best_index(ROWS_BY_NAME['Final Portfolio Value'], [2000.0, 1500.0], initial_aums=aums) == expected
    # A rate row ignores the AUMs
    assert best_index(ROWS_BY_NAME['CAGR'], [0.2, 0.1], initial_aums=aums) == 0


# ---------------------------------------------------------------------------
# Differences
# ---------------------------------------------------------------------------

def test_rows_differ_only_when_their_shown_values_differ():
    a, b = _legacy('a' * 64), _legacy('c' * 64, returns=(10.0, -5.0, 25.0))
    rows = {r['name']: r for r in comparison_rows([a, b])}
    assert not rows['Data source']['differs'] and not rows['Trading Cost']['differs']
    assert rows['Final Portfolio Value']['differs'] and rows['Best Period']['differs']
    assert not rows['Worst Period']['differs']
    # The same analysis twice (as values): nothing differs, nothing is best
    same = comparison_rows([a, _legacy('d' * 64)])
    assert not any(r['differs'] for r in same) and all(r['best'] is None for r in same)


def test_the_table_marks_differences_and_can_show_only_them():
    a, w = _legacy(), _wrds()
    table = comparison_table([a, w], [0, 1], differences_only=False)
    assert list(table.index[:2]) == ['SETTINGS', 'Data source ≠']
    assert 'Initial AUM' in table.index                    # same in both: not marked
    assert table.loc['Total Return ≠'].tolist() == _row([a, w], 'Total Return')['values']
    only = comparison_table([a, w], [0, 1], differences_only=True)
    assert 'Initial AUM' not in only.index and all(n.endswith(' ≠') or n in only.attrs['groups'] for n in only.index)
    assert [n for n in only.index if n in only.attrs['groups']] == ['SETTINGS', 'PERFORMANCE', 'RISK', 'COSTS']
    # Nothing differs: no table
    assert comparison_table([a, _legacy('e' * 64)], [0, 1], differences_only=True) is None
    # A group whose rows are all the same is left out
    b = _legacy('f' * 64, vol_raw_portfolio=0.3)
    only = comparison_table([a, b], [0, 1], differences_only=True)
    assert list(only.index) == ['RISK', 'Volatility (Raw) ≠']


def test_the_table_columns_follow_the_order_and_keep_each_best_cell():
    a, w, l2 = _legacy(), _wrds(), _legacy('c' * 64, returns=(1.0, 1.0, 1.0))
    titles = column_titles([a, w, l2])
    table = comparison_table([a, w, l2], [2, 0, 1], differences_only=False)
    assert list(table.columns) == [titles[2], titles[0], titles[1]]
    assert table.attrs['best']['Sharpe ≠' if 'Sharpe ≠' in table.index else 'Sharpe'] == titles[1]
    assert table.attrs['best']['CAGR ≠'] == titles[0]
    assert table.loc['CAGR ≠', titles[0]] == _row([a, w, l2], 'CAGR')['values'][0]


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def _three():
    return [_legacy('a' * 64, returns=(6.0, 6.0, 6.0), vol_raw_portfolio=0.3, sharpe_portfolio=0.2,
                    max_drawdown_portfolio=-0.30, information_ratio=0.1),
            _wrds('b' * 64, bps=25.0, vol_raw_portfolio=0.1, sharpe_portfolio=0.9, max_drawdown_portfolio=-0.10,
                  information_ratio=0.6),
            _legacy('c' * 64, returns=(10.0, 10.0, 10.0), vol_raw_portfolio=0.2, sharpe_portfolio=0.5,
                    max_drawdown_portfolio=-0.20, information_ratio=0.3, bps=10.0,
                    trading_cost_bps=10.0, cagr_before_costs=0.11, average_turnover_pct=100.0,
                    total_trading_cost=5.0, cost_drag=0.001, turnover_pct=[100] * 3, trading_cost_pct=[0.1] * 3,
                    portfolio_values_before_costs=[1000.0] * 4)]


@pytest.mark.parametrize('option, expected', [
    ('Entered order', [0, 1, 2]),
    ('CAGR', [2, 0, 1]),             # 10% a year, then 6% a year, then the WRDS year (5.05%)
    ('Total return', [2, 0, 1]),
    ('Sharpe', [1, 2, 0]),
    ('Max drawdown', [1, 2, 0]),     # the smallest loss first
    ('Volatility', [1, 2, 0]),       # lowest first
    ('Information ratio', [1, 2, 0]),
    ('Cost drag', [0, 2, 1]),        # 0% (no costs), then 0.10%, then 0.40%
])
def test_every_sort_option_puts_the_best_first(option, expected):
    analyses = _three()
    if option in ('CAGR', 'Total return'):
        assert cagr(analyses[1]) < cagr(analyses[0])
    assert sort_order(analyses, option) == expected


def test_every_menu_option_is_tested():
    assert set(SORT_OPTIONS) == {'Entered order', 'CAGR', 'Total return', 'Sharpe', 'Max drawdown', 'Volatility',
                                 'Information ratio', 'Cost drag'}


def test_sorting_keeps_ties_in_the_order_entered_and_puts_n_a_last():
    a = _legacy('a' * 64, sharpe_portfolio=0.5)
    b = _legacy('b' * 64, sharpe_portfolio=None)
    c = _legacy('c' * 64, sharpe_portfolio=0.50001)     # shown as 0.5000: a tie with a
    d = _legacy('d' * 64, sharpe_portfolio=0.7)
    assert sort_order([a, b, c, d], 'Sharpe') == [3, 0, 2, 1]
    assert sort_order([b, a], 'Sharpe') == [1, 0]
    # n/a last for lower-is-better too
    assert sort_order([_legacy(vol_raw_portfolio=None), _legacy(vol_raw_portfolio=0.4)], 'Volatility') == [1, 0]


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def test_growth_series_put_legacy_years_and_wrds_dates_on_one_axis():
    a, w = _legacy(), _wrds()
    s = growth_series(a)
    assert s['dates'] == ['2002-09-30', '2003-09-30', '2004-09-30', '2005-09-30']
    assert s['portfolio'] == a['result']['portfolio_values']
    assert s['benchmark'] == pytest.approx([1000 * 1.05 ** k for k in range(4)])
    assert s['benchmark_name'] == 'Russell 2000'
    s = growth_series(w)
    assert s['dates'][0] == '2010-03-31' and len(s['dates']) == len(s['portfolio']) == len(s['benchmark'])
    assert s['benchmark_name'] == 'IWM'


def test_the_chart_keeps_dollar_values_when_every_initial_aum_is_the_same():
    analyses = [_legacy('a' * 64), _wrds('b' * 64)]
    series, scaled = chart_series(analyses)
    assert scaled is False
    assert series == [growth_series(a) for a in analyses]
    fig = plot_compare_growth(series, column_titles(analyses), ['a', 'b'])
    assert fig.layout.title.text == '<b>Cumulative Wealth: compared analyses</b>'
    assert fig.layout.yaxis.title.text == 'Portfolio Value (USD)'


def test_the_chart_shows_growth_of_1000_when_initial_aums_differ():
    small = _legacy('a' * 64, aum=1000.0)
    big = _legacy('b' * 64, returns=(5.0, 5.0, 5.0), aum=1_000_000.0)
    series, scaled = chart_series([small, big, _wrds('c' * 64)])
    assert scaled is True and GROWTH_BASE == 1000.0
    for s, a in zip(series, (small, big)):
        raw = growth_series(a)
        # Each portfolio and its own benchmark start at $1,000 and keep their growth
        assert s['portfolio'][0] == pytest.approx(1000.0) and s['benchmark'][0] == pytest.approx(1000.0)
        assert s['portfolio'] == pytest.approx([v * 1000.0 / a['job']['initial_aum'] for v in raw['portfolio']])
        assert s['benchmark'] == pytest.approx([v * 1000.0 / a['job']['initial_aum'] for v in raw['benchmark']])
        assert s['dates'] == raw['dates']
    assert series[1]['portfolio'][-1] == pytest.approx(1000.0 * 1.05 ** 3)
    fig = plot_compare_growth(series, ['a', 'b', 'c'], ['a', 'b', 'c'], growth_of=GROWTH_BASE)
    assert fig.layout.title.text.startswith('<b>Growth of $1,000: compared analyses')
    assert fig.layout.yaxis.title.text == 'Growth of $1,000 (USD)'
    # The loaded analyses are not changed
    assert big['result']['portfolio_values'][0] == 1_000_000.0


def test_the_chart_keeps_each_analysis_color_and_orders_the_legend_by_the_sort():
    analyses = _three()
    series = [growth_series(a) for a in analyses]
    names, tags = column_titles(analyses), ['a', 'b', 'c']
    for order in ([0, 1, 2], [2, 0, 1]):
        fig = plot_compare_growth(series, names, tags, order=order)
        traces = {t.name: t for t in fig.data}
        assert len(fig.data) == 6
        for i, name in enumerate(names):
            line, bench = traces[name], traces[f"{series[i]['benchmark_name']} · {tags[i]}"]
            assert line.line.color == bench.line.color == COMPARE_COLORS[i]
            assert bench.line.dash == 'dash' and line.line.dash is None
            assert line.legendgroup == bench.legendgroup == str(i)
        ranked = sorted((t.legendrank, t.name) for t in fig.data if t.name in names)
        assert [n for _, n in ranked] == [names[i] for i in order]
    assert fig.layout.xaxis.type == 'date' and fig.layout.yaxis.tickprefix == '$'
    assert len(COMPARE_COLORS) == MAX_COMPARED
