"""
How many stocks a WRDS backtest holds (src/wrds_engine.hold_rule): the top X% of the stocks
with a score at each rebalance (in Mixing, of each sleeve), the top N (in Mixing, N in total,
the sleeves taking turns; only scored stocks, even when N covers the filter), every stock in
the filter at 100%, and the job fields (src/wrds_jobs.py) that carry the choice without
changing the cache key of a job at the default 10%.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from src.backtest_jobs import job_hash
from src.wrds_engine import (
    WHOLE_FILTER_LABEL,
    eligible_universe,
    factor_zscores,
    hold_rule,
    rank_stocks,
    run_wrds_backtest,
    select_positions,
)
from src.wrds_jobs import (
    WRDS_JOB_FIELDS,
    build_wrds_job,
    describe_hold,
    describe_wrds_job,
    engine_arguments,
    validate_wrds_job,
)
from src.wrds_service import summarize_wrds_job
from tests.unit.wrds_fixtures import WRDS_SETTINGS, synthetic_frames

FACTORS = ['f_roe', 'mom_12_1']
DIRECTIONS = {'f_roe': 'top', 'mom_12_1': 'bottom'}
LABELS = {'f_roe': 'ROE', 'mom_12_1': '12-1 Mo Momentum'}
RUN = dict(frequency='quarterly', start_year=2010, end_year=2012, initial_aum=1000.0)


@pytest.fixture(scope='module')
def data():
    return synthetic_frames(n_stocks=60, seed=5).data()


@pytest.fixture(scope='module')
def universe(data):
    """Every stock in the filter at each quarterly rebalance date of RUN."""
    dates = pd.DatetimeIndex(run_wrds_backtest(data, FACTORS, DIRECTIONS, **RUN)[0]['rebalance_dates'])
    return eligible_universe(data.panel[data.panel['date'].isin(dates)]).copy()


def _sleeve_scores(universe, combination):
    """Each sleeve's scores, as the engine ranks them: one per factor (Mixing) or the composite."""
    z = {f: factor_zscores(universe, f, DIRECTIONS[f] == 'top') for f in FACTORS}
    if combination == 'mixing':
        return list(z.values())
    both = pd.concat(z.values(), axis=1)
    return [both.mean(axis=1).where(both.notna().all(axis=1))]


def _expected(universe, scores, count):
    """date -> the permnos a sleeve holds: the top count(n) of the n scored stocks, ties by permno."""
    frame = pd.DataFrame({'date': universe['date'], 'permno': universe['permno'], 'score': scores}).dropna()
    picked = {}
    for date, rows in frame.groupby('date'):
        rows = rows.sort_values(['score', 'permno'], ascending=[False, True], kind='mergesort')
        picked[date] = set(rows['permno'].head(count(len(rows))))
    return picked


def _held(positions, sleeve):
    rows = positions[positions['sleeve'] == sleeve]
    return {date: set(group['permno']) for date, group in rows.groupby('date')}


# ---------------------------------------------------------------------------
# Engine: percent, count, count above n, 100%
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
@pytest.mark.parametrize('top_pct', [10.0, 25.0, 3.0])
def test_percent_holds_the_top_share_of_each_sleeves_scored_stocks(universe, combination, top_pct):
    positions = select_positions(universe, FACTORS, DIRECTIONS, combination, top_pct=top_pct)
    for i, scores in enumerate(_sleeve_scores(universe, combination)):
        expected = _expected(universe, scores, lambda n: max(1, math.floor(n * top_pct / 100)))
        assert _held(positions, i) == expected


@pytest.mark.parametrize('top_n', [1, 7, 20])
def test_integrating_count_holds_the_top_n_by_the_combined_score(universe, top_n):
    positions = select_positions(universe, FACTORS, DIRECTIONS, 'integrating', top_n=top_n)
    [composite] = _sleeve_scores(universe, 'integrating')
    assert _held(positions, 0) == _expected(universe, composite, lambda n: min(top_n, n))
    assert (positions.groupby('date').size() == top_n).all()           # every date has more than top_n
    np.testing.assert_allclose(positions['weight'], 1 / top_n, rtol=1e-12)


def _turns(universe, sleeve_scores, top_n):
    """
    The Mixing Top N rule, written out plainly: date -> [(sleeve, permno)] in the order picked.
    The sleeves take turns in order, each adding its next-best stock not yet picked.
    """
    picks = {}
    for date, rows in universe.groupby('date'):
        queues = []
        for scores in sleeve_scores:
            day = pd.DataFrame({'permno': rows['permno'], 'score': scores.loc[rows.index]}).dropna()
            queues.append(list(day.sort_values(['score', 'permno'], ascending=[False, True])['permno']))
        order = []
        while len(order) < top_n and any(queues):
            for i, queue in enumerate(queues):
                while queue and queue[0] in {p for _, p in order}:
                    queue.pop(0)
                if queue and len(order) < top_n:
                    order.append((i, queue.pop(0)))
        picks[date] = order
    return picks


def _picked(positions):
    return {date: set(zip(rows['sleeve'], rows['permno'])) for date, rows in positions.groupby('date')}


@pytest.mark.parametrize('top_n', [1, 7, 20])
@pytest.mark.parametrize('cap_weight', [False, True])
def test_mixing_count_holds_exactly_n_stocks_in_total(universe, top_n, cap_weight):
    positions = select_positions(universe, FACTORS, DIRECTIONS, 'mixing', top_n=top_n,
                                 use_market_cap_weight=cap_weight)
    expected = _turns(universe, _sleeve_scores(universe, 'mixing'), top_n)
    assert _picked(positions) == {date: set(order) for date, order in expected.items()}
    assert (positions.groupby('date')['permno'].nunique() == top_n).all()
    assert not positions.duplicated(['date', 'permno']).any()          # one position per stock
    # Weights across the picked stocks, not a sleeve's share
    caps = universe.set_index(['date', 'permno'])['mkt_cap_musd']
    raw = (caps.reindex(pd.MultiIndex.from_frame(positions[['date', 'permno']])).to_numpy() if cap_weight
           else np.ones(len(positions)))
    totals = pd.Series(raw).groupby(positions['date'].to_numpy()).transform('sum').to_numpy()
    np.testing.assert_allclose(positions['weight'], raw / totals, rtol=1e-12)
    np.testing.assert_allclose(positions.groupby('date')['weight'].sum(), 1.0, rtol=1e-12)


def test_overlapping_top_picks_still_give_exactly_n(universe):
    # Both factors rank the stocks the same way, so every second turn finds its best stock taken
    same = universe.assign(mom_12_1=universe['f_roe'])
    directions = {'f_roe': 'top', 'mom_12_1': 'top'}
    positions = select_positions(same, FACTORS, directions, 'mixing', top_n=7)
    top_7 = _expected(same, factor_zscores(same, 'f_roe', True), lambda n: 7)
    for date, rows in positions.groupby('date'):
        assert len(rows) == 7 and set(rows['permno']) == top_7[date]       # the top 7 by either factor
        assert list(rows['sleeve']) == [0, 1, 0, 1, 0, 1, 0]                # still taking turns


def test_fewer_than_n_scored_stocks_are_all_held(universe):
    date = universe['date'].min()
    sparse = universe.copy()
    on_date = sparse.index[sparse['date'] == date]
    sparse.loc[on_date, ['f_roe', 'mom_12_1']] = np.nan
    sparse.loc[on_date[:3], 'f_roe'] = [1.0, 2.0, 3.0]         # 3 stocks with ROE,
    sparse.loc[on_date[2:5], 'mom_12_1'] = [3.0, 2.0, 1.0]     # 3 with momentum, one of them shared
    assert len(on_date) > 10
    positions = select_positions(sparse, FACTORS, {'f_roe': 'top', 'mom_12_1': 'top'}, 'mixing', top_n=10)
    held = positions[positions['date'] == date]
    assert set(held['permno']) == set(sparse.loc[on_date[:5], 'permno'])      # all 5 scored, none unscored
    np.testing.assert_allclose(held['weight'], 1 / 5, rtol=1e-12)


def _hand_universe():
    """Six stocks on one date: ROE ranks them 1-6, momentum 2, 5, 1, 6, 3, 4."""
    return pd.DataFrame({'date': pd.Timestamp('2010-03-31'), 'permno': [1, 2, 3, 4, 5, 6],
                         'mkt_cap_musd': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                         'f_roe': [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], 'mom_12_1': [4.0, 6.0, 2.0, 1.0, 5.0, 3.0]})


def test_the_turn_order_is_deterministic_and_follows_the_factor_order():
    hand, directions = _hand_universe(), {'f_roe': 'top', 'mom_12_1': 'top'}

    def picks(universe, factors, top_n):
        positions = select_positions(universe, factors, directions, 'mixing', top_n=top_n)
        return list(zip(positions['sleeve'], positions['permno'], positions['rank']))
    # ROE takes 1; momentum takes 2; ROE's next, 2, is taken, so it takes 3; momentum takes 5
    assert picks(hand, ['f_roe', 'mom_12_1'], 4) == [(0, 1, 1), (1, 2, 1), (0, 3, 3), (1, 5, 2)]
    assert picks(hand, ['f_roe', 'mom_12_1'], 3) == [(0, 1, 1), (1, 2, 1), (0, 3, 3)]
    # Momentum first: 2, then ROE's 1, then momentum's next, 5
    assert picks(hand, ['mom_12_1', 'f_roe'], 3) == [(0, 2, 1), (1, 1, 1), (0, 5, 2)]
    # The same inputs always give the same positions
    first = select_positions(hand, FACTORS, directions, 'mixing', top_n=4)
    pd.testing.assert_frame_equal(first, select_positions(hand, FACTORS, directions, 'mixing', top_n=4))
    # Ties within a sleeve go to the lower permno, and the turns go on from there
    tied = hand.assign(f_roe=1.0, mom_12_1=1.0)
    assert picks(tied, FACTORS, 4) == [(0, 1, 1), (1, 2, 2), (0, 3, 3), (1, 4, 4)]


def test_mixing_count_shows_which_sleeve_picked_each_stock(data):
    _, holdings = run_wrds_backtest(data, FACTORS, DIRECTIONS, **RUN, top_n=5, factor_labels=LABELS)
    assert (holdings.groupby('rebalance_date').size() == 5).all()
    assert set(holdings['picked_by']) == set(LABELS.values())         # one sleeve each, never both
    np.testing.assert_allclose(holdings['weight'], 0.2, rtol=1e-12)


def test_count_breaks_ties_by_permno(universe):
    tied = universe.assign(f_roe=1.0)
    positions = select_positions(tied, ['f_roe'], DIRECTIONS, top_n=3)
    for date, rows in positions.groupby('date'):
        assert list(rows['permno']) == sorted(universe.loc[universe['date'] == date, 'permno'])[:3]


def test_a_count_below_the_filter_holds_only_scored_stocks_even_when_few_are_scored(universe):
    # One date with only two stocks scored: a count of 5 holds those two, not the unscored ones
    date = universe['date'].min()
    sparse = universe.copy()
    on_date = sparse.index[sparse['date'] == date]
    sparse.loc[on_date[2:], 'f_roe'] = np.nan
    positions = select_positions(sparse, ['f_roe'], DIRECTIONS, top_n=5)
    assert set(positions.loc[positions['date'] == date, 'permno']) == set(sparse.loc[on_date[:2], 'permno'])


def _whole_filter(universe):
    return {date: set(rows['permno']) for date, rows in universe.groupby('date')}


def _held_any(positions):
    return {date: set(rows['permno']) for date, rows in positions.groupby('date')}


def _scored(universe, combination, factors=FACTORS):
    """date -> the permnos a Top N run can hold: a score in some sleeve (Mixing) or the composite (Integrating)."""
    z = pd.concat([factor_zscores(universe, f, DIRECTIONS[f] == 'top') for f in factors], axis=1)
    has = z.notna().any(axis=1) if combination == 'mixing' else z.notna().all(axis=1)
    rows = universe[has.to_numpy()]
    return {date: set(group['permno']) for date, group in rows.groupby('date')}


@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
@pytest.mark.parametrize('cap_weight', [False, True])
def test_a_count_above_the_filter_holds_every_scored_stock_and_no_other(universe, combination, cap_weight):
    universe = universe.copy()
    universe.loc[universe.index[::17], FACTORS] = np.nan     # stocks with no score at all
    positions = select_positions(universe, FACTORS, DIRECTIONS, combination, top_n=10_000,
                                 use_market_cap_weight=cap_weight)
    held, scored = _held_any(positions), _scored(universe, combination)
    assert held == scored
    for date, permnos in _whole_filter(universe).items():
        assert held[date] < permnos                           # never the whole filter, unlike Top 100%
    assert set(positions['sleeve']) <= set(range(len(FACTORS)))     # every stock picked by a factor
    weights = positions.groupby(['date', 'permno'])['weight'].sum()
    caps = universe.set_index(['date', 'permno'])['mkt_cap_musd'].reindex(weights.index)
    raw = caps if cap_weight else pd.Series(1.0, index=weights.index)
    np.testing.assert_allclose(weights, raw / raw.groupby(level='date').transform('sum'), rtol=1e-12)


@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
@pytest.mark.parametrize('factors', [['f_roe'], FACTORS])
def test_a_count_holds_min_n_and_the_scored_stocks_on_every_date(universe, combination, factors):
    sizes = universe.groupby('date').size()
    top_n = int(sizes.median())                               # at least the filter on some dates
    assert (sizes <= top_n).any() and (sizes > top_n).any()
    positions = select_positions(universe, factors, DIRECTIONS, combination, top_n=top_n)
    held, scored = _held_any(positions), _scored(universe, combination, factors)
    for date in sizes.index:
        assert len(held[date]) == min(top_n, len(scored[date])) and held[date] <= scored[date]


def test_a_count_run_shows_no_whole_filter_holdings(data):
    _, holdings = run_wrds_backtest(data, FACTORS, DIRECTIONS, **RUN, top_n=10_000, factor_labels=LABELS)
    assert WHOLE_FILTER_LABEL not in set(holdings['picked_by']) and holdings['score'].notna().all()


@pytest.mark.parametrize('combination', ['mixing', 'integrating'])
def test_100_percent_holds_every_stock_in_the_filter(data, universe, combination):
    results, holdings = run_wrds_backtest(data, FACTORS, DIRECTIONS, **RUN, top_pct=100.0,
                                          factor_combination=combination)
    held = {pd.Timestamp(d): set(rows['permno']) for d, rows in holdings.groupby('rebalance_date')}
    assert held == _whole_filter(universe)
    weights = holdings.groupby('rebalance_date')['weight']
    np.testing.assert_allclose(holdings['weight'], 1 / weights.transform('size'), rtol=1e-12)
    # The factors no longer matter: the same returns as a run without factors
    no_factors, plain = run_wrds_backtest(data, [], {}, **RUN, top_pct=100.0)
    np.testing.assert_allclose(results['yearly_returns'], no_factors['yearly_returns'], rtol=1e-12)
    assert set(plain['picked_by']) == {WHOLE_FILTER_LABEL} and plain['score'].isna().all()
    assert len(plain) == len(holdings)


def test_100_percent_with_a_gics_filter_holds_the_whole_filter(data):
    gics = {'level': 'sector', 'codes': [10]}             # Energy
    _, holdings = run_wrds_backtest(data, [], {}, **RUN, top_pct=100.0, gics_filter=gics)
    assert set(holdings['sector']) == {'Energy'}
    in_panel = data.panel[data.panel['date'].isin(pd.to_datetime(holdings['rebalance_date']).unique())]
    assert len(holdings) == len(eligible_universe(in_panel, gics_filter=gics))


def test_a_run_needs_a_factor_unless_it_holds_100_percent(data):
    for hold in ({}, {'top_pct': 50.0}, {'top_n': 10_000}):
        with pytest.raises(ValueError, match='at least one factor'):
            run_wrds_backtest(data, [], {}, **RUN, **hold)


@pytest.mark.parametrize('top_pct, top_n, message', [
    (10.0, 5, 'not both'), (None, 0, 'at least 1'), (None, 2.5, 'whole number'), (None, True, 'whole number'),
    (0.0, None, 'above 0'), (100.5, None, 'at most 100'),
])
def test_bad_hold_choices_are_refused(top_pct, top_n, message):
    with pytest.raises(ValueError, match=message):
        hold_rule(top_pct, top_n)


def test_neither_choice_is_the_top_10_percent(universe):
    assert hold_rule() == (10.0, None)
    default = select_positions(universe, FACTORS, DIRECTIONS)
    pd.testing.assert_frame_equal(default, select_positions(universe, FACTORS, DIRECTIONS, top_pct=10.0))


def test_ranked_list_marks_the_top_n_as_held(data, universe):
    date = universe['date'].min()
    table = rank_stocks(data, date, ['f_roe'], DIRECTIONS, top_n=4)
    assert table['held'].sum() == 4 and list(table.loc[table['held'], 'rank']) == [1, 2, 3, 4]
    everything = rank_stocks(data, date, [], {}, top_pct=100.0)
    assert everything['held'].all() and len(everything) == (universe['date'] == date).sum()
    assert everything['score'].isna().all()


# ---------------------------------------------------------------------------
# Jobs: top_n only when set; a job at 10% is exactly as before
# ---------------------------------------------------------------------------

def _job(kind='backtest', **overrides):
    return build_wrds_job(kind, FACTORS, DIRECTIONS, dict(WRDS_SETTINGS, **overrides), factor_labels=LABELS,
                          cohort_pct=20 if kind == 'cohort' else None,
                          rebalance_date='2010-06' if kind == 'ranked' else None)


# job_hash(job, 'data-v1', 'code-a') of these jobs, computed with the code before top_n existed
OLD_HASHES = [
    ('backtest', {}, 'dae796257bbe6d84e42b8dc67711915375c457702be247e8a9b50937865aef92'),
    ('backtest', {'trading_cost_bps': 25, 'gics_filter': {'level': 'sector', 'codes': [35, 10]}},
     'dd851b7264b6001b3f4111b40f33430014ae7a221f5521bf51da191512952962'),
    ('cohort', {}, 'dbaf20c25e2cccceee6ff92444a4a1a064e50a687885b143282bc51f6333cce5'),
    ('ranked', {}, '13b99cc3895899b15522bbf40ddf23ac60fcd130d744a4cc27f9bfe5389ffd2b'),
]


@pytest.mark.parametrize('kind, settings, expected', OLD_HASHES)
@pytest.mark.parametrize('hold', [{}, {'top_pct': 10.0}, {'top_pct': 10}])
def test_a_job_at_10_percent_hashes_exactly_as_before(kind, settings, expected, hold):
    job = _job(kind, **settings, **hold)
    assert 'top_n' not in job and job['top_pct'] == 10.0
    assert job_hash(job, 'data-v1', 'code-a') == expected


def test_a_count_job_has_top_n_and_a_null_top_pct():
    job = _job(top_n=20)
    assert job['top_n'] == 20 and job['top_pct'] is None
    assert tuple(job) == WRDS_JOB_FIELDS + ('top_n',)
    validate_wrds_job(json.loads(json.dumps(job)))
    assert engine_arguments(job)[2]['top_n'] == 20 and engine_arguments(job)[2]['top_pct'] is None
    assert _job(top_n=20.0)['top_n'] == 20                   # a whole float is a count
    assert _job(top_pct=25)['top_pct'] == 25.0 and engine_arguments(_job(top_pct=25))[2]['top_n'] is None
    assert len({job_hash(j, 'd', 'c') for j in (_job(), _job(top_pct=25), _job(top_n=20), _job(top_n=21))}) == 4


def test_a_cohort_job_ignores_the_choice():
    # It holds cohort_pct, so a run's choice must not change its key
    for hold in ({'top_pct': 25.0}, {'top_n': 20}, {'top_pct': 100.0}):
        assert _job('cohort', **hold) == _job('cohort')
    assert _job('ranked', top_n=20)['top_n'] == 20


@pytest.mark.parametrize('change, message', [
    ({'top_n': 20, 'top_pct': 10.0}, 'not both'),
    ({'top_n': 0, 'top_pct': None}, 'at least 1'),
    ({'top_n': 2.5, 'top_pct': None}, 'whole number'),
    ({'top_n': True, 'top_pct': None}, 'whole number'),
    ({'top_n': '20', 'top_pct': None}, 'whole number'),
    ({'top_pct': None}, 'top_pct must be above 0'),
    ({'top_pct': 0}, 'top_pct must be above 0'),
    ({'factors': [], 'factor_directions': {}, 'factor_labels': {}}, 'at least one factor'),
    ({'factors': [], 'factor_directions': {}, 'factor_labels': {}, 'top_pct': 50.0}, 'at least one factor'),
    ({'factors': [], 'factor_directions': {}, 'factor_labels': {}, 'top_pct': None, 'top_n': 9999},
     'at least one factor'),
])
def test_invalid_hold_choices_in_jobs_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        validate_wrds_job(dict(_job(), **change))


def test_a_job_without_factors_runs_only_at_100_percent():
    job = build_wrds_job('backtest', [], {}, dict(WRDS_SETTINGS, top_pct=100.0))
    assert job['factors'] == [] and job['top_pct'] == 100.0
    ranked = build_wrds_job('ranked', [], {}, dict(WRDS_SETTINGS, top_pct=100.0), rebalance_date='2010-06')
    assert ranked['factors'] == []
    with pytest.raises(ValueError, match='at least one factor'):
        build_wrds_job('cohort', [], {}, dict(WRDS_SETTINGS, top_pct=100.0), cohort_pct=20)
    assert summarize_wrds_job(job).startswith('WRDS, Quarterly | No factors | Mixing')


def test_descriptions_name_the_choice():
    assert describe_hold({}) == 'Top 10%' and describe_hold({'top_pct': 12.5}) == 'Top 12.5%'
    assert describe_hold({'top_pct': None, 'top_n': 1}) == 'Top 1 stock'
    assert describe_hold({'top_n': 20}) == 'Top 20 stocks'
    assert describe_hold({'top_pct': 100}) == 'Every stock in the filter (100%)'
    # Logs and listings add it only when it is not the default
    assert 'top' not in describe_wrds_job(_job()).split(':')[0] and 'Top' not in summarize_wrds_job(_job())
    assert describe_wrds_job(_job(top_n=20)).endswith(', top 20 stocks')
    assert summarize_wrds_job(_job(top_pct=25)).endswith('| Top 25%')
