"""
Tests for the GICS filter of WRDS backtests (src/wrds_gics.py): the hierarchy tables, the
filter at each of the four levels, stocks without GICS, and older jobs with selected_sectors.
"""
import pandas as pd
import pytest

from pipeline import config as pipeline_config
from src.wrds_engine import HOLDINGS_GICS_COLUMNS, eligible_universe, rank_stocks, run_wrds_backtest
from src.wrds_factors import WRDS_SECTORS
from src.wrds_gics import (
    GICS_LEVELS,
    GICS_SECTORS,
    codes_under,
    describe_gics_filter,
    gics_mask,
    job_gics_filter,
    make_gics_filter,
    parent_code,
    sector_codes,
    validate_gics_filter,
)
from src.wrds_jobs import build_wrds_job, engine_arguments, validate_wrds_job
from tests.unit.wrds_fixtures import WRDS_SETTINGS, gics_names, synthetic_frames

# Sub-industries the tests give the synthetic stocks: two sectors with two industry groups each
SUB_INDUSTRIES = (10101010, 10102020, 40101015, 40301040, 45103010, 45103020)
FIRST, LAST = '2009-11', '2013-02'


def test_tables_match_the_pipelines_names_and_nest():
    tables = {'sector': pipeline_config.GICS_SECTORS, 'industry_group': pipeline_config.GICS_GROUPS,
              'industry': pipeline_config.GICS_INDUSTRIES, 'sub_industry': pipeline_config.GICS_SUB_INDUSTRIES}
    for level, (_, digits, names) in GICS_LEVELS.items():
        assert all(tables[level][code] == name for code, name in names.items()), level
        assert all(len(str(code)) == digits for code in names), level
        assert len(set(names.values())) == len(names), f"{level} names must be unique (options are names)"
    for code in GICS_LEVELS['sub_industry'][2]:
        for level in ('sector', 'industry_group', 'industry'):
            assert parent_code(code, level) in GICS_LEVELS[level][2]
    assert sorted(GICS_SECTORS.values()) == sorted(WRDS_SECTORS)


def test_parent_codes():
    assert parent_code(45103010, 'sector') == 45 and parent_code(451030, 'industry_group') == 4510
    assert parent_code(4510, 'industry_group') == 4510


def test_codes_under_narrow_to_the_levels_above():
    assert codes_under('industry_group', [40], 'sector') == [4010, 4020, 4030, 4040]
    assert codes_under('industry', [4010], 'industry_group') == [401010, 401020]
    assert codes_under('sub_industry', [40], 'sector')[:2] == [40101010, 40101015]
    assert codes_under('industry', [], 'sector') == []


def _panel():
    """One row per case: GICS stocks, a stock whose sector comes from SIC, and an Unclassified one."""
    rows = [{'gsubind': code, 'sector_name': GICS_SECTORS[code // 1000000]} for code in SUB_INDUSTRIES]
    rows += [{'gsubind': pd.NA, 'sector_name': 'Financials'},           # no GICS, sector from SIC
             {'gsubind': pd.NA, 'sector_name': 'Unclassified'}]
    panel = pd.DataFrame(rows)
    panel['gsubind'] = panel['gsubind'].astype('Int64')
    return panel.assign(prc=10.0, mkt_cap_musd=100.0, siccd=6022.0)


@pytest.mark.parametrize('level, codes, kept', [
    # Sector: sector_name, so the stock whose sector comes from its SIC code is kept
    ('sector', [40], [40101015, 40301040, 'sic']),
    ('industry_group', [4010, 4510], [40101015, 45103010, 45103020]),
    ('industry', [101010, 401010], [10101010, 40101015]),
    ('sub_industry', [45103020], [45103020]),
])
def test_each_level_keeps_its_codes(level, codes, kept):
    panel = _panel()
    out = eligible_universe(panel, gics_filter={'level': level, 'codes': codes})
    got = ['sic' if pd.isna(g) else int(g) for g in out['gsubind']]
    assert got == kept


def test_stocks_without_gics_are_left_out_below_the_sector_level():
    panel = _panel()
    everything = {level: codes_under(level, GICS_SECTORS, 'sector') for level in GICS_LEVELS}
    for level, codes in everything.items():
        mask = gics_mask(panel, {'level': level, 'codes': codes})
        no_gics = panel['gsubind'].isna().to_numpy()
        assert mask[~no_gics].all(), level
        # Only the sector level keeps the stock with a SIC sector; Unclassified is always out
        assert list(mask[no_gics]) == [level == 'sector', False], level
    assert eligible_universe(panel).shape[0] == len(panel)       # no filter: everything is kept


def test_an_old_jobs_selected_sectors_is_the_sector_filter():
    assert job_gics_filter(None, ['Health Care', 'Energy']) == {'level': 'sector', 'codes': [10, 35]}
    assert job_gics_filter(None, []) is None and job_gics_filter(None, None) is None
    with pytest.raises(ValueError, match='not both'):
        job_gics_filter({'level': 'sector', 'codes': [10]}, ['Energy'])
    with pytest.raises(ValueError, match='Unknown GICS sectors'):
        sector_codes(['Technology'])
    panel = _panel()
    old = eligible_universe(panel, selected_sectors=['Financials', 'Energy'])
    new = eligible_universe(panel, gics_filter={'level': 'sector', 'codes': [10, 40]})
    pd.testing.assert_frame_equal(old, new)


@pytest.mark.parametrize('bad, message', [
    ({'level': 'sector'}, 'level and codes'),
    ({'level': 'country', 'codes': [10]}, 'level must be one of'),
    ({'level': 'sector', 'codes': []}, 'at least one'),
    ({'level': 'sector', 'codes': [40, 10]}, 'sorted'),
    ({'level': 'sector', 'codes': [10, 10]}, 'sorted'),
    ({'level': 'sector', 'codes': ['10']}, 'whole numbers'),
    ({'level': 'industry', 'codes': [4010]}, 'Unknown GICS industry codes'),
])
def test_invalid_gics_filters_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_gics_filter(bad)


def test_describe_and_make():
    assert describe_gics_filter(None) == 'All sectors'
    assert describe_gics_filter({'level': 'industry', 'codes': [401010, 403010]}) == 'Industry: Banks, Insurance'
    many = make_gics_filter('sector', GICS_SECTORS)
    assert describe_gics_filter(many, limit=2) == 'Communication Services, Consumer Discretionary and 9 more'
    assert describe_gics_filter({'level': 'sector', 'codes': [10, 40]}) == 'Energy, Financials'
    assert make_gics_filter('industry', []) is None
    assert make_gics_filter('industry', [403010, 401010, 401010]) == {'level': 'industry', 'codes': [401010, 403010]}


# ---- Jobs -------------------------------------------------------------------------------------

def test_a_job_has_gics_filter_only_when_it_is_set():
    plain = build_wrds_job('backtest', ['f_roe'], {'f_roe': 'top'}, WRDS_SETTINGS)
    assert 'gics_filter' not in plain and plain['selected_sectors'] == []
    job = build_wrds_job('backtest', ['f_roe'], {'f_roe': 'top'},
                         {**WRDS_SETTINGS, 'gics_filter': {'level': 'industry', 'codes': [403010, 401010]}})
    assert job['gics_filter'] == {'level': 'industry', 'codes': [401010, 403010]}
    assert engine_arguments(job)[2]['gics_filter'] == job['gics_filter']
    assert engine_arguments(plain)[2]['gics_filter'] is None


def test_an_old_job_with_selected_sectors_still_validates_and_runs():
    old = {'kind': 'backtest', 'data_mode': 'wrds', 'wrds_job_version': 1, 'factors': ['f_roe'],
           'factor_directions': {'f_roe': 'top'}, 'factor_labels': {'f_roe': 'ROE'}, 'frequency': 'quarterly',
           'start_year': 2010, 'end_year': 2012, 'initial_aum': 1000.0, 'top_pct': 10.0,
           'use_market_cap_weight': False, 'factor_combination': 'mixing', 'restrict_fossil_fuels': False,
           'selected_sectors': ['Energy', 'Financials'], 'cohort_pct': None, 'rebalance_date': None}
    validate_wrds_job(old)
    factors, directions, options = engine_arguments(old)
    assert options['selected_sectors'] == ['Energy', 'Financials'] and options['gics_filter'] is None
    data = _frames().data()
    got = run_wrds_backtest(data, factors, directions, first_month=FIRST, last_month=LAST, **options)
    options.update(selected_sectors=[], gics_filter={'level': 'sector', 'codes': [10, 40]})
    same = run_wrds_backtest(data, factors, directions, first_month=FIRST, last_month=LAST, **options)
    assert got[0]['portfolio_values'] == same[0]['portfolio_values']
    pd.testing.assert_frame_equal(got[1], same[1])


@pytest.mark.parametrize('change, message', [
    ({'gics_filter': {'level': 'industry', 'codes': [1]}}, 'Unknown GICS industry codes'),
    ({'gics_filter': {'level': 'sector', 'codes': [10]}, 'selected_sectors': ['Energy']}, 'not both'),
])
def test_invalid_gics_jobs_are_refused(change, message):
    job = {**build_wrds_job('backtest', ['f_roe'], {'f_roe': 'top'}, WRDS_SETTINGS), **change}
    with pytest.raises(ValueError, match=message):
        validate_wrds_job(job)


# ---- Backtests ----------------------------------------------------------------------------------

def _frames():
    """The synthetic data with the stocks spread over SUB_INDUSTRIES (and their sector names)."""
    frames = synthetic_frames(FIRST, LAST, n_stocks=60)
    panel = frames.panel
    codes = pd.Series(SUB_INDUSTRIES).iloc[(panel['permno'] - 10001) % len(SUB_INDUSTRIES)].to_numpy()
    panel['gsubind'] = pd.array(codes, dtype='Int64')
    panel['sector_name'] = [GICS_SECTORS[c // 1000000] for c in codes]
    names = pd.DataFrame([gics_names(c) for c in codes], index=panel.index)
    panel[list(names.columns)] = names
    return frames


@pytest.fixture(scope='module')
def data():
    return _frames().data()


@pytest.mark.parametrize('level, codes, sub_industries', [
    ('sector', [40], {40101015, 40301040}),
    ('industry_group', [1010, 4030], {10101010, 10102020, 40301040}),
    ('industry', [451030], {45103010, 45103020}),
    ('sub_industry', [10102020, 45103020], {10102020, 45103020}),
])
def test_a_backtest_at_each_level_holds_only_those_stocks(data, level, codes, sub_industries):
    gics_filter = {'level': level, 'codes': codes}
    results, holdings = run_wrds_backtest(data, ['f_roe'], {'f_roe': 'top'}, 'quarterly', 2010, 2012, 1000.0,
                                          top_pct=50.0, gics_filter=gics_filter, first_month=FIRST, last_month=LAST)
    names = {gics_names(c)['gsubind_name'] for c in sub_industries}
    assert set(holdings['gics_sub_industry'].astype(str)) == names
    assert not results['empty_rebalance_dates']
    # The four GICS names follow the sector, before Picked By
    columns = list(holdings.columns)
    assert columns[columns.index('sector') + 1:columns.index('sector') + 5] == list(HOLDINGS_GICS_COLUMNS.values())

    ranked = rank_stocks(data, pd.Timestamp(results['rebalance_dates'][0]), ['f_roe'], {'f_roe': 'top'},
                         gics_filter=gics_filter)
    universe = data.panel[data.panel['date'] == pd.Timestamp(results['rebalance_dates'][0])]
    assert set(ranked['permno']) == set(universe.loc[universe['gsubind'].isin(sub_industries), 'permno'])
