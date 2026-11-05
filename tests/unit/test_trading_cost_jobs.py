"""
Trading costs in jobs, cache keys, the Lambda services, saved analyses and their listings.

At 0 bps a job has no trading_cost_bps, so its cache key is main's; above 0 the cost is part
of the job and its key. The reference keys below were computed with main's code (faeca5c)
for the same jobs, data versions and code version string.
"""
import json

import pytest

from src.backtest_jobs import JOB_FIELDS, build_job, decode_saved, job_hash, run_job
from src.backtest_service import result_file, summarize_job
from src.saved_analyses import list_saved_analyses, load_saved_analysis
from src.wrds_jobs import WRDS_JOB_FIELDS, build_wrds_job, describe_wrds_job, engine_arguments, validate_wrds_job
from src.wrds_service import summarize_wrds_job
from tests.unit.test_wrds_service import BUCKET, PREFIX, frames, manager, s3, service, wrds_job  # noqa: F401 (fixtures)
from tests.unit.wrds_fixtures import WRDS_SETTINGS

LEGACY_SETTINGS = dict(start_year=2004, end_year=2020, initial_aum=1000.0)
LEGACY_FACTORS = ['ROE_using_9-30_Data', 'ROA']
LEGACY_DIRECTIONS = {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom'}
WRDS_FACTORS = ['f_roe', 'mom_12_1']
WRDS_DIRECTIONS = {'f_roe': 'top', 'mom_12_1': 'bottom'}
# job_hash(job, data version, 'code-x') of these jobs on main
MAIN_KEYS = {
    'legacy': 'fe9497f10769b905f49b62bfb1c8191b52db4fdfe5f2620d2449f27f36285785',
    'legacy_cohort': 'ccca1760840cbd4ecd2b9bf965b2cbc12956e4d76119e28e59be2901f482c695',
    'wrds_backtest': '0559a6ec6217aeb142dfff1f816bdf95b1f813ced7e6b22b11b7c10400ac78fd',
    'wrds_cohort': '2b01a66ecf9fe2e94c2ab4d144ac056cc9b5cfbe99d43a33021c6b5344dec688',
    'wrds_ranked': 'ae39036456cbc11152be2aeab7174415735f5fc94cc1853c6c8494a9ba787954',
}


def _legacy(kind='backtest', **settings):
    return build_job(kind, LEGACY_FACTORS, LEGACY_DIRECTIONS, dict(LEGACY_SETTINGS, **settings),
                     cohort_pct=20 if kind == 'cohort' else None)


def _wrds(kind='backtest', **settings):
    return build_wrds_job(kind, WRDS_FACTORS, WRDS_DIRECTIONS, dict(WRDS_SETTINGS, **settings),
                          cohort_pct=20 if kind == 'cohort' else None,
                          rebalance_date='2010-06' if kind == 'ranked' else None)


def _legacy_key(job):
    return job_hash(job, 'data-v1', 'code-x')


def _wrds_key(job):
    return job_hash(job, 'wrds-v1', 'code-x')


# ---------------------------------------------------------------------------
# Jobs and cache keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('settings', [{}, {'trading_cost_bps': 0}, {'trading_cost_bps': 0.0}])
def test_cache_keys_at_0_bps_are_mains(settings):
    assert _legacy_key(_legacy(**settings)) == MAIN_KEYS['legacy']
    assert _legacy_key(_legacy('cohort', **settings)) == MAIN_KEYS['legacy_cohort']
    for kind in ('backtest', 'cohort', 'ranked'):
        assert _wrds_key(_wrds(kind, **settings)) == MAIN_KEYS[f'wrds_{kind}']
    # The jobs themselves are main's: no trading_cost_bps key
    assert tuple(_legacy(**settings)) == JOB_FIELDS and tuple(_wrds(**settings)) == WRDS_JOB_FIELDS


def test_an_explicit_0_in_a_job_still_gives_mains_key():
    assert _legacy_key(dict(_legacy(), trading_cost_bps=0)) == MAIN_KEYS['legacy']
    assert _wrds_key(dict(_wrds(), trading_cost_bps=0.0)) == MAIN_KEYS['wrds_backtest']


def test_a_cost_above_0_is_in_the_job_and_changes_the_key():
    legacy, wrds = _legacy(trading_cost_bps=25), _wrds(trading_cost_bps=25)
    assert legacy['trading_cost_bps'] == 25.0 and wrds['trading_cost_bps'] == 25.0
    assert _legacy_key(legacy) != MAIN_KEYS['legacy'] and _wrds_key(wrds) != MAIN_KEYS['wrds_backtest']
    assert _legacy_key(legacy) != _legacy_key(_legacy(trading_cost_bps=30))
    assert _legacy_key(legacy) == _legacy_key(_legacy(trading_cost_bps=25.0))         # int and float alike
    assert _legacy_key(_legacy('cohort', trading_cost_bps=25)) != MAIN_KEYS['legacy_cohort']
    assert _wrds_key(_wrds('cohort', trading_cost_bps=25)) != MAIN_KEYS['wrds_cohort']
    validate_wrds_job(json.loads(json.dumps(wrds)))                                    # plain JSON, as Lambda gets it


def test_ranked_lists_ignore_the_cost_and_keep_their_key():
    ranked = _wrds('ranked', trading_cost_bps=25)
    assert 'trading_cost_bps' not in ranked and _wrds_key(ranked) == MAIN_KEYS['wrds_ranked']
    with pytest.raises(ValueError, match="only for backtests and cohort"):
        validate_wrds_job(dict(ranked, trading_cost_bps=25.0))


@pytest.mark.parametrize('value, message', [
    (600, "from 0 to 500 bps per trade, not 600"), (-5, "from 0 to 500"), ('25', "a number"), (True, "a number"),
])
def test_invalid_costs_are_refused_in_both_job_types(value, message):
    with pytest.raises(ValueError, match=message):
        _legacy(trading_cost_bps=value)
    with pytest.raises(ValueError, match=message):
        _wrds(trading_cost_bps=value)
    with pytest.raises(ValueError, match=message):
        validate_wrds_job(dict(_wrds(), trading_cost_bps=value))


def test_jobs_and_summaries_describe_the_cost_only_above_0():
    assert summarize_job(_legacy()).endswith('| $1,000')
    assert summarize_job(_legacy(trading_cost_bps=25)).endswith('| $1,000 | 25 bps')
    assert summarize_wrds_job(_wrds(trading_cost_bps=12.5)).endswith('| 12.5 bps')
    assert 'bps' not in summarize_wrds_job(_wrds())
    assert describe_wrds_job(_wrds('cohort', trading_cost_bps=25)).endswith('cohort 20%, 25 bps')
    assert engine_arguments(_wrds())[2]['trading_cost_bps'] == 0.0
    assert engine_arguments(_wrds(trading_cost_bps=25))[2]['trading_cost_bps'] == 25.0


def test_run_job_uses_the_cost_for_backtests_and_cohorts(manager):
    raw = manager.fetch_all_data()
    free, costly = run_job(raw, _legacy()), run_job(raw, _legacy(trading_cost_bps=50))
    assert costly['trading_cost_bps'] == 50.0 and 'trading_cost_bps' not in free
    assert costly['final_value'] < free['final_value']
    top_free, top_costly = run_job(raw, _legacy('cohort')), run_job(raw, _legacy('cohort', trading_cost_bps=50))
    assert top_costly['top'][-1] < top_free['top'][-1] and top_costly['bottom'][-1] < top_free['bottom'][-1]


# ---------------------------------------------------------------------------
# The services: cache, holdings, cohorts, saved analyses and listings
# ---------------------------------------------------------------------------

def test_legacy_service_caches_each_cost_separately_and_lists_it(service, manager):
    free = service.handle_event(_legacy())
    costly = service.handle_event(_legacy(trading_cost_bps=25))
    assert free['hash'] != costly['hash'] and not costly['cached']
    assert service.handle_event(json.loads(json.dumps(_legacy(trading_cost_bps=25))))['cached']
    assert costly['result']['trading_cost_bps'] == 25.0 and 'turnover_pct' not in free['result']
    summaries = {e['analysis_id']: e['summary'] for e in list_saved_analyses(manager)['entries']}
    assert summaries[costly['hash']].endswith('| 25 bps') and 'bps' not in summaries[free['hash']]
    saved = load_saved_analysis(costly['hash'], manager)
    assert saved['job']['trading_cost_bps'] == 25.0 and saved['result']['trading_cost_bps'] == 25.0


def test_wrds_service_saves_the_traded_column_and_lists_the_cost(service, s3, manager):
    from src.wrds_holdings import read_holdings_date
    response = service.handle_event(wrds_job(trading_cost_bps=25))
    result = response['result']
    assert result['trading_cost_bps'] == 25.0 and len(result['turnover_pct']) == len(result['yearly_returns'])
    holdings = result['holdings']
    rows = read_holdings_date(manager, holdings['file'], 0, holdings['dates'][0])
    assert rows['traded'].tolist() == pytest.approx(rows['weight'].tolist())         # bought from cash
    free = service.handle_event(wrds_job())['result']
    assert 'traded' not in read_holdings_date(manager, free['holdings']['file'], 0, free['holdings']['dates'][0])
    entry = next(e for e in list_saved_analyses(manager)['entries'] if e['analysis_id'] == response['hash'])
    assert entry['summary'].startswith('WRDS, Quarterly') and entry['summary'].endswith('| 25 bps')
    saved_job, _ = decode_saved(s3.objects[f'{PREFIX}/{result_file(response["hash"])}'][0])
    assert saved_job == wrds_job(trading_cost_bps=25)


def test_wrds_cohort_pays_the_cost_on_both_sides(service):
    free = service.handle_event(wrds_job('cohort'))['result']
    costly = service.handle_event(wrds_job('cohort', trading_cost_bps=50))['result']
    assert costly['top'][-1] < free['top'][-1] and costly['bottom'][-1] < free['bottom'][-1]
    assert costly['top_cagr'] < free['top_cagr']


def test_old_ids_open_at_0_bps(manager, s3):
    """A result saved by main has no trading cost fields; its job opens as a 0 bps job."""
    from pathlib import Path

    from src.trading_costs import has_trading_costs, job_trading_cost
    body = (Path(__file__).resolve().parents[1] / 'fixtures' / 'saved_main_backtest.json').read_bytes()
    old_id = '1' * 64
    s3.objects[f'{PREFIX}/{result_file(old_id)}'] = (body, {})
    saved = load_saved_analysis(old_id, manager)
    assert 'trading_cost_bps' not in saved['job'] and job_trading_cost(saved['job']) == 0.0
    assert not has_trading_costs(saved['result'])
    assert job_hash(saved['job'], 'v', 'c') == job_hash(build_job(
        'backtest', saved['job']['factors'], saved['job']['factor_directions'], saved['job'],
        factor_labels=saved['job']['factor_labels']), 'v', 'c')                      # rebuilt: the same key
