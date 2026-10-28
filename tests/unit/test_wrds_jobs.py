"""
Tests for WRDS jobs (src/wrds_jobs.py): building and validating them, and the cache
keys of both modes (DOCS/WRDS_MODE_DESIGN.md, section 5).
"""
import json

import pytest

from src.backtest_jobs import ENGINE_SOURCES, JOB_FIELDS, build_job, code_version, job_hash
from src.wrds_jobs import (WRDS_ENGINE_SOURCES, WRDS_JOB_FIELDS, build_wrds_job, validate_wrds_job,
                           wrds_code_version)
from tests.unit.wrds_fixtures import WRDS_SETTINGS

FACTORS = ['f_roe', 'mom_12_1']
DIRECTIONS = {'f_roe': 'top', 'mom_12_1': 'bottom'}
LABELS = {'f_roe': 'ROE', 'mom_12_1': '12-1 Mo Momentum'}


def _job(kind='backtest', **overrides):
    settings = dict(WRDS_SETTINGS, **overrides)
    return build_wrds_job(kind, FACTORS, DIRECTIONS, settings, factor_labels=LABELS,
                          cohort_pct=20 if kind == 'cohort' else None,
                          rebalance_date='2010-06' if kind == 'ranked' else None)


def test_a_wrds_job_has_every_field_normalized():
    job = _job(selected_sectors=['Health Care', 'Energy'], start_year='2010', initial_aum=500)
    assert tuple(job) == WRDS_JOB_FIELDS
    assert job['data_mode'] == 'wrds' and job['frequency'] == 'quarterly' and job['top_pct'] == 10.0
    assert job['start_year'] == 2010 and job['initial_aum'] == 500.0
    assert job['selected_sectors'] == ['Energy', 'Health Care']
    assert job['cohort_pct'] is None and job['rebalance_date'] is None
    assert json.loads(json.dumps(job)) == job                          # plain JSON, as Lambda receives it
    validate_wrds_job(json.loads(json.dumps(job)))


def test_cohort_and_ranked_jobs():
    cohort, ranked = _job('cohort'), _job('ranked')
    assert cohort['cohort_pct'] == 20.0 and cohort['rebalance_date'] is None
    assert ranked['rebalance_date'] == '2010-06' and ranked['cohort_pct'] is None


@pytest.mark.parametrize('change, message', [
    ({'kind': 'holdings'}, "kind must be one of"),
    ({'data_mode': 'legacy'}, "needs data_mode 'wrds'"),
    ({'wrds_job_version': 2}, "not supported"),
    ({'factors': []}, "at least one factor"),
    ({'factors': ['f_roe', 'f_roe']}, "only once"),
    ({'factors': ['ret_next_1m'], 'factor_directions': {'ret_next_1m': 'top'},
      'factor_labels': {'ret_next_1m': 'x'}}, "Not WRDS factors"),
    ({'factor_directions': {'f_roe': 'top'}}, "exactly the job's factors"),
    ({'factor_directions': {'f_roe': 'up', 'mom_12_1': 'top'}}, "'top' or 'bottom'"),
    ({'factor_labels': {'f_roe': '', 'mom_12_1': 'x'}}, "must be text"),
    ({'frequency': 'weekly'}, "frequency must be one of"),
    ({'start_year': 2010.5}, "start_year must be a whole number"),
    ({'start_year': 2005}, "Choose years from 2008 to 2026"),
    ({'start_year': 2013, 'end_year': 2012}, "not after the end year"),
    ({'initial_aum': 0}, "greater than \\$0"),
    ({'initial_aum': '1000'}, "greater than \\$0"),
    ({'top_pct': 150}, "top_pct"),
    ({'use_market_cap_weight': 'yes'}, "must be true or false"),
    ({'factor_combination': 'blend'}, "factor_combination"),
    ({'selected_sectors': ['Technology']}, "Unknown GICS sectors"),
    ({'selected_sectors': ['Health Care', 'Energy']}, "sorted list"),
    ({'cohort_pct': 10.0}, "only for cohort comparisons"),
    ({'rebalance_date': '2010-06'}, "only for ranked lists"),
    ({'delisting_strategy': 'zero_return'}, "unknown fields"),
])
def test_invalid_backtest_jobs_are_refused_with_a_readable_message(change, message):
    job = dict(_job(), **change)
    with pytest.raises(ValueError, match=message):
        validate_wrds_job(job)


@pytest.mark.parametrize('kind, change, message', [
    ('cohort', {'cohort_pct': None}, "needs cohort_pct"),
    ('cohort', {'cohort_pct': 75}, "at most 50"),
    ('ranked', {'rebalance_date': None}, "needs rebalance_date"),
    ('ranked', {'rebalance_date': '2010-6'}, "'YYYY-MM'"),
    ('ranked', {'rebalance_date': '2010-05'}, "not a rebalance month"),     # quarterly: Mar, Jun, Sep, Dec
    ('ranked', {'rebalance_date': '2012-12'}, "not a rebalance month"),     # the end of the last period
])
def test_invalid_cohort_and_ranked_jobs_are_refused(kind, change, message):
    job = dict(_job(kind), **change)
    with pytest.raises(ValueError, match=message):
        validate_wrds_job(job)


def test_a_missing_field_is_named():
    job = _job()
    del job['frequency']
    with pytest.raises(ValueError, match=r"missing \['frequency'\]"):
        validate_wrds_job(job)


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------

def test_wrds_cache_key_changes_with_the_job_the_data_and_the_code():
    job, data, code = _job(), 'wrds_2026_10|panel:a|returns:b|market:c', wrds_code_version()
    key = job_hash(job, data, code)
    assert job_hash(_job(), data, code) == key                                     # equal inputs, equal key
    assert job_hash(_job(frequency='monthly'), data, code) != key
    assert job_hash(job, 'wrds_2026_10|panel:a|returns:b|market:NEW', code) != key  # republished data
    assert job_hash(job, data, 'other-code') != key


def test_wrds_code_version_covers_the_wrds_modules_and_legacy_keeps_its_own():
    assert {'wrds_calendar.py', 'wrds_data.py', 'wrds_engine.py', 'wrds_factors.py', 'wrds_holdings.py',
            'wrds_jobs.py', 'wrds_service.py'} <= set(WRDS_ENGINE_SOURCES)
    assert not any(name.startswith('wrds_') for name in ENGINE_SOURCES)          # WRDS edits keep Legacy's keys
    assert wrds_code_version() != code_version()
    assert len(wrds_code_version()) == 16


def test_legacy_jobs_are_unchanged_and_never_share_a_key_with_wrds_jobs():
    legacy = build_job('backtest', FACTORS, DIRECTIONS, dict(WRDS_SETTINGS), factor_labels=LABELS)
    assert tuple(legacy) == JOB_FIELDS and 'data_mode' not in legacy and 'frequency' not in legacy
    assert job_hash(legacy, 'v', 'c') != job_hash(_job(), 'v', 'c')
