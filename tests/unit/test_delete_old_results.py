"""
Tests for deleting saved results older than 1 year (story: "Invalidate cache results on data
changes": saved results are deleted after 1 year by a weekly GitHub Action).

- scripts/delete_old_results.py finds saved results older than RESULT_RETENTION_DAYS (from when
  they were first saved) with their holdings files, only under backtest-results/; a dry run
  lists them, --apply deletes exactly those, and the job summary gives their count and size.
- The deploy script no longer needs an admin's lifecycle rule; it warns only when a rule would
  delete results sooner than 1 year.
- The weekly workflow deletes; a run by hand is a dry run unless "Delete them" is ticked.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from botocore.exceptions import ClientError

import scripts.delete_old_results as cleanup
from scripts.deploy_backtest_lambda import check_result_expiry, set_result_expiry
from src.backtest_service import RESULT_RETENTION_DAYS, RESULT_RETENTION_TEXT, result_file
from src.saved_analyses import FIRST_SAVED_FIELD
from src.wrds_holdings import holdings_file
from tests.unit.test_backtest_lambda import BUCKET, PREFIX, manager, s3  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2027, 11, 15, 6, 17, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=RESULT_RETENTION_DAYS + 1)
RECENT = NOW - timedelta(days=RESULT_RETENTION_DAYS - 1)
FIND = cleanup.find_old_results   # the real one, before tests replace it to fix the date


def _put(s3, filename, when, body=b'{"format": 2}', metadata=None):
    key = f'{PREFIX}/{filename}'
    s3.put_object(BUCKET, key, body, Metadata=metadata or {})
    s3.modified[key] = when
    return key


@pytest.fixture
def bucket(s3):
    """Saved results, holdings and other files of every age the cleanup must tell apart."""
    keys = {
        'old': _put(s3, result_file('a' * 64), OLD, b'x' * 3000),
        'old_holdings': _put(s3, holdings_file('backtest-results', 'a' * 64), RECENT, b'h' * 7000),
        'old_cohort': _put(s3, result_file('b' * 64), OLD, metadata={'kind': 'cohort'}),
        'recent': _put(s3, result_file('c' * 64), RECENT),
        'recent_holdings': _put(s3, holdings_file('backtest-results', 'c' * 64), OLD),       # its result is recent
        'orphan_old': _put(s3, holdings_file('backtest-results', 'd' * 64), OLD, b'o' * 1000),
        'orphan_recent': _put(s3, holdings_file('backtest-results', 'e' * 64), RECENT),
        # Rewritten recently (scores backfill) but first saved more than a year ago
        'rewritten': _put(s3, result_file('f' * 64), RECENT,
                          metadata={FIRST_SAVED_FIELD: OLD.isoformat(), 'kind': 'backtest'}),
        'other_file': _put(s3, 'backtest-results/notes.txt', OLD),
        'market_data': _put(s3, 'factor_universe.parquet', OLD),
        'deploy_package': _put(s3, '_deploy/backtest_lambda-0123.zip', OLD),
    }
    return keys


def test_finds_old_results_with_their_holdings_and_nothing_else(manager, bucket):
    found = cleanup.find_old_results(manager, now=NOW)
    assert {f['key'] for f in found} == {bucket[k] for k in ('old', 'old_holdings', 'old_cohort', 'orphan_old',
                                                              'rewritten')}
    assert all(f['key'].startswith(f'{PREFIX}/backtest-results/') for f in found)
    by_key = {f['key']: f for f in found}
    assert by_key[bucket['old_holdings']]['kind'] == 'holdings' and by_key[bucket['old']]['kind'] == 'result'
    assert by_key[bucket['old_holdings']]['saved_at'] == OLD             # dated by its result
    assert by_key[bucket['rewritten']]['saved_at'] == OLD                # its first save, not the rewrite


def test_a_result_exactly_one_year_old_is_kept(manager, s3):
    _put(s3, result_file('a' * 64), NOW - timedelta(days=RESULT_RETENTION_DAYS))
    assert cleanup.find_old_results(manager, now=NOW) == []


def test_dry_run_lists_them_and_deletes_nothing(manager, s3, bucket, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cleanup, 'S3DataManager', lambda: manager)
    monkeypatch.setattr(cleanup, 'find_old_results', lambda m: _found(m))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(tmp_path / 'summary.md'))
    before = set(s3.objects)
    assert cleanup.main([]) == 0
    assert set(s3.objects) == before
    out = capsys.readouterr().out
    assert 'Dry run: would delete **5 file(s), 0.01 MB**' in out
    assert '3 saved result(s) and 2 holdings file(s)' in out
    assert 'Nothing was deleted. Run with `--apply`' in out
    summary = (tmp_path / 'summary.md').read_text()
    assert summary.startswith(f'### Saved results older than {RESULT_RETENTION_DAYS} days')


def _found(manager):
    return FIND(manager, now=NOW)


def test_apply_deletes_exactly_the_files_listed(manager, s3, bucket, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cleanup, 'S3DataManager', lambda: manager)
    monkeypatch.setattr(cleanup, 'find_old_results', lambda m: _found(m))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(tmp_path / 'summary.md'))
    listed = {f['key'] for f in _found(manager)}
    before = set(s3.objects)
    assert cleanup.main(['--apply']) == 0
    assert set(s3.objects) == before - listed                           # those, and only those
    for kept in ('recent', 'recent_holdings', 'orphan_recent', 'other_file', 'market_data', 'deploy_package'):
        assert bucket[kept] in s3.objects
    summary = (tmp_path / 'summary.md').read_text()
    # 3000 + 7000 + 1000 bytes and two small results: the count and the total size
    assert 'Deleted **5 file(s), 0.01 MB**' in summary and '| `' + 'a' * 64 + '.json` |' in summary
    assert 'Nothing was deleted' not in summary
    assert _found(manager) == []                                        # a second run finds nothing


def test_deletes_in_batches_and_reports_failures(manager, s3, monkeypatch):
    for i in range(5):
        _put(s3, result_file(f'{i:x}' * 64), OLD)
    monkeypatch.setattr(cleanup, 'DELETE_BATCH', 2)
    keys = [f['key'] for f in cleanup.find_old_results(manager, now=NOW)]
    cleanup.delete_files(manager, keys)
    assert s3.delete_requests == 3 and not any(k in s3.objects for k in keys)

    key = _put(s3, result_file('9' * 64), OLD)
    s3.fail_deletes = {key}
    with pytest.raises(RuntimeError, match='could not delete 1 file'):
        cleanup.delete_files(manager, [key])


def test_nothing_to_delete_says_so(manager, s3):
    _put(s3, result_file('c' * 64), RECENT)
    text = cleanup.summary(cleanup.find_old_results(manager, now=NOW), applied=True, manager=manager)
    assert 'Deleted **0 file(s), 0.00 MB**' in text and '|' not in text


# ---------------------------------------------------------------------------
# Retention is 1 year everywhere: the setting, the deploy check, the workflow
# ---------------------------------------------------------------------------

def test_retention_is_one_year():
    assert (RESULT_RETENTION_DAYS, RESULT_RETENTION_TEXT) == (365, '1 year')


class LifecycleS3:
    """The bucket's lifecycle rules as the deploy script reads and writes them."""

    def __init__(self, rules=None, error=None):
        self.rules, self.error, self.written = rules, error, None

    def get_bucket_lifecycle_configuration(self, Bucket):
        if self.error:
            raise ClientError({'Error': {'Code': self.error}}, 'GetBucketLifecycleConfiguration')
        if self.rules is None:
            raise ClientError({'Error': {'Code': 'NoSuchLifecycleConfiguration'}}, 'GetBucketLifecycleConfiguration')
        return {'Rules': self.rules}

    def put_bucket_lifecycle_configuration(self, Bucket, LifecycleConfiguration):
        if self.error:
            raise ClientError({'Error': {'Code': self.error}}, 'PutBucketLifecycleConfiguration')
        self.written = LifecycleConfiguration['Rules']


def _rule(days, prefix=f'{PREFIX}/backtest-results/', status='Enabled'):
    return {'ID': 'r', 'Filter': {'Prefix': prefix}, 'Status': status, 'Expiration': {'Days': days}}


@pytest.mark.parametrize('s3_rules, ok, message', [
    (LifecycleS3(error='AccessDenied'), True, "weekly 'Delete old results' GitHub Action"),   # the class keys
    (LifecycleS3(rules=None), True, "weekly 'Delete old results' GitHub Action"),
    (LifecycleS3(rules=[_rule(365)]), True, "weekly 'Delete old results' GitHub Action"),
    (LifecycleS3(rules=[_rule(90)]), False, 'after 90 days, before the 365 days'),
    (LifecycleS3(rules=[_rule(90, status='Disabled')]), True, "weekly"),
    (LifecycleS3(rules=[_rule(30, prefix='other/')]), True, "weekly"),
])
def test_the_deploy_check_needs_no_admin_and_warns_only_about_a_shorter_rule(s3_rules, ok, message, caplog):
    caplog.set_level('INFO')
    assert check_result_expiry(s3_rules, BUCKET, PREFIX) is ok
    assert message in caplog.text and 'admin to add' not in caplog.text


def test_a_deploy_without_lifecycle_access_still_counts_as_expiring(caplog):
    caplog.set_level('INFO')
    assert set_result_expiry(LifecycleS3(error='AccessDenied'), BUCKET, PREFIX, RESULT_RETENTION_DAYS) is True
    assert "weekly 'Delete old results' GitHub Action" in caplog.text
    allowed = LifecycleS3(rules=[])
    assert set_result_expiry(allowed, BUCKET, PREFIX, RESULT_RETENTION_DAYS) is True
    assert allowed.written[-1]['Expiration'] == {'Days': 365}


def test_the_workflow_deletes_weekly_and_is_a_dry_run_by_hand():
    workflow = yaml.safe_load((ROOT / '.github/workflows/delete-old-results.yml').read_text())
    triggers = workflow[True]                                           # YAML 1.1 reads the key `on` as True
    assert len(triggers['schedule']) == 1 and triggers['schedule'][0]['cron'].endswith('* * 1')   # Mondays
    assert triggers['workflow_dispatch']['inputs']['apply']['default'] is False
    step = workflow['jobs']['delete']['steps'][-1]
    assert step['env']['APPLY'] == "${{ github.event_name == 'schedule' || inputs.apply }}"
    assert 'scripts/delete_old_results.py' in step['run'] and '--apply' in step['run']
    assert set(step['env']) >= {'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'}
