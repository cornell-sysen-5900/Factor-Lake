"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/delete_old_results.py
PURPOSE: Deletes saved backtest results older than 1 year (RESULT_RETENTION_DAYS), with their holdings files.
VERSION: 1.0.0

Usage (from the project root, with the AWS keys in .env or the environment):
    uv run python scripts/delete_old_results.py            # dry run: lists what it would delete
    uv run python scripts/delete_old_results.py --apply    # deletes them

The weekly "Delete old results" GitHub Action (.github/workflows/delete-old-results.yml)
runs it with --apply, so saved results expire without an S3 lifecycle rule (which only an
AWS admin could add).

Only <prefix>/backtest-results/ is read or deleted; the market data is never touched. There:
- A saved result (<ID>.json: an analysis, cohort comparison or ranked list) is deleted once
  it is older than RESULT_RETENTION_DAYS, counted from when it was first saved: its metadata's
  first-saved time when its metadata was rewritten (scripts/score_saved_analyses.py), else
  its LastModified.
- A WRDS holdings file (<ID>.holdings.parquet) is deleted with its result, or on its own
  when it has no result and is older than RESULT_RETENTION_DAYS.
- Any other file there is left alone.

It prints how many files it deleted (or would delete) and their total size, and adds the
same summary to the GitHub job summary when it runs in GitHub Actions.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_service import RESULT_RETENTION_DAYS, RESULTS_DIR  # noqa: E402
from src.s3_client import S3DataManager  # noqa: E402
from src.saved_analyses import HEAD_WORKERS, _metadata, _saved_at  # noqa: E402
from src.wrds_holdings import HOLDINGS_SUFFIX  # noqa: E402

RESULT_NAME = re.compile(r'^(?P<id>[0-9a-f]{64})\.json$')
HOLDINGS_NAME = re.compile(r'^(?P<id>[0-9a-f]{64})' + re.escape(HOLDINGS_SUFFIX) + '$')
# S3 deletes at most this many objects per request
DELETE_BATCH = 1000


def find_old_results(manager: S3DataManager, now: Optional[datetime] = None,
                     days: int = RESULT_RETENTION_DAYS) -> List[Dict[str, Any]]:
    """
    The files to delete (see the module docstring), oldest first: {'key', 'size', 'saved_at',
    'kind' ('result' or 'holdings')}.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    prefix = manager.object_key(RESULTS_DIR) + '/'
    results, holdings = {}, {}
    for obj in _list_objects(manager, prefix):
        name = obj['Key'][len(prefix):]
        if (match := RESULT_NAME.match(name)):
            results[match['id']] = obj
        elif (match := HOLDINGS_NAME.match(name)):
            holdings[match['id']] = obj

    # A result rewritten recently may have been first saved long ago: read the metadata of those
    recent = [i for i, obj in results.items() if obj['LastModified'] >= cutoff]
    with ThreadPoolExecutor(max_workers=HEAD_WORKERS) as pool:
        metadata = dict(zip(recent, pool.map(lambda i: _metadata(manager, i), recent)))
    saved_at = {i: _saved_at(obj['LastModified'], metadata.get(i) or {}) for i, obj in results.items()}

    old = [{'key': results[i]['Key'], 'size': results[i]['Size'], 'saved_at': saved_at[i], 'kind': 'result'}
           for i in results if saved_at[i] < cutoff]
    old_ids = {i for i in results if saved_at[i] < cutoff}
    for i, obj in holdings.items():
        if i in old_ids:
            old.append({'key': obj['Key'], 'size': obj['Size'], 'saved_at': saved_at[i], 'kind': 'holdings'})
        elif i not in results and obj['LastModified'] < cutoff:   # its result is already gone
            old.append({'key': obj['Key'], 'size': obj['Size'], 'saved_at': obj['LastModified'], 'kind': 'holdings'})
    for file in old:   # never anything outside the saved results
        assert file['key'].startswith(prefix), file['key']
    return sorted(old, key=lambda f: (f['saved_at'], f['key']))


def _list_objects(manager: S3DataManager, prefix: str) -> List[Dict[str, Any]]:
    objects, token = [], None
    while True:
        page = manager.client.list_objects_v2(Bucket=manager.bucket, Prefix=prefix,
                                              **({'ContinuationToken': token} if token else {}))
        objects += page.get('Contents', [])
        token = page.get('NextContinuationToken')
        if not page.get('IsTruncated') or not token:
            return objects


def delete_files(manager: S3DataManager, keys: List[str]) -> None:
    """Deletes these keys, DELETE_BATCH per request. Raises RuntimeError when S3 could not delete one."""
    for start in range(0, len(keys), DELETE_BATCH):
        batch = keys[start:start + DELETE_BATCH]
        response = manager.client.delete_objects(
            Bucket=manager.bucket, Delete={'Objects': [{'Key': k} for k in batch], 'Quiet': True})
        if response.get('Errors'):
            failed = ', '.join(f"{e['Key']} ({e.get('Code')})" for e in response['Errors'][:5])
            raise RuntimeError(f"S3 could not delete {len(response['Errors'])} file(s): {failed}")


def summary(files: List[Dict[str, Any]], applied: bool, manager: S3DataManager,
            days: int = RESULT_RETENTION_DAYS) -> str:
    """A Markdown summary of the run: how many files, of which kind, and their total size."""
    results = sum(f['kind'] == 'result' for f in files)
    size_mb = sum(f['size'] for f in files) / 1e6
    where = f"s3://{manager.bucket}/{manager.object_key(RESULTS_DIR)}/"
    verb = "Deleted" if applied else "Dry run: would delete"
    lines = [f"### Saved results older than {days} days", "",
             f"{verb} **{len(files)} file(s), {size_mb:,.2f} MB** from `{where}`: "
             f"{results} saved result(s) and {len(files) - results} holdings file(s)."]
    if files:
        lines += ["", "| Saved (UTC) | File | Size |", "|---|---|---|"]
        lines += [f"| {f['saved_at']:%Y-%m-%d} | `{f['key'].rsplit('/', 1)[-1]}` | {f['size'] / 1e3:,.1f} kB |"
                  for f in files[:50]]
        if len(files) > 50:
            lines.append(f"| | ... and {len(files) - 50} more | |")
    if not applied and files:
        lines += ["", "Nothing was deleted. Run with `--apply` to delete them."]
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Delete saved backtest results older than "
                                                 f"{RESULT_RETENTION_DAYS} days.")
    parser.add_argument('--apply', action='store_true', help="delete them (default: dry run, only list them)")
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / '.env')
    manager = S3DataManager()
    files = find_old_results(manager)
    if args.apply and files:
        delete_files(manager, [f['key'] for f in files])
    text = summary(files, args.apply, manager)
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
