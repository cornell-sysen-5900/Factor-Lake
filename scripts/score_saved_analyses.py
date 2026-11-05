"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/score_saved_analyses.py
PURPOSE: Adds scores to analyses saved before the scoreboard, so it can rank them too.
VERSION: 1.0.0

Usage (from the project root, with the AWS keys in .env):
    uv run python scripts/score_saved_analyses.py            # dry run: what it would change
    uv run python scripts/score_saved_analyses.py --apply    # rewrite their metadata

The backtest Lambda saves each analysis's scores (src/analysis_scores.py) as S3 metadata.
Analyses saved before that have none, and the scoreboard lists them as "Not scored". For
each of them this script reads the saved analysis, computes its scores from the saved result
(nothing is rerun) and rewrites the object's metadata in place (an S3 copy onto itself): the
analysis, its ID and its result are unchanged. Results saved before summaries get their kind
and summary too.

Rewriting metadata changes an object's LastModified, so the time it was first saved is kept
in its metadata (src/saved_analyses.FIRST_SAVED_FIELD) and listings keep showing that.
Analyses saved before sharing have no job, so they cannot be scored and are left as they are.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_scores import SCORES_FIELD, decode_scores, score_metadata  # noqa: E402
from src.backtest_jobs import decode_result, decode_saved  # noqa: E402
from src.backtest_service import result_file, summarize_job  # noqa: E402
from src.s3_client import S3DataManager  # noqa: E402
from src.saved_analyses import FIRST_SAVED_FIELD, NOT_ANALYSES, _list_result_ids, _metadata  # noqa: E402


def _summary(job: Dict) -> str:
    if job.get('data_mode') == 'wrds':
        from src.wrds_service import summarize_wrds_job
        return summarize_wrds_job(job)
    return summarize_job(job)


def score_saved_analyses(manager: S3DataManager, apply: bool = False, log=print) -> Counter:
    """
    Scores every saved analysis that has no scores yet (see the module docstring). Returns
    how many were 'scored', 'labeled' (a cohort or ranked list given its kind and summary),
    'already scored', 'not analyses' (cohorts and ranked lists that have their kind) and
    'no job' (saved before sharing). Without apply, nothing is written.
    """
    counts: Counter = Counter()
    for last_modified, analysis_id in sorted(_list_result_ids(manager), reverse=True):
        metadata = _metadata(manager, analysis_id)
        if metadata is None:
            continue
        if decode_scores(metadata.get(SCORES_FIELD)) is not None:
            counts['already scored'] += 1
            continue
        if metadata.get('kind') in NOT_ANALYSES:
            counts['not analyses'] += 1
            continue
        key = manager.object_key(result_file(analysis_id))
        job, result = decode_saved(manager.client.get_object(Bucket=manager.bucket, Key=key)['Body'].read())
        if job is None:
            counts['no job'] += 1
            continue
        updated = {**metadata, 'kind': job['kind'], 'summary': metadata.get('summary') or _summary(job)}
        updated.setdefault(FIRST_SAVED_FIELD, last_modified.isoformat())
        if job['kind'] == 'backtest':
            updated.update(score_metadata(job, decode_result('backtest', result)))
            counts['scored'] += 1
        else:
            counts['labeled'] += 1
        log(f"{'Updating' if apply else 'Would update'} {analysis_id[:12]} ({job['kind']}): {updated['summary']}")
        if apply:
            manager.client.copy_object(Bucket=manager.bucket, Key=key, CopySource={'Bucket': manager.bucket, 'Key': key},
                                       Metadata=updated, MetadataDirective='REPLACE', ContentType='application/json')
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Add scores to analyses saved before the scoreboard.")
    parser.add_argument('--apply', action='store_true', help="rewrite the metadata (default: dry run)")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
    counts = score_saved_analyses(S3DataManager(), apply=args.apply)
    print(', '.join(f"{name}: {counts[name]}" for name in
                    ('scored', 'labeled', 'already scored', 'not analyses', 'no job')))
    if not args.apply and (counts['scored'] or counts['labeled']):
        print("Dry run: nothing was written. Run again with --apply to update them.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
