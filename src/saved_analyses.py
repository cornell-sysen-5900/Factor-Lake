"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/saved_analyses.py
PURPOSE: Opens a saved analysis (a backtest the Lambda saved in S3) by its analysis ID, and
         lists everyone's saved analyses (with their scores, for the scoreboard).
VERSION: 1.2.0

The analysis ID is the hash the backtest Lambda saves each result under
(see src/backtest_service.py), so anyone with the ID can open the same results.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .analysis_scores import SCORES_FIELD, decode_scores
from .backtest_jobs import decode_result, decode_saved
from .backtest_service import RESULT_RETENTION_DAYS, RESULTS_DIR, result_file
from .s3_client import S3DataManager

ANALYSIS_ID_PATTERN = re.compile(r'^[0-9a-f]{64}$')
# Saved results that are not analyses (cohort comparisons, WRDS ranked stock lists)
NOT_ANALYSES = ('cohort', 'ranked')
# Listings read the metadata of at most this many saved results (the newest), this many at a time
MAX_LISTED = 1000
HEAD_WORKERS = 16
# Metadata of a result whose metadata was rewritten: when it was first saved (UTC, ISO 8601)
FIRST_SAVED_FIELD = 'first-saved'


class SavedAnalysisError(ValueError):
    """
    An analysis ID that cannot be opened, with a message for the user. `reason` says why,
    for pages that show a shorter message: 'empty', 'not_an_id', 'not_found' (mistyped or
    expired), 'unreadable', 'not_shared' (saved before analyses could be shared) or 'kind'
    (a cohort comparison or ranked list, named by `kind`).
    """

    def __init__(self, message: str, reason: str = 'unreadable', kind: Optional[str] = None):
        super().__init__(message)
        self.reason = reason
        self.kind = kind


def normalize_analysis_id(text: Optional[str]) -> str:
    """The ID without surrounding spaces or quotes, lowercased. Raises SavedAnalysisError if it is not an ID."""
    analysis_id = (text or '').strip().strip('`"\'').strip().lower()
    if not analysis_id:
        raise SavedAnalysisError("Paste an analysis ID first.", reason='empty')
    if not ANALYSIS_ID_PATTERN.match(analysis_id):
        raise SavedAnalysisError(
            f"\"{analysis_id[:20]}{'…' if len(analysis_id) > 20 else ''}\" is not an analysis ID. "
            f"An ID is 64 characters of 0-9 and a-f; use the copy button next to a result's Analysis ID.",
            reason='not_an_id',
        )
    return analysis_id


def list_saved_analyses(manager: Optional[S3DataManager] = None, limit: int = 30) -> Dict[str, Any]:
    """
    Everyone's saved analyses in S3, newest first, without downloading them.

    Returns:
        {'entries': [{'analysis_id', 'saved_at' (UTC datetime), 'summary' (None when the
        analysis was saved before summaries were recorded)}], 'total': number of saved
        results found}. At most `limit` entries; cohort comparisons and WRDS ranked lists are
        left out, and so is every file that is not a saved result (e.g. WRDS holdings Parquet).

    Raises:
        SavedAnalysisError: the bucket cannot be listed with these AWS keys.
    """
    saved = _saved_results(manager or S3DataManager())
    entries = [{'analysis_id': r['analysis_id'], 'saved_at': r['saved_at'], 'summary': r['summary']}
               for r in saved if r['kind'] not in NOT_ANALYSES]
    return {'entries': entries[:limit], 'total': len(saved)}


def list_scored_analyses(manager: Optional[S3DataManager] = None) -> Dict[str, Any]:
    """
    Everyone's saved analyses with their scores, for the scoreboard, without downloading them.

    Returns:
        {'entries': [{'analysis_id', 'saved_at', 'summary', 'scores' (src/analysis_scores.py;
        None when the analysis was saved before scores were recorded)}], newest first,
        cohort comparisons and ranked lists left out; 'total': number of saved results found}.

    Raises:
        SavedAnalysisError: the bucket cannot be listed with these AWS keys.
    """
    saved = _saved_results(manager or S3DataManager())
    return {'entries': [{k: r[k] for k in ('analysis_id', 'saved_at', 'summary', 'scores')}
                        for r in saved if r['kind'] not in NOT_ANALYSES],
            'total': len(saved)}


def _saved_results(manager: S3DataManager) -> List[Dict[str, Any]]:
    """
    Every saved result (up to MAX_LISTED, the newest), newest first: {'analysis_id',
    'saved_at', 'kind', 'summary', 'scores'} from its S3 metadata, read HEAD_WORKERS at a time.
    A result deleted between the listing and the metadata read is left out.
    """
    found = sorted(_list_result_ids(manager), reverse=True)[:MAX_LISTED]
    with ThreadPoolExecutor(max_workers=HEAD_WORKERS) as pool:
        metadata = list(pool.map(lambda item: _metadata(manager, item[1]), found))
    results = []
    for (last_modified, analysis_id), meta in zip(found, metadata):
        if meta is None:
            continue
        results.append({'analysis_id': analysis_id, 'saved_at': _saved_at(last_modified, meta),
                        'last_modified': last_modified, 'kind': meta.get('kind'), 'summary': meta.get('summary'),
                        'scores': decode_scores(meta.get(SCORES_FIELD))})
    results.sort(key=lambda r: (r['saved_at'], r['last_modified']), reverse=True)
    return results


def _list_result_ids(manager: S3DataManager) -> List[Tuple[datetime, str]]:
    """(LastModified, analysis ID) of every saved result in S3. Raises SavedAnalysisError."""
    prefix = manager.object_key(RESULTS_DIR) + '/'
    found = []
    token = None
    try:
        while True:
            page = manager.client.list_objects_v2(Bucket=manager.bucket, Prefix=prefix,
                                                  **({'ContinuationToken': token} if token else {}))
            for obj in page.get('Contents', []):
                name = obj['Key'][len(prefix):]
                if name.endswith('.json') and ANALYSIS_ID_PATTERN.match(name[:-5]):
                    found.append((obj['LastModified'], name[:-5]))
            token = page.get('NextContinuationToken')
            if not page.get('IsTruncated') or not token:
                break
    except Exception as e:
        code = str(getattr(e, 'response', {}).get('Error', {}).get('Code', ''))
        if code in ('AccessDenied', '403'):
            raise SavedAnalysisError("This app's AWS key may not list saved analyses (it needs s3:ListBucket on "
                                     f"{manager.s3_uri(RESULTS_DIR)}/). Open one by its ID instead.") from e
        raise SavedAnalysisError(f"Could not list saved analyses in S3: {e}") from e
    return found


def _metadata(manager: S3DataManager, analysis_id: str) -> Optional[Dict[str, str]]:
    """A saved result's S3 metadata, or None when it no longer exists."""
    try:
        head = manager.client.head_object(Bucket=manager.bucket, Key=manager.object_key(result_file(analysis_id)))
    except Exception as e:
        if str(getattr(e, 'response', {}).get('Error', {}).get('Code', '')) in ('NoSuchKey', '404', 'NotFound'):
            return None
        raise
    return head.get('Metadata') or {}


def _saved_at(last_modified: datetime, metadata: Dict[str, str]) -> datetime:
    """
    When a result was saved: its S3 LastModified, or FIRST_SAVED_FIELD when its metadata was
    rewritten later (scripts/score_saved_analyses.py), which changes LastModified.
    """
    try:
        return datetime.fromisoformat(metadata[FIRST_SAVED_FIELD].replace('Z', '+00:00'))
    except (KeyError, ValueError):
        return last_modified


def load_saved_analysis(text: Optional[str], manager: Optional[S3DataManager] = None) -> Dict[str, Any]:
    """
    Reads a saved analysis from S3.

    Returns:
        {'id', 'job', 'result', 'computed_at'}: the analysis ID, the job it ran
        (factors, directions, settings), its result as run_job returned it, and
        when it was computed (None if unknown).

    Raises:
        SavedAnalysisError: the text is not an ID, no analysis has that ID (mistyped,
        or deleted after RESULT_RETENTION_DAYS days), or it cannot be opened.
    """
    analysis_id = normalize_analysis_id(text)
    manager = manager or S3DataManager()
    try:
        response = manager.client.get_object(Bucket=manager.bucket,
                                             Key=manager.object_key(result_file(analysis_id)))
        body = response['Body'].read()
    except Exception as e:
        code = str(getattr(e, 'response', {}).get('Error', {}).get('Code', ''))
        # Without s3:ListBucket, S3 answers AccessDenied instead of NoSuchKey for a missing object
        if code in ('NoSuchKey', '404', 'NotFound', 'AccessDenied', '403'):
            raise SavedAnalysisError(
                f"No saved analysis has the ID {analysis_id[:8]}…{analysis_id[-4:]}. "
                "Check that the whole ID was copied. "
                f"Saved analyses are deleted after {RESULT_RETENTION_DAYS} days, so an older one has expired; "
                f"run it again to get a new ID.",
                reason='not_found',
            ) from e
        raise SavedAnalysisError(f"Could not read the saved analysis from S3: {e}") from e

    job, result = decode_saved(body)
    if job is None:
        raise SavedAnalysisError(
            "This analysis was saved before analyses could be shared, so its settings were not kept. "
            "Run it again to get an ID you can share.",
            reason='not_shared',
        )
    if job.get('kind') != 'backtest':
        what = {'cohort': 'a cohort comparison', 'ranked': 'a ranked stock list'}.get(job.get('kind'), 'another job')
        raise SavedAnalysisError(
            f"This ID belongs to {what}, not an analysis. Share the ID shown with the analysis instead.",
            reason='kind', kind=job.get('kind'),
        )
    metadata = response.get('Metadata') or {}
    return {
        'id': analysis_id,
        'job': job,
        'result': decode_result('backtest', result),
        'computed_at': metadata.get('computed_at'),
    }
