"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/saved_analyses.py
PURPOSE: Opens a saved analysis (a backtest the Lambda saved in S3) by its analysis ID.
VERSION: 1.0.0

The analysis ID is the hash the backtest Lambda saves each result under
(see src/backtest_service.py), so anyone with the ID can open the same results.
"""

import re
from typing import Any, Dict, Optional

from .backtest_jobs import decode_result, decode_saved
from .backtest_service import RESULT_RETENTION_DAYS, result_file
from .s3_client import S3DataManager

ANALYSIS_ID_PATTERN = re.compile(r'^[0-9a-f]{64}$')


class SavedAnalysisError(ValueError):
    """An analysis ID that cannot be opened, with a message for the user."""


def normalize_analysis_id(text: Optional[str]) -> str:
    """The ID without surrounding spaces or quotes, lowercased. Raises SavedAnalysisError if it is not an ID."""
    analysis_id = (text or '').strip().strip('`"\'').strip().lower()
    if not analysis_id:
        raise SavedAnalysisError("Paste an analysis ID first.")
    if not ANALYSIS_ID_PATTERN.match(analysis_id):
        raise SavedAnalysisError(
            f"\"{analysis_id[:20]}{'…' if len(analysis_id) > 20 else ''}\" is not an analysis ID. "
            f"An ID is 64 characters of 0-9 and a-f; use the copy button next to a result's Analysis ID."
        )
    return analysis_id


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
                f"No saved analysis has the ID {analysis_id[:12]}…. Check that the whole ID was copied. "
                f"Saved analyses are deleted after {RESULT_RETENTION_DAYS} days, so an older one has expired; "
                f"run it again to get a new ID."
            ) from e
        raise SavedAnalysisError(f"Could not read the saved analysis from S3: {e}") from e

    job, result = decode_saved(body)
    if job is None:
        raise SavedAnalysisError(
            "This analysis was saved before analyses could be shared, so its settings were not kept. "
            "Run it again to get an ID you can share."
        )
    if job.get('kind') != 'backtest':
        raise SavedAnalysisError(
            "This ID belongs to a cohort comparison, not an analysis. Share the ID shown with the analysis instead."
        )
    metadata = response.get('Metadata') or {}
    return {
        'id': analysis_id,
        'job': job,
        'result': decode_result('backtest', result),
        'computed_at': metadata.get('computed_at'),
    }
