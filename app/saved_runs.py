"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/saved_runs.py
PURPOSE: Pure helpers for the list of saved backtest runs shown as Results tabs.
VERSION: 1.0.0
"""

from typing import Any, Dict, List


def build_run_label(run_id: int, factor_labels: List[str]) -> str:
    """
    Builds the tab label for a saved run, e.g. "Run 3: ROA % +1".

    The run id keeps labels unique, which the stateful Results tabs need.
    """
    if not factor_labels:
        return f"Run {run_id}"
    label = f"Run {run_id}: {factor_labels[0]}"
    if len(factor_labels) > 1:
        label += f" +{len(factor_labels) - 1}"
    return label


def add_saved_run(saved_runs: List[Dict[str, Any]],
                  run: Dict[str, Any],
                  max_runs: int) -> List[Dict[str, Any]]:
    """
    Returns a new list with `run` first (newest first), keeping at most `max_runs`.
    The input list is not modified.
    """
    return ([run] + list(saved_runs))[:max_runs]


def remove_saved_run(saved_runs: List[Dict[str, Any]],
                     run_id: int) -> List[Dict[str, Any]]:
    """
    Returns a new list without the run whose id is `run_id` (no-op if absent).
    The input list is not modified.
    """
    return [run for run in saved_runs if run['id'] != run_id]
