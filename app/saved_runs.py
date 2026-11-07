"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/saved_runs.py
PURPOSE: Pure helpers for the list of saved backtest runs shown as Results tabs.
VERSION: 1.0.0
"""

from typing import Any, Dict, List


def build_run_label(run_id: int, factor_labels: List[str], trading_cost_bps: float = 0.0) -> str:
    """
    Builds the tab label for a saved run, e.g. "Run 3: ROA % +1", or
    "Run 3: ROA % +1 | 25 bps" with a trading cost.

    The run id keeps labels unique, which the stateful Results tabs need.
    """
    if not factor_labels:
        label = f"Run {run_id}"
    else:
        label = f"Run {run_id}: {factor_labels[0]}"
        if len(factor_labels) > 1:
            label += f" +{len(factor_labels) - 1}"
    return label + trading_cost_suffix(trading_cost_bps)


def named_run_label(run_id: int, name: str, default_label: str) -> str:
    """
    A named run's tab label: the name, then the default label's details in brackets, e.g.
    "Run 3: Value test (WRDS, Quarterly, 25 bps)" for "Run 3: WRDS, Quarterly | 25 bps",
    or "Run 1: Value test (ROE +1, opened)" for "Run 1: ROE +1 (opened)".
    """
    prefix = f"Run {run_id}"
    details = default_label[len(prefix):] if default_label.startswith(prefix) else default_label
    details = details.removeprefix(': ').replace(' | ', ', ').removeprefix(', ')
    if details.endswith(' (opened)'):
        details = details[:-len(' (opened)')] + ', opened'
    return f"{prefix}: {name} ({details})" if details else f"{prefix}: {name}"


def trading_cost_suffix(trading_cost_bps: float) -> str:
    """A run label's trading cost, e.g. ' | 25 bps', or an empty string at 0 bps."""
    return f" | {trading_cost_bps:g} bps" if trading_cost_bps and trading_cost_bps > 0 else ''


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
