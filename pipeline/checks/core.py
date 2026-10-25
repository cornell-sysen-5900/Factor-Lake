"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/checks/core.py
PURPOSE: Check results (PASS / WARN / FAIL) and the checks shared by stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def overall_status(checks: Iterable[Check]) -> str:
    """Worst status of the checks (PASS when there are none)."""
    return max((c.status for c in checks), key=_RANK.__getitem__, default=PASS)


def _sample(values: Iterable, n: int = 10) -> str:
    values = sorted(values, key=str)
    return ", ".join(map(str, values[:n])) + (f" (+{len(values) - n} more)" if len(values) > n else "")


def columns_present(name: str, columns: Iterable[str], required: Iterable[str]) -> Check:
    """FAIL if a required column is missing."""
    missing = set(required) - set(columns)
    if missing:
        return Check(f"{name}: required columns", FAIL, f"missing {_sample(missing)}")
    return Check(f"{name}: required columns", PASS)


def not_empty(name: str, rows: int) -> Check:
    return Check(f"{name}: has rows", PASS if rows > 0 else FAIL, f"{rows:,} rows")


def values_present(name: str, column: str, observed: Iterable[str], expected: Iterable[str]) -> list[Check]:
    """FAIL if an expected value never appears in the column; WARN for values that were not expected."""
    observed, expected = set(observed), set(expected)
    missing, extra = expected - observed, observed - expected
    checks = [Check(f"{name}: {column} values", FAIL if missing else PASS,
                    f"missing {_sample(missing)}" if missing else f"all {len(expected)} present")]
    if extra:
        checks.append(Check(f"{name}: unexpected {column} values", WARN, _sample(extra)))
    return checks
