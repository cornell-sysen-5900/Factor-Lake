"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s04_fundamentals_a.py
PURPOSE: Stage s04_fundamentals_a: annual Compustat statements for DCF, fiscal-year based.

Not written yet (phase 2). Inputs, outputs, rules and checks are in pipeline/PLAN.md.
"""

from pipeline.config import DataPaths

STAGE = "s04_fundamentals_a"


def run(paths: DataPaths) -> dict:
    raise NotImplementedError(f"{STAGE} is planned in pipeline/PLAN.md but not written yet")
