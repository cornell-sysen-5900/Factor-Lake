"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s03_fundamentals_q.py
PURPOSE: Stage s03_fundamentals_q: quarterly Compustat factors, TTM sums and DCF/RV building blocks.

Not written yet (phase 2). Inputs, outputs, rules and checks are in pipeline/PLAN.md.
"""

from pipeline.config import DataPaths

STAGE = "s03_fundamentals_q"


def run(paths: DataPaths) -> dict:
    raise NotImplementedError(f"{STAGE} is planned in pipeline/PLAN.md but not written yet")
