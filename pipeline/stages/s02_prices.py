"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s02_prices.py
PURPOSE: Stage s02_prices: daily CRSP CIZ cleaning, momentum and volatility, monthly returns with delisting rows.

Not written yet (phase 2). Inputs, outputs, rules and checks are in pipeline/PLAN.md.
"""

from pipeline.config import DataPaths

STAGE = "s02_prices"


def run(paths: DataPaths) -> dict:
    raise NotImplementedError(f"{STAGE} is planned in pipeline/PLAN.md but not written yet")
