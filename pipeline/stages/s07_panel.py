"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s07_panel.py
PURPOSE: Stage s07_panel: monthly point-in-time panel and the all-stocks monthly returns table.

Not written yet (phase 2). Inputs, outputs, rules and checks are in pipeline/PLAN.md.
"""

from pipeline.config import DataPaths

STAGE = "s07_panel"


def run(paths: DataPaths) -> dict:
    raise NotImplementedError(f"{STAGE} is planned in pipeline/PLAN.md but not written yet")
