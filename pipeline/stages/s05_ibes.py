"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s05_ibes.py
PURPOSE: Stage s05_ibes: IBES consensus by measure and FPI, linked to permno, raw analyst factors.

Not written yet (phase 2). Inputs, outputs, rules and checks are in pipeline/PLAN.md.
"""

from pipeline.config import DataPaths

STAGE = "s05_ibes"


def run(paths: DataPaths) -> dict:
    raise NotImplementedError(f"{STAGE} is planned in pipeline/PLAN.md but not written yet")
