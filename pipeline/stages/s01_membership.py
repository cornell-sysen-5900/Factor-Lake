"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s01_membership.py
PURPOSE: Stage s01_membership: IWM snapshots, membership periods, permno fill from CUSIP, known gaps.

Not written yet (phase 2). Inputs, outputs, rules and checks are in pipeline/PLAN.md.
"""

from pipeline.config import DataPaths

STAGE = "s01_membership"


def run(paths: DataPaths) -> dict:
    raise NotImplementedError(f"{STAGE} is planned in pipeline/PLAN.md but not written yet")
