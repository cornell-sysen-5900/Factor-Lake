"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s08_export.py
PURPOSE: Stage s08_export: final Parquet files, manifest.json, DATA_DICTIONARY.md, UPLOAD.md.

Not written yet (phase 2). Inputs, outputs, rules and checks are in pipeline/PLAN.md.
"""

from pipeline.config import DataPaths

STAGE = "s08_export"


def run(paths: DataPaths) -> dict:
    raise NotImplementedError(f"{STAGE} is planned in pipeline/PLAN.md but not written yet")
