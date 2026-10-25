"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/outputs.py
PURPOSE: Writing a stage's outputs: a fresh processed/<stage>/ folder, Parquet files and the JSON summary.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.checks import Check, overall_status
from pipeline.config import PARQUET_COMPRESSION, DataPaths


def fresh_stage_dir(paths: DataPaths, stage: str) -> Path:
    """Empty processed/<stage>/ so a rerun overwrites only this stage's outputs."""
    out = paths.stage_dir(stage).resolve()
    if out.parent != paths.processed.resolve():
        raise ValueError(f"{out} is not a stage folder inside {paths.processed}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame as zstd Parquet and check the row count reads back."""
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression=PARQUET_COMPRESSION)
    if pq.read_metadata(path).num_rows != len(df):
        raise IOError(f"{path} did not read back with {len(df):,} rows")
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_summary(out_dir: Path, stage: str, tables: dict[str, dict], checks: Iterable[Check]) -> dict:
    """summary.json every stage writes: per output table rows, date range and permnos, plus the checks."""
    checks = list(checks)
    summary = {"stage": stage, "created_at": utc_now(), "status": overall_status(checks), "tables": tables,
               "checks": [c.to_dict() for c in checks]}
    write_json(summary, out_dir / "summary.json")
    return summary
