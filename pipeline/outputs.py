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


def _num(v: Any) -> Any:
    return None if pd.isna(v) else round(float(v), 6)


def table_info(df: pd.DataFrame, date_col: str | None = None, stats: Iterable[str] = ()) -> dict:
    """Rows, date range, permnos, and count / missing % / mean / p1 / p50 / p99 of the given numeric columns."""
    info: dict[str, Any] = {"rows": len(df)}
    if date_col:
        info["date_range"] = [str(df[date_col].min())[:10], str(df[date_col].max())[:10]] if len(df) else None
    if "permno" in df.columns:
        info["permnos"] = int(df["permno"].nunique())
    info["stats"] = {}
    for c in stats:
        x = pd.to_numeric(df[c], errors="coerce").astype("float64")
        q = x.quantile([0.01, 0.5, 0.99]) if x.notna().any() else pd.Series([None] * 3, index=[0.01, 0.5, 0.99])
        info["stats"][c] = {"count": int(x.notna().sum()),
                            "missing_pct": round(100 * float(x.isna().mean()), 3) if len(x) else None,
                            "mean": _num(x.mean()), "p1": _num(q[0.01]), "p50": _num(q[0.5]), "p99": _num(q[0.99])}
    return info


def write_summary(out_dir: Path, stage: str, tables: dict[str, dict], checks: Iterable[Check],
                  extra: dict | None = None) -> dict:
    """summary.json every stage writes: per output table rows, date range, permnos and column stats, the checks,
    and any stage-specific lists in `extra`."""
    checks = list(checks)
    summary = {"stage": stage, "created_at": utc_now(), "status": overall_status(checks), "tables": tables,
               "checks": [c.to_dict() for c in checks], **(extra or {})}
    write_json(summary, out_dir / "summary.json")
    return summary
