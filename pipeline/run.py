"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/run.py
PURPOSE: Command line entry point for the WRDS data pipeline.

Usage (from the repo root):
    uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" --stage s00_inventory
    uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" --stage all

--stage takes a full stage name (s02_prices), its number (s02), its short name (prices) or all.
Exit code: 0 all checks PASS or WARN, 1 a check FAILed, 2 a stage is not written yet.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from pipeline.config import DataPaths
from pipeline.stages import STAGES


def resolve_stages(name: str) -> list[str]:
    if name == "all":
        return list(STAGES)
    for stage in STAGES:
        number, short = stage.split("_", 1)
        if name in (stage, number, short):
            return [stage]
    raise SystemExit(f"unknown stage {name!r}; choose one of: all, {', '.join(STAGES)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.run", description="Factor-Lake WRDS data pipeline")
    parser.add_argument("--data-dir", required=True, help="data folder that contains raw/ (processed/ is written)")
    parser.add_argument("--stage", required=True, help="stage name, number, short name, or all")
    args = parser.parse_args(argv)

    paths = DataPaths.from_data_dir(args.data_dir)
    for stage in resolve_stages(args.stage):
        print(f"== {stage}", flush=True)
        started = time.time()
        try:
            summary = importlib.import_module(f"pipeline.stages.{stage}").run(paths)
        except NotImplementedError as exc:
            print(f"{stage}: {exc}")
            return 2
        print(f"{stage}: {summary['status']} in {time.time() - started:.0f}s -> {paths.stage_dir(stage)}")
        if summary["status"] == "FAIL":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
