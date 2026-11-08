"""
Read-only checks of an existing pipeline run on the real WRDS data. They never run the pipeline: CI has no data,
so every test here is skipped unless FACTOR_LAKE_DATA_DIR points at the data folder (the one holding raw/ and
processed/). FACTOR_LAKE_PROCESSED_DIR picks another processed folder (relative to the data folder, or absolute;
default processed/), e.g. processed_v2 for data version 2.

    FACTOR_LAKE_DATA_DIR="C:/path/to/WRDS FA26" uv run pytest tests/pipeline/test_real_data.py
    FACTOR_LAKE_DATA_DIR="C:/path/to/WRDS FA26" FACTOR_LAKE_PROCESSED_DIR=processed_v2 uv run pytest ...
"""
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

DATA = Path(os.environ["FACTOR_LAKE_DATA_DIR"]) if os.environ.get("FACTOR_LAKE_DATA_DIR") else None
PROCESSED = (DATA / os.environ.get("FACTOR_LAKE_PROCESSED_DIR", "processed")) if DATA else None
pytestmark = pytest.mark.skipif(
    PROCESSED is None or not (PROCESSED / "final" / "manifest.json").is_file(),
    reason="needs the WRDS data folder with a finished pipeline run: set FACTOR_LAKE_DATA_DIR (not available in CI)")


def summary(folder):
    return json.loads((PROCESSED / folder / "summary.json").read_text(encoding="utf-8"))


def test_export_and_validation_did_not_fail():
    assert summary("s08_export")["status"] != "FAIL"
    if (PROCESSED / "s09_validation" / "summary.json").is_file():
        assert summary("s09_validation")["status"] != "FAIL"


def test_final_files_match_the_manifest():
    final = PROCESSED / "final"
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    for name, o in manifest["outputs"].items():
        files = sorted(final.glob(o["path"]))
        assert len(files) == o["files"] and sum(pq.read_metadata(f).num_rows for f in files) == o["rows"], name


def test_repo_dictionary_holds_no_value_from_the_data():
    import pandas as pd

    from pipeline import column_stats as cs
    stats = pd.read_parquet(PROCESSED / "final" / "column_stats.parquet")
    schema = (Path(__file__).resolve().parents[2] / "pipeline" / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert cs.leaked(schema, cs.distinctive_values(stats)) == []
