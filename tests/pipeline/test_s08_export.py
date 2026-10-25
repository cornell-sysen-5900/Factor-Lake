"""
Tests for pipeline/stages/s08_export.py and pipeline/dictionary.py on small synthetic stage outputs (no real data).
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pipeline import dictionary
from pipeline.checks import FAIL, PASS
from pipeline.config import DataPaths
from pipeline.stages import s08_export as s08


def fake_outputs():
    return {n: {"rows": 10, "size_mb": 1.0, "files": 1, "date_range": ["2020-01-31", "2020-02-28"],
                "columns": {c: "double" for c in dictionary.columns(n)}} for n in dictionary.FILES}


def fake_manifest():
    return {"data_version": "2026-10-25", "pipeline_git_commit": "abc123", "outputs": fake_outputs()}


def test_types_are_standardized():
    df = pd.DataFrame({"permno": np.array([1, 2], dtype="int64"), "permco": [10.0, np.nan],
                       "date": pd.to_datetime(["2020-01-31", "2020-02-28"]).tz_localize("UTC"),
                       "ret": np.array([0.1, -0.2], dtype="float32"), "ticker": ["A", "B"]})
    out = s08.standardize(df)
    assert out["permno"].dtype == "int32" and str(out["permco"].dtype) == "Int32" and out["permco"].isna()[1]
    assert out["date"].dtype == "datetime64[ns]" and out["date"].dt.tz is None
    assert out["ret"].dtype == "float64"
    with pytest.raises(ValueError):
        s08.standardize(pd.DataFrame({"permno": [1.0, np.nan]}))


def test_dictionary_renders_every_column_and_parses_back():
    md = dictionary.render(fake_manifest())
    parsed = dictionary.parse_columns(md)
    assert set(parsed) == set(dictionary.FILES)
    for name in dictionary.FILES:
        assert parsed[name] == list(dictionary.columns(name))
    assert all(len(v) == 4 and all(v) for n in dictionary.FILES for v in dictionary.columns(n).values())


def test_undocumented_and_stale_columns_fail():
    md = dictionary.render(fake_manifest())
    ok = s08.dictionary_checks(md, fake_outputs())
    assert [c.status for c in ok] == [PASS, PASS]
    extra = fake_outputs()
    extra["returns_monthly.parquet"]["columns"]["helper_col"] = "double"
    del extra["security_master.parquet"]["columns"]["ticker"]
    bad = s08.dictionary_checks(md, extra)
    assert [c.status for c in bad] == [FAIL, FAIL]
    assert "helper_col" in bad[0].detail and "ticker" in bad[1].detail


def test_daily_files_keep_the_publish_layout(tmp_path):
    (tmp_path / "raw").mkdir()
    paths = DataPaths.from_data_dir(tmp_path)
    src = paths.stage_dir("s02_prices") / "daily"
    for year in (2019, 2020):
        d = src / f"year={year}"
        d.mkdir(parents=True)
        pd.DataFrame({"date": pd.to_datetime([f"{year}-12-31", f"{year}-12-30", f"{year}-12-31"]),
                      "permno": np.array([2, 1, 1], dtype="int32"), "permco": 5, "ticker": ["B", "A", "A"],
                      "prc": [1.0, 2.0, 3.0], "prc_adj": 1.0, "cfacpr": 1.0, "cfacshr": 1.0, "shrout": 100.0,
                      "mkt_cap": 1.0, "mkt_cap_musd": 0.1, "ret": 0.01, "vol": 5.0, "is_delisting_row": False,
                      "mom_1m": 0.0}).to_parquet(d / "data.parquet")
    out = tmp_path / "final"
    info = s08.export_daily(paths, out)
    f = out / "prices_daily" / "year=2020" / "data.parquet"
    df = pd.read_parquet(f)
    assert list(df.columns) == list(s08.DAILY_COLS) and df["permno"].tolist() == [1, 1, 2]
    assert str(pq.read_schema(f).field("permno").type) == "int32"
    manifest = json.loads((out / "prices_daily" / "_manifest.json").read_text())
    assert manifest["rows_written"] == 6 and set(manifest["years"]) == {"2019", "2020"}
    assert info["source_rows"] == {"2019": 3, "2020": 3}


def test_upload_instructions_target_a_new_folder_and_the_script_compiles():
    md = s08.upload_md("2026-10-25", fake_outputs())
    assert "s3://sysen-5900-factor-lake/factor-lake-data/wrds_2026_10/" in md
    code = md.split("```python\n")[1].split("```")[0]
    compile(code, "upload_wrds.py", "exec")
    assert "print(os.environ" not in code and "AWS_SECRET_ACCESS_KEY\"]" not in code
