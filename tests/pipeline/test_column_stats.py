"""
Tests for pipeline/column_stats.py and the column stats in s08_export (synthetic data only).
"""
import json

import numpy as np
import pandas as pd
import pytest

from pipeline import column_stats as cs
from pipeline import dictionary
from pipeline.stages import s08_export as s08


def write_years(tmp_path):
    files = []
    for year, vals in ((2019, [1.0, 2.0, np.nan]), (2020, [3.0, 4.0, 5.0])):
        f = tmp_path / f"year={year}" / "data.parquet"
        f.parent.mkdir()
        pd.DataFrame({"date": pd.to_datetime([f"{year}-01-02", f"{year}-01-03", f"{year}-01-03"]),
                      "ticker": ["A", "A", "B"] if year == 2019 else ["A", None, "C"], "prc": vals,
                      "is_delisting_row": [False, False, True]}).to_parquet(f)
        files.append(f)
    return files


def test_stats_combine_files_and_cover_every_kind(tmp_path):
    s = cs.stats_frame(cs.file_stats("prices_daily/year=YYYY/data.parquet", write_years(tmp_path))).set_index("column")
    assert s.loc["prc", "rows"] == 6 and s.loc["prc", "non_null_pct"] == pytest.approx(500 / 6)
    assert s.loc["prc", "min"] == 1.0 and s.loc["prc", "max"] == 5.0 and s.loc["prc", "median"] == 3.0
    assert s.loc["date", "unique"] == 4 and s.loc["date", "date_max"] == pd.Timestamp("2020-01-03")
    assert s.loc["ticker", "unique"] == 3 and json.loads(s.loc["ticker", "top_values"])[0] == ["A", 50.0]
    assert s.loc["is_delisting_row", "true_pct"] == pytest.approx(100 / 3)
    assert set(s["kind"]) == {"numeric", "date", "text", "bool"}


def test_four_significant_figures_and_factor_percentiles():
    assert cs.fmt(123456.789) == "1.235e+05" and cs.fmt(0.0123456) == "0.01235" and cs.fmt(np.nan) == ""
    row = pd.Series({"column": "f_roa", "kind": "numeric", "non_null_pct": 95.0, "min": -1.0, "p1": -0.5,
                     "median": 0.01, "mean": 0.02, "p99": 0.3, "max": 2.0})
    assert cs.stats_cells(row)[3] == "-0.5"
    assert cs.stats_cells(row.copy().replace({"f_roa": "shrout"}))[3] == ""      # not a factor: no p1


def test_coverage_by_year():
    panel = pd.DataFrame({"date": pd.to_datetime(["2015-01-30", "2015-02-27", "2016-01-29"]),
                          "f_ltg": [0.1, np.nan, np.nan], "ticker": "A"})
    cov = cs.coverage_by_year(panel)
    assert list(cov.index) == ["f_ltg"] and cov.loc["f_ltg", 2015] == 50 and cov.loc["f_ltg", 2016] == 0


def test_stats_checks_catch_a_missing_row_and_a_changed_number():
    stats = cs.stats_frame([cs.column_stat("t.parquet", "x", __import__("pyarrow").chunked_array([[1.0, 2.0]]))])
    outputs = {"t.parquet": {"columns": {"x": "double"}}, "column_stats.parquet": {"columns": {}}}
    md = "\n".join(["## t.parquet", "", *cs.stats_table(stats, "t.parquet")])
    assert [c.status for c in s08.stats_checks(md, stats, outputs)] == ["PASS", "PASS"]
    outputs["t.parquet"]["columns"]["y"] = "double"
    assert s08.stats_checks(md, stats, outputs)[0].status == "FAIL"
    assert s08.stats_checks(md.replace("| 1 |", "| 9 |"), stats, outputs)[1].status == "FAIL"


def test_column_stats_file_is_documented():
    assert set(dictionary.columns("column_stats.parquet")) == set(cs.COLUMNS)


def test_schema_dictionary_holds_nothing_from_the_data():
    import pyarrow as pa
    stats = cs.stats_frame([cs.column_stat("t.parquet", "x", pa.chunked_array([[1.23456, 98765.4321, 3.0]])),
                            cs.column_stat("t.parquet", "ticker", pa.chunked_array([["ZZQX", "ZZQX", "AB"]]))])
    values = cs.distinctive_values(stats)
    assert {"1.235", "9.877e+04", "ZZQX", "AB"} <= values and "3" not in values      # short numbers are not tested
    schema = dictionary.render_schema()
    assert not cs.leaked(schema, values)
    assert cs.leaked(schema + "\n| x | 1.235 |", values) == ["1.235"]
    assert "### Stats" not in schema and "Coverage by year" not in schema and "rows" not in schema.split("## Files")[1].split("\n")[2]


def test_schema_and_full_dictionaries_list_the_same_columns():
    outputs = {n: {"rows": 10, "size_mb": 1.0, "files": 1, "date_range": None,
                   "columns": {c: "double" for c in dictionary.columns(n)}} for n in dictionary.FILES}
    full = dictionary.render({"data_version": "2026-10-25", "pipeline_git_commit": "abc", "outputs": outputs})
    stats = cs.stats_frame([])
    checks = s08.schema_checks(full, dictionary.render_schema(), stats)
    assert [c.status for c in checks] == ["PASS", "PASS"]
    broken = dictionary.render_schema().replace("| `permno` |", "| `permno_x` |", 1)
    assert s08.schema_checks(full, broken, stats)[0].status == "FAIL"
