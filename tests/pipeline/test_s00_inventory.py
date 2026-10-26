"""
Tests for pipeline/stages/s00_inventory.py.

Small synthetic CSVs in a temporary data folder stand in for the WRDS downloads; no real data is used.
"""
import hashlib
import json

import pytest

from pipeline.config import DataPaths, RawFile
from pipeline.stages import s00_inventory as s00

HOLDINGS = RawFile("holdings.csv", "report_dt", ("report_dt", "permno", "crsp_portno"),
                   unique=("permno",), groups=(("crsp_portno",),), blanks=("permno",),
                   expected_values={"crsp_portno": ("1", "2")}, first_date_by="2008-02-29",
                   upload_list=("permno_list.txt", "permno"))
HEADER = "Report_DT,PERMNO,crsp_portno,security_name\n"      # mixed case: columns are lowercased
ROWS = [
    "2008-02-29,10001,1,ALPHA",
    "2008-02-29,10002.0,1,BETA",        # float-looking id: same as 10002
    "2008-03-31,,1,CASH",               # missing permno
    "2008-03-31,10002,2,BETA",
    "not-a-date,10003,2,GAMMA",         # bad date: counted, left out of the range
    ",10004,3,DELTA",                   # blank date; portno 3 is unexpected
    "2026-06-30,10001,2,ALPHA",
]


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "raw" / "query_inputs").mkdir(parents=True)
    (tmp_path / "raw" / "holdings.csv").write_text(HEADER + "\n".join(ROWS) + "\n")
    (tmp_path / "raw" / "query_inputs" / "permno_list.txt").write_text("10001\n10002\n10009\n")
    return tmp_path


def scan(data_dir, spec=HOLDINGS):
    return s00.scan_csv(data_dir / "raw" / spec.name, spec)


@pytest.mark.parametrize("block_bytes", [1 << 26, 64])
def test_scan_counts_rows_dates_ids_and_groups(data_dir, monkeypatch, block_bytes):
    # A 64-byte block splits the file into several batches, so the totals must add up across batches
    monkeypatch.setattr(s00, "_BLOCK_BYTES", block_bytes)
    out = scan(data_dir)
    assert out["columns"] == ["report_dt", "permno", "crsp_portno", "security_name"]
    assert out["rows"] == 7
    assert out["date_range"] == ["2008-02-29", "2026-06-30"]
    assert (out["bad_dates"], out["blank_dates"]) == (1, 1)
    assert out["distinct"]["permno"] == {"10001", "10002", "10003", "10004"}
    assert out["counts"]["crsp_portno"] == [{"crsp_portno": "1", "rows": 3}, {"crsp_portno": "2", "rows": 3},
                                            {"crsp_portno": "3", "rows": 1}]
    assert out["missing_values"] == {"permno": 1}


def test_scan_counts_by_two_columns(tmp_path):
    spec = RawFile("ibes.csv", "statpers", ("statpers",), groups=(("measure", "fpi"),))
    (tmp_path / "ibes.csv").write_text("STATPERS,MEASURE,FPI\n2020-01-16,EPS,1\n2020-01-16,EPS,1\n2020-01-16,SAL,0\n")
    out = s00.scan_csv(tmp_path / "ibes.csv", spec)
    assert out["counts"]["measure, fpi"] == [{"measure": "EPS", "fpi": "1", "rows": 2},
                                             {"measure": "SAL", "fpi": "0", "rows": 1}]


def test_inventory_file_checks_and_upload_list(data_dir):
    lists = {"permno_list.txt": s00.scan_list(data_dir / "raw" / "query_inputs" / "permno_list.txt")}
    info, checks = s00.inventory_file(data_dir / "raw" / "holdings.csv", HOLDINGS, lists)
    status = {c.name: c.status for c in checks}
    assert status["holdings.csv: required columns"] == "PASS"
    assert status["holdings.csv: report_dt format"] == "WARN"
    assert status["holdings.csv: crsp_portno values"] == "PASS"
    assert status["holdings.csv: unexpected crsp_portno values"] == "WARN"
    assert "holdings.csv: starts late" not in status
    assert info["sha256"] == hashlib.sha256((data_dir / "raw" / "holdings.csv").read_bytes()).hexdigest()
    assert info["upload_list"]["listed_not_in_file"] == 1                # 10009
    assert info["upload_list"]["examples_listed_not_in_file"] == ["10009"]
    assert info["upload_list"]["in_file_not_listed"] == 2                # 10003, 10004


def test_missing_file_column_and_value_fail(data_dir):
    _, checks = s00.inventory_file(data_dir / "raw" / "nope.csv", RawFile("nope.csv", None, ()), {})
    assert [(c.name, c.status) for c in checks] == [("nope.csv: exists", "FAIL")]

    strict = RawFile("holdings.csv", "report_dt", ("report_dt", "cusip"), expected_values={"crsp_portno": ("9",)},
                     first_date_by="2007-12-31")
    _, checks = s00.inventory_file(data_dir / "raw" / "holdings.csv", strict, {})
    status = {c.name: (c.status, c.detail) for c in checks}
    assert status["holdings.csv: required columns"] == ("FAIL", "missing cusip")
    assert status["holdings.csv: crsp_portno values"] == ("FAIL", "missing 9")
    assert status["holdings.csv: starts late"][0] == "WARN"


def test_scan_list_counts_duplicates_and_blanks(tmp_path):
    (tmp_path / "ids.txt").write_text("10001\n10001\n\n10002.0\n")
    out = s00.scan_list(tmp_path / "ids.txt")
    assert (out["lines"], out["distinct"], out["duplicates"], out["blank_lines"]) == (3, 2, 1, 1)
    assert out["ids"] == {"10001", "10002"}


def test_run_writes_only_its_own_folder_and_reruns(data_dir, monkeypatch, capsys):
    monkeypatch.setattr(s00, "RAW_FILES", (HOLDINGS,))
    monkeypatch.setattr(s00, "QUERY_INPUTS", ("permno_list.txt",))
    (data_dir / "processed" / "s01_membership").mkdir(parents=True)
    (data_dir / "processed" / "s01_membership" / "keep.txt").write_text("other stage")
    raw_before = {p: p.read_bytes() for p in (data_dir / "raw").rglob("*") if p.is_file()}
    paths = DataPaths.from_data_dir(data_dir)

    summary = s00.run(paths)
    stage_dir = data_dir / "processed" / "s00_inventory"
    (stage_dir / "stale.json").write_text("{}")
    summary = s00.run(paths)                                                # rerun

    assert sorted(p.name for p in stage_dir.iterdir()) == ["inventory.json", "summary.json"]
    assert (data_dir / "processed" / "s01_membership" / "keep.txt").read_text() == "other stage"
    assert raw_before == {p: p.read_bytes() for p in (data_dir / "raw").rglob("*") if p.is_file()}
    inv = json.loads((stage_dir / "inventory.json").read_text())
    assert inv["files"]["holdings.csv"]["rows"] == 7
    assert inv["query_inputs"]["permno_list.txt"]["lines"] == 3
    assert inv["status"] == summary["status"] == "WARN"
    assert summary["tables"]["holdings.csv"] == {"rows": 7, "date_range": ["2008-02-29", "2026-06-30"],
                                                 "permnos": 4}
    assert "holdings.csv" in capsys.readouterr().out
