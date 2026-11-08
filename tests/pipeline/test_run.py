"""
Tests for the pipeline's command line (pipeline/run.py), output helpers (pipeline/outputs.py) and checks.
"""
import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from pipeline import run
from pipeline.checks import Check, columns_present, overall_status, values_present
from pipeline.config import DataPaths
from pipeline.outputs import fresh_stage_dir, write_parquet, write_summary
from pipeline.stages import STAGES


@pytest.fixture
def paths(tmp_path):
    (tmp_path / "raw").mkdir()
    return DataPaths.from_data_dir(tmp_path)


@pytest.mark.parametrize("name", ["s02_prices", "s02", "prices"])
def test_stage_names(name):
    assert run.resolve_stages(name) == ["s02_prices"]


def test_all_runs_every_stage_in_order():
    assert run.resolve_stages("all") == list(STAGES)
    assert STAGES[0] == "s00_inventory" and STAGES[-1] == "s09_validate"


def test_unknown_stage_exits():
    with pytest.raises(SystemExit):
        run.resolve_stages("s99")


def test_data_dir_needs_raw(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataPaths.from_data_dir(tmp_path)


def test_folder_options_default_to_the_standard_layout_and_move_everything(tmp_path):
    (tmp_path / "raw").mkdir()
    std = DataPaths.from_data_dir(tmp_path)
    assert (std.raw, std.processed) == (tmp_path / "raw", tmp_path / "processed")
    assert std.client == tmp_path / "raw" / "client"
    assert std.permno_list == tmp_path / "raw" / "query_inputs" / "permno_list.txt"
    assert std.cli_args() == ["--data-dir", str(tmp_path)]
    (tmp_path / "raw_v2").mkdir()
    v2 = DataPaths.from_data_dir(tmp_path, "processed_v2", raw_dir="raw_v2", client_dir="raw/client",
                                 permno_list="raw/query_inputs/permno_list_v2.txt",
                                 ibes_tickers=tmp_path / "x" / "ibes_tickers_v2.txt")
    assert (v2.raw, v2.processed) == (tmp_path / "raw_v2", tmp_path / "processed_v2")
    assert v2.stage_dir("s01_membership") == tmp_path / "processed_v2" / "s01_membership"
    assert v2.list_path("permno_list.txt") == tmp_path / "raw" / "query_inputs" / "permno_list_v2.txt"
    assert v2.ibes_tickers == tmp_path / "x" / "ibes_tickers_v2.txt"
    assert v2.relative(v2.permno_list) == "raw/query_inputs/permno_list_v2.txt"
    again = DataPaths.from_data_dir(*v2.cli_args()[1:2], **dict(zip(
        ["raw_dir", "client_dir", "permno_list", "ibes_tickers"], v2.cli_args()[3::2])))
    assert again.raw == v2.raw and again.ibes_tickers == v2.ibes_tickers
    with pytest.raises(FileNotFoundError):
        DataPaths.from_data_dir(tmp_path, raw_dir="raw_v3")


def test_main_passes_the_folder_options(tmp_path, monkeypatch):
    (tmp_path / "raw_v2").mkdir()
    seen = {}
    from pipeline.stages import s00_inventory

    def fake(paths):
        seen["paths"] = paths
        return {"status": "PASS"}
    monkeypatch.setattr(s00_inventory, "run", fake)
    assert run.main(["--data-dir", str(tmp_path), "--raw-dir", "raw_v2", "--processed-dir", "processed_v2",
                     "--permno-list", "raw/query_inputs/permno_list_v2.txt", "--stage", "s00"]) == 0
    p = seen["paths"]
    assert (p.raw, p.processed) == (tmp_path / "raw_v2", tmp_path / "processed_v2")
    assert p.permno_list.name == "permno_list_v2.txt"


def test_unwritten_stage_returns_exit_code_2(paths, capsys, monkeypatch):
    from pipeline.stages import s08_export

    def not_written(_):
        raise NotImplementedError("s08_export is not written yet")
    monkeypatch.setattr(s08_export, "run", not_written)
    assert run.main(["--data-dir", str(paths.root), "--stage", "s08"]) == 2
    assert "not written yet" in capsys.readouterr().out


def test_fresh_stage_dir_stays_inside_processed(paths):
    out = fresh_stage_dir(paths, "s01_membership")
    (out / "old.parquet").write_text("x")
    assert list(fresh_stage_dir(paths, "s01_membership").iterdir()) == []
    with pytest.raises(ValueError):
        fresh_stage_dir(paths, "../raw")
    assert (paths.raw).is_dir()


def test_write_parquet_uses_zstd(paths, tmp_path):
    path = write_parquet(pd.DataFrame({"permno": [1, 2], "ret": [0.1, None]}), tmp_path / "t.parquet")
    assert pq.read_metadata(path).row_group(0).column(0).compression == "ZSTD"
    assert pq.read_table(path).num_rows == 2


def test_summary_status_is_the_worst_check(tmp_path):
    checks = [Check("a", "PASS"), Check("b", "WARN"), Check("c", "PASS")]
    summary = write_summary(tmp_path, "s99_test", {"t": {"rows": 1}}, checks)
    assert summary["status"] == "WARN"
    assert json.loads((tmp_path / "summary.json").read_text())["checks"][1] == {"name": "b", "status": "WARN",
                                                                                "detail": ""}
    assert overall_status([]) == "PASS"
    assert overall_status(checks + [Check("d", "FAIL")]) == "FAIL"


def test_shared_checks():
    assert columns_present("f", ["a", "b"], ["a"]).status == "PASS"
    assert columns_present("f", ["a"], ["a", "c"]).detail == "missing c"
    assert [c.status for c in values_present("f", "tic", ["x", "y", "z"], ["x", "y"])] == ["PASS", "WARN"]
    assert values_present("f", "tic", ["x"], ["x", "y"])[0].status == "FAIL"


def test_last_run_values_come_from_the_data_folder(tmp_path):
    from pipeline import config
    from pipeline.checks import vs_last_run
    saved = (dict(config.LAST_RUN), dict(config.LAST_RUN_COVERAGE_2015))
    try:
        config.LAST_RUN.clear()
        assert vs_last_run("rows", 100, "rows").status == "WARN"                 # no value: a WARN, not a crash
        assert not config.load_last_run(tmp_path / "pipeline_last_run.json")
        (tmp_path / "pipeline_last_run.json").write_text(json.dumps({"counts": {"rows": 101},
                                                                     "coverage_2015": {"f_roa": 0.9}}))
        assert config.load_last_run(tmp_path / "pipeline_last_run.json")
        assert vs_last_run("rows", 100, "rows").status == "PASS" and config.LAST_RUN_COVERAGE_2015 == {"f_roa": 0.9}
    finally:
        config.LAST_RUN.clear(), config.LAST_RUN.update(saved[0])
        config.LAST_RUN_COVERAGE_2015.clear(), config.LAST_RUN_COVERAGE_2015.update(saved[1])
