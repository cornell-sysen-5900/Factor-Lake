"""
Tests for pipeline/stages/s09_validate.py and scripts/wrds_independent_recompute.py on small synthetic data.
"""
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.stages import s08_export as s08
from pipeline.stages import s09_validate as s09

ROOT = Path(__file__).resolve().parents[2]
ENDS = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])


def rets_frame(rows):
    return pd.DataFrame(rows, columns=["month_end", "permno", "ret_m"]).astype({"month_end": "datetime64[ns]"})


def test_cap_weighted_next_month_return():
    panel = pd.DataFrame({"date": [ENDS[0], ENDS[0]], "permno": [1, 2], "mkt_cap_musd": [300.0, 100.0]})
    rets = rets_frame([(ENDS[1], 1, 0.10), (ENDS[1], 2, -0.10), (ENDS[0], 1, 0.5)])
    r = s09.weighted(s09.next_returns(panel, rets), "mkt_cap_musd")
    assert r.index.tolist() == [pd.Period("2020-02", "M")] and r.iloc[0] == pytest.approx(0.05)


def test_tracking_statistics():
    idx = pd.period_range("2020-01", periods=24, freq="M")
    bench = pd.Series(np.linspace(-0.02, 0.03, 24), index=idx)
    t = s09.tracking(bench + 0.001, bench)
    assert t["correlation"] == pytest.approx(1.0) and t["tracking_error_annual"] == pytest.approx(0.0, abs=1e-9)
    assert t["mean_annual_difference"] == pytest.approx(0.012) and set(t["by_year"]) == {2020, 2021}


def test_long_short_top_minus_bottom_and_low_minus_high():
    panel = pd.DataFrame({"date": ENDS[0], "permno": range(1, 11), "mkt_cap_musd": 1.0,
                          "f": np.arange(10.0)})
    rets = rets_frame([(ENDS[1], p, 0.01 * p) for p in range(1, 11)])
    m = s09.next_returns(panel, rets)
    hi_lo = s09.long_short(m, "f", False).iloc[0]
    assert hi_lo == pytest.approx(np.mean([0.08, 0.09, 0.10]) - np.mean([0.01, 0.02, 0.03]))
    assert s09.long_short(m, "f", True).iloc[0] == pytest.approx(-hi_lo)


def test_june_portfolio_holds_july_to_june():
    dates = pd.date_range("2019-06-30", periods=14, freq="ME")
    panel = pd.DataFrame([(d, p, float(p), 0.01 * p) for d in dates for p in range(1, 21)],
                         columns=["date", "permno", "f_roa", "ret_next_1m"])
    r, picks = s09.june_top_roa(panel)
    assert picks[2019] == {19, 20} and picks[2020] == {19, 20}
    assert r.loc[pd.Period("2019-07", "M")] == pytest.approx(0.195)
    assert r.index.min() == pd.Period("2019-07", "M")


def write_final(final: Path, value: float):
    final.mkdir(parents=True)
    pd.DataFrame({"date": ENDS, "permno": [1, 2, 3], "x": [1.0, 2.0, value]}).to_parquet(
        final / "factor_panel_monthly.parquet")
    (final / "manifest.json").write_text(json.dumps({"created_at": str(value), "rows": 3}))


def test_rerun_comparison_finds_changed_data_and_ignores_created_at(tmp_path):
    write_final(tmp_path / "a", 3.0)
    write_final(tmp_path / "b", 3.0)
    same = s09.compare_finals(tmp_path / "a", tmp_path / "b")
    assert all(v["same"] for v in same.values())
    write_final(tmp_path / "c", 3.5)
    diff = s09.compare_finals(tmp_path / "a", tmp_path / "c")
    assert not diff["factor_panel_monthly.parquet"]["same"] and diff["manifest.json"]["same"]          # only created_at differs


def test_rerun_hash_ignores_row_order(tmp_path):
    df = pd.DataFrame({"date": ENDS, "permno": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    df.to_parquet(tmp_path / "a.parquet")
    df.iloc[::-1].to_parquet(tmp_path / "b.parquet")
    assert s09.table_hash(tmp_path / "a.parquet", ["date", "permno"]) == \
        s09.table_hash(tmp_path / "b.parquet", ["date", "permno"])


def test_upload_dry_run_works_offline(tmp_path, monkeypatch):
    final, out = tmp_path / "final", tmp_path / "out"
    final.mkdir()
    out.mkdir()
    outputs = {"x.parquet": {"size_mb": 0.1, "files": 32}}
    (final / "UPLOAD.md").write_text(s08.upload_md("2026-10-25", outputs), encoding="utf-8")
    for i in range(34):
        (final / f"f{i}.parquet").write_bytes(b"x")
    (final / "manifest.json").write_text("{}")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "should-not-be-needed")
    manifest = {"s3_target": "s3://sysen-5900-factor-lake/factor-lake-data/wrds_2026_10/",
                "outputs": {"tables": {"files": 34}}}                       # Parquet files only
    checks, res = s09.check_upload(final, out, manifest)
    assert res["exit_code"] == 0, res["stderr"]
    assert res["keys"] == 34 and not res["outside_prefix"] and checks[0].status == "PASS"
    assert all(k.endswith(".parquet") for k in res["first_keys"])     # no JSON or Markdown


def test_independent_recompute_script_on_tiny_raw_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    days = pd.bdate_range("2019-01-01", "2020-03-31")
    rets = np.full(len(days), 0.001)
    ciz = pd.DataFrame({"permno": 1, "HdrCUSIP": "X", "SecurityNm": "ACME INC; COM NONE; CONS", "DelReasonType": "",
                        "Ticker": "ACME", "PERMCO": 10, "SICCD": 3571, "DlyCalDt": days.strftime("%Y-%m-%d"),
                        "DlyDelFlg": "N", "DlyPrc": 10.0, "DlyRet": rets, "DlyVol": 1, "DlyClose": 10.0,
                        "DlyLow": 10.0, "DlyHigh": 10.0, "DlyOpen": 10.0, "ShrOut": 2000.0, "DlyCumFacPr": 1.0,
                        "DlyCumFacShr": 1.0, "sprtrn": 0.0})
    ciz.to_csv(raw / "r2000_price_crsp_ciz.csv", index=False)
    pd.DataFrame(columns=["delactiontype", "permno", "delistingdt", "delreasontype", "delpaymenttype",
                          "delret"]).to_csv(raw / "delistings.csv", index=False)
    q = []
    for k, end in enumerate(pd.to_datetime(["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31"])):
        q.append({"GVKEY": "001", "LPERMNO": 1, "datadate": end.date(), "fyearq": 2019, "fqtr": k + 1,
                  "curcdq": "USD", "rdq": (end + pd.Timedelta(days=30)).date(), "niq": 1.0, "ibq": 1.0,
                  "oibdpq": 3.0, "atq": 100.0, "ltq": 60.0, "seqq": 40.0, "ceqq": 40.0, "pstkq": 0.0,
                  "txditcq": 0.0, "dlttq": 5.0, "dlcq": 0.0, "cheq": 1.0, "mibtq": 0.0, "cshoq": 2.0})
    pd.DataFrame(q).to_csv(raw / "fundamentals_raw.csv", index=False)
    pd.DataFrame({"permno": [1], "date": ["2020-02-28"]}).to_csv(tmp_path / "sample.csv", index=False)
    mod = runpy.run_path(str(ROOT / "scripts" / "wrds_independent_recompute.py"))
    mod["main"](str(raw), str(tmp_path / "sample.csv"), str(tmp_path / "out.csv"))
    r = pd.read_csv(tmp_path / "out.csv").iloc[0]
    assert r.roa == pytest.approx(0.04) and r.b_p == pytest.approx(40 / 20)        # cap 10 x 2,000 / 1000 = $20M
    assert r.e_p == pytest.approx(4 / 20) and r.ev_ebitda == pytest.approx((20 + 5 - 1) / 12)
    assert r.mom_12_1 == pytest.approx(1.001 ** 231 - 1)
    n_feb = sum(1 for d in days if d.month == 2 and d.year == 2020)
    assert r.ret_m == pytest.approx(1.001 ** n_feb - 1)
    code = (ROOT / "scripts" / "wrds_independent_recompute.py").read_text(encoding="utf-8")
    imports = [l for l in code.splitlines() if l.startswith(("import ", "from "))]
    assert imports and not any("pipeline" in l for l in imports)
