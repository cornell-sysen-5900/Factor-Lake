"""
Tests for pipeline/stages/s06_market.py on small synthetic market files (no real data).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.config import DataPaths
from pipeline.stages import s06_market as s06


def write_inputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    days = pd.bdate_range("2012-10-01", "2012-12-31")
    lvl = {"I0124": 100.0, "I0120": 200.0, "I0122": 50.0}
    rows = [(t, d.date(), 1, v * 1.001 ** i) for i, d in enumerate(days) for t, v in lvl.items()]
    pd.DataFrame(rows, columns=["tic", "datadate", "gvkeyx", "prccd"]).to_csv(raw / "index_close.csv", index=False)
    crsp_days = days[~days.isin(pd.to_datetime(["2012-10-29", "2012-10-30"]))]
    etf = [(p, d.date(), 0.001) for d in crsp_days for p in (88222, 88399, 88401)]
    pd.DataFrame(etf, columns=["PERMNO", "DlyCalDt", "DlyRet"]).to_csv(raw / "benchmark_etfs.csv", index=False)
    ff = pd.DataFrame({"date": crsp_days, **{c: 0.0005 for c in s06.FF_COLS}})
    ff["rf"] = 0.0001
    ff.to_csv(raw / "ff_daily.csv", index=False)
    comp = (1 + ff.set_index("date")).groupby(ff["date"].dt.to_period("M").values).prod() - 1
    mm = comp.reset_index(drop=True).assign(dateff=[pd.Timestamp("2012-10-31"), pd.Timestamp("2012-11-30"),
                                                    pd.Timestamp("2012-12-31")])
    mm.to_csv(raw / "ff_monthly.csv", index=False)
    for series, pct in (("DGS10", 1.7), ("DGS3MO", 0.1), ("DBAA", 4.6), ("DAAA", 3.6)):
        vals = [pct] * len(days)
        vals[days.get_loc(pd.Timestamp("2012-11-12"))] = None          # Veterans Day: bond market closed
        pd.DataFrame({"observation_date": days.date, series: vals}).to_csv(raw / f"fred_{series}.csv", index=False)
    cal_dir = tmp_path / "processed" / "s02_prices"
    cal_dir.mkdir(parents=True)
    cal = pd.DataFrame({"month": ["2012-10", "2012-11", "2012-12"],
                        "month_end": pd.to_datetime(["2012-10-31", "2012-11-30", "2012-12-31"]), "trading_days": 21})
    cal.to_parquet(cal_dir / "calendar.parquet")
    return DataPaths.from_data_dir(tmp_path), crsp_days


def test_market_daily_and_monthly(tmp_path):
    paths, crsp_days = write_inputs(tmp_path)
    summary = s06.run(paths)
    out = tmp_path / "processed" / "s06_market"
    daily = pd.read_parquet(out / "market_daily.parquet").set_index("date")
    monthly = pd.read_parquet(out / "market_monthly.parquet").set_index("month")

    # Sandy days are dropped from the index levels, so the 2012-10-31 return spans 10-26 to 10-31
    assert np.isnan(daily.loc["2012-10-29", "r2000_price_ret"]) and np.isnan(daily.loc["2012-10-30", "r2000_price_ret"])
    assert daily.loc["2012-10-31", "r2000_price_ret"] == pytest.approx(1.001 ** 3 - 1)
    # FRED in decimals, the holiday forward-filled and flagged, spreads
    vet = daily.loc["2012-11-12"]
    assert vet.ust_10y == pytest.approx(0.017) and vet.was_filled and not daily.loc["2012-11-13", "was_filled"]
    assert vet.term_spread == pytest.approx(0.016) and vet.credit_spread == pytest.approx(0.01)
    # monthly: first month dropped; returns compounded; yields at month end; FF published monthly
    assert list(monthly.index) == ["2012-11", "2012-12"]
    nov = monthly.loc["2012-11"]
    n_nov = int(((crsp_days >= "2012-11-01") & (crsp_days <= "2012-11-30")).sum())
    assert nov.iwm_total_ret == pytest.approx(1.001 ** n_nov - 1)
    n_lvl = len(pd.bdate_range("2012-11-01", "2012-11-30"))
    assert nov.r2000_price_ret == pytest.approx(1.001 ** n_lvl - 1)
    assert nov.moodys_baa == pytest.approx(0.046) and nov.rf == pytest.approx(1.0001 ** n_nov - 1)
    status = {c["name"]: c["status"] for c in summary["checks"]}
    assert status["ust_10y within 0.0%-6%"] == "PASS"
    assert status["published monthly mktrf and rf within 50 bp of compounded daily"] == "PASS"
    assert status["monthly table has every month"] == "PASS"
