"""
Tests for pipeline/stages/s04_fundamentals_a.py on small synthetic CCM annual data (no real data).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.config import DataPaths
from pipeline.stages import s04_fundamentals_a as s04

NUMERIC = ["act", "ap", "aqc", "at", "capx", "ceq", "che", "cogs", "cshfd", "csho", "dlc", "dltis", "dltr", "dltt",
           "dp", "dv", "dvc", "emp", "epsfx", "epspx", "gdwl", "ib", "intan", "invt", "lct", "lt", "mib", "ni",
           "oancf", "oiadp", "oibdp", "pi", "ppent", "prstkc", "pstk", "rect", "revt", "sale", "seq", "sstk",
           "txditc", "txt", "xint", "xintd", "xrd", "xsga", "dvpsx_f", "prcc_f"]


def years(fyears=(2020, 2021, 2022), permno=1, **cols):
    rows = [{"permno": permno, "gvkey": f"{permno:06d}", "tic": "TIC", "conm": "CO", "fyear": y, "fyr": 12,
             "datadate": pd.Timestamp(f"{y}-12-31"), "curcd": "USD", "ajex": 1.0, "sich": 7372, "gsector": 45,
             "ggroup": 4510, "gind": 451030, "gsubind": 45103010, "naics": 511210, **{c: 1.0 for c in NUMERIC}}
            for y in fyears]
    df = pd.DataFrame(rows)
    for c, v in cols.items():
        df[c] = v
    return df


def q4(gvkey="000001", fyears=(2020, 2021, 2022), lag_days=40, ends=None):
    ends = ends or [pd.Timestamp(f"{y}-12-31") for y in fyears]
    return pd.DataFrame({"gvkey": gvkey, "fyearq": list(fyears), "fqtr": 4, "period_end": ends,
                         "rdq": [e + pd.Timedelta(days=lag_days) for e in ends], "revenue_ttm": 1.0})


def built(raw, quarterly=None):
    return s04.build(raw, q4() if quarterly is None else quarterly)[0].set_index("fyear")


def test_report_date_from_matching_q4_rdq_else_90_days():
    quarterly = q4(fyears=(2020, 2021), ends=[pd.Timestamp("2020-12-31"), pd.Timestamp("2021-06-30")])
    out, info = s04.build(years(), quarterly)
    out = out.set_index("fyear")
    assert out.loc[2020, "report_date"] == pd.Timestamp("2021-02-09") and out.loc[2020, "report_date_source"] == "q4_rdq"
    # 2021: the Q4 of fiscal 2021 ends in June (fiscal-year change), so its rdq is not used
    assert out.loc[2021, "report_date"] == pd.Timestamp("2022-03-31") and out.loc[2021, "report_date_source"] == "proxy"
    assert out.loc[2022, "report_date_source"] == "proxy"                  # no Q4 row at all
    assert info["q4_rejected_period_end_differs"] == 1


def test_fcff_formula_and_missing_parts():
    raw = years(oiadp=100.0, pi=80.0, txt=20.0, dp=10.0, capx=30.0, act=[50.0, 60.0, 60.0], che=0.0,
                lct=[20.0, 20.0, np.nan], dlc=0.0)
    out = built(raw)
    # 2021: EBIT 100 x (1 - 25%) + D&A 10 - capex 30 - change in NWC (40 - 30 = 10) = 45
    assert out.loc[2021, "tax_rate"] == pytest.approx(0.25) and out.loc[2021, "d_nwc"] == pytest.approx(10.0)
    assert out.loc[2021, "fcff"] == pytest.approx(45.0)
    assert np.isnan(out.loc[2020, "fcff"])                                 # no previous year for the NWC change
    assert np.isnan(out.loc[2022, "fcff"])                                 # current liabilities missing


def test_changes_only_vs_exactly_the_previous_fiscal_year():
    out = built(years((2020, 2022, 2023), sale=[100.0, 150.0, 180.0], oibdp=[10.0, 12.0, 15.0]))
    assert np.isnan(out.loc[2022, "revenue_growth"]) and np.isnan(out.loc[2022, "d_nwc"])   # 2021 is missing
    assert out.loc[2023, "revenue_growth"] == pytest.approx(0.2) and out.loc[2023, "ebitda_growth"] == pytest.approx(0.25)


@pytest.mark.parametrize("xint, xintd, value, source", [(5.0, 4.0, 5.0, "xint"), (np.nan, 4.0, 4.0, "xintd"),
                                                        (np.nan, np.nan, np.nan, "missing")])
def test_interest_xint_then_xintd_never_zero(xint, xintd, value, source):
    out = built(years((2020,), xint=xint, xintd=xintd))
    got = out.loc[2020, "interest_expense"]
    assert ((np.isnan(got) and np.isnan(value)) or got == value) and out.loc[2020, "interest_source"] == source


@pytest.mark.parametrize("pi, txt, rate", [(100.0, 21.0, 0.21), (100.0, 90.0, 0.60), (-5.0, 1.0, np.nan)])
def test_tax_rate_rule(pi, txt, rate):
    got = built(years((2020,), pi=pi, txt=txt)).loc[2020, "tax_rate"]
    assert (np.isnan(got) and np.isnan(rate)) or got == pytest.approx(rate)


def test_duplicate_year_keeps_most_complete_row():
    raw = years((2020, 2021))
    dup = raw.iloc[[1]].assign(sale=99.0, emp=np.nan)
    out, info = s04.build(pd.concat([raw, dup], ignore_index=True), q4())
    assert info["duplicates_removed"] == 1 and out["revenue"].tolist() == [1.0, 1.0]


def test_run_end_to_end(tmp_path):
    (tmp_path / "raw").mkdir()
    years(sale=[1.0, 1.0, 1.0]).to_csv(tmp_path / "raw" / s04.RAW_FILE, index=False)
    qdir = tmp_path / "processed" / "s03_fundamentals_q"
    qdir.mkdir(parents=True)
    q4().to_parquet(qdir / "fundamentals_quarterly.parquet")
    summary = s04.run(DataPaths.from_data_dir(tmp_path))
    out = pd.read_parquet(tmp_path / "processed" / "s04_fundamentals_a" / s04.OUTPUT)
    assert len(out) == 3 and (out["report_date_source"] == "q4_rdq").all()
    assert summary["report_date_source"] == {"q4_rdq": 3}
    assert {c["status"] for c in summary["checks"]} == {"PASS"}
    assert set(s04.BLOCKS) <= set(summary["tables"][s04.OUTPUT]["stats"])
