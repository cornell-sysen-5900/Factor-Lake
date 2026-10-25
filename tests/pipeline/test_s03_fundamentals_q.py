"""
Tests for pipeline/stages/s03_fundamentals_q.py on small synthetic CCM quarterly data (no real data).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.config import DataPaths, YTD_ITEMS
from pipeline.stages import s03_fundamentals_q as s03

NUMERIC = ["atq", "ltq", "actq", "lctq", "cheq", "invtq", "rectq", "apq", "ppentq", "gdwlq", "intanq", "dlcq",
           "dlttq", "pstkq", "mibtq", "seqq", "ceqq", "txditcq", "saleq", "revtq", "cogsq", "xsgaq", "xrdq", "oibdpq",
           "dpq", "oiadpq", "xintq", "piq", "txtq", "niq", "ibq", "epsfxq", "epspxq", "dvpsxq", "prccq", "cshoq",
           "cshfdq", *YTD_ITEMS]


def quarters(n=8, permno=1, start_year=2020, **cols):
    """n consecutive fiscal quarters (calendar = fiscal), every number 1.0 unless given; cols may be lists."""
    rows = []
    for i in range(n):
        y, q = start_year + i // 4, i % 4 + 1
        end = pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)
        rows.append({"permno": permno, "gvkey": f"{permno:06d}", "tic": "TIC", "conm": "CO", "datadate": end,
                     "rdq": end + pd.Timedelta(days=30), "fyearq": y, "fqtr": q, "curcdq": "USD", "ajexq": 1.0,
                     "gsector": 45, "ggroup": 4510, "gind": 451030, "gsubind": 45103010, "naics": 511210,
                     **{c: 1.0 for c in NUMERIC}})
    df = pd.DataFrame(rows)
    df["prccq"], df["cshoq"] = 10.0, 100.0
    for c, v in cols.items():
        df[c] = v
    return df


def built(raw):
    return s03.build(raw)[0].set_index(["fyearq", "fqtr"])


def test_ttm_needs_4_consecutive_quarters():
    raw = quarters(8, saleq=[10.0] * 8)
    raw = raw.drop(index=4)                                       # 2021 Q1 missing
    out = built(raw)
    assert out.loc[(2020, 4), "revenue_ttm"] == 40.0
    for q in (2, 3, 4):                                           # every window that needs 2021 Q1
        assert np.isnan(out.loc[(2021, q), "revenue_ttm"])


def test_interest_never_filled_with_zero():
    out = built(quarters(5, xintq=[2.0, 2.0, np.nan, 2.0, 2.0]))
    assert out["interest_ttm"].isna().all()                       # every 4-quarter window includes the gap
    out = built(quarters(5, xintq=[2.0] * 5))
    assert out.loc[(2021, 1), "interest_ttm"] == 8.0


def test_ytd_cash_flows_to_quarterly():
    raw = quarters(5, capxy=[5.0, 12.0, 20.0, 30.0, 6.0])
    out = built(raw)
    assert out["capx_q"].tolist() == [5.0, 7.0, 8.0, 10.0, 6.0]
    assert out.loc[(2020, 4), "capx_ttm"] == 30.0
    gap = built(raw.drop(index=1))                                # Q2 missing: Q3 cannot be differenced
    assert np.isnan(gap.loc[(2020, 3), "capx_q"]) and gap.loc[(2020, 4), "capx_q"] == 10.0


def test_growth_vs_exactly_4_quarters_earlier():
    raw = quarters(9, atq=[100.0] * 4 + [110.0] * 5)
    out = built(raw)
    assert out.loc[(2021, 1), "f_asset_growth_1yr"] == pytest.approx(0.10)
    out = built(raw.drop(index=0))                                # 2020 Q1 missing
    assert np.isnan(out.loc[(2021, 1), "f_asset_growth_1yr"])
    assert out.loc[(2021, 2), "f_asset_growth_1yr"] == pytest.approx(0.10)


def test_split_adjustment_with_ajexq():
    # 2-for-1 split before the 4th quarter: Compustat restates the older quarters' factor to 2
    raw = quarters(4, ajexq=[2.0, 2.0, 2.0, 1.0], epsfxq=[1.0, 1.0, 1.0, 0.5], dvpsxq=[0.2, 0.2, 0.2, 0.1],
                   cshoq=[100.0, 100.0, 100.0, 200.0])
    out = built(raw)
    last = out.loc[(2020, 4)]
    assert last.epsfxq_ttm_adj == pytest.approx(2.0)              # 4 quarters x $0.50 per post-split share
    assert last.dvpsxq_ttm_adj == pytest.approx(0.4)
    assert last.f_earnings_to_price == pytest.approx(2.0 / 10.0)
    assert last.csho_adj == pytest.approx(200.0) and out.loc[(2020, 1), "csho_adj"] == pytest.approx(200.0)


@pytest.mark.parametrize("pi, txt, rate", [(10.0, 2.5, 0.25), (10.0, 8.0, 0.60), (10.0, -1.0, 0.0),
                                           (-10.0, 1.0, np.nan), (0.0, 0.0, np.nan)])
def test_tax_rate_only_when_pretax_positive_and_clipped(pi, txt, rate):
    out = built(quarters(4, piq=[pi / 4] * 4, txtq=[txt / 4] * 4))
    got = out.loc[(2020, 4), "tax_rate_ttm"]
    assert (np.isnan(got) and np.isnan(rate)) or got == pytest.approx(rate)


def test_non_usd_and_tiny_price_lose_price_ratios_only():
    raw = pd.concat([quarters(4, permno=1, curcdq="CAD"), quarters(4, permno=2, prccq=0.005),
                     quarters(4, permno=3)], ignore_index=True)
    out = s03.build(raw)[0].set_index(["permno", "fyearq", "fqtr"])
    for p in (1, 2):
        row = out.loc[(p, 2020, 4)]
        assert row[list(s03.PRICE_BASED_FACTORS)].isna().all()
        assert row.f_roa == pytest.approx(4.0) and row.f_leverage == pytest.approx(2.0)
    assert np.isnan(out.loc[(1, 2020, 4), "mkt_cap_qe_musd"]) and out.loc[(2, 2020, 4), "mkt_cap_qe_musd"] > 0
    assert out.loc[(3, 2020, 4), list(s03.PRICE_BASED_FACTORS)].notna().all()


def test_duplicate_quarter_keeps_the_most_complete_row():
    raw = quarters(4)
    dup = raw.iloc[[3]].assign(saleq=99.0, invtq=np.nan)          # one field fewer
    out, info = s03.build(pd.concat([raw, dup], ignore_index=True))
    assert info["duplicates_removed"] == 1 and out["saleq"].tolist() == [1.0] * 4


def test_report_date_rdq_or_45_90_day_proxy():
    out = built(quarters(4, rdq=pd.NaT))
    ends = out["period_end"]
    assert (out["report_date"] - ends).dt.days.tolist() == [45, 45, 45, 90]
    assert (out["report_date_source"] == "proxy").all()


def test_book_equity_and_sue():
    out = built(quarters(4, seqq=[np.nan, 50.0, 50.0, 50.0], ceqq=40.0, pstkq=5.0, txditcq=2.0))
    assert out["book_equity"].tolist() == [42.0, 47.0, 47.0, 47.0]   # ceq + pstk, else seq; + txditc - pstk
    eps = [0.1 * (i % 3) for i in range(16)]                          # changing EPS gives a non-zero std
    out = built(quarters(16, epspxq=eps))
    d = [eps[i] - eps[i - 4] for i in range(4, 16)]
    hist = d[14 - 4 - 8:14 - 4]                                       # the 8 changes before 2023 Q3
    assert out.loc[(2023, 3), "f_sue"] == pytest.approx(d[14 - 4] / np.std(hist, ddof=1))


def test_run_writes_output_and_summary(tmp_path):
    (tmp_path / "raw").mkdir()
    quarters(8, saleq=[10.0] * 8).to_csv(tmp_path / "raw" / s03.RAW_FILE, index=False)
    summary = s03.run(DataPaths.from_data_dir(tmp_path))
    out = pd.read_parquet(tmp_path / "processed" / "s03_fundamentals_q" / s03.OUTPUT)
    assert len(out) == 8 and out["gind_name"].iloc[0] == "Software" and out["gsector_name"].iloc[0] == (
        "Information Technology")
    stats = summary["tables"][s03.OUTPUT]["stats"]
    assert set(s03.FACTORS) | {"revenue_ttm", "fcf_ttm", "tax_rate_ttm"} <= set(stats)
    status = {c["name"]: c["status"] for c in summary["checks"]}
    assert status["no price-based ratio on non-USD rows or prices below $0.01"] == "PASS"
    assert status["revenue_ttm equals the sum of 4 fiscal quarters (200 random rows)"] == "PASS"


def test_near_zero_denominators_give_nan():
    # prior TTM sales 0.8 (< $1M): no sales growth; prior TTM capex 0.4: no capex growth; current sales 0.8: no R&D ratio
    raw = quarters(8, saleq=[0.2] * 4 + [5.0] * 4, capxy=[0.1, 0.2, 0.3, 0.4, 2.0, 4.0, 6.0, 8.0], xrdq=1.0)
    out = built(raw)
    assert np.isnan(out.loc[(2021, 4), "f_sales_growth_1yr"]) and np.isnan(out.loc[(2021, 4), "f_capex_growth_1yr"])
    assert np.isnan(out.loc[(2020, 4), "f_rd_to_sales"]) and out.loc[(2021, 4), "f_rd_to_sales"] == pytest.approx(0.2)
    big = built(quarters(8, saleq=[1.0] * 4 + [5.0] * 4, capxy=[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]))
    assert big.loc[(2021, 4), "f_sales_growth_1yr"] == pytest.approx(4.0)
    assert big.loc[(2021, 4), "f_capex_growth_1yr"] == pytest.approx(1.0)


def test_sue_needs_a_std_of_at_least_one_cent():
    flat = [1.0 + 0.001 * (i % 2) for i in range(16)]          # EPS changes of a tenth of a cent
    assert built(quarters(16, epspxq=flat))["f_sue"].isna().all()
