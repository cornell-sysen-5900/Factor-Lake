"""
Tests for pipeline/stages/s05_ibes.py on small synthetic IBES and link data (no real data).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.stages import s05_ibes as s05


def estimates(rows):
    """rows: (ticker, statpers, measure, fpi, meanest, numest[, curcode])."""
    out = []
    for i, r in enumerate(rows):
        t, sp, meas, fpi, mean, n = r[:6]
        out.append({"ticker": t, "statpers": pd.Timestamp(sp), "measure": meas, "fpi": fpi, "meanest": mean,
                    "medest": mean, "stdev": 0.1, "highest": mean, "lowest": mean, "numest": n, "numup": 1,
                    "numdown": 0, "curcode": r[6] if len(r) > 6 else "USD", "fpedats": pd.Timestamp("2021-12-31"),
                    "actual": 1.0, "anndats_act": pd.Timestamp("2022-02-01"), "row": i})
    return pd.DataFrame(out)


def links(rows):
    """rows: (ticker, permno, sdate, edate, score)."""
    return pd.DataFrame(rows, columns=["TICKER", "PERMNO", "sdate", "edate", "SCORE"]).assign(
        sdate=lambda d: pd.to_datetime(d["sdate"]), edate=lambda d: pd.to_datetime(d["edate"]))


def test_usd_only_fpi_0_to_3_and_duplicates_keep_more_analysts():
    ib = estimates([("AAA", "2021-01-14", "EPS", 1, 1.0, 3), ("AAA", "2021-01-14", "EPS", 1, 2.0, 5),   # dup
                    ("AAA", "2021-01-14", "EPS", 4, 9.0, 2),                                            # FY4
                    ("BBB", "2021-01-14", "EPS", 1, 1.5, 4, "CAD")])                                     # non-USD
    out, info = s05.filter_and_dedupe(ib)
    assert out["meanest"].tolist() == [2.0]
    assert (info["duplicates_dropped"], info["fpi_outside_0_3_dropped"], info["non_usd_dropped"]) == (1, 1, 1)


def test_conflicting_actuals_are_blanked():
    ib = estimates([("AAA", "2021-06-17", "EPS", 1, 1.0, 3), ("AAA", "2021-07-15", "EPS", 1, 1.1, 3)])
    ib.loc[1, "actual"] = 1.2                         # IBES gives two actuals for the same fiscal year
    out, info = s05.filter_and_dedupe(ib)
    assert out["actual"].isna().all() and info["conflicting_actuals_blanked"] == 2


def test_link_windows_scores_and_open_ended_latest_link():
    lk = s05.clean_links(links([("AAA", 1, "2010-01-01", "2015-12-31", 1),     # old link, closed
                                ("AAA", 1, "2018-01-01", "2025-12-31", 1),     # latest link: open-ended
                                ("CCC", 3, "2010-01-01", "2030-12-31", 4),     # score 4: not used
                                ("DDD", 9, "2010-01-01", "2030-12-31", 1)]),   # not one of our permnos
                         ours={1, 3})
    assert lk["ticker"].tolist() == ["AAA", "AAA"]
    assert lk["edate_used"].tolist() == [pd.Timestamp("2015-12-31"), pd.Timestamp(s05.IBES_OPEN_LINK_END)]
    ib = estimates([("AAA", d, "EPS", 1, 1.0, 3) for d in ("2012-05-17", "2016-06-16", "2026-08-20")]
                   + [("CCC", "2012-05-17", "EPS", 1, 1.0, 3)])
    out, _ = s05.link(ib, lk)
    assert out["statpers"].dt.strftime("%Y-%m-%d").tolist() == ["2012-05-17", "2026-08-20"]   # 2016: between links


def test_two_tickers_for_one_permno_keep_the_better_link():
    lk = s05.clean_links(links([("AAA", 1, "2010-01-01", "2030-12-31", 2), ("AAB", 1, "2010-01-01", "2030-12-31", 1)]),
                         ours={1})
    out, dropped = s05.link(estimates([("AAA", "2021-01-14", "EPS", 1, 1.0, 9), ("AAB", "2021-01-14", "EPS", 1, 2.0, 3)]),
                            lk)
    assert dropped == 1 and out["ticker"].tolist() == ["AAB"]


def test_long_term_growth_to_decimal_and_neighbour_rule_for_totals():
    sal = [400.0, 410.0, 52_000.0, 420.0, 430.0, 60_000.0]          # a glitch in the middle and one at the end
    capx = [0.2, 0.3, 30.0, 32.0, 35.0, 40.0]                       # real growth from near zero: not flagged
    ib = estimates([("AAA", "2021-01-14", "EPS", 0, 15.0, 3)]
                   + [("AAA", f"2021-{m + 2:02d}-14", "SAL", 1, v, 3) for m, v in enumerate(sal)]
                   + [("BBB", f"2021-{m + 2:02d}-14", "CPX", 1, v, 3) for m, v in enumerate(capx)])
    ib["fpi"] = ib["fpi"].astype("int8")
    out, counts, sample = s05.convert_units(ib)
    assert out.loc[0, "meanest"] == pytest.approx(0.15) and out.loc[0, "stdev"] == pytest.approx(0.001)
    assert counts == {"SAL": 2}
    flagged = out[out["suspect_total"]]
    assert flagged["statpers"].dt.month.tolist() == [4, 7] and flagged["meanest"].isna().all()
    assert sample[0] == {"ticker": "AAA", "statpers": "2021-07-14", "measure": "SAL", "fpi": 1, "meanest": 60_000.0,
                         "previous": 430.0, "next": None}
    assert out.loc[out["ticker"] == "BBB", "meanest"].notna().all()


def test_crsp_basis_factor_lines_up_a_split():
    # 4-for-1 split on 2022-07-22: CRSP cfacpr 4 before, 1 after; IBES restates the old estimates to new shares
    daily = pd.DataFrame({"date": pd.to_datetime(["2022-07-14", "2022-07-22", "2022-09-30"]), "permno": 1,
                          "prc": [150.0, 36.0, 25.0], "cfacpr": [4.0, 1.0, 1.0]})
    ib = estimates([("AAA", "2022-07-14", "EPS", 1, -0.92, 5), ("AAA", "2022-07-18", "EPS", 1, -0.92, 5),
                    ("AAA", "2022-08-18", "EPS", 1, -0.92, 5)]).assign(permno=1)
    out = s05.add_crsp_basis(ib, daily).set_index("statpers")
    first = out.loc["2022-07-14"]
    assert first.crsp_basis_factor == 4.0 and first.price_statpers == 150.0
    assert first.meanest * first.crsp_basis_factor / first.price_statpers == pytest.approx(-0.92 / 37.5)
    assert out.loc["2022-07-18", "price_statpers"] == 150.0                      # last trading day within 7 days
    assert np.isnan(out.loc["2022-08-18", "price_statpers"])                     # 27 days old: no price


def test_scale_rule_uses_company_size_known_at_statpers():
    ib = pd.DataFrame({
        "ticker": "ABC", "permno": 1, "measure": ["SAL", "SAL", "EBI", "SAL", "EPS", "SAL"], "fpi": 1,
        "statpers": pd.to_datetime(["2020-02-20", "2020-06-18", "2020-06-18", "2019-01-17", "2020-06-18",
                                    "2020-06-18"]),
        "meanest": [500.0, 2e6, 150_000.0, 2e6, 9e9, 5e5], "medest": 1.0, "stdev": 1.0, "highest": 1.0,
        "lowest": 1.0, "actual": [np.nan, np.nan, np.nan, np.nan, np.nan, 3e5]})
    ib.loc[5, "permno"] = 2                                               # permno 2: no size known at all
    fund = pd.DataFrame({"permno": 1, "report_date": pd.to_datetime(["2020-01-15", "2020-07-15"]),
                         "revenue_ttm": [1000.0, 1e9], "atq": [800.0, 1e9]})  # the July filing must not be used
    caps = pd.DataFrame({"permno": 1, "date": pd.to_datetime(["2020-02-19", "2020-06-17"]),
                         "company_mkt_cap_musd": [1500.0, 1400.0]})
    out, info = s05.apply_scale_rule(ib, fund, caps)
    # size at 2020-06-18: max(1000, 800, 1400) = 1400; 100x = 140,000
    assert out.loc[1, "suspect_scale"] and np.isnan(out.loc[1, "meanest"])          # 2e6 > 140,000
    assert out.loc[2, "suspect_scale"] and np.isnan(out.loc[2, "medest"])           # 150,000 > 140,000
    assert not out.loc[0, "suspect_scale"] and out.loc[0, "meanest"] == 500.0
    assert not out.loc[3, "suspect_scale"]            # 2019: no fundamentals or price known yet, rule skipped
    assert not out.loc[4, "suspect_scale"]            # per-share measures are not totals
    assert not out.loc[5, "suspect_scale"]            # permno 2: no size known
    assert info["estimates_by_measure"] == {"EBI": 1, "SAL": 1} and info["no_size_known_rows"] == 2


def test_margin_rule_catches_ebit_and_capex_far_above_revenue_but_spares_losses_and_biotechs():
    # size is large (cap $50B), so the scale rule never fires here; only the margin rule can
    est = [("BIG", 1, "EBI", 5_000.0),      # 50x revenue of $100M: margin
           ("BIG", 1, "EBT", 1_500.0),      # 15x: kept
           ("BIG", 1, "CPX", 3_000.0),      # capex 30x revenue: margin
           ("BIG", 1, "EBI", -5_000.0),     # a loss far beyond revenue: never touched
           ("BIG", 1, "SAL", 5_000.0),      # sales are not in the margin rule
           ("BIO", 2, "EBT", 1_000.0)]      # pre-revenue biotech ($0.5M revenue): kept
    ib = pd.DataFrame([{"ticker": t, "permno": p, "measure": m, "fpi": 2, "statpers": pd.Timestamp("2021-03-18"),
                        "meanest": v, "medest": v, "stdev": 1.0, "highest": v, "lowest": v, "actual": np.nan}
                       for t, p, m, v in est])
    fund = pd.DataFrame({"permno": [1, 2], "report_date": pd.Timestamp("2021-02-10"), "revenue_ttm": [100.0, 0.5],
                         "atq": [400.0, 300.0]})
    caps = pd.DataFrame({"permno": [1, 2], "date": pd.Timestamp("2021-03-17"), "company_mkt_cap_musd": 50_000.0})
    out, info = s05.apply_scale_rule(ib, fund, caps)
    assert out["suspect_scale_reason"].tolist() == ["margin", None, "margin", None, None, None]
    assert np.isnan(out.loc[0, "meanest"]) and np.isnan(out.loc[2, "highest"])
    assert out.loc[3, "meanest"] == -5_000.0 and out.loc[5, "meanest"] == 1_000.0 and out.loc[1, "meanest"] == 1_500
    assert info["margin_by_measure"] == {"CPX": 1, "EBI": 1} and not info["estimates_by_measure"]
