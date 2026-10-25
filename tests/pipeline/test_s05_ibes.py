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


def test_long_term_growth_to_decimal_and_100x_totals():
    ib = estimates([("AAA", "2021-01-14", "EPS", 0, 15.0, 3)]
                   + [("AAA", f"2021-{m:02d}-14", "SAL", 1, 400.0, 3) for m in range(2, 12)]
                   + [("AAA", "2021-12-16", "SAL", 1, 52_000.0, 3)])                  # 130x the median
    ib["fpi"] = ib["fpi"].astype("int8")
    out, counts = s05.convert_units(ib)
    assert out.loc[0, "meanest"] == pytest.approx(0.15) and out.loc[0, "stdev"] == pytest.approx(0.001)
    assert counts == {"SAL": 1}
    last = out.iloc[-1]
    assert last.suspect_total and np.isnan(last.meanest) and np.isnan(last.highest)
    assert out.loc[out["measure"] == "SAL", "meanest"].notna().sum() == 10


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
