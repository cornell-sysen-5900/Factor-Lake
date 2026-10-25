"""
Tests for pipeline/stages/s07_panel.py on small synthetic stage outputs (no real data).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.config import PRICE_BASED_FACTORS
from pipeline.stages import s07_panel as s07

T = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"])


def test_universe_uses_the_latest_snapshot_whose_every_line_is_effective():
    snaps = pd.DataFrame({
        "report_dt": pd.to_datetime(["2019-12-31", "2019-12-31", "2020-01-31", "2020-01-31"]),
        "eff_dt": pd.to_datetime(["2020-01-20", "2020-01-25", "2020-02-20", "2020-03-05"]),   # Jan report: Mar 5
        "permno": [1, 2, 1, 3], "weight": 0.5, "ticker": "T", "security_name": "N", "in_universe": [True, True, True, False]})
    cal = pd.DataFrame({"month_end": T})
    dates = s07.rebalance_dates(cal, snaps)
    assert dates.tolist() == list(T)                                   # first snapshot effective by 2020-01-31
    u = s07.build_universe(dates, snaps).set_index(["date", "permno"])
    assert u.loc[T[1]].index.tolist() == [1, 2]                        # Feb: Jan report not fully effective yet
    assert u.loc[T[2]].index.tolist() == [1]                           # Mar: Jan report; permno 3 excluded
    assert u.loc[(T[2], 1), "snapshot_eff_dt"] == pd.Timestamp("2020-03-05")
    assert not u.loc[(T[2], 1), "new_entrant"] and not u.loc[(T[0], 1), "new_entrant"]


def quarter(permno, end, report, **vals):
    end = pd.Timestamp(end)
    row = {"permno": permno, "gvkey": "001", "period_end": end, "report_date": pd.Timestamp(report),
           "fyearq": end.year, "fqtr": (end.month - 1) // 3 + 1, "is_usd": True, "gsector": 45, "ggroup": 4510, "gind": 451030, "gsubind": 45103010,
           **{c: 1.0 for c in (*s07.BLOCKS, *s07.STATEMENT_FACTORS, *PRICE_BASED_FACTORS)}}
    row.update(vals)
    return row


def test_fundamentals_as_of_late_filed_and_stale():
    fq = pd.DataFrame([quarter(1, "2019-06-30", "2019-08-10", f_roa=0.1),
                       quarter(1, "2019-09-30", "2019-11-10", f_roa=0.2),
                       quarter(1, "2019-03-31", "2019-12-15", f_roa=0.9),      # an old quarter filed late
                       quarter(2, "2018-12-31", "2019-02-10", f_roa=0.5)])     # never updated: goes stale
    u = pd.DataFrame({"date": pd.to_datetime(["2019-10-31", "2019-12-31", "2019-12-31", "2020-01-31"]),
                      "permno": [1, 1, 2, 2], "siccd": 7372.0})
    out, info = s07.add_fundamentals(u, fq)
    out = out.set_index(["permno", "date"])
    assert out.loc[(1, pd.Timestamp("2019-10-31")), "f_roa"] == 0.1        # Q3 not reported until Nov 10
    assert out.loc[(1, pd.Timestamp("2019-12-31")), "f_roa"] == 0.2        # the late-filed Q1 does not replace Q3
    assert info["late_filed_quarters_dropped"] == 1
    assert out.loc[(2, pd.Timestamp("2019-12-31")), "f_roa"] == 0.5        # 365 days old: still used
    stale = out.loc[(2, pd.Timestamp("2020-01-31"))]
    assert stale.fund_stale and np.isnan(stale.f_roa) and np.isnan(stale.book_equity) and info["stale_rows"] == 1
    assert out.loc[(1, pd.Timestamp("2019-12-31")), "sector_name"] == "Information Technology"


@pytest.mark.parametrize("sic, sector", [(2834, "Health Care"), (3845, "Health Care"), (2821, "Materials"),
                                         (2844, "Consumer Staples"), (1311, "Energy"), (6798, "Real Estate"),
                                         (6021, "Financials"), (3674, "Information Technology"),
                                         (3711, "Consumer Discretionary"), (3531, "Industrials"),
                                         (9995, "Unclassified"), (np.nan, "Unclassified")])
def test_sic_fallback_to_gics_names(sic, sector):
    assert s07.sic_to_sector(pd.Series([sic])).item() == sector


def test_rolling_beta_on_a_known_series():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 0.01, 300))
    y = 1.5 * x + 0.0002
    g = pd.Series(np.ones(300, dtype=int))
    b = s07.rolling_beta(y, x, g, 252, 0.8 * 252)
    assert b.iloc[-1] == pytest.approx(1.5) and np.isnan(b.iloc[200]) and b.iloc[202] == pytest.approx(1.5)
    y_gappy = y.copy()
    y_gappy.iloc[-252:-150] = np.nan                                      # only 150 of 252 days
    assert np.isnan(s07.rolling_beta(y_gappy, x, g, 252, 0.8 * 252).iloc[-1])


def panel_rows(**cols):
    base = {"date": T[0], "permno": 1, "gsector": 45, "gind": 451030, "prc": 20.0, "company_mkt_cap_musd": 1000.0,
            "company_cap_from_compustat": False, "fund_is_usd": True, "ev_claims": 200.0, "book_equity": 500.0, "ib_ttm": 50.0, "revenue_ttm": 800.0,
            "dv_ttm": 10.0, "fcf_ttm": 40.0, "ebitda_ttm": 120.0, "ebit_ttm": 90.0, "eps1_crsp": 1.0, "eps2_crsp": 1.25,
            "ebt1_meanest": 130.0, "sal1_meanest": 850.0, "ltg_meanest": 0.10,
            **{c: np.nan for c in s07.QE_FACTORS}}
    base.update(cols)
    return pd.DataFrame([base])


def test_ev_and_multiples_with_non_positive_denominators():
    u = s07.add_price_scaled(pd.concat([panel_rows(), panel_rows(permno=2, ib_ttm=-5.0, book_equity=-1.0),
                                        panel_rows(permno=3, ev_claims=-1500.0), panel_rows(permno=4, prc=0.005),
                                        panel_rows(permno=5, fund_is_usd=False)], ignore_index=True))
    assert u.loc[0, "ev"] == 1200.0 and u.loc[0, "f_ebitda_to_ev"] == pytest.approx(0.1)
    assert u.loc[0, "f_earnings_to_price"] == pytest.approx(0.05) and u.loc[0, "f_book_to_price"] == pytest.approx(0.5)
    assert np.isnan(u.loc[2, "ev"]) and np.isnan(u.loc[2, "f_ebitda_to_ev"])          # EV <= 0
    assert u.loc[[3, 4], list(s07.PRICE_SCALED)].isna().all().all()                    # sub-$0.01 and non-USD
    rv = s07.build_rv(u).set_index("permno")
    assert rv.loc[1, "pe"] == pytest.approx(20.0) and rv.loc[1, "pb"] == pytest.approx(2.0)
    assert rv.loc[1, "ev_ebitda"] == pytest.approx(10.0) and rv.loc[1, "fwd_pe_fy1"] == pytest.approx(20.0)
    assert rv.loc[1, "peg"] == pytest.approx(2.0) and rv.loc[1, "fwd_ev_sales"] == pytest.approx(1200 / 850)
    assert np.isnan(rv.loc[2, "pe"]) and np.isnan(rv.loc[2, "pb"])                    # negative earnings and book
    assert rv.loc[2, "earnings_yield"] == pytest.approx(-0.005)                        # yields may be negative


def test_percentile_ranks_need_5_peers():
    rows = [panel_rows(permno=p, ib_ttm=float(p), gind=111) for p in range(1, 7)]       # 6 stocks in industry 111
    rows += [panel_rows(permno=p, ib_ttm=float(p), gind=222) for p in range(7, 11)]     # 4 stocks in industry 222
    rv = s07.build_rv(s07.add_price_scaled(pd.concat(rows, ignore_index=True))).set_index("permno")
    assert rv.loc[1:6, "pe_pct_gind"].tolist() == pytest.approx([6 / 6, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6])
    assert rv.loc[7:10, "pe_pct_gind"].isna().all()
    assert rv["pe_pct_gsector"].notna().all()                                           # 10 stocks in sector 45
    peers = s07.build_peers(rv.reset_index()).set_index("gind")
    assert peers.loc[111, "n_peers"] == 6 and peers.loc[111, "pe_p50"] == pytest.approx(rv.loc[1:6, "pe"].median())
    assert np.isnan(peers.loc[222, "pe_p50"])


def test_forward_ep_on_the_crsp_basis_across_a_split():
    # IBES restated FY1 EPS -0.92 (post-split shares); at t before the 4-for-1 split CRSP cfacpr is 4, last is 1
    u = pd.DataFrame({"date": [pd.Timestamp("2022-06-30")], "permno": [1], "prc": [150.0], "cfacpr_t": [4.0],
                      "ev": [np.nan]})
    est = pd.DataFrame({"permno": 1, "statpers": pd.Timestamp("2022-06-16"), "measure": "EPS", "fpi": [1, 2],
                        "meanest": [-0.92, -0.5], "stdev": 0.1, "numest": 5.0, "numup": 1.0, "numdown": 3.0,
                        "cfacpr_last": 1.0, "fpedats": pd.Timestamp("2023-01-31"), "actual": np.nan,
                        "anndats_act": pd.NaT})
    out = s07.add_analyst(s07.add_estimates(u, est), s07.surprise_events(est, pd.DataFrame(
        columns=["date", "permno", "prc", "cfacpr"]).astype({"date": "datetime64[ns]", "permno": int,
                                                             "prc": float, "cfacpr": float})))
    r = out.iloc[0]
    assert r.f_forward_ep == pytest.approx(-0.92 * 4 / 150) and r.f_forward_ep_fy2 == pytest.approx(-0.5 * 4 / 150)
    assert r.f_eps_revision == pytest.approx(-0.4) and r.numest_fy1 == 5


def test_surprise_uses_the_price_before_the_announcement():
    est = pd.DataFrame({"permno": 1, "statpers": pd.to_datetime(["2022-01-20", "2022-02-17"]), "measure": "EPS",
                        "fpi": 1, "meanest": [1.0, 1.1], "stdev": 0.1, "numest": 5.0, "numup": 1.0, "numdown": 0.0,
                        "cfacpr_last": 1.0, "fpedats": pd.Timestamp("2021-12-31"), "actual": 1.3,
                        "anndats_act": pd.Timestamp("2022-03-01")})
    prices = pd.DataFrame({"date": pd.to_datetime(["2022-02-25", "2022-02-28", "2022-03-01"]), "permno": 1,
                           "prc": [40.0, 44.0, 99.0], "cfacpr": [2.0, 2.0, 2.0]})
    ev = s07.surprise_events(est, prices)
    # last consensus before the announcement 1.1; price of 2022-02-28 (not the announcement day), share basis x2
    assert ev["f_analyst_surprise"].item() == pytest.approx((1.3 - 1.1) * 2 / 44.0)


def test_fiscal_year_change_takes_the_new_numbering():
    # the quarter ending 2008-12-31 appears as fiscal 2008 Q4 (old) and fiscal 2009 Q2 (new), same report date
    old = quarter(1, "2008-12-31", "2009-02-12", fyearq=2008, fqtr=4, f_roa=-0.19)
    new = quarter(1, "2008-12-31", "2009-02-12", fyearq=2009, fqtr=2, f_roa=-0.18)
    for rows in ([old, new], [new, old]):
        out, _ = s07.add_fundamentals(pd.DataFrame({"date": [pd.Timestamp("2009-03-31")], "permno": [1],
                                                    "siccd": [7372.0]}), pd.DataFrame(rows))
        assert out["f_roa"].item() == -0.18


def test_company_cap_from_compustat_shares_on_the_crsp_basis():
    # company 10: Compustat 30M shares at 2019-12-31, before a 2-for-1 split (cfacshr 2 -> 1): 60M at t, CRSP 40M
    # company 20: Compustat 41M vs CRSP 40M (2.5% more): stays the CRSP market cap
    t = pd.Timestamp("2020-03-31")
    u = pd.DataFrame({"date": t, "permno": [1, 2], "permco": [10, 20], "fund_period_end": pd.Timestamp("2019-12-31"),
                      "cshoq": [30.0, 41.0], "cfacshr_t": 1.0, "company_shares_crsp_m": 40.0,
                      "company_main_prc": 10.0, "company_mkt_cap_musd": 400.0})
    me = pd.DataFrame({"permno": [1, 2], "month": pd.Period("2019-12", "M"), "cfacshr_t": [2.0, 1.0]})
    out = s07.add_company_cap(u, me).set_index("permno")
    assert out.loc[1, "company_shares_compustat_m"] == 60.0 and out.loc[1, "company_cap_from_compustat"]
    assert out.loc[1, "company_mkt_cap_musd"] == 600.0 and out.loc[1, "company_mkt_cap_crsp_musd"] == 400.0
    assert not out.loc[2, "company_cap_from_compustat"] and out.loc[2, "company_mkt_cap_musd"] == 400.0


def test_multiples_need_1m_denominators_and_1_cent_forward_eps():
    u = s07.add_price_scaled(pd.concat([panel_rows(), panel_rows(permno=2, ib_ttm=0.5, ebitda_ttm=0.9, eps1_crsp=0.005,
                                                                 ebt1_meanest=0.4, sal1_meanest=0.99),
                                        panel_rows(permno=3, ib_ttm=1.0, eps1_crsp=0.01)], ignore_index=True))
    rv = s07.build_rv(u).set_index("permno")
    assert rv.loc[2, ["pe", "ev_ebitda", "fwd_pe_fy1", "fwd_ev_ebitda", "fwd_ev_sales", "peg"]].isna().all()
    assert rv.loc[2, "ev_ebit"] == pytest.approx(1200 / 90)                            # 90 is above the floor
    assert rv.loc[3, "pe"] == pytest.approx(1000.0) and rv.loc[3, "fwd_pe_fy1"] == pytest.approx(2000.0)


def test_forward_yields_and_surprise_are_capped():
    u = pd.DataFrame({"f_forward_ep": [0.1, -6.0, 5.0], "f_forward_ep_fy2": [0.1, 0.2, 7.0],
                      "f_forward_cfp": [0.1, np.nan, 0.3], "f_analyst_surprise": [0.02, -1.5, 1.0]})
    out, counts = s07.cap_yields(u)
    assert counts == {"f_forward_ep": 1, "f_forward_ep_fy2": 1, "f_forward_cfp": 0, "f_analyst_surprise": 1}
    assert np.isnan(out.loc[1, "f_forward_ep"]) and out.loc[2, "f_forward_ep"] == 5.0
    assert np.isnan(out.loc[1, "f_analyst_surprise"]) and out.loc[2, "f_analyst_surprise"] == 1.0
