"""
Tests for pipeline/stages/s02_prices.py on a small synthetic CIZ daily file (no real data).
"""
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.config import DataPaths
from pipeline.stages import s02_prices as s02

HEADER = ["permno", "PERMCO", "Ticker", "SICCD", "DlyCalDt", "DlyDelFlg", "DlyPrc", "DlyRet", "DlyVol", "ShrOut",
          "DlyOpen", "DlyHigh", "DlyLow", "DlyCumFacPr", "DlyCumFacShr"]


def ciz_rows(permno, dates, rets, prc=10.0, flag="N", sic=3571, **over):
    rows = []
    for d, r in zip(dates, rets):
        row = {"permno": permno, "PERMCO": permno + 50000, "Ticker": f"T{permno}", "SICCD": sic,
               "DlyCalDt": d.date(), "DlyDelFlg": flag, "DlyPrc": prc, "DlyRet": r, "DlyVol": 1000,
               "ShrOut": 2000, "DlyOpen": prc, "DlyHigh": prc * 1.02, "DlyLow": prc * 0.98, "DlyCumFacPr": 1.0,
               "DlyCumFacShr": 1.0}
        row.update(over)
        rows.append(row)
    return rows


def frame(rows):
    """CIZ rows -> the cleaned frame s02 works on (via a CSV round trip, like the real file)."""
    df = pd.DataFrame(rows, columns=HEADER)
    return df


def write_ciz(path, rows):
    frame(rows).to_csv(path, index=False)


def load(tmp_path, rows, delist=None):
    write_ciz(tmp_path / "ciz.csv", rows)
    px = s02.read_ciz(tmp_path / "ciz.csv")
    px, n_dup, n_conf = s02.drop_duplicates(px)
    dl = delist if delist is not None else pd.DataFrame({"permno": pd.Series([], dtype="int32"),
                                                          "delactiontype": pd.Series([], dtype=str)})
    px = s02.fill_delisting_returns(s02.clean_prices(px), dl)
    return px, n_dup, n_conf


DAYS = pd.bdate_range("2020-01-02", periods=300)


def test_duplicates_exact_dropped_conflicts_counted(tmp_path):
    rows = ciz_rows(1, DAYS[:3], [0.01, 0.02, 0.03])
    rows.append(dict(rows[1]))                                    # exact copy
    conflict = dict(rows[2])
    conflict["DlyPrc"] = 99.0                                     # same key, different price
    px, n_dup, n_conf = load(tmp_path, rows + [conflict])
    assert len(px) == 3 and n_dup == 2 and n_conf == 1
    assert px.loc[px.date == DAYS[2], "prc"].item() == 10.0       # first row in file order kept


def test_price_cleaning_and_missing_codes(tmp_path):
    rows = (ciz_rows(1, DAYS[:1], [0.01], prc=-12.5)              # bid/ask midpoint
            + ciz_rows(1, DAYS[1:2], [-99.0], prc=0.0)            # no price, -99 return code
            + ciz_rows(1, DAYS[2:3], [0.0], ShrOut=0))            # shares 0 = missing
    px, _, _ = load(tmp_path, rows)
    assert px["prc"].tolist()[0] == 12.5 and np.isnan(px["prc"].iloc[1])
    assert np.isnan(px["ret"].iloc[1])
    assert np.isnan(px["mkt_cap"].iloc[2])
    assert px["mkt_cap"].iloc[0] == pytest.approx(12.5 * 2000)    # $ thousands
    assert px["mkt_cap_musd"].iloc[0] == pytest.approx(25.0)


def test_delisting_row_return_kept_or_filled(tmp_path):
    dates = DAYS[:3]
    rows = (ciz_rows(1, dates[:2], [0.01, 0.01]) + ciz_rows(1, dates[2:], [0.25], flag="Y", prc=7.0, ShrOut="")
            + ciz_rows(2, dates[:2], [0.01, 0.01]) + ciz_rows(2, dates[2:], [""], flag="Y", prc=0.0, ShrOut="")
            + ciz_rows(3, dates[:2], [0.01, 0.01]) + ciz_rows(3, dates[2:], [""], flag="Y", prc=0.0, ShrOut=""))
    dl = pd.DataFrame({"permno": np.array([1, 2, 3], dtype="int32"), "delactiontype": ["MER", "GDR", "MER"]})
    px, _, _ = load(tmp_path, rows, dl)
    flag = px[px.is_delisting_row].set_index("permno")
    assert flag.loc[1, "ret"] == 0.25 and not flag.loc[1, "delist_ret_filled"]     # never added again
    assert flag.loc[2, "ret"] == -0.30 and flag.loc[2, "delist_ret_filled"]        # GDR
    assert flag.loc[3, "ret"] == 0.0 and flag.loc[3, "delist_ret_filled"]          # anything else
    assert flag["prc"].isna().all() and flag["mkt_cap"].isna().all()               # no price from a delisting row


def test_momentum_window_and_80pct_rule(tmp_path):
    r = 0.01
    rets = [r] * 40
    rets_gappy = list(rets)
    for i in (25, 26, 27, 28):          # 4 missing of the 21 rows ending at row 30: 17 >= 16.8, still valid
        rets_gappy[i] = ""
    rets_gappier = list(rets_gappy)
    rets_gappier[29] = ""               # 5 missing: 16 < 16.8, NaN
    rows = ciz_rows(1, DAYS[:40], rets) + ciz_rows(2, DAYS[:40], rets_gappy) + ciz_rows(3, DAYS[:40], rets_gappier)
    px, _, _ = load(tmp_path, rows)
    px = s02.add_momentum(px)
    a = px[px.permno == 1].reset_index(drop=True)
    assert np.isnan(a.loc[20, "mom_1m"])                          # 21 rows need a 22nd row to difference against
    assert a.loc[21, "mom_1m"] == pytest.approx((1 + r) ** 21 - 1)
    assert a["mom_6m"].isna().all()                                # 126 rows not reached
    b = px[px.permno == 2].reset_index(drop=True)
    c = px[px.permno == 3].reset_index(drop=True)
    assert b.loc[30, "mom_1m"] == pytest.approx((1 + r) ** 17 - 1)
    assert np.isnan(c.loc[30, "mom_1m"])


def test_momentum_ignores_splits(tmp_path):
    # a 2-for-1 split halves the price and doubles cfacpr; the return on that day is the real return (0)
    rows = (ciz_rows(1, DAYS[:15], [0.01] * 15, prc=20.0) + ciz_rows(1, DAYS[15:30], [0.0] + [0.01] * 14, prc=10.0,
                                                                     DlyCumFacPr=2.0))
    px, _, _ = load(tmp_path, rows)
    px = s02.add_momentum(px)
    assert px["mom_1m"].iloc[29] == pytest.approx(1.01 ** 20 - 1)


def test_volatility_matches_hand_values(tmp_path):
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.02, 60).round(6)
    rows = ciz_rows(1, DAYS[:60], rets)
    px, _, _ = load(tmp_path, rows)
    px = s02.add_volatility(px)
    lr = np.log1p(rets)
    assert px["vol_close_1m"].iloc[40] == pytest.approx(np.std(lr[20:41], ddof=1) * np.sqrt(252))
    # rows before the 21st use everything so far once 80% of the window (17 rows) is there
    assert px["vol_close_1m"].iloc[16] == pytest.approx(np.std(lr[:17], ddof=1) * np.sqrt(252))
    assert np.isnan(px["vol_close_1m"].iloc[15])
    gk = 0.5 * np.log(1.02 / 0.98) ** 2                            # close = open, so the second term is 0
    assert px["vol_gk_1m"].iloc[40] == pytest.approx(np.sqrt(gk * 252))


def test_monthly_compounding_and_month_end_from_last_trading_day(tmp_path):
    jan = pd.bdate_range("2021-01-04", "2021-01-29")
    feb = pd.bdate_range("2021-02-01", "2021-02-26")
    # permno 1 trades through Feb 25 and delists Feb 26 (flag row carries a post-delisting price of 3.0)
    rows = (ciz_rows(1, jan, [0.01] * len(jan), prc=10.0) + ciz_rows(1, feb[:-1], [0.02] * (len(feb) - 1), prc=11.0)
            + ciz_rows(1, feb[-1:], [-0.5], flag="Y", prc=3.0, ShrOut="")
            # permno 2 trades through Jan 29; its flag row falls in February alone
            + ciz_rows(2, jan, [0.0] * len(jan), prc=8.0) + ciz_rows(2, feb[:1], [""], flag="Y", prc=0.0, ShrOut="")
            # permno 3 trades every day, so Feb 26 is a trading day in the calendar
            + ciz_rows(3, jan.append(feb), [0.0] * (len(jan) + len(feb))))
    dl = pd.DataFrame({"permno": np.array([1, 2], dtype="int32"), "delactiontype": ["MER", "GDR"]})
    px, _, _ = load(tmp_path, rows, dl)
    px = s02.add_volatility(s02.add_momentum(px))
    cal = s02.build_calendar(px)
    m = s02.build_monthly(px, cal).set_index(["permno", "month_end"])
    jan_end, feb_end = pd.Timestamp("2021-01-29"), pd.Timestamp("2021-02-26")
    assert cal["month_end"].tolist() == [jan_end, feb_end]
    one = m.loc[(1, feb_end)]
    assert one.ret_m == pytest.approx(1.02 ** (len(feb) - 1) * 0.5 - 1)          # delisting return compounded in
    assert (one.prc, one.last_trade_date, one.trading_days) == (11.0, feb[-2], len(feb) - 1)
    assert one.mkt_cap_musd == pytest.approx(11.0 * 2000 / 1000) and one.delisted_in_month
    assert m.loc[(1, jan_end)].ret_m == pytest.approx(1.01 ** len(jan) - 1)
    two = m.loc[(2, feb_end)]
    assert two.ret_m == -0.30 and two.trading_days == 0 and two.delist_ret_filled
    assert np.isnan(two.prc) and np.isnan(two.mkt_cap_musd) and pd.isna(two.last_trade_date)
    caps = s02.month_end_market_caps(px).set_index(["permno", "month"])
    assert caps.loc[(1, pd.Period("2021-02", "M")), "mkt_cap_musd"] == pytest.approx(22.0)   # same as monthly


def test_run_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    (raw / "query_inputs").mkdir(parents=True)
    days = pd.bdate_range("2020-01-02", "2020-04-30")
    rows = (ciz_rows(1, days, [0.001] * len(days)) + ciz_rows(2, days[:-5], [0.002] * (len(days) - 5))
            + ciz_rows(2, days[-5:-4], [""], flag="Y", prc=0.0, ShrOut=""))
    rows.append(dict(rows[3]))                                                 # one exact duplicate
    write_ciz(raw / "r2000_price_crsp_ciz.csv", rows)
    (raw / "delistings.csv").write_text("delactiontype,permno,delistingdt,delreasontype,delpaymenttype,delret\n"
                                        f"GDR,2,{days[-6].date()},UNAV,CASH,\n")
    (raw / "query_inputs" / "permno_list.txt").write_text("1\n2\n3\n")
    summary = s02.run(DataPaths.from_data_dir(tmp_path))
    out = tmp_path / "processed" / "s02_prices"
    status = {c["name"]: c["status"] for c in summary["checks"]}
    for name in ("conflicting duplicate rows", "one delisting row per delisted permno, and it is the last row",
                 "delisting rows = delistings.csv", "every delisting falls in exactly one month",
                 "filled delisting returns = blank delret in delistings.csv",
                 "ret_m equals the compounded daily returns (25 random permnos)",
                 "no month-end price or market cap from a delisting row", "calendar has every month"):
        assert status[name] == "PASS", name
    assert status["permnos in permno_list.txt have CIZ rows"] == "WARN"         # permno 3 has no rows
    daily = pd.read_parquet(out / "daily")
    assert len(daily) == len(rows) - 1 and daily["is_delisting_row"].sum() == 1
    assert sorted(p.name for p in (out / "daily").iterdir()) == ["year=2020"]
    monthly = pd.read_parquet(out / "monthly.parquet")
    assert len(monthly) == 8 and monthly["delist_ret_filled"].sum() == 1
    dl = pd.read_parquet(out / "delistings.parquet")
    assert dl["delret_used"].tolist() == [-0.30]
    assert summary["tables"]["monthly.parquet"]["stats"]["ret_m"]["count"] == 8
    assert json.loads((out / "summary.json").read_text())["delisting_fills"] == {"total": 1, "gdr_minus_30pct": 1,
                                                                                 "zero": 0}
