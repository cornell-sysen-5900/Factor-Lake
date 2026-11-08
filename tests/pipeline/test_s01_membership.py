"""
Tests for pipeline/stages/s01_membership.py on small synthetic IWM holdings (no real data).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline import config
from pipeline.config import DataPaths
from pipeline.stages import s00c_client_map as s00c
from pipeline.stages import s01_membership as s01

D = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31", "2020-06-30"])


def holdings(rows):
    """rows: (report_dt index, permno or None, cusip, ticker, name, weight). eff_dt = report_dt + 20 days."""
    h = pd.DataFrame(rows, columns=["k", "permno", "cusip", "ticker", "security_name", "percent_tna"])
    h["report_dt"] = D[h["k"]]
    h["eff_dt"] = h["report_dt"] + pd.Timedelta(days=20)
    h["permno"] = h["permno"].astype("float64")
    h["nbr_shares"] = 100.0
    h["line"] = np.arange(len(h))
    return h.drop(columns="k")


def test_cusip_fill_and_known_gaps():
    h = holdings([
        (0, 101, "AAA11111", "AAA", "ZORBLAX INC", 0.5),
        (1, 101, "AAA11111", "AAA", "ZORBLAX INC", 0.5),
        (2, None, "AAA11111", "AAA", "ZORBLAX INC", 0.5),       # filled from the same CUSIP
        (0, None, "GGG22222", "GGG", "GAMMA CORP", 0.1),      # no permno anywhere: a known gap
        (1, None, "GGG22222", "GGG", "GAMMA CORP", 0.2),
        (0, None, "CSH00000", "USD", "CASH COLLATERAL USD", 1.0),   # not a stock
        (0, None, None, None, "S&P E-MINI FUTURE", 0.0),
    ])
    h, fill = s01.fill_permno_from_cusip(h)
    assert fill == {"lines": 1, "securities": 1, "cusips_with_two_permnos": []}
    assert h.loc[2, "permno"] == 101 and h.loc[2, "permno_from_cusip"]
    gaps = s01.known_gaps(h)
    assert gaps["cusip"].tolist() == ["GGG22222"]
    assert (gaps.loc[0, "snapshots"], gaps.loc[0, "max_weight"]) == (2, 0.2)


def test_snapshot_sums_share_classes_and_names_from_largest_line():
    rows = [(0, 7, "BBB0000A", "BBBA", "BETA CL A", 0.30), (0, 7, "BBB0000B", "BBBB", "BETA CL B", 0.70)]
    for order in (rows, rows[::-1]):                          # same result whatever the row order
        h = holdings(order)
        h.loc[h["ticker"] == "BBBB", "eff_dt"] -= pd.Timedelta(days=3)
        h, _ = s01.fill_permno_from_cusip(h)
        snap = s01.build_snapshots(h)
        assert len(snap) == 1
        r = snap.iloc[0]
        assert (r.ticker, r.cusip, r.n_lines) == ("BBBB", "BBB0000B", 2)
        assert r.weight == pytest.approx(1.0)
        assert r.eff_dt == D[0] + pd.Timedelta(days=17)


def test_equal_weights_break_ties_on_shares():
    # a SPAC and its new name listed on the same date with the same weight: the line with more shares wins
    for order in ([0, 1], [1, 0]):
        rows = [(0, 9, "OLD00000", "OLD", "OLD SPAC CORP", 0.02), (0, 9, "NEW00000", "NEW", "NEW NAME INC", 0.02)]
        h = holdings([rows[i] for i in order])
        h.loc[h["ticker"] == "NEW", "nbr_shares"] = 101.0
        h, _ = s01.fill_permno_from_cusip(h)
        assert s01.build_snapshots(h)["ticker"].item() == "NEW"


def snaps(present, weight=0.5, permno=1):
    return pd.DataFrame({"report_dt": D[present], "eff_dt": D[present] + pd.Timedelta(days=20), "permno": permno,
                         "weight": weight, "ticker": "T", "security_name": "N", "in_universe": True})


def test_one_missing_snapshot_is_bridged_two_start_a_new_period():
    # all 6 dates exist (another stock is in every snapshot)
    base = snaps(list(range(6)), permno=999)
    bridged = s01.build_periods(pd.concat([base, snaps([0, 1, 3], permno=1)]))
    split = s01.build_periods(pd.concat([base, snaps([0, 1, 4], permno=2)]))
    p1, p2 = bridged[bridged.permno == 1], split[split.permno == 2]
    assert len(p1) == 1 and p1.iloc[0].index_through == D[3] and p1.iloc[0].n_snapshots == 3
    assert len(p2) == 2 and p2.index_effective.tolist() == [D[0], D[4]]
    assert p2.first_known.tolist() == [D[0] + pd.Timedelta(days=20), D[4] + pd.Timedelta(days=20)]


def caps_for(snap, values):
    return pd.DataFrame({"permno": list(values), "month": snap["report_dt"].iloc[0].to_period("M"),
                         "mkt_cap_musd": list(values.values())})


@pytest.mark.parametrize("big, weight, excluded", [
    (310.0, 0.04, True),       # 3.1x the next largest, weight below 0.05%
    (290.0, 0.04, False),      # only 2.9x
    (310.0, 0.05, False),      # big but with a weight that fits its size (a real member that grew, e.g. SMCI)
])
def test_rule_a_market_cap_above_3x_and_small_weight(big, weight, excluded):
    snap = pd.concat([snaps([0], permno=p) for p in (1, 2, 3)]).assign(weight=1.0)
    snap.loc[snap.permno == 1, "weight"] = weight
    for p in (1, 2, 3):            # long histories, so rule b never applies
        snap = pd.concat([snap, snaps([1, 2, 3, 4], permno=p).assign(weight=1.0)])
    out = s01.apply_non_member_rules(snap.reset_index(drop=True), caps_for(snap, {1: big, 2: 100.0, 3: 50.0}))
    first = out[out.report_dt == D[0]].set_index("permno")
    assert first.loc[1, "in_universe"] is not excluded and first.loc[1, "exclusion_rule"] == ("a" if excluded else "")
    assert first.loc[1, "max_other_mkt_cap_musd"] == 100.0 and first.loc[2, "max_other_mkt_cap_musd"] == big
    assert first.loc[[2, 3], "in_universe"].all()


@pytest.mark.parametrize("present, weights, cap, excluded", [
    ([0, 1, 2], [0.0, 0.0, 0.004], 50.0, True),          # 3 snapshots, all 0.00% after rounding, above p95
    ([0, 1, 2], [0.0, 0.0, 0.0], 3.0, False),            # same, but a small stock (below p95)
    ([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0], 50.0, False),   # 4 snapshots
    ([0, 1, 2], [0.0, 0.01, 0.0], 50.0, False),          # one non-zero weight
])
def test_rule_b_few_zero_weight_snapshots_and_large(present, weights, cap, excluded):
    others = pd.concat([snaps(list(range(6)), permno=p) for p in range(100, 120)])     # caps 1 to 20
    stock = snaps(present, permno=5).assign(weight=weights)
    snap = pd.concat([others, stock]).reset_index(drop=True)
    caps = pd.DataFrame([(p, d.to_period("M"), float(p - 99)) for p in range(100, 120) for d in D]
                        + [(5, D[k].to_period("M"), cap) for k in present], columns=["permno", "month", "mkt_cap_musd"])
    out = s01.apply_non_member_rules(snap, caps)
    s = out[out.permno == 5]
    assert (~s["in_universe"]).all() == excluded and (s["exclusion_rule"] == "b").all() == excluded
    assert (s["p95_mkt_cap_musd"] < 50).all()                                      # p95 of 1-20 (+ the stock)
    periods = s01.build_periods(out)
    assert periods.loc[periods.permno == 5, "excluded_non_member"].item() == excluded


def names_history(rows):
    """rows: (permno, start, end, cusip, ticker, issuer); one names row each."""
    n = pd.DataFrame(rows, columns=["permno", "secinfostartdt", "secinfoenddt", "cusip", "ticker", "issuernm"])
    n["secinfostartdt"], n["secinfoenddt"] = pd.to_datetime(n["secinfostartdt"]), pd.to_datetime(n["secinfoenddt"])
    n["permco"] = n["permno"]
    n["shareclass"] = n["sharetype"] = n["securitytype"] = None
    return s00c.Names(n[s00c.NAMES_COLUMNS])


@pytest.mark.parametrize("year, day", [(2002, "2002-06-28"), (2006, "2006-06-23"), (2007, "2007-06-22"),
                                       (2013, "2013-06-28"), (2018, "2018-06-22")])
def test_reconstitution_is_the_last_june_friday_unless_29th_or_30th(year, day):
    assert str(config.reconstitution_date(year)) == day


def test_list_in_force_switches_at_the_first_month_end_after_the_reconstitution():
    ts = pd.Timestamp
    assert config.client_list_year(ts("2003-05-30")) == 2002
    assert config.client_list_year(ts("2003-06-30")) == 2003            # reconstitution 2003-06-27
    assert config.client_list_year(ts("2002-06-28")) == 2001            # reconstitution on June's last trading day
    assert config.client_list_year(ts("2002-07-31")) == 2002
    assert config.client_list_year(ts("2003-12-31").date()) == 2003


def client_map(rows):
    """rows: (list year, permno, weight)."""
    c = pd.DataFrame(rows, columns=["year", "final_permno", "weight"])
    c["date"] = pd.to_datetime(c["year"].astype(str) + "-09-30")
    return c.assign(final_permno=c["final_permno"].astype("Int64"), cusip8=None, ticker=None, name=None)


def test_list_members_count_while_trading_and_ipos_from_september():
    names = names_history([(1, "2010-01-01", "2030-12-31", "A0000001", "AAA", "ZORBLAX"),
                           (2, "2019-07-15", "2030-12-31", "B0000001", "BBB", "QUIVERLY"),      # IPO after June
                           (3, "2010-01-01", "2019-08-15", "C0000001", "CCC", "FROBNITZ")])    # stops trading
    first = pd.Series({1: pd.Timestamp("2010-01-04"), 2: pd.Timestamp("2019-07-15"), 3: pd.Timestamp("2010-01-04")})
    lists = s01.ClientLists(client_map([(2019, 1, 0.5), (2019, 2, 0.3), (2019, 3, 0.2), (2018, 1, 1.0)]), names, first)
    held = {m: sorted(lists.at(pd.Timestamp(m))[0]["permno"]) for m in ("2019-06-28", "2019-07-31", "2019-08-30",
                                                                         "2019-09-30")}
    assert held == {"2019-06-28": [1], "2019-07-31": [1, 3], "2019-08-30": [1], "2019-09-30": [1, 2]}
    _, st = lists.at(pd.Timestamp("2019-08-30"))
    assert (st["list_year"], st["not_trading"], st["ipo_after_reconstitution"], st["ipo_held_back"]) == (2019, 1, 1, 1)


def test_client_list_snapshots_and_fill(monkeypatch):
    monkeypatch.setattr(s01, "CLIENT_LIST_FIRST_MONTH", "2019-10")
    monkeypatch.setattr(s01, "IWM_FIRST_MONTH", "2020-01")
    monkeypatch.setattr(s01, "CLIENT_FILL_LAST_MONTH", "2020-02")
    names = names_history([(p, "2010-01-01", "2030-12-31", f"X000000{p}", f"T{p}", f"CO {p}") for p in (1, 2, 3)])
    first = pd.Series({p: pd.Timestamp("2010-01-04") for p in (1, 2, 3)})
    lists = s01.ClientLists(client_map([(2019, 1, 0.6), (2019, 2, 0.4)]), names, first)
    month_ends = pd.Series(pd.to_datetime(["2019-10-31", "2019-11-29", "2019-12-31"]),
                           index=pd.period_range("2019-10", "2019-12", freq="M"))
    cl, _ = s01.client_list_snapshots(lists, month_ends)
    assert cl.groupby("report_dt")["permno"].apply(sorted).tolist() == [[1, 2]] * 3
    assert (cl["eff_dt"] == cl["report_dt"]).all() and set(cl["membership_source"]) == {"client_list"}

    iwm = snaps([0, 1, 2], permno=1).assign(membership_source="iwm")          # 2020-01..03; permno 2 is hidden
    iwm.loc[iwm.report_dt == D[0], "eff_dt"] = D[0] + pd.Timedelta(days=40)
    iwm = pd.concat([iwm, snaps([0], permno=3).assign(membership_source="iwm")])
    fill, st = s01.client_fill(iwm, lists)
    assert fill[["report_dt", "permno"]].values.tolist() == [[D[0], 2], [D[1], 2]]    # not 2020-03: fill ends 02
    assert fill.loc[fill.report_dt == D[0], "eff_dt"].item() == D[0] + pd.Timedelta(days=40)
    assert [x["fill"] for x in st] == [1, 1]
    rows = s01.with_ids(fill, names.n)
    assert rows["ticker"].tolist() == ["T2", "T2"] and (rows["n_lines"] == 1).all()


def test_periods_never_mix_sources():
    a = snaps([0, 1, 2], permno=1).assign(membership_source="client_list")
    b = snaps([3, 4, 5], permno=1).assign(membership_source="iwm")
    p = s01.build_periods(pd.concat([a, b]).reset_index(drop=True))
    assert p["membership_source"].tolist() == ["client_list", "iwm"]
    assert p["index_through"].tolist()[0] == D[2] and p["index_effective"].tolist()[1] == D[3]


def test_gap_fixes():
    gaps = pd.DataFrame({"cusip": ["AAA11111", "BBB22222", "CCC33333", "DDD44444", "EEE55555"],
                         "security_name": ["ZORBLAX INC", "QUIVERLY CORP", "FROBNITZ LTD", "WUMPEX INC", "GRIZZOP INC"],
                         "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                         "first": pd.to_datetime(["2015-01-31"] * 5), "last": pd.to_datetime(["2016-12-31"] * 5),
                         "snapshots": 24, "max_weight": 0.1})
    names = names_history([(1, "2010-01-01", "2030-12-31", "AAA11111", "AAA", "ZORBLAX INC"),   # trades then: CRSP
                           (2, "2001-01-01", "2005-12-31", "BBB22222", "BBB", "QUIVERLY CORP"),   # traded long before
                           (3, "2001-01-01", "2005-12-31", "CCC33333", "CCC", "FROBNITZ LTD"),
                           (5, "2010-01-01", "2030-12-31", "EEE55555", "EEE", "GRIZZOP INC")])
    cmap = client_map([(2015, 2, 0.1), (2002, 3, 0.1)])
    cmap["cusip8"] = ["BBB22222", "CCC33333"]
    h = pd.DataFrame({"permno": np.nan, "cusip": gaps["cusip"], "nbr_shares": [500.0, 500.0, 500.0, 500.0, 1.0]})
    fx = s01.gap_fixes(gaps, h, names, cmap).set_index("cusip")
    assert fx["fix_permno"].astype("float").fillna(-1).to_dict() == {"AAA11111": 1, "BBB22222": 2, "CCC33333": -1,
                                                                    "DDD44444": -1, "EEE55555": -1}
    assert fx.loc["AAA11111", "fix_source"] == "crsp_cusip"
    assert fx.loc["BBB22222", "fix_source"] == "client_map (cusip)"                # 9/30/2015 within the IWM dates
    assert fx.loc["CCC33333", "fix_source"] == "CUSIP in CRSP, permno not trading on the IWM dates"   # 2002: too far
    assert fx.loc["DDD44444", "fix_source"] == "no fix"
    assert fx.loc["EEE55555", "fix_source"].startswith("stray line")
    fixed = s01.apply_gap_fixes(h, fx.reset_index())
    assert fixed["permno"].fillna(-1).tolist() == [1, 2, -1, -1, -1] and fixed["permno_from_gap_fix"].sum() == 2


CIZ_HEADER = "permno,DlyCalDt,DlyDelFlg,DlyPrc,ShrOut\n"


def test_run_writes_outputs_and_is_rerunnable(tmp_path, monkeypatch):
    # client lists for 2019-10..2019-12, IWM from 2020-01 with the fill until 2020-02 (synthetic months)
    monkeypatch.setattr(s01, "CLIENT_LIST_FIRST_MONTH", "2019-10")
    monkeypatch.setattr(s01, "IWM_FIRST_MONTH", "2020-01")
    monkeypatch.setattr(s01, "CLIENT_FILL_LAST_MONTH", "2020-02")
    raw = tmp_path / "raw"
    (raw / "query_inputs").mkdir(parents=True)
    (raw / s00c.NAMES_FILE).write_text(
        "permno,permco,secinfostartdt,secinfoenddt,cusip,ticker,issuernm,shareclass,securitytype,sharetype\n"
        + "".join(f"{p},{p},2010-01-01,2030-12-31,X000{p},T{p},CO {p},,EQTY,NS\n" for p in (101, 102, 103, 104)))
    cm_dir = tmp_path / "processed" / "s00c_client_map"
    cm_dir.mkdir(parents=True)
    client_map([(2019, 101, 0.5), (2019, 102, 0.3), (2019, 104, 0.2)]).to_parquet(cm_dir / s00c.OUTPUT)
    lines = ["crsp_portno,report_dt,eff_dt,percent_tna,nbr_shares,security_name,cusip,permno,ticker"]
    for i, d in enumerate(D):
        e = (d + pd.Timedelta(days=20)).date()
        lines += [f"1,{d.date()},{e},0.60,100,ZORBLAX INC,AAA11111,101,AAA",
                  f"1,{d.date()},{e},0.40,100,BETA INC,BBB22222,{'' if i == 2 else 102},BBB"]
    lines.append(f"1,{D[0].date()},{(D[0] + pd.Timedelta(days=20)).date()},0.00,5,HUGE CORP,HHH33333,103,HUG")
    (raw / "iwm_holdings_raw.csv").write_text("\n".join(lines) + "\n")
    days = list(pd.to_datetime(["2019-10-31", "2019-11-29", "2019-12-31"])) + list(D)
    ciz = [f"{p},{d.date()},N,{prc},{sh}"
           for d in days for p, prc, sh in ((101, 10, 1000), (102, 5, 1000), (103, 50, 100000), (104, 2, 1000))]
    (raw / "r2000_price_crsp_ciz.csv").write_text(CIZ_HEADER + "\n".join(ciz) + "\n")
    (raw / "query_inputs" / "permno_list.txt").write_text("101\n102\n103\n104\n")
    paths = DataPaths.from_data_dir(tmp_path)

    first = s01.run(paths)
    out = paths.stage_dir("s01_membership")
    snap1 = pd.read_parquet(out / "snapshots.parquet")
    second = s01.run(paths)
    assert pd.read_parquet(out / "snapshots.parquet").equals(snap1)
    assert first["tables"] == second["tables"]
    assert sorted(p.name for p in out.iterdir()) == ["excluded_non_members.parquet", "gap_fixes.parquet",
                                                     "known_gaps.md", "known_gaps.parquet", "periods.parquet",
                                                     "permno_list.txt", "snapshots.parquet", "summary.json"]
    assert (out / "permno_list.txt").read_text().split() == ["101", "102", "103", "104"]
    src = snap1.groupby("membership_source")["report_dt"].agg(["min", "max", "size"])
    assert src.loc["client_list", "size"] == 9 and src.loc["client_list", "max"] == pd.Timestamp("2019-12-31")
    assert src.loc["client_fill"].tolist() == [D[0], D[1], 2]                    # 104 is not in IWM: filled
    periods = pd.read_parquet(out / "periods.parquet")
    assert periods[periods.permno == 104]["membership_source"].tolist() == ["client_list", "client_fill"]
    assert first["status"] in ("PASS", "WARN")
    assert snap1.loc[(snap1.permno == 102) & (snap1.report_dt == D[2]), "permno_from_cusip"].item()
    excl = pd.read_parquet(out / "excluded_non_members.parquet")
    assert excl["permno"].tolist() == [103] and excl["rule"].tolist() == ["a+b"]     # $5bn vs $10m, 1 snapshot
    assert first["excluded_non_members"][0]["ticker"] == "HUG"
