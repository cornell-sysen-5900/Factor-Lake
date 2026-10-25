"""
Tests for pipeline/stages/s01_membership.py on small synthetic IWM holdings (no real data).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.config import DataPaths
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
        (0, 101, "AAA11111", "AAA", "ALPHA INC", 0.5),
        (1, 101, "AAA11111", "AAA", "ALPHA INC", 0.5),
        (2, None, "AAA11111", "AAA", "ALPHA INC", 0.5),       # filled from the same CUSIP
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


@pytest.mark.parametrize("big, excluded", [(310.0, True), (290.0, False)])
def test_rule_a_market_cap_above_3x_the_largest_other(big, excluded):
    snap = pd.concat([snaps([0], permno=p) for p in (1, 2, 3)]).assign(weight=1.0)
    for p in (1, 2, 3):            # long histories, so rule b never applies
        snap = pd.concat([snap, snaps([1, 2, 3, 4], permno=p).assign(weight=1.0)])
    out = s01.apply_non_member_rules(snap.reset_index(drop=True), caps_for(snap, {1: big, 2: 100.0, 3: 50.0}))
    first = out[out.report_dt == D[0]].set_index("permno")
    assert first.loc[1, "in_universe"] is not excluded and first.loc[1, "exclusion_rule"] == ("a" if excluded else "")
    assert first.loc[1, "max_other_mkt_cap_musd"] == 100.0 and first.loc[2, "max_other_mkt_cap_musd"] == big
    assert first.loc[[2, 3], "in_universe"].all()


@pytest.mark.parametrize("present, weights, excluded", [
    ([0, 1, 2], [0.0, 0.0, 0.004], True),          # 3 snapshots, all 0.00% after rounding
    ([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0], False),   # 4 snapshots
    ([0, 1, 2], [0.0, 0.01, 0.0], False),          # one non-zero weight
])
def test_rule_b_few_zero_weight_snapshots(present, weights, excluded):
    other = snaps(list(range(6)), permno=999)
    stock = snaps(present, permno=5).assign(weight=weights)
    out = s01.apply_non_member_rules(pd.concat([other, stock]).reset_index(drop=True),
                                     pd.DataFrame(columns=["permno", "month", "mkt_cap_musd"]))
    s = out[out.permno == 5]
    assert (~s["in_universe"]).all() == excluded and (s["exclusion_rule"] == "b").all() == excluded
    periods = s01.build_periods(out)
    assert periods.loc[periods.permno == 5, "excluded_non_member"].item() == excluded


CIZ_HEADER = "permno,DlyCalDt,DlyDelFlg,DlyPrc,ShrOut\n"


def test_run_writes_outputs_and_is_rerunnable(tmp_path):
    raw = tmp_path / "raw"
    (raw / "query_inputs").mkdir(parents=True)
    lines = ["crsp_portno,report_dt,eff_dt,percent_tna,nbr_shares,security_name,cusip,permno,ticker"]
    for i, d in enumerate(D):
        e = (d + pd.Timedelta(days=20)).date()
        lines += [f"1,{d.date()},{e},0.60,100,ALPHA INC,AAA11111,101,AAA",
                  f"1,{d.date()},{e},0.40,100,BETA INC,BBB22222,{'' if i == 2 else 102},BBB"]
    lines.append(f"1,{D[0].date()},{(D[0] + pd.Timedelta(days=20)).date()},0.00,5,HUGE CORP,HHH33333,103,HUG")
    (raw / "iwm_holdings_raw.csv").write_text("\n".join(lines) + "\n")
    ciz = [f"{p},{d.date()},N,{prc},{sh}" for d in D for p, prc, sh in ((101, 10, 1000), (102, 5, 1000), (103, 50, 100000))]
    (raw / "r2000_price_crsp_ciz.csv").write_text(CIZ_HEADER + "\n".join(ciz) + "\n")
    (raw / "query_inputs" / "permno_list.txt").write_text("101\n102\n103\n")
    paths = DataPaths.from_data_dir(tmp_path)

    first = s01.run(paths)
    out = paths.stage_dir("s01_membership")
    snap1 = pd.read_parquet(out / "snapshots.parquet")
    second = s01.run(paths)
    assert pd.read_parquet(out / "snapshots.parquet").equals(snap1)
    assert first["tables"] == second["tables"]
    assert sorted(p.name for p in out.iterdir()) == ["excluded_non_members.parquet", "known_gaps.md",
                                                     "known_gaps.parquet", "periods.parquet", "permno_list.txt",
                                                     "snapshots.parquet", "summary.json"]
    assert (out / "permno_list.txt").read_text().split() == ["101", "102", "103"]
    assert snap1.loc[(snap1.permno == 102) & (snap1.report_dt == D[2]), "permno_from_cusip"].item()
    excl = pd.read_parquet(out / "excluded_non_members.parquet")
    assert excl["permno"].tolist() == [103] and excl["rule"].tolist() == ["a+b"]     # $5bn vs $10m, 1 snapshot
    assert first["excluded_non_members"][0]["ticker"] == "HUG"
