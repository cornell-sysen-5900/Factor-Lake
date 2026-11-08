"""
Tests for pipeline/stages/s00b_client.py and s00c_client_map.py on small synthetic lists and CRSP names (no real
data: invented companies, CUSIPs and permnos).
"""
import pandas as pd
import pytest

from pipeline.config import DataPaths
from pipeline.stages import s00b_client as s00b
from pipeline.stages import s00c_client_map as s00c

CLIENT = ('"Russell 2000 stocks as of 9/30, synthetic",,,,,\n'
          ",,,----- data -----,,\n"
          "Security Name,Ticker-Region,CUSIP,Russell 2000 Port. Weight,Date,Next FY Earns/P\n"
          "Zorblax Inc.,ALP-US,00000a101,0.5,9/30/2020,0.1\n"
          "Quiverly Corp,BET.XX1-US,#Calc,0.25,9/30/2020,\n"
          "Frobnitz Holdings,--,#Calc,0.25,9/30/2020,\n"
          ",,,,,\n"
          "Zorblax Inc.,ALP-US,00000A101,1.0,9/30/2021,0.2\n")


def write_client(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / s00b.CLIENT_FILE).write_text(CLIENT)


def test_client_file_cleaned(tmp_path):
    write_client(tmp_path)
    c = s00b.load_client_file(tmp_path / s00b.CLIENT_FILE)
    assert len(c) == 4                                                  # the blank row is dropped
    assert c["row"].tolist() == [5, 6, 7, 9]                           # pandas index + 5, the map's row key
    assert c["ticker"].tolist() == ["ALP", "BET.XX1", "--", "ALP"]
    assert c["method"].tolist() == ["cusip", "ticker", "ticker", "cusip"]
    assert c["cusip8"].tolist()[0] == "00000A10" and c["cusip8"].isna().tolist()[1:3] == [True, True]
    assert c["weight"].tolist() == [0.5, 0.25, 0.25, 1.0]
    assert c["year"].tolist() == [2020, 2020, 2020, 2021]
    assert c["date"].dt.strftime("%m-%d").unique().tolist() == ["09-30"]
    status = {k.name: k.status for k in s00b.run_checks(c)}
    assert status["every list is dated 9/30"] == "PASS" and status["rows per list within 1,800-2,150"] == "WARN"


def test_list_not_dated_9_30_fails(tmp_path):
    write_client(tmp_path)
    c = s00b.load_client_file(tmp_path / s00b.CLIENT_FILE)
    c.loc[0, "date"] = pd.Timestamp("2020-06-30")
    assert {k.name: k.status for k in s00b.run_checks(c)}["every list is dated 9/30"] == "FAIL"


@pytest.mark.parametrize("a, b, ok", [
    ("Zorblax Inc.", "ZORBLAX INC", True),
    ("Q R X Corp", "QRX CORP", True),                          # single letters joined
    ("First Zorblax", "FIRST QUIVERLY CORP", False),           # generic first word needs the second to match
    ("First Zorblax", "FIRST ZORBLAX PR", True),
    ("Wumpextech Industries", "WUMPEX TECH", True),            # one inside the other, 5+ letters
    ("Plonkly Air", "SNORKELO FOODS", False),
])
def test_name_rule(a, b, ok):
    assert s00c.name_ok(a, b) is ok


def names_history():
    """permno, start, end, issuer, ticker, cusip, shareclass."""
    rows = [
        (1, "2010-01-01", "2030-12-31", "ZORBLAX INC", "ALP", "00000A10", None),
        (2, "2010-01-01", "2021-06-30", "BLORPCO LTD", "BET", "99999X10", None),   # BET belongs to Blorpco on 9/30/20
        (3, "2021-07-01", "2030-12-31", "QUIVERLY CORP", "BET", "11111B10", None),     # ... Quiverly has it only later
        (3, "2015-01-01", "2021-06-30", "QUIVERLY CORP", "BTA", "11111B10", None),     # Quiverly under its old ticker
        (4, "2010-01-01", "2030-12-31", "WUMPEX CO", "DLT", "22222D10", "A"),       # two classes, one ticker
        (5, "2010-01-01", "2030-12-31", "WUMPEX CO", "DLT", "22222D20", "B"),
        (6, "2010-01-01", "2030-12-31", "GRIZZOP INC", "EPS", "33333E10", None),
        (7, "2010-01-01", "2030-12-31", "SNORKELO CORP", "ZET", "44444Z10", None),
    ]
    n = pd.DataFrame(rows, columns=["permno", "secinfostartdt", "secinfoenddt", "issuernm", "ticker", "cusip",
                                    "shareclass"])
    n["secinfostartdt"], n["secinfoenddt"] = pd.to_datetime(n["secinfostartdt"]), pd.to_datetime(n["secinfoenddt"])
    n["permco"] = n["permno"]
    n["sharetype"] = n["securitytype"] = None
    return s00c.Names(n[s00c.NAMES_COLUMNS])


def client_rows(rows):
    c = pd.DataFrame(rows, columns=["row", "name", "ticker", "cusip8"])
    c["date"] = pd.Timestamp("2020-09-30")
    c["year"] = 2020
    c["method"] = c["cusip8"].isna().map({True: "ticker", False: "cusip"})
    c["ticker_region"] = c["ticker"] + "-US"
    c["cusip9"] = c["cusip8"].fillna("#Calc")
    c["weight"] = 0.1
    return c


def test_matching_rules():
    c = client_rows([
        (1, "Zorblax Inc.", "ALP", "00000A10"),          # rule 1: CUSIP valid on 9/30
        (2, "Quiverly Corp", "BET.XX1", None),        # rule 2 fails (BET is Blorpco on 9/30); rule 4 finds it
        (3, "Wumpex Co Class B", "DLT", None),          # rule 5: two permnos, the class in the name
        (4, "Snorkelo Corp", "ZET", "33333E10"),       # rule 3: the CUSIP is Grizzop's, ticker + name say Snorkelo
        (5, "Unknownco Inc", "UNK", None),               # rule 7: nothing
    ])
    priced = {(2020, 3), (2020, 7)}
    out, info = s00c.build(c, names_history(), priced)
    got = out.set_index("row")[["final_permno", "final_method"]].astype(str).agg(" ".join, axis=1).to_dict()
    assert got == {1: "1 cusip", 2: "3 ticker_any_date", 3: "5 ticker_shareclass_B",
                   4: "7 cusip_corrected_by_ticker", 5: "<NA> unmatched"}
    assert out.set_index("row").loc[2, "status"] == "unmatched"           # BET.XX1 is no ticker on 9/30

    out, _ = s00c.build(c, names_history(), {(2020, 7)})                       # Quiverly has no price on 9/30
    r = out.set_index("row").loc[2]
    assert (pd.isna(r.final_permno), r.final_method, r.fallback_rejected) == (True, "unmatched",
                                                                               "no CRSP price on 9/30")
    gaps = s00c.known_gaps(out)
    assert "ticker fallback rejected" in gaps.set_index("name").loc["Quiverly Corp", "why"]


def test_same_permno_twice_keeps_the_name_match_else_the_cusip_row():
    c = client_rows([(1, "Zorblax Inc.", "ALP", None), (2, "Mismatched Name Corp", "XXX", "00000A10")])
    out, info = s00c.build(c, names_history(), set())
    got = out.set_index("row")[["final_permno", "final_method"]]
    assert got.loc[1, "final_permno"] == 1 and got.loc[2, "final_method"] == "dropped_duplicate"
    assert info["dropped_duplicates"] == 1

    c = client_rows([(1, "Zorblax Inc.", "ALP", None), (2, "Zorblax Inc.", "ALP", "00000A10")])    # both match
    out, _ = s00c.build(c, names_history(), set())
    assert out.set_index("row")["final_method"].to_dict() == {1: "dropped_duplicate", 2: "cusip"}


def test_september_prices_use_the_last_real_trading_day(tmp_path):
    csv = ("permno,DlyCalDt,DlyDelFlg,DlyPrc\n"
           "1,2020-09-29,N,10\n2,2020-09-29,N,5\n1,2020-09-30,N,10.5\n2,2020-09-30,Y,5\n"
           "3,2020-09-30,N,0\n4,2020-09-30,N,-2.5\n5,2020-10-01,N,1\n6,2020-09-15,N,1\n")
    (tmp_path / "ciz.csv").write_text(csv)
    assert s00c.september_prices(tmp_path / "ciz.csv") == {(2020, 1), (2020, 4)}


def test_run_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    write_client(raw / "client")
    (raw / "query_inputs").mkdir()
    (raw / "query_inputs" / "permno_list.txt").write_text("1\n3\n")
    (raw / s00c.NAMES_FILE).write_text(
        "permno,permco,secinfostartdt,secinfoenddt,cusip,ticker,issuernm,shareclass,securitytype,sharetype\n"
        "1,1,2010-01-01,2030-12-31,00000A10,ALP,ZORBLAX INC,,EQTY,NS\n"
        "3,3,2015-01-01,2030-12-31,11111B10,BET,QUIVERLY CORP,,EQTY,NS\n")
    (raw / "r2000_price_crsp_ciz.csv").write_text("permno,DlyCalDt,DlyDelFlg,DlyPrc\n1,2020-09-30,N,10\n"
                                                  "3,2020-09-30,N,4\n1,2021-09-30,N,11\n")
    paths = DataPaths.from_data_dir(tmp_path)
    s00b.run(paths)
    summary = s00c.run(paths)
    out = pd.read_parquet(paths.stage_dir("s00c_client_map") / s00c.OUTPUT)
    assert out.set_index("row")["final_permno"].astype("float").fillna(-1).to_dict() == {5: 1, 6: 3, 7: -1, 9: 1}
    assert summary["status"] == "WARN"                                     # the 2020 list matches 2 of 3 rows
    assert "Frobnitz Holdings" in (paths.stage_dir("s00c_client_map") / "known_gaps.md").read_text()
