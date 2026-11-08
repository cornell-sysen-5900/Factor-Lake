"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s00b_client.py
PURPOSE: Stage s00b_client: the client's Russell 2000 lists (one per 9/30, from FactSet) as a clean table.

In:  <client folder>/FR2000 Annual Quant Data(Data).csv   (licensed: never committed, nor anything derived from it)
Out: processed/s00b_client/client_lists.parquet   one row per list row: row, date, year, name, ticker_region,
                                                   ticker, cusip9, cusip8, method, weight
     processed/s00b_client/summary.json

Rules (pipeline/PLAN.md, s00b_client):
    1. the header is on the third line; rows that are blank in every column are dropped. `row` = the pandas row
       index + 5, the key the client map uses
    2. ticker = Ticker-Region without its "-US" suffix; CUSIP "#Calc" (FactSet has none) -> method "ticker",
       otherwise method "cusip" with cusip8 = the first 8 characters, upper case
    3. weight = Russell 2000 Port. Weight on that 9/30 (percent of the index); only the identity columns and the
       weight are kept, the FactSet factor columns are not used
"""

from __future__ import annotations

import pandas as pd

from pipeline.checks import WARN, Check, check
from pipeline.config import (
    CLIENT_COLUMNS,
    CLIENT_FILE,
    CLIENT_LIST_MONTH_DAY,
    CLIENT_NO_CUSIP,
    CLIENT_NO_TICKER,
    SNAPSHOT_SIZE_RANGE,
    DataPaths,
    RawFile,
)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary

STAGE = "s00b_client"
OUTPUT = "client_lists.parquet"
SPEC = RawFile(CLIENT_FILE, "date", tuple(CLIENT_COLUMNS), skip_lines=2)
ROW_OFFSET = 5          # the client map's row key: pandas index + 5
WEIGHT_SUM_RANGE = (99.0, 101.0)


def load_client_file(path) -> pd.DataFrame:
    """Rules 1-3. Everything is read as text; the weight is converted to a number."""
    raw = pd.read_csv(path, header=SPEC.skip_lines, dtype=str, encoding="utf-8", keep_default_na=False,
                      na_values=[""])
    raw = raw.dropna(how="all")
    cols = {c: CLIENT_COLUMNS[c.strip().lower()] for c in raw.columns if c.strip().lower() in CLIENT_COLUMNS}
    c = raw[list(cols)].rename(columns=cols)
    out = pd.DataFrame({"row": (raw.index + ROW_OFFSET).astype("int64"),
                        "date": pd.to_datetime(c["date"], format="%m/%d/%Y"),
                        "name": c["name"].str.strip(),
                        "ticker_region": c["ticker_region"].str.strip(),
                        "cusip9": c["cusip9"].str.strip()})
    out["year"] = out["date"].dt.year.astype("int32")
    out["ticker"] = out["ticker_region"].str.replace(r"-US$", "", regex=True)
    out["method"] = (out["cusip9"] == CLIENT_NO_CUSIP).map({True: "ticker", False: "cusip"})
    out["cusip8"] = out["cusip9"].where(out["method"] == "cusip").str[:8].str.upper()
    out["weight"] = pd.to_numeric(c["weight"].str.strip(), errors="coerce").to_numpy()
    cols = ["row", "date", "year", "name", "ticker_region", "ticker", "cusip9", "cusip8", "method", "weight"]
    return out[cols].reset_index(drop=True)


def run_checks(c: pd.DataFrame) -> list[Check]:
    month, day = CLIENT_LIST_MONTH_DAY
    dates = c["date"].drop_duplicates()
    bad = dates[(dates.dt.month != month) | (dates.dt.day != day)]
    years = sorted(c["year"].unique())
    lo, hi = SNAPSHOT_SIZE_RANGE
    size = c.groupby("year").size()
    w = c.groupby("year")["weight"].sum()
    wlo, whi = WEIGHT_SUM_RANGE
    cusip_ok = c["cusip9"].str.fullmatch(r"[0-9A-Z]{9}") | c["cusip9"].eq(CLIENT_NO_CUSIP)
    dup = c[c["method"].eq("cusip")].duplicated(["year", "cusip8"])
    return [
        check("unique row key", not c["row"].duplicated().any()),
        check(f"every list is dated {month}/{day}", bad.empty, ", ".join(bad.dt.strftime("%Y-%m-%d")) or
              f"{len(dates)} lists"),
        check("one list per year, no year missing", years == list(range(years[0], years[-1] + 1)),
              f"{years[0]}-{years[-1]}"),
        check(f"rows per list within {lo:,}-{hi:,}", size.between(lo, hi).all(),
              f"{size.min():,} to {size.max():,}", level=WARN),
        check(f"weights per list sum to {wlo:.0f}-{whi:.0f}%", w.between(wlo, whi).all(),
              f"{w.min():.2f}% to {w.max():.2f}%", level=WARN),
        check("every row has a weight", c["weight"].notna().all(), f"{c['weight'].isna().sum()} without",
              level=WARN),
        check(f"CUSIPs are 9 characters or {CLIENT_NO_CUSIP}", cusip_ok.all(),
              f"{(~cusip_ok).sum()} other values", level=WARN),
        check("no CUSIP twice in one list", not dup.any(), f"{int(dup.sum())} repeats", level=WARN),
        Check("rows by matching method", "PASS",
              f"{c['method'].eq('cusip').sum():,} by CUSIP, {c['method'].eq('ticker').sum():,} by ticker + name "
              f"({c['ticker'].eq(CLIENT_NO_TICKER).sum()} of them without a ticker)"),
    ]


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    c = load_client_file(paths.client / CLIENT_FILE)
    checks = run_checks(c)
    write_parquet(c, out / OUTPUT)
    by_year = c.groupby("year").agg(rows=("row", "size"), by_ticker=("method", lambda m: int(m.eq("ticker").sum())),
                                    weight_sum=("weight", "sum"))
    extra = {"by_year": {int(y): {"rows": int(r.rows), "by_ticker": int(r.by_ticker),
                                  "weight_sum": round(float(r.weight_sum), 2)} for y, r in by_year.iterrows()}}
    summary = write_summary(out, STAGE, {OUTPUT: table_info(c, "date", ["weight"])}, checks, extra)
    print(f"  {len(c):,} rows, {c['year'].nunique()} lists ({c['year'].min()}-{c['year'].max()}), "
          f"{c['method'].eq('ticker').sum()} without a CUSIP")
    for ck in checks:
        if ck.status != "PASS":
            print(f"  {ck.status}  {ck.name}: {ck.detail}")
    return summary
