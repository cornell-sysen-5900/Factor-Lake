# WRDS data pipeline

Turns the yearly WRDS and FRED downloads into the cleaned Parquet files the Factor-Lake app uses
(monthly point-in-time panel, monthly returns, daily prices, fundamentals, IBES, benchmarks, rates).
It replaces `WRDS_Cleaning.ipynb`; the notebook's rules are ported stage by stage (see [PLAN.md](PLAN.md)).

The pipeline is not part of the app. Its dependencies are in the `pipeline` uv group, which Streamlit Cloud
does not install, and it never uploads anything.

## Run it

From the repo root:

```bash
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" --stage s00_inventory
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" --stage all
```

`--stage` takes a full name (`s02_prices`), a number (`s02`), a short name (`prices`) or `all` (every stage in
order, stopping at the first FAIL). Exit code 0 = all checks PASS or WARN, 1 = a check FAILed, 2 = a stage is not
written yet.

| stage | what it does | status |
|---|---|---|
| s00_inventory | size, sha256, rows, columns, date range and expected columns of every raw file | done |
| s01_membership | IWM snapshots, membership periods, permno fill from CUSIP, known gaps, non-member exclusions | done |
| s02_prices | daily cleaning, momentum, volatility, monthly returns incl. delisting rows | done |
| s03_fundamentals_q | quarterly factors, TTM, split adjustment, RV/DCF building blocks | planned |
| s04_fundamentals_a | annual statements for DCF | planned |
| s05_ibes | consensus by measure and FPI, linked to permno, analyst factors | planned |
| s06_rates_benchmarks | FRED yields, Fama-French, index and ETF returns | planned |
| s07_panel | monthly point-in-time panel and the all-stocks monthly returns table | planned |
| s08_export | final Parquet set in processed/final/, manifest.json, DATA_DICTIONARY.md, UPLOAD.md | planned |

Tests use small synthetic data only: `uv run pytest tests/pipeline`.

## Data folder

The data never goes in the repo (`.gitignore` blocks `*.csv`, `*.parquet`, `*.zip`, `*.gz` and every `raw/`,
`processed/` and `data/` folder). Pass its location with `--data-dir`:

```
<data folder>/
  raw/                  WRDS and FRED downloads, exactly as downloaded. Read only.
    query_inputs/       permno_list.txt, ibes_tickers.txt: the upload lists the WRDS queries were run with
  archive/              old notebook and its outputs. Reference only; the pipeline never reads it.
  processed/
    s00_inventory/      one folder per stage, written only by that stage
    s01_membership/
    ...
```

## Rules every stage follows

1. **Inputs:** read only from `raw/` or from earlier stages' folders in `processed/`. Never write to `raw/`.
2. **Outputs:** write only to `processed/<stage>/`. The folder is emptied at the start of the stage, so running a
   stage again overwrites only its own outputs and never leaves stale files.
3. **Format:** tables are Parquet (pyarrow, zstd). Each stage also writes `summary.json`: status, and per output
   table its rows, date range and permnos, plus every check with PASS / WARN / FAIL and a short detail.
4. **Large files:** read only the needed columns, with pyarrow, and stream in blocks where the whole file is not
   needed in memory (the CIZ daily file is 2.9 GB, the IBES file 1.1 GB).
5. **Checks:** a FAIL stops `--stage all`. A WARN is reported and the run continues. Shared checks live in
   `pipeline/checks/`.
6. **Constants** (date ranges, sector mapping, delisting fill rule, units, windows) live in `pipeline/config.py`,
   never inline in a stage.
7. **Types:** `permno` int32, dates tz-naive `datetime64[ns]`, numbers float64, returns and yields as decimals.
8. **Short output:** a stage prints a few summary lines, not tables.

A stage module has `STAGE = "<name>"` and `run(paths: DataPaths) -> dict` that returns its summary.

## WRDS refresh checklist

Goal: rerun the queries below once a year, save each output under the same file name in `raw/`, then run
`--stage all` and read each `summary.json`. Saved query IDs are not recorded yet; fill them in after the next pull.
Menu paths are as of 2026.

**Order matters for the two upload lists.** permno_list.txt comes from the holdings, ibes_tickers.txt from the
link table. The pipeline writes both to `processed/`; copy them to `raw/query_inputs/` yourself before the queries
that use them.

1. Run query 1. Run `--stage s01`, copy `processed/s01_membership/permno_list.txt` to `raw/query_inputs/`.
2. Run queries 2, 3, 6, 9, 11 with `permno_list.txt`, and queries 4, 5, 7, 8 (no codes file). Download the FRED
   series (12). Run `--stage s05` up to the link step, copy `processed/s05_ibes/ibes_tickers.txt` to
   `raw/query_inputs/`.
3. Run query 10 with `ibes_tickers.txt`, then `--stage all`.

| # | source | date range | codes upload | variables | file in raw/ |
|---|---|---|---|---|---|
| 1 | CRSP > Mutual Funds > Portfolio Holdings | report dates 2008-02 to latest | crsp_portno 1004185, 1025818 (IWM) | crsp_portno, report_dt, eff_dt, percent_tna, nbr_shares, security_name, cusip, permno, ticker | iwm_holdings_raw.csv |
| 2 | CRSP > Stock (Version 2, CIZ) > Daily Stock File | 2006-12-01 to latest | permno_list.txt (PERMNO) | permno, HdrCUSIP, SecurityNm, DelReasonType, Ticker, PERMCO, SICCD, DlyCalDt, DlyDelFlg, DlyPrc, DlyRet, DlyVol, DlyClose, DlyLow, DlyHigh, DlyOpen, ShrOut, DlyCumFacPr, DlyCumFacShr, sprtrn | r2000_price_crsp_ciz.csv |
| 3 | CRSP/Compustat Merged > Fundamentals Quarterly | datadate 2005-01 to latest | permno_list.txt (LPERMNO) | identifiers (GVKEY, LPERMNO, datadate, fyearq, fqtr, indfmt, consol, popsrc, datafmt, tic, conm, ajexq, curcdq, datacqtr, datafqtr, rdq, costat), balance sheet (actq, apq, atq, ceqq, cheq, cshfdq, cshoq, dlcq, dlttq, gdwlq, intanq, invtq, lctq, ltq, mibtq, ppentq, pstkq, rectq, seqq, txditcq), income (cogsq, dpq, epsfxq, epspxq, ibq, niq, oiadpq, oibdpq, piq, revtq, saleq, txtq, xintq, xrdq, xsgaq), year-to-date cash flow (aqcy, capxy, dltisy, dltry, dvy, oancfy, prstkcy, sstky), dvpsxq, prccq, industry (ggroup, gind, gsector, gsubind, naics) | fundamentals_raw.csv |
| 4 | Compustat > North America > Index Prices (Daily) | 2006-12-01 to latest | tic I0124 (Russell 2000), I0120 (Growth), I0122 (Value) | tic, datadate, gvkeyx, prccd | index_close.csv |
| 5 | CRSP > Stock (Version 2, CIZ) > Daily Stock File | 2006-12-01 to latest | PERMNO 88222 (IWM), 88399 (IWN), 88401 (IWO) | PERMNO, HdrCUSIP, SecurityNm, Ticker, PERMCO, DlyCalDt, DlyPrc, DlyRet | benchmark_etfs.csv |
| 6 | Linking Suite by WRDS > IBES-CRSP Link | all | permno_list.txt (PERMNO) | TICKER, PERMNO, ncusip, sdate, edate, SCORE | crsp_ibes_link.csv |
| 7 | Fama-French Portfolios and Factors > 5 Factors + Momentum, Daily | 2006-12-01 to latest | none | date, mktrf, smb, hml, rmw, cma, rf, umd | ff_daily.csv |
| 8 | Fama-French Portfolios and Factors > 5 Factors + Momentum, Monthly | 2006-12 to latest | none | dateff, mktrf, smb, hml, rmw, cma, rf, umd | ff_monthly.csv |
| 9 | CRSP > Stock (Version 2, CIZ) > Delisting events | all | permno_list.txt (permno) | delactiontype, permno, delistingdt, delreasontype, delpaymenttype, delret | delistings.csv |
| 10 | IBES > Summary History > Summary Statistics (adjusted) | statpers 2005-01 to latest | ibes_tickers.txt (TICKER) | ticker, CUSIP, OFTIC, CNAME, STATPERS, MEASURE, FISCALP, FPI, CURCODE, NUMEST, NUMUP, NUMDOWN, MEDEST, MEANEST, STDEV, HIGHEST, LOWEST, FPEDATS, ACTUAL, ANNDATS_ACT, CURR_ACT; measures EPS, CPS, DPS, EBT (EBITDA), SAL, CPX, EBI (EBIT); FPI 0 (long-term growth) and 1-3 (FY1-FY3; FY4 is not used) | forward_estimates.csv |
| 11 | CRSP/Compustat Merged > Fundamentals Annual | datadate 2005-01 to latest | permno_list.txt (LPERMNO) | the annual versions of query 3 (GVKEY, LPERMNO, datadate, fyear, fyr, ajex, curcd, act ... xsga, dvpsx_f, prcc_f, sich, emp, xintd, dvc) and the same industry columns | fundamentals_annual_raw.csv |
| 12 | FRED (fred.stlouisfed.org, not WRDS): DGS10, DGS3MO, DBAA, DAAA, daily, "Download CSV" | 2006-01-01 to latest | none | observation_date, the series (percent) | fred_DGS10.csv, fred_DGS3MO.csv, fred_DBAA.csv, fred_DAAA.csv |

The exact column lists the pipeline expects are `RAW_FILES` in `pipeline/config.py`; s00_inventory FAILs on a
missing column, so a changed query shows up before any cleaning runs.

After a refresh: compare each stage's `summary.json` with the previous run (row counts, date ranges, WARNs), and
note the run in the PR that updates the data.
