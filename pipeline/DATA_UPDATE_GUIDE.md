# WRDS data update guide

A step-by-step runbook for the next refresh of the WRDS data, written for a teammate who has never done one. It
builds a new data version (`wrds_YYYY_MM`), checks it, publishes it to S3 next to the old ones, and switches the app
to it. Expect a day of work spread over two or three days, most of it waiting for WRDS queries.

The exact WRDS queries (forms, codes, dates, screens and variables) are in [PULLS_V2.md](PULLS_V2.md); this guide
links to it instead of repeating them. The rules behind every stage are in [PLAN.md](PLAN.md), the commands in
[README.md](README.md), and the columns in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Before you start

**You need:**

- **WRDS access** through the Cornell library proxy. Open a query link from PULLS_V2.md and log in with your
  NetID; you need the CRSP, Compustat (CCM), IBES, Fama-French and Linking Suite subscriptions.
- **The repo**, with the `pipeline` uv group installed (`uv sync --group pipeline`), and your own
  `.streamlit/secrets.toml` with the AWS keys (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`).
  Never print, commit or paste the keys anywhere.
- **The data folder** (not in the repo). Each version gets its own raw and processed folders:

  ```
  <data folder>/
    raw/                  the files of data version 1 (read only)
    raw/client/           the client's Russell 2000 lists (FactSet, licensed) and an older names history
    raw/query_inputs/     the upload lists: permno_list_vN.txt, ibes_tickers_vN.txt
    raw_v2/ ... raw_vN/   the raw files of each later version, under the exact file names below
    processed_v2/ ...     each version's pipeline output; processed_vN/final/ is what gets uploaded
    pipeline_last_run.json   the counts of the last run, which the next run compares with
  ```

- **About 15 GB free disk**: a raw folder takes about 6 GB (the daily price file alone about 5 GB) and a processed
  folder about 4.5 GB. And a machine with at least 16 GB of RAM (s02 holds the whole daily file in memory).

**Which version comes next.** A version is named after the year and month of its pull: `wrds_YYYY_MM`. Look at
"Latest WRDS data" in the root [README.md](../README.md) for the current one (for example `wrds_2026_11`); yours is
the year and month you pull, and its folders are the next `raw_vN/` and `processed_vN/`. Never reuse a folder.

## Step 1. The stock list

The permno-based queries (prices, delistings, fundamentals, the IBES link) are run on an upload list of CRSP
permnos: every stock that was ever in the index. It has to be rebuilt every refresh, because new IWM months bring
new members.

1. Pull the IWM holdings first (CRSP > Mutual Funds > Portfolio Holdings, query 1 in the refresh checklist of
   [README.md](README.md)), to the latest report date, and save it as `raw_vN/iwm_holdings_raw.csv`.
2. Copy every other file of the previous version's raw folder into `raw_vN/` (they are replaced in step 2; s01 needs
   a price file to run, and the old one is fine for this).
3. Run the client and membership stages:

   ```bash
   uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" \
     --raw-dir raw_vN --processed-dir processed_vN --client-dir raw/client \
     --permno-list raw/query_inputs/permno_list_vPREV.txt --ibes-tickers raw/query_inputs/ibes_tickers_vPREV.txt \
     --stage s00b
   # then the same with --stage s00c, then --stage s01
   ```

   s01 WARNs that some snapshot stocks have no market cap: that is expected (the old price file does not have the
   new stocks yet) and goes away in step 4.
4. Copy `processed_vN/s01_membership/permno_list.txt` to `raw/query_inputs/permno_list_vN.txt`. It holds every
   permno of every snapshot: IWM members, the client-list members before 2008 and the client fill.

**New client lists.** The client's lists only define membership before 2008-02 (and the fill to 2010-05). A new
client file (for example with a newer 9/30 list) changes nothing there unless it revises old lists; it only feeds
the checks. If you do get one, save it under `raw/client/` with its exact name, pull the names history again (step
2), and rerun s00b and s00c before s01.

## Step 2. The pulls

Open each query from the table in [PULLS_V2.md](PULLS_V2.md) (each ID links to the WRDS Query Manager), then:

1. Click **View** to open the saved query's form.
2. Change the **end date** to the latest available. A rerun repeats the old date range unless you change it.
3. Where the query uploads a code list, upload the new list (`permno_list_vN.txt`, or `ibes_tickers_vN.txt` for
   IBES).
4. Run it, wait for the email, download the file and save it under `raw_vN/` with **exactly** the file name in
   PULLS_V2.md (the pipeline finds files by name).

Order:

1. **The names history** (`crsp_names_history.csv`), only if you got new client lists.
2. **The IBES link** (`crsp_ibes_link.csv`) with `permno_list_vN.txt`.
3. **IBES** (`forward_estimates.csv`), after you have built `ibes_tickers_vN.txt` from the new link: every IBES ticker
   linked to a permno of the list whose link ends on or after 2001-01-01 or has no end date, any score:

   ```python
   import pandas as pd
   permnos = {int(x) for x in open("raw/query_inputs/permno_list_vN.txt").read().split()}
   lk = pd.read_csv("raw_vN/crsp_ibes_link.csv", dtype={"TICKER": str}, keep_default_na=False, na_values=[""],
                    parse_dates=["edate"])
   lk = lk[lk["PERMNO"].isin(permnos) & (lk["edate"].isna() | (lk["edate"] >= "2001-01-01"))]
   open("raw/query_inputs/ibes_tickers_vN.txt", "w").write("\n".join(sorted(set(lk["TICKER"].str.strip()))) + "\n")
   ```

4. **Everything else in parallel:** prices, delistings, ETFs, fundamentals quarterly and annual, index levels,
   Fama-French daily and monthly, and the four FRED series (FRED links in PULLS_V2.md, not WRDS).

**Rules:**

- **Prices, delistings and ETFs come from the Monthly Update library** (`crsp_m_stock`), not the Annual Update. The
  annual one ends at the last 12/31 and restates history for its version.
- **Choose compressed output** for the big files (prices, IBES): they are several GB uncompressed.
- **Untick the market-return columns** (`ewretd`, `ewretx`, `sprtrn`) in the price query: the pipeline does not use
  them.
- **Never open and save a CSV in Excel.** Excel rewrites CUSIPs, tickers and gvkeys (leading zeros, dates, `NA`),
  silently. Look at the files with a text editor or pandas only.
- **WRDS may expire old saved queries.** If a link no longer opens, build the query again from the form path,
  codes, dates, screens and variables in PULLS_V2.md, and record the new ID (step 8).

## Step 3. Check the raw files

```bash
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" \
  --raw-dir raw_vN --processed-dir processed_vN --client-dir raw/client \
  --permno-list raw/query_inputs/permno_list_vN.txt --ibes-tickers raw/query_inputs/ibes_tickers_vN.txt \
  --stage s00
```

s00 lists every file with its size, rows, columns and date range, and checks the columns each stage needs. Fix
everything it flags before you go on:

- **FAIL** (a missing file or column): the pull is wrong. Compare the query's variables with PULLS_V2.md and pull it
  again.
- **"starts late"** WARN: the date range was not set back far enough.
- **upload list** counts: `listed_not_in_file` in `processed_vN/s00_inventory/inventory.json` should be small (a
  few permnos without data at WRDS); a large number means the wrong list was uploaded.

## Step 4. Run the pipeline

```bash
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" \
  --raw-dir raw_vN --processed-dir processed_vN --client-dir raw/client \
  --permno-list raw/query_inputs/permno_list_vN.txt --ibes-tickers raw/query_inputs/ibes_tickers_vN.txt \
  --stage all
```

It runs s00 to s09 in order and stops at the first FAIL. It takes about 25 minutes: s02 (prices) and s07 (the
panel) about 3 minutes each, s05 (IBES) about 2, and s09 about 10, because it reruns every stage into a temporary
folder to check the run is reproducible.

Each stage writes `processed_vN/<stage>/summary.json`: its status, the rows and date range of each output, and every
check with its status and a short detail. The run prints the checks that did not pass.

- **PASS:** fine.
- **WARN:** unusual; read it. Most are explained in the check's detail. The run continues.
- **FAIL:** an output is wrong. The run stops; fix the cause (usually a raw file) and run again from that stage.

**Known WARNs** (not problems):

- **"vs last run"** checks compare counts with `pipeline_last_run.json` (the previous version), within 2%. A refresh
  adds about a year of data, so some counts move more; check that they grew, not shrank.
- **s01:** `permno_list.txt` differs from the upload list by a few permnos (client-list stocks that are never index
  members), and a few client-fill rows share a company with an IWM holding (another share class).
- **s05:** its own `ibes_tickers.txt` differs a little from the upload list (links that ended before 2001, scores
  above 2).
- **s09:** gross profitability vs RMW (the measures differ), app memory of the full panel (the app loads only the
  columns it needs), and WRDS vs Legacy price (FactSet adjusts its prices for later splits).

## Step 5. Review the validation

Open `processed_vN/s09_validation/VALIDATION.md`. Look at:

1. **Index tracking (section 1):** the panel, weighted by market cap, against IWM. The correlation should be at least
   0.99 overall and in each membership era, and the tracking error at most about 2% a year. A year far off is
   explained under the table.
2. **Membership by source (section 8):** 1,800 to 2,150 stocks every month, each source only in its own months.
3. **Factor sanity (section 2):** value, momentum and investment long-short returns should correlate with the
   Fama-French factors as before.
4. **Client lists vs IWM (section 9)** and **the independent recompute (section 4).**

**Stop and investigate** when any check FAILs, when a WARN appears that is not in the list above, when tracking gets
worse than in the previous version's VALIDATION.md, or when the stocks per month fall outside the range. Compare with
the previous version's file: the same checks should give similar results.

## Step 6. Publish

Upload `processed_vN/final/` to a **new** S3 folder, `factor-lake-data/wrds_YYYY_MM/` (the script reads the folder
from `final/manifest.json`). Never overwrite an old version: the app and saved analyses may still read it.

```bash
uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_YYYY_MM/ --no-head      # must be empty
uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed_vN/final"              # dry run
uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed_vN/final" --upload     # upload
uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed_vN/final" --check      # verify
uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_YYYY_MM/ --no-head      # look at it
```

`--check` compares every key's size and ETag with the local file and downloads two files back. The upload is
complete when it reports every file matching.

## Step 7. Switch the app

1. On a new branch, set `DEFAULT_WRDS_DIR = 'wrds_YYYY_MM'` in `src/wrds_data.py`. If the panel's first or last
   month changed, set `FIRST_MONTH` and `LAST_MONTH` in `src/wrds_calendar.py`, and update the tests that pin the
   range (`tests/unit/test_wrds_calendar.py`, `test_wrds_engine.py`, `test_wrds_mode_switch.py`).
2. Test on the **dev** Lambda, never the live one:

   ```bash
   uv run python scripts/deploy_backtest_lambda.py --function-name sysen5900-factor-lake-backtest-dev \
     --role-arn <the course role's ARN> --leave-lifecycle-rules --wrds-dir wrds_YYYY_MM
   FACTOR_LAKE_BACKTEST_LAMBDA=sysen5900-factor-lake-backtest-dev uv run python scripts/check_backtest_lambda.py
   ```

   The deploy script reads the AWS keys from the environment; the role ARN is the repo's `LAMBDA_ROLE_ARN`
   Actions variable. Point a local app at the dev function (`FACTOR_LAKE_BACKTEST_LAMBDA` in your
   `secrets.toml`) and run a WRDS backtest over the whole range and a Legacy one.
3. Open a PR. **Merging deploys the live Lambda** through CI (the Deploy Backtest Lambda workflow), whose check step
   runs a WRDS backtest on the new folder. Watch it pass, then run one WRDS and one Legacy job on the live app.

**Rollback:** set `DEFAULT_WRDS_DIR` (and the calendar months) back to the previous folder and merge; CI redeploys
with it. The old folder is still in S3, and saved analyses keep opening by their ID.

## Step 8. Record it

1. **The queries:** write `pipeline/PULLS_VN.md` (copy PULLS_V2.md) with the new query IDs, dates and row counts,
   and the superseded IDs.
2. **Decisions:** add any new decision to the Decisions table in [PLAN.md](PLAN.md), and the version to its "Data
   versions" table.
3. **Docs:** update "Latest WRDS data" in the root `README.md` and `DOCS/AWS_S3_DATA.md` (folder, data version,
   coverage, what is new, the previous version for rollback).
4. **Tag** the commit that built the version (the `pipeline_git_commit` in `processed_vN/final/manifest.json`) as
   `data-wrds_YYYY_MM` and push the tag.
5. **Baseline:** keep the old `pipeline_last_run.json` as `pipeline_last_run_vPREV.json` and save this run's counts
   in `pipeline_last_run.json` (same keys), so the next refresh compares with this one.

To re-export a published version later (for example after a documentation change), set
`FACTOR_LAKE_DATA_VERSION=YYYY-MM-DD` to its original date; otherwise s08 stamps today's date.

## Licensing

- WRDS data (CRSP, Compustat, IBES) and the client's FactSet lists are licensed. **Nothing from them goes into git**:
  no raw or processed files, no statistics or counts of the data in code, docs, PR text or commit messages, and no
  company names from the client lists (tests use invented names). The only numbers in the repo are row counts in
  PULLS_VN.md.
- The full data dictionary (`processed_vN/DATA_DICTIONARY.md`, with statistics) and `VALIDATION.md` are shared with
  the team privately, never committed. The repo keeps the schema-only [DATA_DICTIONARY.md](DATA_DICTIONARY.md).
- Keep the S3 bucket private and give the app read-only keys; the licensed data must never be publicly readable.
