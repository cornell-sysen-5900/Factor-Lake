# WRDS data update guide

A step-by-step runbook for the next refresh of the WRDS data, for a teammate who has never done one: build a new data
version (`wrds_YYYY_MM`), check it, publish it to S3 next to the old ones, and switch the app to it. Expect a day of
work over two or three days, mostly waiting for WRDS. The queries are in [PULLS_V2.md](PULLS_V2.md), the stage rules
in [PLAN.md](PLAN.md), the columns in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Before you start

- **WRDS access** through the Cornell library proxy (log in with your NetID when you open a query link): CRSP,
  Compustat (CCM), IBES, Fama-French and the Linking Suite.
- **The repo** with `uv sync --group pipeline`, and your own `.streamlit/secrets.toml` with `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION`. Never print, commit or paste the keys.
- **The data folder** (never in the repo), one raw and one processed folder per version:

  ```
  <data folder>/
    raw/                  data version 1 (read only); raw/client/ holds the client's lists (licensed)
    raw/query_inputs/     the upload lists: permno_list_vN.txt, ibes_tickers_vN.txt
    raw_vN/               this version's raw files, under the exact names in PULLS_V2.md
    processed_vN/         this version's pipeline output; processed_vN/final/ is what gets uploaded
    pipeline_last_run.json   the counts of the last run, which the next run compares with
  ```

- **About 15 GB free disk** (a raw folder about 6 GB, a processed folder about 4.5 GB) and **16 GB of RAM** (s02 holds
  the whole daily price file in memory).
- **The version name:** the year and month of your pull, `wrds_YYYY_MM`. The current one is in "Latest WRDS data" in
  the root [README.md](../README.md); yours gets the next `raw_vN/` and `processed_vN/`. Never reuse a folder.

Every pipeline run below is this command from the repo root; `run <stage>` means it with that `--stage`:

```bash
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" \
  --raw-dir raw_vN --processed-dir processed_vN --client-dir raw/client \
  --permno-list raw/query_inputs/permno_list_vN.txt --ibes-tickers raw/query_inputs/ibes_tickers_vN.txt \
  --stage <stage>
```

## Step 1. The stock list

The permno-based queries run on a list of every stock ever in the index, which grows with every new IWM month.

1. Pull the IWM holdings to the latest report date ([PULLS_V2.md, section 12](PULLS_V2.md)) and save them as
   `raw_vN/iwm_holdings_raw.csv`.
2. Copy the previous version's other raw files into `raw_vN/` (step 2 replaces them; s01 needs a price file).
3. For this one step, point `--permno-list` and `--ibes-tickers` at the previous version's lists, then run
   `run s00b`, `run s00c` and `run s01`. s01 WARNs that new stocks have no market cap: expected, the old price file
   lacks them.
4. Copy `processed_vN/s01_membership/permno_list.txt` to `raw/query_inputs/permno_list_vN.txt`.

The client's lists only set membership before 2008-02 (and the fill to 2010-05). Only with a new client file: save
it under `raw/client/` with its exact name, pull the names history (step 2) and rerun s00b and s00c before s01.

## Step 2. The pulls

For each query in [PULLS_V2.md](PULLS_V2.md): open its Query Manager link, click **View**, set the **end date** to the
latest (a rerun repeats the old range otherwise), upload the new code list if it uses one, run it, and save the
download under `raw_vN/` with **exactly** its file name in PULLS_V2.md. Order:

1. The names history, only with a new client file.
2. The IBES link, with `permno_list_vN.txt`. Then build `ibes_tickers_vN.txt` (the rule in PULLS_V2.md, "Upload
   lists"):

   ```python
   import pandas as pd
   permnos = {int(x) for x in open("raw/query_inputs/permno_list_vN.txt").read().split()}
   lk = pd.read_csv("raw_vN/crsp_ibes_link.csv", dtype={"TICKER": str}, keep_default_na=False, na_values=[""],
                    parse_dates=["edate"])
   lk = lk[lk["PERMNO"].isin(permnos) & (lk["edate"].isna() | (lk["edate"] >= "2001-01-01"))]
   open("raw/query_inputs/ibes_tickers_vN.txt", "w").write("\n".join(sorted(set(lk["TICKER"].str.strip()))) + "\n")
   ```

3. IBES, with `ibes_tickers_vN.txt`.
4. Everything else in parallel: prices, delistings, ETFs, both fundamentals, index levels, both Fama-French files,
   and the four FRED series.

Rules: prices, delistings and ETFs from the **Monthly Update** library; **compressed output** for the big files;
**untick** `ewretd`, `ewretx` and `sprtrn`; **never open and save a CSV in Excel** (it silently rewrites CUSIPs,
tickers and gvkeys). If WRDS has expired a saved query, rebuild it from PULLS_V2.md and record the new ID (step 8).

## Step 3. Check the raw files

Run `run s00` and fix everything it flags before going on. A FAIL (missing file or column) means a wrong pull:
compare the query with PULLS_V2.md and pull again. "starts late" means the start date was not set back far enough.
In `processed_vN/s00_inventory/inventory.json`, `listed_not_in_file` should be a few permnos, not hundreds (a wrong
upload list).

## Step 4. Run the pipeline

`run all` runs s00 to s09 and stops at the first FAIL, in about 25 minutes (s02 and s07 about 3 minutes each, s09 about 10,
because it reruns every stage to check the run is reproducible). Each stage writes `processed_vN/<stage>/summary.json`
with its status, the rows and dates of each output, and every check. **PASS** is fine; **WARN** is unusual, so read
its detail; **FAIL** stops the run: fix the cause (usually a raw file) and rerun from that stage.

Known WARNs, not problems: **"vs last run"** (counts vs `pipeline_last_run.json` within 2%; a year more of data
moves some more, so check they grew); **s01** (a few upload-list permnos are never members, a few client-fill rows
share a company with an IWM holding); **s05** (its `ibes_tickers.txt` differs a little from the upload list); **s09**
(gross profitability vs RMW, app memory, WRDS vs Legacy price).

## Step 5. Review the validation

Open `processed_vN/s09_validation/VALIDATION.md` next to the previous version's. Check **index tracking** (section
1: correlation with IWM at least 0.99 overall and in each membership era, tracking error at most about 2% a year),
**membership by source** (section 8: 1,800 to 2,150 stocks every month), and **factor sanity**, **client lists vs
IWM** and **the independent recompute** (sections 2, 9, 4). **Stop and investigate** on any FAIL, on a WARN not listed above, when tracking is worse than in the previous
version, or when the stocks per month leave that range.

## Step 6. Publish

Upload to a **new** folder `factor-lake-data/wrds_YYYY_MM/` (taken from `final/manifest.json`); never overwrite an old
one, the app and saved analyses may read it.

```bash
uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_YYYY_MM/ --no-head      # must be empty
uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed_vN/final"              # dry run
uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed_vN/final" --upload     # upload
uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed_vN/final" --check      # every file must match
uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_YYYY_MM/ --no-head      # look at it
```

## Step 7. Switch the app

1. On a new branch, set `DEFAULT_WRDS_DIR = 'wrds_YYYY_MM'` in `src/wrds_data.py`. If the panel's first or last month
   changed, set `FIRST_MONTH` and `LAST_MONTH` in `src/wrds_calendar.py` and update the tests that pin them
   (`tests/unit/test_wrds_calendar.py`, `test_wrds_engine.py`, `test_wrds_mode_switch.py`).
2. Test on the **dev** Lambda, never the live one (AWS keys in the environment; the role ARN is the repo's
   `LAMBDA_ROLE_ARN` Actions variable):

   ```bash
   uv run python scripts/deploy_backtest_lambda.py --function-name sysen5900-factor-lake-backtest-dev \
     --role-arn <the course role's ARN> --leave-lifecycle-rules --wrds-dir wrds_YYYY_MM
   FACTOR_LAKE_BACKTEST_LAMBDA=sysen5900-factor-lake-backtest-dev uv run python scripts/check_backtest_lambda.py
   ```

   Then point a local app at it (`FACTOR_LAKE_BACKTEST_LAMBDA` in `secrets.toml`) and run a full-range WRDS
   backtest and a Legacy one.
3. Open a PR. **Merging deploys the live Lambda** through CI (Deploy Backtest Lambda), whose check step runs a WRDS
   backtest on the new folder. Watch it pass, then run one WRDS and one Legacy job on the live app.

**Rollback:** set `DEFAULT_WRDS_DIR` and the calendar months back and merge; CI redeploys. The old folder stays in S3
and saved analyses keep opening by their ID.

## Step 8. Record it

1. `pipeline/PULLS_VN.md` (a copy of PULLS_V2.md) with the new query IDs, dates, row counts and superseded IDs.
2. New decisions and the version's row in the "Data versions" table of [PLAN.md](PLAN.md); "Latest WRDS data" in the
   root `README.md` and `DOCS/AWS_S3_DATA.md`.
3. Tag the commit that built the version (`pipeline_git_commit` in `processed_vN/final/manifest.json`) as
   `data-wrds_YYYY_MM` and push the tag.
4. Keep the old `pipeline_last_run.json` as `pipeline_last_run_vPREV.json` and save this run's counts in
   `pipeline_last_run.json` (same keys), so the next refresh compares with this one.

To re-export a published version later, set `FACTOR_LAKE_DATA_VERSION=YYYY-MM-DD` to its date; otherwise s08 stamps
today's.

## Licensing

WRDS data and the client's FactSet lists are licensed: **nothing from them goes into git** (no data files, no
statistics or counts in code, docs, PRs or commits, no company names from the client lists; the only numbers are the
row counts in PULLS_VN.md). Share the full data dictionary (`processed_vN/DATA_DICTIONARY.md`) and `VALIDATION.md`
with the team privately. Keep the S3 bucket private and give the app read-only keys.
