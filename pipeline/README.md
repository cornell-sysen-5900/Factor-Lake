# WRDS data pipeline

Turns the yearly WRDS and FRED downloads into the cleaned Parquet files the Factor-Lake app uses
(monthly point-in-time panel, monthly returns, daily prices, fundamentals, IBES, benchmarks, rates).
It replaces `WRDS_Cleaning.ipynb`; the notebook's rules are ported stage by stage (see [PLAN.md](PLAN.md)).
Every exported file and column is described in [DATA_DICTIONARY.md](DATA_DICTIONARY.md); where the latest data is in
S3 is in [DOCS/AWS_S3_DATA.md](../DOCS/AWS_S3_DATA.md#latest-wrds-data).

The pipeline is not part of the app. Its dependencies are in the `pipeline` uv group, which Streamlit Cloud
does not install, and it never uploads anything: the upload is a separate script, run by hand.

**Each new data pull gets its own folder**, `s3://sysen-5900-factor-lake/factor-lake-data/wrds_YYYY_MM/`
(year and month of the pull), next to the files the app reads; nothing older is overwritten. After an upload, update
"Latest WRDS data" in the root README.md and DOCS/AWS_S3_DATA.md to point at the new folder.

## Run it

From the repo root:

```bash
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" --stage s00_inventory
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" --stage all
```

`--stage` takes a full name (`s02_prices`), a number (`s02`), a short name (`prices`) or `all` (every stage in
order, stopping at the first FAIL).

A second data version runs in place with the folder options (relative paths are relative to `--data-dir`; the
defaults are the layout below):

```bash
uv run --group pipeline python -m pipeline.run --data-dir "C:/path/to/WRDS FA26" --stage all \
  --raw-dir raw_v2 --processed-dir processed_v2 --client-dir raw/client \
  --permno-list raw/query_inputs/permno_list_v2.txt --ibes-tickers raw/query_inputs/ibes_tickers_v2.txt
```

Exit code 0 = all checks PASS or WARN, 1 = a check FAILed. A full run of data version 2 takes about 25 minutes;
s09_validate reruns every stage into a temporary folder to check the run is reproducible (`S09_SKIP_REPRO=1` skips
that part).

| stage | what it does | status |
|---|---|---|
| s00_inventory | size, sha256, rows, columns, date range and expected columns of every raw file | done |
| s00b_client | the client's 9/30 Russell 2000 lists as a clean table | done |
| s00c_client_map | a CRSP permno for every client list row (CUSIP, then ticker + name), known gaps | done |
| s01_membership | index snapshots from the client lists (before 2008-02), IWM and the client fill (2008-02 to 2010-05), membership periods, permno fill from CUSIP, known gaps and their fixes, non-member exclusions | done |
| s02_prices | daily cleaning, momentum, volatility, monthly returns incl. delisting rows | done |
| s03_fundamentals_q | quarterly factors, TTM, split adjustment, RV/DCF building blocks | done |
| s04_fundamentals_a | annual statements for DCF (dcf_inputs_annual) | done |
| s05_ibes | full IBES consensus (estimates_consensus), linked to permno, with the CRSP share-basis factor | done |
| s06_market | index and ETF returns, Fama-French, FRED yields and spreads (market_daily, market_monthly) | done |
| s07_panel | factor_panel_monthly, rv_multiples_monthly, rv_peer_stats_monthly, security_master | done |
| s08_export | final Parquet set in processed/final/ with manifest.json and UPLOAD.md; processed/DATA_DICTIONARY.md | done |
| s09_validate | validation against IWM, Fama-French, the old panel and the raw CSVs; full rerun; VALIDATION.md | done |

Every exported file and column (meaning, unit, source, stage) is described in
[DATA_DICTIONARY.md](DATA_DICTIONARY.md), written by s08 from `dictionary.py`. That version is schema only, because
the repo is public and the data is licensed from WRDS; the full one with statistics is `processed/DATA_DICTIONARY.md`
next to the data. **No statistics of the data go into the repo** (code, docs, PR text or commit messages).

Tests use small synthetic data only: `uv run pytest tests/pipeline`. `tests/pipeline/test_real_data.py` checks a
finished run read-only when `FACTOR_LAKE_DATA_DIR` points at the data folder, and is skipped otherwise (as in CI).

**Windows:** if `uv sync` fails with "Access is denied" on a `.pyd` file, a running Python still has it open: stop
the running app, Jupyter kernels or the IDE's Python first, then sync again.

## Data folder

The data never goes in the repo (`.gitignore` blocks `*.csv`, `*.parquet`, `*.zip`, `*.gz` and every `raw/`,
`processed/` and `data/` folder). Pass its location with `--data-dir`:

```
<data folder>/
  raw/                  WRDS and FRED downloads, exactly as downloaded. Read only.
    query_inputs/       permno_list.txt, ibes_tickers.txt: the upload lists the WRDS queries were run with
    client/             the client's Russell 2000 lists (FactSet, licensed: never commit it or anything from it)
  archive/              old notebook and its outputs. Reference only; the pipeline never reads it.
  pipeline_last_run.json  the notebook's last-run counts the "vs last run" checks compare with (kept out of git)
  processed/
    s00_inventory/      one folder per stage, written only by that stage
    s01_membership/
    ...
    s08_export/         summary.json only
    s09_validation/     summary.json, VALIDATION.md (stage s09_validate)
    final/              what s08 exports: the files to share and upload (see final/UPLOAD.md)
    DATA_DICTIONARY.md  every final column with meaning, unit, source and stats (written by s08, not uploaded)
```

## Upload

`scripts/upload_wrds.py` uploads `processed/final/` to `factor-lake-data/wrds_YYYY_MM/` (from final/manifest.json):
the Parquet files with their relative paths, then `prices_daily/_manifest.json` last, once every year file is
verified. It refuses any other prefix, never deletes, skips objects that are already the same, and stops on the
first AWS error.

    uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed/final"            # dry run
    uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed/final" --offline  # no keys, no network
    uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed/final" --upload
    uv run python scripts/upload_wrds.py "C:/path/to/WRDS FA26/processed/final" --check    # every key vs the local file

`--check` compares Parquet files by size and ETag and JSON files by content (created_at ignored), and downloads the
daily manifest and the smallest Parquet file back. final/UPLOAD.md repeats this for the folder at hand.

## How to inspect S3

`scripts/s3_inspect.py` shows what is in the bucket without changing anything (it makes no put, copy or delete
calls) and without downloading whole files: a tree with sizes and dates, and the head of each file (Parquet: rows,
columns, types and the first rows from the footer and the first row group, read with ranged GETs; text files: the
first 10 lines).

    uv run python scripts/s3_inspect.py --no-head                                     # the whole factor-lake-data/ tree
    uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_2026_11/        # with heads
    uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_2026_11/ --all-partitions --rows 3

Options: `--prefix` (default `factor-lake-data/`), `--rows` (default 5), `--max-files`, `--no-head`,
`--all-partitions` (every `year=YYYY` head, not only the newest), `--head-mb` (column data read per head, default 8).

The AWS keys come from `.streamlit/secrets.toml` in the repo root (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`), read with tomllib and never printed. That file is local only: it is in .gitignore and has
never been on GitHub, so each teammate needs their own copy at the same path.

## Rules every stage follows

1. **Inputs:** read only from `raw/` or from earlier stages' folders in `processed/`. Never write to `raw/`.
2. **Outputs:** write only to `processed/<stage>/` (s08 also to `processed/final/`). The folder is emptied at the start of the stage, so running a
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

## Data refresh

Follow [DATA_UPDATE_GUIDE.md](DATA_UPDATE_GUIDE.md) step by step; the exact WRDS queries are in
[PULLS_V2.md](PULLS_V2.md).
