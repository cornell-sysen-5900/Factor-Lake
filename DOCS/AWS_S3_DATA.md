# AWS S3 Data Source

Use this guide when you need to configure, update, or troubleshoot where the Factor-Lake app loads its market data from.

## 1. Know the setup

1. The Streamlit app still runs on Streamlit Community Cloud.
2. The market data lives in AWS S3 as two Parquet files, not in a database.
3. S3 is always on. It does not pause after a week without traffic the way the free-tier Supabase project did.
4. The app downloads the data once per server process (about 1–2 seconds) and shares that copy with every user session for 6 hours.
5. Supabase is still supported as a fallback data source.

| File | Former Supabase table | Contents |
|---|---|---|
| `s3://sysen-5900-factor-lake/factor-lake-data/full_precision_test.parquet` | `Full Precision Test` | Yearly factor data for the investment universe |
| `s3://sysen-5900-factor-lake/factor-lake-data/last_price_mapping.parquet` | `last_price_mapping` | Delisting dates and last prices |

Every publish also keeps a copy under `factor-lake-data/archive/<UTC timestamp>/` for rollback.

## 2. Configure the app

Add these to the Streamlit Cloud app's **Secrets** (or `.streamlit/secrets.toml` / `.env` locally):

```toml
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_DEFAULT_REGION = "us-east-1"
```

1. With AWS credentials present, the app uses S3 automatically. The sidebar's Data Source line shows "AWS S3 (Parquet files)".
2. Set `FACTOR_LAKE_DATA_SOURCE = "supabase"` to switch back to Supabase without a code change, or `"s3"` to force S3.
3. Optional: `FACTOR_LAKE_S3_BUCKET` (default `sysen-5900-factor-lake`) and `FACTOR_LAKE_S3_PREFIX` (default `factor-lake-data`) point the app at other files.
4. The app only needs read access. Ask the AWS account owner for a key whose policy allows just `s3:GetObject` on `arn:aws:s3:::sysen-5900-factor-lake/factor-lake-data/*`, rather than reusing a key that can write or delete.
5. Never commit keys. `.env` and `.streamlit/secrets.toml` are in `.gitignore`.

## 3. Publish new data

Use `scripts/publish_data_to_s3.py` with a key that can write to the bucket.

1. Put `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`.
2. Check the files first without uploading:

```bash
uv run python scripts/publish_data_to_s3.py --universe new_data.xlsx --delisting last_price_mapping.csv --dry-run
```

3. Publish:

```bash
uv run python scripts/publish_data_to_s3.py --universe new_data.xlsx --delisting last_price_mapping.csv
```

4. Files can be `.csv`, `.xlsx`, `.xls` or `.parquet`, with the same columns as the former Supabase tables.
5. The script refuses files the app could not use (for example, no `Date` column), uploads, and reads the file back to confirm the row count.
6. The app picks up the new data within 6 hours. To see it immediately, reboot the app from Streamlit Cloud (Manage app → Reboot app).
7. To copy the current Supabase tables instead, use `--from-supabase` (needs `SUPABASE_URL` and `SUPABASE_KEY`).

## 4. Publish the daily stock data

The daily CRSP prices for the Russell 2000 universe (`russell2000_cleaned.csv`, about 7.5 million rows) are published separately with `scripts/publish_daily_data_to_s3.py`. The app does not use them yet.

1. Put `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (a key that can write to the bucket) in `.env`.
2. Check the file first. This cleans it and writes the Parquet files locally without uploading:

```bash
uv run python scripts/publish_daily_data_to_s3.py path/to/russell2000_cleaned.csv --dry-run --output-dir daily_parquet
```

3. Publish:

```bash
uv run python scripts/publish_daily_data_to_s3.py path/to/russell2000_cleaned.csv
```

4. Expect about 1–2 minutes and under 1 GB of memory for 7.5 million rows.

The cleaning matches the old daily loader and the yearly data:

- Column names can be upper or lower case: `date`, `permno`, `ticker`, `prc`, `cfacpr`, `shrout`, `ret`, `vol`. Other columns are not published, and the script lists them.
- Placeholders such as `--` and CRSP letter codes in the numbers become missing values.
- Rows without a date, a price or an identifier are dropped. A price of 0 counts as missing.
- `prc` is made positive, because CRSP marks bid/ask midpoints with a negative price.
- `prc_adj` = `prc / cfacpr` (split-adjusted).
- `mkt_cap` = `prc × shrout`, in $ thousands.
- For duplicate security/date rows, the first one is kept.

What ends up in S3:

| File | Contents |
|---|---|
| `factor-lake-data/daily_prices/year=YYYY/data.parquet` | One file per year, sorted by security and date |
| `factor-lake-data/daily_last_prices.parquet` | Each security's last trading day and price, with a `delisted` flag (stopped trading more than a week before the data ends) |
| `factor-lake-data/daily_prices/_manifest.json` | Years, row counts, columns and the cleaning summary. Uploaded last, so readers never see a half-published set |

Read it in Python with `S3DataManager().read_daily_prices(years=[2020, 2021])`. Leave out `years` to load everything.

## 5. Roll back a bad publish

1. Find the previous good copy under `factor-lake-data/archive/` in the S3 console.
2. Download its two files and publish them again with step 3.

## 6. Check the app under load

`scripts/load_test_sessions.py` opens 5 browser sessions at the same time, and each runs a full backtest: Load Market Data, Run Portfolio Analysis, Results, and the cohort comparison.

1. Run it against the live app from the Actions tab (Load Test → Run workflow, `.github/workflows/load-test.yml`). It only runs when someone starts it. A failed run means at least one session failed; screenshots are attached to the run.
2. Or run it locally:

```bash
pip install playwright && playwright install chromium
python scripts/load_test_sessions.py --url http://localhost:8501 --rounds 3
```

## 7. Troubleshoot

1. **"AWS credentials are not configured"**: add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to the app's secrets.
2. **"Could not read s3://... (AccessDenied)"**: the key cannot read that file. Check its IAM policy.
3. **"Could not read s3://... (NoSuchKey)"**: the file was not published, or the bucket or prefix setting is wrong.
4. After fixing secrets, reboot the app so the new values load.

## 8. Cost

At standard S3 prices (us-east-1), costs are well under $1 per month, far inside the $50 per month budget:

- Storage is about $0.023 per GB-month. The yearly data is about 8 MB per publish and the daily data about 300 MB.
- Downloads cost about $0.0004 per 1,000 requests.
- Data transfer out stays within the AWS free tier at this volume.
