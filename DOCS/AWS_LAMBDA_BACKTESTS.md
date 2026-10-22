# Backtests in AWS Lambda

Backtests can run in AWS Lambda instead of inside the Streamlit app. The app sends only the inputs (factors, directions and sidebar settings). Lambda reads the market data from S3, runs the same backtest code, and saves the result in S3, so the app never has to download or hold the market data to run a backtest.

## 1. How it works

```
Streamlit app ──inputs──▶ Lambda factor-lake-backtest ──reads──▶ s3://.../factor-lake-data/*.parquet
      ▲                         │
      └──result or S3 link──────┴──saves──▶ s3://.../factor-lake-data/backtest-results/<hash>.json
```

1. **Inputs only.** Run Portfolio Analysis (and Generate Comparison) sends a small JSON job: factors, directions, years, AUM, weighting, delisting strategy, factor combination, sectors and the fossil fuel filter. See `src/backtest_jobs.py`.
2. **Same code, same results.** Lambda filters the data with `src/universe_selection.py` (the code behind Load Market Data) and runs `src/backtest_engine.py`, the code the app runs locally.
3. **Cached by hash.** The result is saved under a SHA-256 hash of the inputs, the data version (the S3 ETags of the market data files) and the backtest code version (a hash of the engine's source files). Republishing the data or deploying new backtest code gives fresh results.
4. **Repeats are instant.** When a result with that hash exists, Lambda returns it without rerunning the backtest. Results are shared, so a run by any user is cached for everyone.
5. **Result or link.** Results up to 5 MB come back in the Lambda response (Lambda's limit is 6 MB). Bigger results come back as a temporary S3 link (valid for 1 hour), which the app downloads.
6. **90-day expiry.** An S3 lifecycle rule deletes saved results after 90 days.

The Results tab shows how each run was produced, for example "Computed: Returned from the AWS Lambda result cache in 0.4 s (first computed 2026-10-22 18:03:11 UTC)".

## 2. Deploy the Lambda

Needs AWS keys that can create IAM roles and Lambda functions, and change the bucket's lifecycle rules.

```bash
uv run python scripts/deploy_backtest_lambda.py
```

It is safe to run again: it creates what is missing and updates the rest. It sets up:

| Resource | Details |
|---|---|
| IAM role `factor-lake-backtest-lambda` | Read `factor-lake-data/*`, write `factor-lake-data/backtest-results/*`, CloudWatch logs |
| Lambda `factor-lake-backtest` | Python 3.12, 3008 MB, 10-minute timeout; pandas, numpy and pyarrow come from AWS's public [AWS SDK for pandas layer](https://aws-sdk-pandas.readthedocs.io/en/stable/layers.html), so the upload is only this project's code (about 30 KB) |
| S3 lifecycle rule `factor-lake-expire-backtest-results` | Deletes `factor-lake-data/backtest-results/` objects after 90 days. Other rules on the bucket are kept |

It finishes by pinging the function. Options: `--memory`, `--timeout`, `--function-name`, `--layer-arn` (if the newest layer version cannot be looked up), `--package-only out.zip` (build the zip without deploying).

## 3. Turn it on in the app

Set the function name next to the AWS keys, in `.env` locally or in the Streamlit Cloud secrets:

```
FACTOR_LAKE_BACKTEST_LAMBDA="factor-lake-backtest"
```

The app's AWS key also needs `lambda:InvokeFunction` on that function. When the variable is set, the sidebar shows **Run backtests in AWS Lambda** (on by default). Turn it off to run backtests in the app as before (Load Market Data first). Without the variable, the app works exactly as before.

## 4. Check it

```bash
uv run python scripts/check_backtest_lambda.py
```

For four sample backtests (Mixing and Integrating, every delisting strategy, equal and market cap weighting, the fossil fuel filter and a cohort comparison), it runs each one locally on the S3 data, then in Lambda, and compares every number. It then runs each one in Lambda again and checks that it came from the cache, unchanged. It uses new inputs each time, so the first Lambda call always computes. The Lambda runs on Linux with AWS's pandas build, so numbers may differ in the last binary digit. The check allows a relative difference of 1e-9 and prints the largest difference it found.

The unit tests (`tests/unit/test_backtest_lambda.py`) run the same Lambda code against a fake S3 and drive the real app, checking that a Lambda run equals a local run for both factor combinations, each delisting strategy and the cohort comparison, and that a repeated run is served from the cache.

## 5. Demo script (about 5 minutes)

Before the call: deploy (section 2), add `FACTOR_LAKE_BACKTEST_LAMBDA` to `.env`, and start the app with `uv run streamlit run app/streamlit_app.py`. To warm the Lambda up, run one backtest. To keep the "first run computes" moment for the call, pick inputs you have not run yet (for example a new AUM).

1. **The switch.** Point out **Run backtests in AWS Lambda** in the sidebar and the caption with the function name. On the Analysis tab, the caption says the market data does not need to be loaded.
2. **First run computes in Lambda.** Add two factors (for example ROE using 9/30 Data and ROA %), pick Integrating, and click **Run Portfolio Analysis** without loading data. The confirmation says "computed in AWS Lambda in N s". In the Results tab, the **Computed** line shows the result ID (the first 12 characters of its hash).
3. **The repeat is cached.** Click **Run Portfolio Analysis** again with the same settings. The new run shows the same result ID and "returned from the AWS Lambda result cache in 0.N s", with the time it was first computed.
4. **Any change is a new backtest.** Change the AUM or a factor direction and run again: it computes (new result ID).
5. **Same results as before.** Turn the Lambda switch off, click **Load Market Data**, and run the same settings as step 2. The new run tab shows the same final value, CAGR and holdings as the Lambda run. For the precise check, run `scripts/check_backtest_lambda.py` in a terminal: "match: yes" with the largest difference for each backtest, and "cached=True" for every repeat.
6. **Cohort comparison.** In a Lambda run's tab, open **Run Cohort Comparison** and click **Generate Comparison**: the caption says it ran in Lambda.
7. **Where results live.** In the AWS console: S3 → `sysen-5900-factor-lake` → `factor-lake-data/backtest-results/` shows one JSON file per result hash. Management → Lifecycle rules shows the 90-day expiry. Lambda → `factor-lake-backtest` → Monitor shows the invocations, and its CloudWatch logs show "Computed ..." and "Cache hit ..." lines.

## 6. Troubleshoot

| Message | Fix |
|---|---|
| `Could not call the backtest Lambda ... AccessDeniedException` | The app's AWS key needs `lambda:InvokeFunction` on the function |
| `... ResourceNotFoundException` | Deploy the function (section 2) or fix `FACTOR_LAKE_BACKTEST_LAMBDA` |
| `The backtest Lambda failed: Task timed out` | Raise `--timeout` (up to 900 seconds) and redeploy |
| `... Runtime.OutOfMemory` | Raise `--memory` and redeploy |
| `The backtest Lambda could not run this backtest: ...` | The backtest itself failed; the message is the engine's error |

## 7. Cost

Each uncached backtest is one Lambda invocation of a few seconds. A cold start adds the time to download the 8 MB market data file. Cached repeats take well under a second, and saved results are small JSON files deleted after 90 days. Class use stays far inside the Lambda and S3 free tiers.
