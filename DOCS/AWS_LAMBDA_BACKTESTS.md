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
| Lambda `factor-lake-backtest` | Python 3.12, 3008 MB, 10-minute timeout. pandas, numpy and pyarrow are bundled in the package at the versions your environment runs (Linux builds fetched with `uv`; unused parts such as test suites and pyarrow's Flight libraries removed), so Lambda computes with the same libraries as the app and needs no Lambda layer. The package (about 62 MB zipped, 199 MB unzipped; Lambda allows 250 MB) is uploaded to `factor-lake-data/_deploy/` first, because direct uploads are limited to 50 MB |
| S3 lifecycle rule `factor-lake-expire-backtest-results` | Deletes `factor-lake-data/backtest-results/` objects after 90 days. Other rules on the bucket are kept |

**Keys without IAM access** (such as the class `Sysen5900Student` user, which is denied all IAM actions) cannot create the role. Deploy with an existing role instead: `--role-arn <the role's ARN>` (the script then makes no IAM calls). In the class account that is the course's Lambda role:

```bash
uv run python scripts/deploy_backtest_lambda.py --role-arn arn:aws:iam::620832638024:role/Sysen5900LambdaExecRole --function-name sysen5900-factor-lake-backtest
```

Without a usable role, run `uv run python scripts/deploy_backtest_lambda.py --print-role-policy` and send the output to an AWS admin. If the keys may not change the bucket's lifecycle rules, the deploy still finishes and says what to ask an admin for.

It finishes by pinging the function. Options: `--memory`, `--timeout`, `--function-name`, `--role-arn`, `--package-only out.zip` (build the zip without deploying).

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

For four sample backtests (Mixing and Integrating, every delisting strategy, equal and market cap weighting, the fossil fuel filter and a cohort comparison), it runs each one locally on the S3 data, then in Lambda, and compares every number. It then runs each one in Lambda again and checks that it came from the cache, unchanged. It uses new inputs each time, so the first Lambda call always computes. The Lambda uses the same library versions but runs on Linux x86, so numbers may differ in the last binary digit. The check allows a relative difference of 1e-9 and prints the largest difference it found.

Holdings are compared stock by stock, leaving out **Rank** and **Score**. Winsorizing gives the top 1% of each factor exactly the same score, and the engine sorts those ties in an order that depends on the machine. So a Mac and Lambda can list tied stocks in a different order, and for a stock bought by several Mixing sleeves, show the Rank and Score from a different sleeve. The stocks held, shares, prices, weights, returns and contributions still have to match. When a percentage cuts through a tied block (for example a 1% cohort), the engine's choice of which tied stocks to buy also depends on the machine and on row order. That is a known engine issue, to be fixed separately with deterministic tie-breaking.

The unit tests (`tests/unit/test_backtest_lambda.py`) run the same Lambda code against a fake S3 and drive the real app, checking that a Lambda run equals a local run for both factor combinations, each delisting strategy and the cohort comparison, and that a repeated run is served from the cache.

## 5. Demo script (about 5 minutes)

Before the call: deploy (section 2), add `FACTOR_LAKE_BACKTEST_LAMBDA` to `.env`, and start the app with `uv run streamlit run app/streamlit_app.py`. To warm the Lambda up, run one backtest. To keep the "first run computes" moment for the call, pick inputs you have not run yet (for example a new AUM).

1. **The switch.** Point out **Run backtests in AWS Lambda** in the sidebar and the caption with the function name. On the Analysis tab, the caption says the market data does not need to be loaded.
2. **First run computes in Lambda.** Add two factors (for example ROE using 9/30 Data and ROA %), pick Integrating, and click **Run Portfolio Analysis** without loading data. The confirmation says "computed in AWS Lambda in N s". In the Results tab, the **Computed** line shows the result ID (the first 12 characters of its hash).
3. **The repeat is cached.** Click **Run Portfolio Analysis** again with the same settings. The new run shows the same result ID and "returned from the AWS Lambda result cache in 0.N s", with the time it was first computed.
4. **Any change is a new backtest.** Change the AUM or a factor direction and run again: it computes (new result ID).
5. **Same results as before.** Turn the Lambda switch off, click **Load Market Data**, and run the same settings as step 2. The new run tab shows the same final value, CAGR and holdings as the Lambda run. For the precise check, run `scripts/check_backtest_lambda.py` in a terminal: "match: yes" with the largest difference for each backtest, and "cached=True" for every repeat.
6. **Cohort comparison.** In a Lambda run's tab, open **Run Cohort Comparison** and click **Generate Comparison**: the caption says it ran in Lambda.
7. **Where results live.** In the AWS console: S3 → `sysen-5900-factor-lake` → `factor-lake-data/backtest-results/` shows one JSON file per result hash. Management → Lifecycle rules shows the 90-day expiry (in the class account an admin has to add it; see section 2). Lambda → `factor-lake-backtest` (`sysen5900-factor-lake-backtest` in the class account) → Monitor shows the invocations, and its CloudWatch logs show "Computed ..." and "Cache hit ..." lines.

## 6. Share an analysis by its ID

The hash a result is saved under doubles as its **analysis ID**. Each saved file holds the job (factors, directions, settings) next to the result, so the ID alone is enough to show the analysis again. The app reads it straight from S3 (`src/saved_analyses.py`), without calling the Lambda, so opening never reruns a backtest.

- Each Lambda run's tab shows the full ID with a copy button.
- **Results → Open a saved analysis** opens an ID as a new tab with the original results and settings.
- **This session's analyses** lists only the runs of the current browser session. Nothing is listed from S3, so nobody sees other people's analyses unless they're given an ID.
- Errors: text that isn't a 64-character ID, an ID with no saved analysis (mistyped, or deleted by the 90-day rule), a result saved before sharing existed, or a cohort comparison's ID.

Demo (about 2 minutes, after the steps in section 5):

1. In a run's tab, copy its **Analysis ID**.
2. Open the app in a second browser window (a private window works) to stand in for a colleague. Its **Results** tab shows **Open a saved analysis** and no runs.
3. Paste the ID and click **Open analysis**. The same final value, CAGR, holdings and charts appear in a tab labelled "(opened)", with "Opened from a saved analysis (first computed …)". Generate Comparison also works there.
4. Paste a wrong ID (for example drop the last character, or change one character) to show the errors: "is not an analysis ID" and "No saved analysis has the ID … deleted after 90 days".
5. Back in the first window, **This session's analyses** lists your runs with their IDs. The second window lists only the analysis it opened.

## 7. Troubleshoot

| Message | Fix |
|---|---|
| `Could not call the backtest Lambda ... AccessDeniedException` | The app's AWS key needs `lambda:InvokeFunction` on the function |
| `... ResourceNotFoundException` | Deploy the function (section 2) or fix `FACTOR_LAKE_BACKTEST_LAMBDA` |
| `The backtest Lambda failed: Task timed out` | Raise `--timeout` (up to 900 seconds) and redeploy |
| `... Runtime.OutOfMemory` | Raise `--memory` and redeploy |
| `The backtest Lambda could not run this backtest: ...` | The backtest itself failed; the message is the engine's error |

## 8. Cost

Each uncached backtest is one Lambda invocation of a few seconds. A cold start adds the time to download the 8 MB market data file. Cached repeats take well under a second, and saved results are small JSON files deleted after 90 days. Class use stays far inside the Lambda and S3 free tiers.

## 9. WRDS mode

WRDS backtests (see the user guide, section 15, and `DOCS/WRDS_MODE_DESIGN.md`) run in the same function. A job with `data_mode: "wrds"` goes to `src/wrds_service.py`; every other job is a Legacy job, handled exactly as before.

- **Data.** The function reads three files under `factor-lake-data/wrds_2026_10/` (`factor_panel_monthly`, `returns_monthly`, `market_monthly`), only the columns it needs, once per container. The folder is set by the function's `FACTOR_LAKE_WRDS_DIR` variable (the deploy script's `--wrds-dir`). A new WRDS pull goes into a new folder.
- **Cache.** The key hashes the job, the S3 ETags of the three files and a WRDS code version. Legacy keeps its own code version, so WRDS changes never recompute Legacy results.
- **Holdings.** A WRDS backtest writes `backtest-results/<hash>.holdings.parquet` (one row group per rebalance date) **before** `backtest-results/<hash>.json`. The app reads one date at a time with its own S3 keys, in two ranged GETs. A cached result whose holdings file is missing is computed again.

### Deploy a dev function

Test changes on a separate function and leave the live one alone:

```bash
uv run python scripts/deploy_backtest_lambda.py --function-name sysen5900-factor-lake-backtest-dev \
    --role-arn <the course role's ARN> --leave-lifecycle-rules
```

- `--leave-lifecycle-rules` does not write the bucket's lifecycle rules (a normal deploy rewrites them, and the bucket is shared with the live app). It only reads them and says whether `backtest-results/` expires.
- `--wrds-dir wrds_YYYY_MM` points the function at another WRDS folder (default `wrds_2026_10`).

To point a local app at the dev function, set the function name next to the AWS keys in `.streamlit/secrets.toml` (top level):

```
FACTOR_LAKE_BACKTEST_LAMBDA = "sysen5900-factor-lake-backtest-dev"
```

To check the WRDS engine locally on the pipeline's files (IWM tracking, run time and memory), run `uv run python scripts/check_wrds_engine.py "<path>/processed/final"`. It prints its results and stores none.

### Validation on the dev function (2026-10-28)

Lambda's own durations (from the invocation log tail), 3008 MB function, course role. All runs were computed (no cache) unless marked.

| Job | Container | Duration | Max memory | Result JSON | Holdings Parquet | Result |
|---|---|---|---|---|---|---|
| Monthly, whole universe, cap weighted (IWM tracking check) | cold: init 2.5 s, plus loading the WRDS data | 10.2 s | 1154 MB | 85 KB | 47.9 MB | PASS (correlation ≥ 0.99, tracking error ≤ 3%) |
| Monthly, Mixing, 3 factors | warm | 3.0 s | 1154 MB | 85 KB | 22.1 MB | |
| Monthly, Integrating, 2 factors | warm | 1.5 s | 1154 MB | 85 KB | 6.0 MB | |
| Monthly, Mixing, 3 factors, cap weighted | warm | 2.9 s | 1154 MB | 85 KB | 23.3 MB | |
| Monthly, Mixing, 4 factors | warm | 3.5 s | 1154 MB | 85 KB | 27.6 MB | |
| Monthly, Mixing, 10 factors | warm | 6.5 s | 1154 MB | 85 KB | 49.7 MB | |
| The first one again | warm, cache hit | 0.11 s | 1154 MB | 85 KB | | |
| 3 jobs at once: semi-annual / quarterly / monthly | 1 warm, 2 cold (init 2.6 s) | 0.5 / 6.9 / 8.6 s | 1154 / 890 / 996 MB | 16 / 29 / 84 KB | 2.3 / 5.1 / 16.8 MB | all finished, 9.6 s wall |
| From the app: semi-annual / quarterly / monthly backtest | warm | 0.7 / 1.1 / 2.9 s | 996 MB | 16 / 30 / 85 KB | 3.3 / 7.0 / 22.1 MB | PASS |
| From the app: cohort comparison (monthly, 10%) | warm | 3.6 s | 996 MB | 11 KB | | PASS |
| From the app: ranked list, one date | warm | 0.3 s | 996 MB | 0.9 MB | | PASS |
| From the app: Legacy backtest | warm | 2.4 s | 996 MB | 2.3 MB | | equals `main`'s result (≤ 1e-9 relative) |

- **Warm repeats do not slow down on Linux.** Five different monthly jobs in a row on one container took time in line with their number of factors, not their order. (On Windows, repeated engine runs in one process did get slower.)
- **Cold start** for WRDS: about 2.5 s of Lambda init, plus a few seconds the first WRDS job of a container spends loading the WRDS data.
- **Memory**: at most about 1.15 GB of the function's 3008 MB, with the Legacy and WRDS data both loaded.
- **Package**: 61.7 MB zipped, 199 MB unzipped (Lambda allows 250 MB).

### Admin: confirm the 90-day rule

Saved results and WRDS holdings both live under `factor-lake-data/backtest-results/`, so one lifecycle rule on that prefix expires both. The class keys cannot read the bucket's lifecycle rules, so **an AWS admin must confirm** that the rule `factor-lake-expire-backtest-results` (prefix `factor-lake-data/backtest-results/`, expire after 90 days) is enabled on `sysen-5900-factor-lake`.
