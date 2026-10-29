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
- **Earlier analyses on this browser** (`app/analysis_history.py`) lists IDs from earlier sessions on the same browser. The app has no logins, so they are kept in the browser's `localStorage` (key `factor-lake-analyses`, newest 200). A small inline JavaScript component (`st.components.v2`) adds each session's IDs and reports the stored list to the app once per page load. Reporting more often would rerun the app and clear messages like "Analysis complete".
- **All saved analyses** lists everyone's newest 20 from S3 (`list_saved_analyses`). It reads only the files' metadata (the Lambda saves a short `summary` and the `kind`, so cohort comparisons are left out), and only while expanded. The app's key needs `s3:ListBucket` on `backtest-results/`; without it the list explains that. Results saved before summaries existed show "Details not recorded".
- Errors: text that isn't a 64-character ID, an ID with no saved analysis (mistyped, or deleted by the 90-day rule), a result saved before sharing existed, or a cohort comparison's ID.

Demo (about 2 minutes, after the steps in section 5):

1. In a run's tab, copy its **Analysis ID**.
2. Open the app in a second browser window (a private window works) to stand in for a colleague. Its **Results** tab shows **Open a saved analysis** and no runs.
3. Paste the ID and click **Open analysis**. The same final value, CAGR, holdings and charts appear in a tab labelled "(opened)", with "Opened from a saved analysis (first computed …)". Generate Comparison also works there.
4. Paste a wrong ID (for example drop the last character, or change one character) to show the errors: "is not an analysis ID" and "No saved analysis has the ID … deleted after 90 days".
5. Back in the first window, **This session's analyses** lists your runs with their IDs. The second window lists only the analysis it opened.
6. Close both windows, open the app again in the first browser, and go to **Results → Earlier analyses on this browser**: the analyses from before are listed with their IDs. Click **Open** on one, and it opens without pasting anything.
7. Expand **All saved analyses** to show everyone's recent analyses with their factors and dates, and open one from there.

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
