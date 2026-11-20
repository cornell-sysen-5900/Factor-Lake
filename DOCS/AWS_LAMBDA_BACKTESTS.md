# Backtests in AWS Lambda

Every backtest runs in AWS Lambda. The app sends only the inputs (factors, directions and sidebar settings). Lambda reads the WRDS data from S3, runs the backtest, and saves the result in S3, so the app never downloads or holds the market data. New analyses use WRDS only; older Legacy analyses still open by ID (see [Older saved runs](#older-saved-runs-legacy)).

## 1. How it works

```
Streamlit app ──inputs──▶ Lambda factor-lake-backtest ──reads──▶ s3://.../factor-lake-data/wrds_2026_11/*.parquet
      ▲                         │
      └──result or S3 link──────┴──saves──▶ s3://.../factor-lake-data/backtest-results/<hash>.json
                                                                       (+ <hash>.holdings.parquet)
```

1. **Inputs only.** Run Portfolio Analysis (and Generate Comparison) sends a small JSON job: factors, directions, rebalancing frequency, years, AUM, weighting, Hold, factor combination, the GICS filter, the fossil fuel filter, the liquidity limit and, above 0 bps, the trading cost (section 10). See `src/wrds_jobs.py`.
2. **One engine, checked against a local run.** Lambda runs `src/wrds_service.py` and the vectorized engine `src/wrds_engine.py`. After each deploy, `scripts/check_backtest_lambda.py` runs the same job locally on the S3 data and compares every number (section 4).
3. **Cached by hash.** The result is saved under a SHA-256 hash of the inputs, the data version (the S3 ETags of the WRDS files) and the backtest code version (a hash of the engine's source files). Republishing the data or deploying new backtest code gives fresh results.
4. **Repeats are instant.** When a result with that hash exists, Lambda returns it without rerunning the backtest. Results are shared, so a run by any user is cached for everyone.
5. **Result or link.** Results up to 5 MB come back in the Lambda response (Lambda's limit is 6 MB). Bigger results come back as a temporary S3 link (valid for 1 hour), which the app downloads.
6. **1-year expiry.** The weekly "Delete old results" GitHub Action deletes saved results older than 1 year, with their holdings files (section 12).

The Results tab shows how each run was produced, for example "Computed: Returned from the AWS Lambda result cache in 0.4 s (first computed 2026-10-22 18:03:11 UTC)".

## 2. Deploy the Lambda

Needs AWS keys that can create IAM roles and Lambda functions.

```bash
uv run python scripts/deploy_backtest_lambda.py
```

It is safe to run again: it creates what is missing and updates the rest. It sets up:

| Resource | Details |
|---|---|
| IAM role `factor-lake-backtest-lambda` | Read `factor-lake-data/*`, write `factor-lake-data/backtest-results/*`, CloudWatch logs |
| Lambda `factor-lake-backtest` | Python 3.12, 3008 MB, 10-minute timeout. pandas, numpy and pyarrow are bundled in the package at the versions your environment runs (Linux builds fetched with `uv`; unused parts such as test suites and pyarrow's Flight libraries removed), so Lambda computes with the same libraries as the app and needs no Lambda layer. The package (about 62 MB zipped, 199 MB unzipped; Lambda allows 250 MB) is uploaded to `factor-lake-data/_deploy/` first, because direct uploads are limited to 50 MB |
| S3 lifecycle rule `factor-lake-expire-backtest-results` (optional) | Only when the keys may change the bucket's lifecycle rules: deletes `factor-lake-data/backtest-results/` objects after 1 year, like the weekly cleanup. Other rules on the bucket are kept |

**Keys without IAM access** (such as the class `Sysen5900Student` user, which is denied all IAM actions) cannot create the role. Deploy with an existing role instead: `--role-arn <the role's ARN>` (the script then makes no IAM calls). In the class account that is the course's Lambda role:

```bash
uv run python scripts/deploy_backtest_lambda.py --role-arn arn:aws:iam::620832638024:role/Sysen5900LambdaExecRole --function-name sysen5900-factor-lake-backtest
```

Without a usable role, run `uv run python scripts/deploy_backtest_lambda.py --print-role-policy` and send the output to an AWS admin. If the keys may not change the bucket's lifecycle rules, the deploy still finishes: the weekly cleanup deletes old results, so no admin is needed for that.

It finishes by pinging the function. Options: `--memory`, `--timeout`, `--function-name`, `--role-arn`, `--package-only out.zip` (build the zip without deploying).

### Deploy from CI

The app on Streamlit Cloud redeploys from `main` on every merge, and `.github/workflows/deploy-lambda.yml` keeps the Lambda in step with it. A push to `main` that changes `src/`, `aws_lambda/`, `scripts/deploy_backtest_lambda.py` or `uv.lock` (which sets the bundled pandas, numpy and pyarrow versions) runs:

1. **Unit tests.** The workflow calls `tests.yml`. Nothing is deployed unless the tests pass.
2. **Deploy.** It runs `uv sync --locked`, then the deploy command above with `--role-arn` and `--function-name` from the repository variables, plus `--leave-lifecycle-rules`, so CI never rewrites the shared bucket's lifecycle settings (it only checks them; saved results are deleted after 1 year by the weekly cleanup, section 12). The Python version comes from `.python-version`, so the Lambda gets the library versions locked in `uv.lock`. The WRDS data folder is the code's default (`DEFAULT_WRDS_DIR` in `src/wrds_data.py`), so a new WRDS pull is one code change.
3. **Check.** It runs `scripts/check_backtest_lambda.py` (section 4) against the new function: one quarterly WRDS backtest end to end, plus the Legacy samples that keep older saved runs reproducible.

The run fails if any step fails, so a red run means the Lambda may be behind `main`. To redeploy by hand, open Actions → Deploy Backtest Lambda → Run workflow (only `main` deploys). Deploys never overlap. A push that arrives during a deploy waits for it to finish.

A repository admin has to add these under Settings → Secrets and variables → Actions:

| Kind | Name | Value |
|---|---|---|
| Secret | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Keys that may update the function, pass its role, upload to `factor-lake-data/_deploy/` and read the market data (the class `Sysen5900Student` keys can) |
| Variable | `LAMBDA_ROLE_ARN` | `arn:aws:iam::620832638024:role/Sysen5900LambdaExecRole` |
| Variable | `LAMBDA_FUNCTION_NAME` | `sysen5900-factor-lake-backtest` |

Until all four are set, the workflow stops at its first step and names what is missing. It never falls back to the script's default function name.

Old packages are not deleted. Each package in `factor-lake-data/_deploy/` is named by its content hash, so a deploy with unchanged code reuses the same file, and only real changes add a new one (about 62 MB each). To clear them out, delete all but the newest few `backtest-lambda-*.zip` files in the S3 console, or ask an AWS admin for a lifecycle rule on `factor-lake-data/_deploy/`.

## 3. Turn it on in the app

Set the function name next to the AWS keys, in `.env` locally or in the Streamlit Cloud secrets:

```
FACTOR_LAKE_BACKTEST_LAMBDA="factor-lake-backtest"
```

The app's AWS key also needs `lambda:InvokeFunction` on that function. The sidebar then names the function ("WRDS backtests run in AWS Lambda: …"). Without the variable, the sidebar says "WRDS mode runs in AWS Lambda, which is not configured for this app." and Run Portfolio Analysis is disabled; saved analyses still open by ID if the app can read S3.

## 4. Check it

```bash
uv run python scripts/check_backtest_lambda.py
```

It checks one quarterly WRDS backtest (2002-2026, the whole data, two factors) end to end, at 0 bps and again at 25 bps. The function must read the WRDS folder the code expects (its `FACTOR_LAKE_WRDS_DIR`). The Lambda's result must match the same job run locally on the WRDS data in S3. The newest rebalance date's holdings, read from the Lambda's holdings file the way the app reads them (two ranged GETs), must match the local holdings of that date. A second Lambda call must come from the cache, unchanged. Each 25 bps run must also come back with its trading cost fields and a lower final value than the same job at 0 bps; the check prints its average turnover, total cost and annual cost drag.

The Lambda uses the same library versions but runs on Linux x86, so numbers may differ in the last binary digit. The check allows a relative difference of 1e-9 and prints the largest difference it found. It also runs the Legacy samples that keep older saved runs reproducible ([Older saved runs](#older-saved-runs-legacy)). `--skip-wrds` leaves the WRDS part out.

The unit tests (`tests/unit/test_wrds_service.py`, `test_wrds_saved.py`, `test_wrds_check_script.py`) run the same Lambda code against a fake S3 with synthetic WRDS data and drive the real app, checking the holdings files, the cache, and that a repeated run is served from the cache.

## 5. Demo script (about 5 minutes)

Before the call: deploy (section 2), add `FACTOR_LAKE_BACKTEST_LAMBDA` to `.env`, and start the app with `uv run streamlit run app/streamlit_app.py`. To warm the Lambda up, run one backtest. To keep the "first run computes" moment for the call, pick inputs you have not run yet (for example a new AUM).

1. **Where it runs.** Point out the sidebar caption naming the Lambda function. On the Analysis tab, the caption says the backtest runs in AWS Lambda on the WRDS data, with nothing to load in the app.
2. **First run computes in Lambda.** Add two factors (for example ROE and 12-1 Mo Momentum), keep Quarterly, and click **Run Portfolio Analysis**. The confirmation says "computed in AWS Lambda in N s". In the Results tab, the header shows how it was computed and its Analysis ID.
3. **The repeat is cached.** Click **Run Portfolio Analysis** again with the same settings. The new run shows the same Analysis ID and "returned from the AWS Lambda result cache in 0.N s", with the time it was first computed.
4. **Any change is a new backtest.** Change the AUM or a factor direction and run again: it computes (new Analysis ID).
5. **Same results as a local run.** Run `scripts/check_backtest_lambda.py` in a terminal: it runs a WRDS job locally on the S3 data and in Lambda and prints "match: yes" with the largest difference, and "cached=True" for the repeat.
6. **Cohort comparison.** In a run's tab, open **Run Cohort Comparison** and click **Generate Comparison**: the caption says it ran in Lambda.
7. **Where results live.** In the AWS console: S3 → `sysen-5900-factor-lake` → `factor-lake-data/backtest-results/` shows one JSON file per result hash, with a `.holdings.parquet` file next to each WRDS backtest. Old results are deleted by the weekly "Delete old results" GitHub Action (section 12). Lambda → `factor-lake-backtest` (`sysen5900-factor-lake-backtest` in the class account) → Monitor shows the invocations, and its CloudWatch logs show "Computed ..." and "Cache hit ..." lines.

## 6. Share an analysis by its ID

The hash a result is saved under doubles as its **analysis ID**. Each saved file holds the job (factors, directions, settings) next to the result, so the ID alone is enough to show the analysis again. The app reads it straight from S3 (`src/saved_analyses.py`), without calling the Lambda, so opening never reruns a backtest.

- Each Lambda run's tab shows the full ID with a copy button.
- **Results → Open a saved analysis** opens an ID as a new tab with the original results and settings.
- **This session's analyses** lists only the runs of the current browser session. Nothing is listed from S3, so nobody sees other people's analyses unless they're given an ID.
- **Earlier analyses on this browser** (`app/analysis_history.py`) lists IDs from earlier sessions on the same browser. The app has no logins, so they are kept in the browser's `localStorage` (key `factor-lake-analyses`, newest 200). A small inline JavaScript component (`st.components.v2`) adds each session's IDs and reports the stored list to the app once per page load. Reporting more often would rerun the app and clear messages like "Analysis complete".
- **All saved analyses** lists everyone's newest 20 from S3 (`list_saved_analyses`). It reads only the files' metadata (the Lambda saves a short `summary` and the `kind`, so cohort comparisons and WRDS ranked lists are left out), and only while expanded. Only `<ID>.json` results are listed, never the WRDS holdings files next to them. A WRDS analysis's summary starts with its data source and frequency, e.g. "WRDS, Quarterly | ROE (higher), ROA (higher) | Mixing | 2008-2026 | $1,000", and the session and browser lists show the same line under a WRDS run's label. The app's key needs `s3:ListBucket` on `backtest-results/`; without it the list explains that. Results saved before summaries existed show "Details not recorded".
- The **Scoreboard** tab ranks everyone's saved analyses by the scores saved in their metadata (section 11).
- Errors: text that isn't a 64-character ID, an ID with no saved analysis (mistyped, or deleted after 1 year by the weekly cleanup, section 12), a result saved before sharing existed, or a cohort comparison's ID.

Demo (about 2 minutes, after the steps in section 5):

1. In a run's tab, copy its **Analysis ID**.
2. Open the app in a second browser window (a private window works) to stand in for a colleague. Its **Results** tab shows **Open a saved analysis** and no runs.
3. Paste the ID and click **Open analysis**. The same final value, CAGR, holdings and charts appear in a tab labelled "(opened)", with "Opened from a saved analysis (first computed …)". Generate Comparison also works there.
4. Paste a wrong ID (for example drop the last character, or change one character) to show the errors: "is not an analysis ID" and "No saved analysis has the ID … Saved analyses are deleted after 1 year".
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

Each uncached backtest is one Lambda invocation of a few seconds. A cold start adds a few seconds to load the WRDS data (section 9). Cached repeats take well under a second. Saved results (small JSON files, plus a holdings Parquet file per WRDS backtest) are deleted after 1 year (section 12). Class use stays far inside the Lambda and S3 free tiers.

## 9. WRDS mode

WRDS backtests (see the user guide, section 15, and `DOCS/WRDS_MODE_DESIGN.md`) are the only kind the app runs. Every job it sends has `data_mode: "wrds"` and goes to `src/wrds_service.py`.

- **Data.** The function reads three files under `factor-lake-data/wrds_2026_11/` (`factor_panel_monthly`, `returns_monthly`, `market_monthly`; the panel runs from 2002-07), only the columns it needs, once per container. The folder is set by the function's `FACTOR_LAKE_WRDS_DIR` variable (the deploy script's `--wrds-dir`, which defaults to `DEFAULT_WRDS_DIR` in `src/wrds_data.py`, as CI deploys it). A new WRDS pull goes into a new folder; the previous one, `wrds_2026_10/`, stays for rollback (set `DEFAULT_WRDS_DIR` back and merge).
- **Cache.** The key hashes the job, the S3 ETags of the three files and a WRDS code version. Legacy keeps its own code version, so WRDS changes never recompute Legacy results.
- **Holdings.** A WRDS backtest writes `backtest-results/<hash>.holdings.parquet` (one row group per rebalance date) **before** `backtest-results/<hash>.json`. The app reads one date at a time with its own S3 keys, in two ranged GETs. A cached result whose holdings file is missing is computed again.
- **Listings.** The result JSON's metadata has the job's `kind` and a summary starting "WRDS, <frequency>" (section 6). WRDS results saved before this (on the dev function) have none and show "Details not recorded" until the weekly cleanup deletes them after 1 year (section 12).
- **Live deploys** come from CI (section 2, "Deploy from CI"), whose check step runs a WRDS backtest end to end.

### Older saved runs (Legacy)

Analyses saved before the app became WRDS-only (PR #87) still open by ID in Results and Compare: the app reads them from S3 and never recomputes them. The function still accepts Legacy jobs (a job without `data_mode`, run by `src/backtest_engine.py`), and the post-deploy check's Legacy samples and `tests/unit/test_legacy_pinned.py` keep those results reproducible. The app no longer sends Legacy jobs.

### Deploy a dev function

Test changes on a separate function and leave the live one alone:

```bash
uv run python scripts/deploy_backtest_lambda.py --function-name sysen5900-factor-lake-backtest-dev \
    --role-arn <the course role's ARN> --leave-lifecycle-rules
```

- `--leave-lifecycle-rules` does not write the bucket's lifecycle rules (a normal deploy rewrites them, and the bucket is shared with the live app). It only reads them, and warns if a rule would delete `backtest-results/` before 1 year.
- `--wrds-dir wrds_YYYY_MM` points the function at another WRDS folder (default `DEFAULT_WRDS_DIR` in `src/wrds_data.py`, now `wrds_2026_11`).

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

### Factor risk on the dev function (2026-11-14)

Two WRDS backtests, 2003-2025, Mixing, equal weight, 10 bps, computed (no cache) on the dev function. Each result's `factor_risk` was recomputed with statsmodels (OLS, HAC with the same lag) from its returned monthly returns and `market_monthly.parquet`: coefficients agree to 1e-15 and t-stats to 5e-14. The monthly returns compound to the period returns to 1e-13 percentage points. The lines are the ones the app builds from those saved numbers (Size is left out of them).

| Job | Months | Alpha / yr (t) | R² | Exposures (t): Mkt, SMB, HML, RMW, CMA, UMD | Line | Compute | Result JSON |
|---|---|---|---|---|---|---|---|
| Monthly: Book/Price + 12-1 Mo Momentum | 276 (lag 5) | -0.19% (-0.14) | 0.94 | 1.20 (31.0), 1.08 (20.3), 0.17 (3.4), -0.27 (-3.7), 0.07 (0.8), -0.18 (-3.1) | Mostly a low-profitability and value tilt; alpha not significant. | 8.7 s | 144 KB |
| Quarterly: ROE + Earnings/Price | 276 (lag 5) | -0.82% (-0.81) | 0.96 | 1.05 (30.6), 0.90 (22.8), 0.26 (6.2), 0.28 (4.9), -0.11 (-2.0), -0.18 (-5.2) | Mostly a value and anti-momentum tilt; alpha not significant. | 1.2 s | 56 KB |

### Hold size on the dev function (2026-11-15)

One sub-industry job, Regional Banks (GICS 40101015), Book/Price (higher is better), quarterly 2003-2025 (92 rebalances), Mixing, equal weight, 10 bps, at four Hold choices, each computed (no cache) on the dev function. The holdings counts were read from each run's holdings file in S3 and checked against the filter computed separately from `factor_panel_monthly.parquet` (members with a positive price and market cap in the sub-industry): 10% is `max(1, floor(0.1 n))` and Top 20 is `min(20, n)` at every date (n = stocks with a Book/Price), and 100%, with or without the factor, is every stock in the filter, including the few without a Book/Price (for example 223 held, 220 scored, on 2003-06-30). The two 100% runs have identical returns.

| Hold | Stocks held per rebalance | CAGR | Compute | Result JSON |
|---|---|---|---|---|
| Top 10% (default) | 14 to 27 | 3.77% | 4.7 s (the container's first job, with the WRDS data load) | 56 KB |
| Top 100% | 146 to 270 (the whole filter) | 6.38% | 0.8 s | 56 KB |
| Top 100%, no factors | 146 to 270 (the whole filter) | 6.38% | 0.7 s | 56 KB |
| Top 20 stocks | 20 | 4.82% | 0.7 s | 56 KB |

Top N in Mixing holds N stocks in total (the factors take turns). The same Regional Banks job with two factors, Book/Price then 12-1 Mo Momentum, at Top 20 (computed fresh, 5.0 s as the first job after the deploy, CAGR 3.55%) held exactly 20 stocks at all 92 rebalances, 10 picked by each factor. Each date's picks, and the factor that picked each stock, match the turn rule recomputed separately from the S3 panel's z-scores. The two factors' top 20s shared stocks on 39 of the 92 dates (up to 4).

### Expiry of WRDS holdings

Saved results and WRDS holdings both live under `factor-lake-data/backtest-results/`. The weekly cleanup (section 12) deletes a holdings file with its result, and a holdings file whose result is already gone once it is older than 1 year.

## 10. Trading costs

Both job types take an optional **`trading_cost_bps`**: the cost of every buy and sell in basis points of its value, a number from 0 to 500 (anything else is refused with a readable error). The engines charge it on each rebalance's turnover; see the user guide, section 16, and `src/trading_costs.py`.

```json
{"kind": "backtest", "data_mode": "wrds", "factors": ["f_roe", "mom_12_1"], "...": "...", "trading_cost_bps": 25.0}
```

- **Only above 0.** The app adds the field only when the cost is above 0, and `job_hash` drops an explicit `0`. A 0 bps job is therefore exactly the job it was before trading costs existed, and so is its hash input. WRDS ranked lists never carry the field (ranks do not depend on costs).
- **Cache keys.** The job part of the key is unchanged at 0 bps (pinned in `tests/unit/test_trading_cost_jobs.py` against keys computed with the previous `main`). The key also includes the code version (a hash of the engine files), which this change updates, as every engine change does: after this deploy, a 0 bps job is computed once more and gets a new ID, with the same results as before. IDs saved earlier still open, because opening reads S3 by ID and never recomputes.
- **Old IDs** have no `trading_cost_bps` in their job and no cost fields in their result. They open at 0 bps, with no turnover views.
- **Results** above 0 bps add `trading_cost_bps`, `turnover_pct`, `trading_cost_pct` and `trading_cost_dollars` (per period), `returns_before_costs`, `portfolio_values_before_costs`, `average_turnover_pct`, `total_trading_cost`, `cagr_before_costs`, `cagr_after_costs` and `cost_drag`. Legacy holdings rows add `Traded %`; WRDS holdings Parquet files add a `traded` column. At 0 bps none of these are added, and results are identical to before.
- **Summaries** (the S3 metadata the saved-analysis list reads) end with "| 25 bps" above 0.

### Validation on the dev function (2026-11-03)

Deployed to `sysen5900-factor-lake-backtest-dev` with `--leave-lifecycle-rules` (Legacy code version `c4daff7be0a3c34a`). Runs from the local app pointed at the dev function, two factors (Legacy: ROE using 9/30 Data and ROA %, 2002-2024; WRDS: ROE and 12-1 Mo Momentum, 2008-2026), Mixing, equal weight, $1,000. Times are the app's round trip, computed (not cached), with the backtest's own time in brackets.

| Run | Cost | Final value | CAGR | Avg turnover / rebalance | Total cost | Cost drag / yr | Time | 0 bps vs `main` |
|---|---|---|---|---|---|---|---|---|
| Legacy, annual | 0 bps | $9,460.16 | 10.75% | | | | 2.3 s (1.6 s) | equal (≤ 5e-16 relative) |
| Legacy, annual | 25 bps | $8,931.01 | 10.46% | 104.5% | $228.36 | 0.29% | 2.3 s (1.7 s) | |
| WRDS, quarterly | 0 bps | $4,857.67 | 9.05% | | | | 1.2 s (1.0 s) | equal (≤ 2.2e-16), all 73 holdings dates equal |
| WRDS, quarterly | 25 bps | $4,229.49 | 8.22% | 75.8% | $264.19 | 0.82% | 1.2 s (1.0 s) | |
| WRDS, monthly | 25 bps | $5,510.90 | 9.71% | 44.4% | $575.07 | 1.47% | 3.2 s (3.0 s) | |
| Cohort of the quarterly 25 bps run, 10% | 25 bps | top $4,229.49 / bottom $1,970.82 | | | | | 1.6 s (1.4 s) | |

- "0 bps vs `main`" ran `main`'s code (faeca5c) locally on the same S3 data and compared every number with the dev function's result (relative 1e-9, as the check script).
- The before-cost values of the 25 bps runs equal the 0 bps runs' final values ($9,460.16 and $4,857.67).
- `scripts/check_backtest_lambda.py --function-name sysen5900-factor-lake-backtest-dev`: all checks passed, including the Legacy (102.9% average turnover, 0.28% drag) and WRDS (75.8%, 0.82%) runs at 25 bps.

### Validation on the dev function (2026-11-08): GICS filter

Deployed to `sysen5900-factor-lake-backtest-dev` with `--leave-lifecycle-rules` (WRDS code version `2799f38f32311ed9`). WRDS, quarterly 2002-2026 (96 rebalances), ROE and 12-1 Mo Momentum, Mixing, equal weight, a fresh AUM so every run was computed. Each run matched the same job run locally on the S3 data (relative difference 0) and came from the cache the second time (0.15 s).

| Job | Filter | Compute (round trip) | Holdings rows (per date) |
|---|---|---|---|
| Sector | Financials | 5.5 s (5.9 s), first run after the deploy: includes loading the data | 6,154 (53-75) |
| Industry group | Software & Services | 0.7 s (1.0 s) | 2,087 (14-32) |
| Industry | Banks | 0.8 s (0.9 s) | 3,338 (25-47) |
| Sub-industry | Application Software, Regional Banks | 0.7 s (0.9 s) | 4,438 (32-59) |
| Older job, `selected_sectors` | Financials | 0.8 s (0.9 s) | 6,154 (53-75), the same result as the Sector job |

## 11. Scoreboard scores

The Scoreboard tab (user guide, section 18) ranks every saved analysis without downloading any. When the Lambda saves an analysis (`kind` `backtest`, Legacy or WRDS), it adds its **scores** to the result's S3 metadata, next to `kind` and `summary` (`src/analysis_scores.py`). For example (numbers rounded):

```json
{"v":1,"mode":"legacy","frequency":"Annual","combination":"mixing","start_year":2002,"end_year":2024,
 "initial_aum":1000.0,"trading_cost_bps":0.0,"factors":["ROE using 9/30 Data","ROA %"],
 "metrics":{"final_value":9460.16,"total_return":8.46,"cagr":0.1075,"outperformance":0.52,"sharpe":0.61,
            "max_drawdown":-0.42,"volatility":0.21,"win_rate":0.57}}
```

- **Metrics** use the Results tab's formulas (the same as `app/compare.py`; `tests/unit/test_scoreboard.py` checks they are equal): CAGR counts years as periods / periods per year, outperformance is cumulative against the analysis's own benchmark, and Sharpe, max drawdown, raw volatility and win rate are the result's own. A missing, NaN or infinite value is saved as `null` (n/a).
- **Size.** S3 allows 2 KB of metadata in all. The scores are compact ASCII JSON of at most 1,200 characters; when long factor labels would pass that, `factors` is saved as `null` (the summary still names them).
- **Listing.** `list_scored_analyses` (`src/saved_analyses.py`) lists `backtest-results/` and reads each result's metadata with `head_object`, 16 at a time (about 1 second for 160 results; one at a time took 11 s). "All saved analyses" uses the same listing. Each listing reads the newest 1,000 results at most.
- **Code versions.** `src/analysis_scores.py` and `src/backtest_service.py` are not in `ENGINE_SOURCES`, so the Legacy code version and Legacy analysis IDs do not change. `src/wrds_service.py` is in `WRDS_ENGINE_SOURCES`, so this change gives WRDS a new code version: a WRDS job run again after the deploy is computed once more and gets a new ID, with the same results. IDs saved earlier still open.
- **Cache hits** return the saved result as it is, so an analysis computed before the scores existed gets none by being run again.
- **Checked after each deploy.** `scripts/check_backtest_lambda.py` reads the metadata of every analysis it computes and checks that the scores were saved and equal the scores of its local run (relative 1e-9), and that cohort comparisons have none.

### Score analyses saved before the scoreboard

`scripts/score_saved_analyses.py` adds scores to the analyses that have none. It reads each one, computes its scores from the saved result (nothing is rerun) and rewrites only its metadata, with an S3 copy onto itself: the result, the analysis and its ID do not change. Results saved before summaries also get their `kind` and `summary`, so a cohort comparison saved then leaves the saved-analysis lists. Analyses saved before sharing have no job, so they cannot be scored and are left as they are.

```bash
uv run python scripts/score_saved_analyses.py            # dry run: lists what it would change
uv run python scripts/score_saved_analyses.py --apply    # rewrite their metadata
```

A metadata rewrite changes an object's `LastModified`, so its first save time is kept in its metadata (`first-saved`), which the listings show and sort by, and from which the weekly cleanup counts its year (section 12). The key needs `s3:GetObject` and `s3:PutObject` on `backtest-results/`. Running it twice changes nothing: scored analyses are skipped.

## 12. Deleting old results

Saved results last **1 year** (`RESULT_RETENTION_DAYS = 365` in `src/backtest_service.py`). The **Delete old results** GitHub Action (`.github/workflows/delete-old-results.yml`) runs `scripts/delete_old_results.py` every Monday at 06:17 UTC and deletes older ones, so the bucket doesn't grow forever and no S3 lifecycle rule, and so no AWS admin, is needed.

- **What it deletes.** Only files under `factor-lake-data/backtest-results/`; the market data is never touched. A saved result (`<ID>.json`: an analysis, cohort comparison or ranked list) is deleted once it is more than 365 days old, counted from when it was first saved (its `first-saved` metadata if the scoreboard backfill rewrote it, else its `LastModified`). Its WRDS holdings file (`<ID>.holdings.parquet`) goes with it; a holdings file whose result is already gone is deleted once it is more than 365 days old. Any other file there is left alone.
- **Job summary.** Each run's summary (on the run's page in GitHub Actions) says how many files it deleted, results and holdings, and their total size, and lists up to 50 of them.
- **By hand.** Actions → **Delete old results** → **Run workflow** is a dry run that only lists what it would delete, unless **Delete them** is ticked. Locally:

```bash
uv run python scripts/delete_old_results.py            # dry run
uv run python scripts/delete_old_results.py --apply    # delete
```

- **Keys.** It uses the repository secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (the deploy's), which need `s3:ListBucket` on the prefix and `s3:GetObject` and `s3:DeleteObject` on `backtest-results/`.
- **Scheduled runs** use the workflow file on `main`, and GitHub pauses schedules in a repository with no activity for 60 days.
- **The deploy** (section 2) no longer needs an admin for expiry. It only warns if a bucket lifecycle rule would delete saved results before 1 year.
