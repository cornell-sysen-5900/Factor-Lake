# WRDS data preview and single-source transition

New analyses use WRDS only. The Analysis source switch and local market-data loading controls are removed.
Saved Legacy results still open by ID or share link in Results and Compare, with their original settings and
metrics. Legacy settings exports remain readable, but importing one for a new run or using a Legacy run's
settings is refused with an explanation. The Legacy engines and saved-result decoders remain in the codebase.
See the [Legacy-to-WRDS table](WRDS_FACTOR_REVIEW.md#legacy-to-wrds) for each Legacy factor's WRDS replacement.

## Data Viewer

After sample display is approved and enabled, choose a year range and any of the four cascading GICS lists:
Sector, Industry group, Industry, Sub-industry. The deepest selected level defines the filter; empty lists mean
all choices within the selected parent. These preview controls are independent of the Analysis sidebar.
Click **Load preview**. The viewer shows at most five matching rows per year, ordered by date and PERMNO.
Fewer rows are shown if fewer exist; missing years and empty results are explained. Null values are shown as a dash.
These are deterministic examples of fields, not a statistically representative sample or a portfolio.

Only identifiers, classification fields, price, market cap and allowlisted factor fields are included. The raw
column names and native units are preserved (for example, ROA is a ratio). Forward returns are excluded.
GICS is the company's current classification across its history. Sector uses the existing SIC fallback;
finer levels require a GICS code, exactly as in the backtest filter.

There is no CSV download button or dataframe download toolbar. The table is escaped static HTML. Displaying
rows necessarily makes those displayed values accessible in the browser; this is not a copy-prevention system.
Changing filters, clearing the preview or leaving the tab discards the current preview. Only one bounded result
is kept in the app session; neither the panel nor the sample index is loaded into Streamlit.

## Architecture and performance

The Lambda action `wrds_preview` accepts only `start_year`, `end_year`, and `gics_filter` in `request`.
It never calls the backtest engine, creates an analysis ID, or downloads the returns or market files.
The server enforces the year bounds and five-row-per-year cap. The client validates the response too.

A preparation script scans projected panel batches and retains five rows per year/sector/sub-industry cell.
Every supported GICS filter is a union of these cells; merging the first five candidates per cell produces the
same first five rows as filtering the full panel first. Thus rare classifications can still be previewed.
The compact Parquet index is tied to the panel ETag and an algorithm version. Lambda reads only that index,
keeps one version in memory, and checks the panel ETag on every request. A changed panel without a rebuilt index
returns an unavailable message, not stale rows. No per-filter cache accumulates.

Limits: index at most 100,000 rows / 16 MiB compressed; response at most 2 MiB and five rows per requested year.
Measure cold and warm Lambda latency and Streamlit RSS on the development deployment. Do not infer cold-start
latency from a warm request or from the small final row count. The performance target is a few seconds; actual
acceptance depends on deployed measurements, including Lambda initialization and the network.

## Preparation and release

**Sample display defaults to disabled in both Lambda and Streamlit. Prof. Steward must approve showing sample
rows before either is enabled on a user-facing deployment.** A request cannot override the server gate.
Record that approval in the team's release checklist; this document does not constitute approval.

1. Configure your existing AWS credentials locally; never add them or real sample files to git.
2. Prepare the compact index without changing AWS:

   ```bash
   uv run python scripts/build_wrds_preview.py
   ```

3. Upload the index explicitly when the team is ready:

   ```bash
   uv run python scripts/build_wrds_preview.py --upload
   ```

   Only a versioned key under the current WRDS folder's `preview/` is written. Existing indexes, raw data and
   saved results are never overwritten. The uploader verifies the uploaded bytes. Rebuild after each new
   panel version. This upload alone does not enable display.

4. Deploy the reviewed code to the **development** Lambda using the team's normal deployment process.
   `scripts/deploy_backtest_lambda.py` creates a new function with preview disabled and, without
   `--preview-approved` or `--preview-disabled`, keeps an existing function's setting. Keep production unchanged
   during acceptance.
   The Lambda role needs read access to the panel metadata and its versioned preview index under the WRDS prefix;
   the existing read-prefix policy covers both. The preparer's credentials need write access to the index key.
5. Before approval, verify the action returns `status: disabled` and no rows. Run the synthetic tests locally:

   ```bash
   uv run pytest tests/unit/test_wrds_preview.py tests/unit/test_wrds_preview_app.py
   ```

6. **Only after approval:** deploy with `--preview-approved` or explicitly set the Lambda environment variable
   `FACTOR_LAKE_WRDS_PREVIEW_APPROVED=true`. Set the same variable in Streamlit's environment or secrets, then
   restart it. A frontend-only flag cannot bypass the backend gate. Later redeploys without a preview option
   keep the backend flag as it is.
7. Verify unfiltered previews, every GICS level, rare and empty filters, partial years, timeouts, missing index,
   cold/warm latency, and repeated-filter app memory. Reopen real saved WRDS and Legacy IDs in Results and Compare.
8. To disable display, deploy with `--preview-disabled` (or set the Lambda variable to `false`), unset the
   Streamlit flag or set it to `false`, and restart the app. It will clear
   its session's preview on the next viewer render. Already displayed values cannot be recalled from a browser.

All committed tests use synthetic data. Keep real rows, empirical statistics, credentials and private measurement
logs out of this public repository, its PR text and commit messages.
