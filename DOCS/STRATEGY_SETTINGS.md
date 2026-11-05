# Save, import and name backtests

On **Analysis**, open **Save or import strategy settings** after selecting factors.
Enter an optional backtest name (up to 100 characters) and note (up to 1,000 characters).
These are copied into each successful run; blank names keep the existing automatic labels.

- **Export settings (JSON)** downloads the current inputs, name and note.
- To restore a configuration, choose a file with **Import strategy JSON**, then click
  **Apply imported settings**. Review the restored inputs before running. Import does
  not execute a backtest or fetch market data. Local backtests require **Load Market Data**.
- In **Results**, open **Name, note and settings file** to edit that run's annotations
  or export the exact settings used for that run, independently of the current sidebar.
  Names and notes are session-only; downloaded files preserve them for later use.
- **Clear Parameters** preserves saved runs and their names, notes and results.

Both Legacy and WRDS are supported, including factors and directions, years, investment,
weighting, factor combination, ESG and sector filters, trading costs, and the applicable
Legacy delisting strategy or WRDS rebalancing frequency. Import restores the file's data
mode. WRDS import requires a configured backtest Lambda. The Legacy execution preference
is local to the app and is not exported or overwritten.

Files contain inputs and annotations only: no credentials, source data, calculation
results, AWS configuration or analysis IDs. Exporting and importing the same file restores
the same settings. Identical results also require the same market data and calculation
logic. Import invalidates the currently filtered local data and Data Viewer while keeping
cached source data and saved results.

## File format and validation

Version 1 uses `schema_version`, `data_mode`, `name`, `note`, `settings`, and `factors`.
Factors are identified by data columns rather than display labels. All required fields,
types, modes, supported factors/sectors, directions, ranges and date ordering are checked
before any inputs change. Files larger than 64 KB, duplicate JSON keys, extra fields,
unknown schema versions, nonfinite values and malformed JSON are rejected with an error.

This version supports the portfolio percentage and benchmark exposed by the current app
(default 10% and benchmark index 1). Export explicitly rejects previously saved Legacy
runs with other values instead of silently dropping them. Trading costs must be whole
basis points, matching the current input control.

Names retain the unique run number in tab labels, so duplicate names are safe. Annotation
edits affect session presentation only; they do not alter engine inputs, results, shared
analysis IDs or cache keys. Historical/shared analysis records retain their original labels.

## Demo (local Legacy mode)

1. Select **ROA %**, choose **Lower is better**, and set 2010-2020 and $2,500.
2. Enter **Value hypothesis** and a short note; export the settings JSON.
3. Load market data, run the analysis, and show the named Results tab.
4. Rename the run and change its note; show that its metrics stay the same.
5. Return to Analysis and click Clear Parameters. The saved run remains available.
6. Import the downloaded file and click Apply imported settings. Show that the original
   factor, direction, dates, investment, name and note return.
7. Load market data and rerun. With the same underlying data, the metrics match the first run.
8. Upload an invalid JSON file and apply it. The error appears and current inputs remain intact.

## Developer checks

```bash
uv sync --locked --group dev
uv run pytest tests/unit/test_strategy_config.py
uv run pytest
uv run streamlit run app/streamlit_app.py
```

The new tests use synthetic data and fake AWS services. AppTest supplies file bytes at
its unsupported uploader boundary and exercises the real Apply callback, widget updates,
backtest execution, round trips and annotation edits.
