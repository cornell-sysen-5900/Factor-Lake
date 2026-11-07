# Save, import and name backtests

Each run in **Results** has a row of actions beside **Remove this run**:

- **Rename** names the run (up to 100 characters) and adds a note (up to 1,000 characters).
  The name replaces the automatic label after the run number; leave it empty to restore the
  automatic label. The note is shown under the actions. Names and notes are session-only;
  exported settings files keep them.
- **Export settings (JSON)** downloads the exact inputs used for that run, with its name and
  note, independently of the current sidebar.
- **Use these settings** loads the run's inputs into the Analysis tab and opens it, so you can
  change them and run again. It runs nothing.

To restore a configuration from a file, open the **Saved analyses** panel in **Results**
(point at the **Saved analyses** tab on the right edge of the page; the panel slides over the
results and hides again when you move away) and choose the file under
**Import settings from file**. The settings are applied as soon as the file is chosen and the
Analysis tab opens to show them. Import does not execute a backtest or fetch market data;
local backtests require **Load Market Data**. If the file has a name, the next run gets it
(shown above the run buttons); **Clear Parameters** forgets it. **Clear Parameters**
preserves saved runs and their names, notes and results.

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
(default 10% and benchmark index 1). Export and Use these settings are unavailable (with the
reason on hover) for previously saved Legacy runs with other values, instead of silently
dropping them. Trading costs must be whole basis points, matching the current input control.

Names retain the unique run number in tab labels, so duplicate names are safe. Renaming
keeps the run's tab open and renames it in **This session's analyses** too. Annotation
edits affect session presentation only; they do not alter engine inputs, results, shared
analysis IDs or cache keys. Historical/shared analysis records retain their original labels.

## Demo (local Legacy mode)

1. Select **ROA %**, choose **Lower is better**, and set 2010-2020 and $2,500.
2. Load market data and run the analysis.
3. **Rename** the run to **Value hypothesis** with a short note; show that its metrics stay the
   same. Click **Export settings (JSON)**.
4. Return to Analysis and click Clear Parameters. The saved run remains available.
5. In Results, click **Use these settings**: the Analysis tab opens with the run's factor,
   direction, dates and investment.
6. Clear Parameters again, then import the downloaded file in the **Saved analyses** panel.
   The same inputs return, and the next run will be named **Value hypothesis**.
7. Load market data and rerun. With the same underlying data, the metrics match the first run.
8. Import an invalid JSON file. The error appears in the panel and current inputs remain intact.

## Developer checks

```bash
uv sync --locked --group dev
uv run pytest tests/unit/test_strategy_config.py
uv run pytest
uv run streamlit run app/streamlit_app.py
```

The tests use synthetic data and fake AWS services. AppTest supplies file bytes at its
unsupported uploader boundary and exercises the panel's import and rerun, Use these settings,
widget updates, backtest execution, round trips and annotation edits.
