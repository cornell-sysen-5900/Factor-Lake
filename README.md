# Factor-Lake

An interactive factor-investing toolkit with a clean Streamlit UI, market data stored in AWS S3, and a pytest test suite. The codebase uses a modern `src/` layout and a clean UX.

## Use the App

- Hosted: Share your Streamlit Community Cloud app URL. Users only need the link to use the app (open access, no password required).
- Data: Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` in Streamlit secrets for cloud deploys (or `.env` for local runs). The app then loads its data from S3; see `DOCS/AWS_S3_DATA.md`. Supabase (`SUPABASE_URL`, `SUPABASE_KEY`) still works as a fallback.

Example secrets (TOML):
```
AWS_ACCESS_KEY_ID = "your-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-secret-access-key"
AWS_DEFAULT_REGION = "us-east-1"

# Optional fallback
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-public-key"
```

## Quick Start (Local)

### 1. Install uv (one-time)

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and run

```bash
git clone https://github.com/cornell-sysen-5900/Factor-Lake.git
cd Factor-Lake
uv sync --group dev
uv run pytest                                      # run unit tests
uv run streamlit run app/streamlit_app.py           # launch the app
```

Then open http://localhost:8501

## Features

- Clean factor selection (13 core factors: Momentum, Value, Quality, Growth, Profitability)
- ESG exclusion (fossil fuel filter)
- Sector filtering (configurable sector universe)
- Market data loaded from AWS S3 (Parquet), shared by all sessions, with Supabase as a fallback
- Annual rebalancing backtest (configurable period, currently 2002-2024 in UI)
- WRDS mode: point-in-time WRDS data with semi-annual, quarterly or monthly rebalancing, run in AWS Lambda (user guide section 15; design in `DOCS/WRDS_MODE_DESIGN.md`)
- Benchmark comparison vs Russell 2000, Growth, and Value
- Performance metrics: CAGR, yearly returns, drawdown, Sharpe, Information Ratio, win rate
- Ranked-stock table and top-vs-bottom cohort analysis
- Saved runs: each backtest run is kept as its own tab in Results (newest first, last 5; cleared when the page is refreshed), shows the settings that produced it, and can be removed
- Strategy files and names: export/import Legacy or WRDS settings as JSON, name backtests and add notes, and edit annotations in Results without changing calculations. See [Save, import and name backtests](DOCS/STRATEGY_SETTINGS.md).
- Clear and rerun: on the Analysis tab, click **Clear Parameters** to remove selected factors and restore the analysis settings to their defaults. Choose new inputs, then **Run Portfolio Analysis**. For local backtests, click **Load Market Data** first to apply the new filters using the cached source data. Saved runs, analysis history, the Lambda execution preference and the data source (Legacy or WRDS) are kept; in WRDS mode the frequency and years return to Quarterly, 2008-2026. No page refresh or app restart is needed.
- Data Viewer tab: filter the loaded market data by date range, sector and industry, view it, and download it as CSV (no extra data downloads)

## Project Layout

```
Factor-Lake/
├── app/                    # Streamlit app and UI components
│   ├── streamlit_app.py    # Main entrypoint
│   ├── streamlit_utils.py  # Session state / orchestration helpers
│   ├── streamlit_config.py # Factor and UI metadata
│   ├── saved_runs.py       # Saved-run list helpers (Results tabs)
│   ├── viewer_data.py      # Data Viewer filtering and CSV helpers
│   └── components/         # Sidebar, factor selection, results, data viewer, about
├── src/                    # Library & core logic
│   ├── backtest_engine.py
│   ├── benchmarks.py
│   ├── performance_metrics.py
│   ├── portfolio.py
│   ├── portfolio_filters.py
│   ├── factor_registry.py
│   ├── factor_utils.py
│   ├── factors_doc.py
│   ├── s3_client.py        # S3 data source (default)
│   ├── supabase_client.py  # Supabase data source (fallback)
│   ├── data_standardization.py
│   └── ...
├── Visualizations/         # Plot helpers
├── tests/
│   ├── unit/               # Unit tests (default pytest target)
│   └── integration/        # Integration tests (require Supabase creds)
├── scripts/                # CI / helper scripts
├── DOCS/                   # Supplementary documentation
├── pyproject.toml          # Dependencies, project metadata, pytest config
├── uv.lock                 # Pinned dependency lockfile
└── README.md
```

## Live Calculation Preview

The Analysis tab shows a **Calculation Preview** directly below Factor Selection, before
Load Market Data. It updates immediately when factors, directions, combination method,
weighting, delisting strategy, ESG, sectors, years or initial investment change. No market
data or cloud credentials are needed to preview the strategy. With no factors, a placeholder
is shown. Expand **Step-by-step calculation** for normalization, selection, allocation,
delisting and compounding details.

After running, **Calculation Used in This Run** in Results preserves the formula from that
run. Current sidebar changes do not change historical formulas. Opened saved analyses use
their stored inputs. For local runs, changing universe filters or years after loading data
reapplies those selections to the cached raw data before execution.

Run the acceptance tests with `uv run pytest tests/unit/test_calculation_preview.py`.

## Import Conventions

Always import from `src`:
```python
from src.backtest_engine import rebalance_portfolio
from src.portfolio import Portfolio
from src.factor_registry import get_factor_column
```

## Code Style

The Python code follows [PEP 8](https://peps.python.org/pep-0008/), checked by [ruff](https://docs.astral.sh/ruff/) on every pull request (the CI **Lint** step fails on any error):

```bash
uv run ruff check .          # list problems
uv run ruff check --fix .    # fix whitespace, import order and other safe fixes
```

The rules are in `pyproject.toml` (`[tool.ruff]`): PEP 8's errors and warnings (`E`, `W`), its naming rules (`N`) and import order (`I`), with lines up to **120 characters**, the width the code was already written at. Three exceptions are allowed, each explained there: the fake AWS clients in `tests/` take boto3's argument names (`Bucket=`, `Key=`), `app/streamlit_app.py` sets up the environment before its imports, and the `Visualizations` package keeps its capitalized name.

Changing a backtest engine file (`ENGINE_SOURCES` in `src/backtest_jobs.py`, `WRDS_ENGINE_SOURCES` in `src/wrds_jobs.py`), even only its formatting, changes the backtest code version: after the deploy, each backtest run again is computed once more and gets a new analysis ID. Old IDs still open.

## Documentation

- `DOCS/index.md` - Documentation home
- `DOCS/CONTRIBUTING.md` - Contribution guidelines
- `DOCS/DEPLOYMENT.md` - Streamlit Community Cloud deployment
- `DOCS/SUPABASE_SETUP.md` - Environment and credentials setup
- `DOCS/STREAMLIT_STYLING_GUIDE.md` - Styling and UI customization
- `DOCS/Bandit & Safety.md` - Security scanning notes
- `DOCS/REORGANIZATION_SUMMARY.md` - Historical refactor summary (contains legacy context)

## Deployment

For detailed deployment instructions (Streamlit Community Cloud, secrets management, and troubleshooting), see `DOCS/DEPLOYMENT.md`.

Run locally with:

```bash
uv run streamlit run app/streamlit_app.py
```

## Contributing

1. Create a feature branch from `main`.
2. Add/modify tests in `tests/unit/` and/or `tests/integration/`.
3. Run unit tests: `uv run pytest`, and the PEP 8 check: `uv run ruff check .` (see [Code Style](#code-style))
4. Run integration tests (requires Supabase and/or AWS creds):
   ```bash
   SUPABASE_URL="..." SUPABASE_KEY="..." uv run pytest tests/integration -v
   ```
5. Submit a PR describing UX/data impacts.

## Market Data in S3

The app reads its data from Parquet files in AWS S3 (`s3://sysen-5900-factor-lake/factor-lake-data/`). Publish new data with `scripts/publish_data_to_s3.py` and the daily CRSP prices with `scripts/publish_daily_data_to_s3.py`, and check the live app with 5 simultaneous sessions using `scripts/load_test_sessions.py` (or on demand from the Load Test workflow in GitHub Actions). See `DOCS/AWS_S3_DATA.md`.

### Latest WRDS data

The app's WRDS mode and the backtest Lambda read **`s3://sysen-5900-factor-lake/factor-lake-data/wrds_2026_11/`**
(data version 2026-11-07), a point-in-time Russell 2000 panel from **2002-07 to 2026-09**, built by the WRDS pipeline
in [`pipeline/`](pipeline/README.md). Files, new columns, source dates and rollback:
[`DOCS/AWS_S3_DATA.md`](DOCS/AWS_S3_DATA.md#latest-wrds-data). Columns:
[`pipeline/DATA_DICTIONARY.md`](pipeline/DATA_DICTIONARY.md). Next refresh:
[`pipeline/DATA_UPDATE_GUIDE.md`](pipeline/DATA_UPDATE_GUIDE.md).

## Supabase Archiver

The `scripts/archive_supabase_tables.py` script automatically discovers, downloads, and archives all of your Supabase tables to Parquet format. It creates a GitHub Release to store the large `.parquet` binary files efficiently without bloating your repository history.

**Usage:**
1. Copy `.env.example` to `.env` and fill in your `SUPABASE_URL` and `SUPABASE_KEY` (must be the `service_role` secret to bypass RLS and fetch the schema).
2. Ensure you have the GitHub CLI (`gh`) installed and authenticated (`gh auth login`).
3. Run the archiver:

```pwsh
python scripts/archive_supabase_tables.py
```
