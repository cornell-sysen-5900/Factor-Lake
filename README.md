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

The newest data version is **`s3://sysen-5900-factor-lake/factor-lake-data/wrds_2026_11/`** (data version
2026-11-07), built by the WRDS pipeline in [`pipeline/`](pipeline/README.md) from the November 2026 WRDS,
Fama-French and FRED pulls ([`pipeline/PULLS_V2.md`](pipeline/PULLS_V2.md) lists every query). The point-in-time
panel runs from **2002-07 to 2026-09**. Every column is described in
[`pipeline/DATA_DICTIONARY.md`](pipeline/DATA_DICTIONARY.md). **The app's WRDS mode and the backtest Lambda read
this folder** (`DEFAULT_WRDS_DIR` in `src/wrds_data.py`); Legacy reads its own files and is unchanged.

| key under wrds_2026_11/ | contents |
|---|---|
| `factor_panel_monthly.parquet` | point-in-time factor panel: every Russell 2000 member at each month end from 2002-07, with fundamental, price and analyst factors, betas and next-month return |
| `universe_periods.parquet` | each stock's Russell 2000 membership periods, with their source |
| `returns_monthly.parquet` | monthly returns of every stock ever in the index, delisting returns included |
| `prices_daily/year=YYYY/data.parquet` | daily CRSP prices, returns, shares and market cap, one file per year from 1997 |
| `prices_daily/_manifest.json` | the years, rows and columns of the daily files (uploaded last) |
| `market_daily.parquet`, `market_monthly.parquet` | Russell 2000 index and ETF returns, Fama-French factors, Treasury and Moody's yields, from 1997 (IWM from 2000-05, IWN and IWO from 2000-07) |
| `rv_multiples_monthly.parquet` | valuation multiples with their GICS industry and sector percentile ranks |
| `rv_peer_stats_monthly.parquet` | peer percentiles of every multiple by GICS industry and month |
| `dcf_inputs_annual.parquet` | annual statements and FCFF for DCF valuation |
| `estimates_consensus.parquet` | the full IBES consensus (7 measures, FY1-FY3 and long-term growth), linked to permno |
| `fundamentals_quarterly.parquet` | quarterly Compustat statements, TTM sums and quarter-end factors |
| `security_master.parquet` | ids, names and GICS classification of every stock |
| `column_stats.parquet` | statistics of every column of the other files |

New in this version:
- **Membership from 2002.** `membership_source` says where a stock's membership comes from: `client_list` (the
  annual Russell 2000 lists on each 9/30, used before 2008-02; each list counts from the June reconstitution until the
  next, with no mid-year additions), `iwm` (IWM's holdings, from 2008-02) or `client_fill` (list members the old IWM
  fund id hides, 2008-02 to 2010-05).
- **`index_weight_source`**: `iwm`, or `client_list_static_930` before 2008-02, where `index_weight` is the list's 9/30
  weight held for the year. Don't weight portfolios with it; backtests use market cap.
- **GICS at four levels** with names (`gsector_name`, `ggroup_name`, `gind_name`, `gsubind_name`) and `gics_source`
  (`current_header`: each company's current classification applied to all history; there is no historical GICS).

Where each source ends: CRSP prices 2026-09-30, IWM holdings 2026-06-30, Compustat 2026-11 (datadate), IBES
2026-08-20, Fama-French 2026-09-30, FRED 2026-11-05.

**Previous version.** `wrds_2026_10/` (data version 2026-10-25, panel 2008-04 to 2026-09) stays in S3 unchanged.
To roll back, set `DEFAULT_WRDS_DIR = 'wrds_2026_10'` and `FIRST_MONTH = '2008-04'` (`src/wrds_calendar.py`) and
merge: CI redeploys the Lambda with that folder. Analyses saved on either version keep opening by their ID. Each new
data pull gets its own `wrds_YYYY_MM/` folder, and this section is updated to point at it. How the folder is built,
uploaded and inspected: [`pipeline/README.md`](pipeline/README.md); the S3 setup:
[`DOCS/AWS_S3_DATA.md`](DOCS/AWS_S3_DATA.md).

## Supabase Archiver

The `scripts/archive_supabase_tables.py` script automatically discovers, downloads, and archives all of your Supabase tables to Parquet format. It creates a GitHub Release to store the large `.parquet` binary files efficiently without bloating your repository history.

**Usage:**
1. Copy `.env.example` to `.env` and fill in your `SUPABASE_URL` and `SUPABASE_KEY` (must be the `service_role` secret to bypass RLS and fetch the schema).
2. Ensure you have the GitHub CLI (`gh`) installed and authenticated (`gh auth login`).
3. Run the archiver:

```pwsh
python scripts/archive_supabase_tables.py
```
