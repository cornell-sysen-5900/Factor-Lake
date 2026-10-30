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
3. Run unit tests: `uv run pytest`
4. Run integration tests (requires Supabase and/or AWS creds):
   ```bash
   SUPABASE_URL="..." SUPABASE_KEY="..." uv run pytest tests/integration -v
   ```
5. Submit a PR describing UX/data impacts.

## Market Data in S3

The app reads its data from Parquet files in AWS S3 (`s3://sysen-5900-factor-lake/factor-lake-data/`). Publish new data with `scripts/publish_data_to_s3.py` and the daily CRSP prices with `scripts/publish_daily_data_to_s3.py`, and check the live app with 5 simultaneous sessions using `scripts/load_test_sessions.py` (or on demand from the Load Test workflow in GitHub Actions). See `DOCS/AWS_S3_DATA.md`.

### Latest WRDS data

The newest data version is **`s3://sysen-5900-factor-lake/factor-lake-data/wrds_2026_10/`** (data version
2026-10-25), built by the WRDS pipeline in [`pipeline/`](pipeline/README.md) from the October 2026 WRDS, Fama-French
and FRED pulls. Every column is described in [`pipeline/DATA_DICTIONARY.md`](pipeline/DATA_DICTIONARY.md).

| key under wrds_2026_10/ | contents |
|---|---|
| `factor_panel_monthly.parquet` | point-in-time factor panel: every Russell 2000 member (IWM holding) at each month end, with fundamental, price and analyst factors, betas and next-month return |
| `universe_periods.parquet` | each stock's Russell 2000 membership periods |
| `returns_monthly.parquet` | monthly returns of every stock ever in the index, delisting returns included |
| `prices_daily/year=YYYY/data.parquet` | daily CRSP prices, returns, shares and market cap, one file per year |
| `prices_daily/_manifest.json` | the years, rows and columns of the daily files (uploaded last) |
| `market_daily.parquet`, `market_monthly.parquet` | Russell 2000 index and ETF returns, Fama-French factors, Treasury and Moody's yields |
| `rv_multiples_monthly.parquet` | valuation multiples with their GICS industry and sector percentile ranks |
| `rv_peer_stats_monthly.parquet` | peer percentiles of every multiple by GICS industry and month |
| `dcf_inputs_annual.parquet` | annual statements and FCFF for DCF valuation |
| `estimates_consensus.parquet` | the full IBES consensus (7 measures, FY1-FY3 and long-term growth), linked to permno |
| `fundamentals_quarterly.parquet` | quarterly Compustat statements, TTM sums and quarter-end factors |
| `security_master.parquet` | ids, names and GICS classification of every stock |
| `column_stats.parquet` | statistics of every column of the other files |

Where each source ends: CRSP prices 2026-09-30, IWM holdings 2026-06-30, Compustat 2026-09-30, IBES 2026-08-20,
Fama-French 2026-08-31, FRED 2026-10-22.

**The live app still reads the older files** (`full_precision_test.parquet`, `last_price_mapping.parquet`,
`daily_prices/`) until the "switch app to new data" story moves it to this folder. Each new data pull gets its own
`wrds_YYYY_MM/` folder, and this section is updated to point at it. How the folder is built, uploaded and inspected:
[`pipeline/README.md`](pipeline/README.md); the S3 setup: [`DOCS/AWS_S3_DATA.md`](DOCS/AWS_S3_DATA.md).

## Supabase Archiver

The `scripts/archive_supabase_tables.py` script automatically discovers, downloads, and archives all of your Supabase tables to Parquet format. It creates a GitHub Release to store the large `.parquet` binary files efficiently without bloating your repository history.

**Usage:**
1. Copy `.env.example` to `.env` and fill in your `SUPABASE_URL` and `SUPABASE_KEY` (must be the `service_role` secret to bypass RLS and fetch the schema).
2. Ensure you have the GitHub CLI (`gh`) installed and authenticated (`gh auth login`).
3. Run the archiver:

```pwsh
python scripts/archive_supabase_tables.py
```
