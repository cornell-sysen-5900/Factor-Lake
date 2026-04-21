# Factor-Lake

An interactive factor-investing toolkit with a clean Streamlit UI, Supabase data integration, and a pytest test suite. The codebase uses a modern `src/` layout and a clean UX.

## Use the App

- Hosted: Share your Streamlit Community Cloud app URL. Users only need the link to use the app (open access, no password required).
- Supabase: Set `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit secrets for cloud deploys (or `.env` for local runs).

Example secrets (TOML):
```
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-public-key"
```

## Quick Start (Local)

```pwsh
git clone https://github.com/cornell-sysen-5900/Factor-Lake.git
cd Factor-Lake
pip install -r requirements.txt
python -m streamlit run .\app\streamlit_app.py
```

Then open http://localhost:8501

## Features

- Clean factor selection (13 core factors: Momentum, Value, Quality, Growth, Profitability)
- ESG exclusion (fossil fuel filter)
- Sector filtering (configurable sector universe)
- Supabase data loading with column normalization
- Annual rebalancing backtest (configurable period, currently 2002-2024 in UI)
- Benchmark comparison vs Russell 2000, Growth, and Value
- Performance metrics: CAGR, yearly returns, drawdown, Sharpe, Information Ratio, win rate
- Ranked-stock table and top-vs-bottom cohort analysis

## Project Layout

```
Factor-Lake/
├── app/                    # Streamlit app and UI components
│   ├── streamlit_app.py    # Main entrypoint
│   ├── streamlit_utils.py  # Session state / orchestration helpers
│   ├── streamlit_config.py # Factor and UI metadata
│   └── components/         # Sidebar, factor selection, results, about
├── src/                    # Library & core logic
│   ├── backtest_engine.py
│   ├── benchmarks.py
│   ├── performance_metrics.py
│   ├── portfolio.py
│   ├── portfolio_filters.py
│   ├── factor_registry.py
│   ├── factor_utils.py
│   ├── factors_doc.py
│   ├── supabase_client.py
│   └── ...
├── Visualizations/         # Plot helpers
├── UnitTests/              # Unit test suite (default pytest target)
├── IntegrationTests/       # Integration tests (run explicitly)
├── scripts/                # CI / helper scripts
├── DOCS/                   # Supplementary documentation
├── requirements.txt
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

```pwsh
python -m streamlit run .\app\streamlit_app.py
```

## Contributing

1. Create a feature branch from `main`.
2. Add/modify tests in `UnitTests/` and/or `IntegrationTests/`.
3. Run unit tests with `pytest` (defaults to `UnitTests/` via `pytest.ini`).
4. Run integration tests explicitly when needed:

```pwsh
pytest IntegrationTests/ -m integration -v
```

5. Optionally use the helper runner:

```pwsh
python .\scripts\run_tests.py all
```

6. Submit a PR describing UX/data impacts.

