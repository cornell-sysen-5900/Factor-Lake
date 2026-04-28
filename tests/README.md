# Factor-Lake Test Suite

## Setup

```bash
uv sync --group dev
```

## Test Layout

```
tests/
├── README.md
├── conftest.py
├── unit/                      # No credentials needed
│   ├── test_backtest.py       # Portfolio construction, rebalance loop, benchmarks
│   ├── test_delisting.py      # Time-adjusted delisting strategies
│   ├── test_factors.py        # Factor registry and signal normalization
│   ├── test_filters.py        # Sector and ESG universe filters
│   └── test_portfolio.py      # Portfolio object CRUD and valuation
└── integration/               # Require SUPABASE_URL + SUPABASE_KEY
    ├── test_backtest_regression.py  # Regression: known-good portfolio values
    └── test_supabase.py             # Connection, pagination, data quality
```

## Running Tests

All commands run from the project root (`Factor-Lake/`).

### Unit tests (default)

```bash
uv run pytest
```

`pyproject.toml` sets `testpaths = ["tests/unit"]`, so bare `uv run pytest` runs unit tests only.

### Integration tests

```bash
uv run pytest tests/integration -v
```

Integration tests hit live Supabase and require credentials.

**Linux / macOS (bash/zsh):**
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
uv run pytest tests/integration -v
```

**Windows PowerShell:**
```powershell
$env:SUPABASE_URL = "https://your-project.supabase.co"
$env:SUPABASE_KEY = "your-anon-key"
uv run pytest tests/integration -v
```

**Windows CMD:**
```cmd
set SUPABASE_URL=https://your-project.supabase.co
set SUPABASE_KEY=your-anon-key
uv run pytest tests/integration -v
```

### All tests (unit + integration)

```bash
uv run pytest tests
```

### A single file or test

```bash
uv run pytest tests/unit/test_portfolio.py
uv run pytest tests/unit/test_portfolio.py::TestPortfolio::test_add_investment -v
```

### By marker

```bash
uv run pytest -m unit                  # unit-marked tests only
uv run pytest -m integration           # integration-marked tests only
uv run pytest -m "not slow"            # skip slow tests
```

### Coverage

```bash
uv run pytest --cov=src --cov-report=term --cov-report=html
```

HTML report lands in `htmlcov/index.html`.

## Debugging Failures

```bash
uv run pytest tests/unit/test_factors.py -v -s     # show print output
uv run pytest tests/unit -v -l                      # show locals on failure
uv run pytest tests/unit --pdb                      # drop into debugger on failure
```

## Markers

Defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker        | Purpose                             |
|---------------|-------------------------------------|
| `unit`        | No external services needed         |
| `integration` | Requires Supabase credentials       |
| `slow`        | Long-running tests                  |
| `fast`        | Quick-running tests                 |

## Troubleshooting

**Import errors** -- run from project root, not from inside `tests/`.

**Integration tests skipped** -- `SUPABASE_URL` and `SUPABASE_KEY` are not set. See the env var commands above.

**Regression value mismatch** -- expected values in `test_backtest_regression.py` were calibrated against live Supabase data. If upstream data changes, recalibrate the expected constants.
