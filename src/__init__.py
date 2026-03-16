"""Factor-Lake src package initialization.

Exports primary modules for backtesting and portfolio analytics. 
Following the 2026 refactor, the engine logic is centralized in backtest_engine.

Usage:
    from src.backtest_engine import rebalance_portfolio, run_cohort_comparison
    from src.portfolio_filters import filter_universe
"""

from . import backtest_engine    # noqa: F401
from . import portfolio          # noqa: F401
from . import portfolio_filters  # noqa: F401
from . import benchmarks         # noqa: F401
from . import factor_registry    # noqa: F401
from . import factors_doc        # noqa: F401
from . import factor_utils       # noqa: F401
from . import performance_metrics # noqa: F401
from . import supabase_client    # noqa: F401