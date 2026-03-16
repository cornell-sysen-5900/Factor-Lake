"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/__init__.py
PURPOSE: Package initialization for core analytical engines and data providers.
VERSION: 3.0.0
"""

# The following modules are exposed as sub-packages within the src namespace.
# Direct submodule imports are preserved to maintain compatibility with existing logic.

from . import backtest_engine      # noqa: F401
from . import portfolio            # noqa: F401
from . import portfolio_filters    # noqa: F401
from . import benchmarks           # noqa: F401
from . import factor_registry      # noqa: F401
from . import factors_doc          # noqa: F401
from . import factor_utils         # noqa: F401
from . import performance_metrics  # noqa: F401
from . import supabase_client      # noqa: F401