#marking as python package
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: visualizations/__init__.py
PURPOSE: Package initialization for the portfolio reporting and visualization suite.
VERSION: 2.1.0
"""

# The following modules are exposed as sub-packages within the visualizations namespace.
# Direct submodule imports are preserved to maintain compatibility with existing logic.

from . import portfolio_growth_plot     # noqa: F401
from . import top_bottom_portfolio_plot  # noqa: F401