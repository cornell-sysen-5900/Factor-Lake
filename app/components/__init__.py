"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/__init__.py
PURPOSE: Package initialization for Streamlit UI component submodules.
VERSION: 2.1.0
"""

# The following modules are exposed as sub-packages within the components namespace.
# Direct submodule imports are preserved to maintain compatibility with existing logic.

from . import about             # noqa: F401
from . import factor_selection  # noqa: F401
from . import results_tab       # noqa: F401
from . import sidebar           # noqa: F401