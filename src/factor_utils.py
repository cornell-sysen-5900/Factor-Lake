"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/factor_utils.py
PURPOSE: Numerical transformation and normalization utilities for quantitative signals.
VERSION: 2.1.0
"""

import numpy as np
import pandas as pd
from typing import Optional

def normalize_series(
    s: pd.Series,
    higher_is_better: bool = True,
    method: str = 'negate',
    winsorize_pct: Optional[float] = 0.01,
    zscore: bool = True
) -> pd.Series:
    """
    Transforms a raw factor distribution into a standardized, comparable signal.

    This utility processes a cross-section of financial data by mitigating the 
    influence of outliers and adjusting the directional polarity of the signal. 
    Standardization is achieved via Z-score normalization:

    $$Z = \frac{x - \mu}{\sigma}$$

    Args:
        s: The raw numerical factor values.
        higher_is_better: If True, larger values represent a stronger signal.
        method: The inversion strategy ('negate' or 'reciprocal').
        winsorize_pct: The tail fraction to cap at both ends (e.g., 0.01 for 1%).
        zscore: If True, centers the distribution at zero with unit variance.

    Returns:
        pd.Series: A standardized series of factor scores.
    """
    # Numerical validation and initial cleanup
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).copy()
    
    if s.dropna().empty:
        return s

    # 1. Outlier Mitigation (Winsorization)
    if winsorize_pct and winsorize_pct > 0:
        lower_limit = s.quantile(winsorize_pct)
        upper_limit = s.quantile(1 - winsorize_pct)
        s = s.clip(lower=lower_limit, upper=upper_limit)

    # 2. Directional Alignment (Polarity)
    if not higher_is_better:
        if method == 'reciprocal':
            # Replace zero with NaN to prevent infinite values
            s = (1.0 / s.replace(0, np.nan))
        else:
            s = -s

    # 3. Statistical Standardization
    if zscore:
        mu = s.mean()
        sigma = s.std(ddof=1)
        
        # Guard against zero-variance to prevent division by zero errors
        if sigma > 1e-12:
            s = (s - mu) / sigma
        else:
            s = s - mu  # Center the data if scaling is mathematically impossible

    return s