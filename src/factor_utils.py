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

    This utility processes a cross-section of financial data by addressing 
    extreme outliers through winsorization and adjusting the directional 
    polarity of the signal. The resulting output is typically a Z-score, 
    calculated as:
    
    $$Z = \frac{x - \mu}{\sigma}$$

    Args:
        s (pd.Series): The raw numerical factor values.
        higher_is_better (bool): If True, larger values represent a stronger signal.
        method (str): The inversion strategy ('negate' or 'reciprocal').
        winsorize_pct (Optional[float]): The tail fraction to cap at both ends.
        zscore (bool): If True, center the distribution around zero with unit variance.

    Returns:
        pd.Series: A standardized series of factor scores.
    """
    # Ensure numerical consistency
    s = pd.to_numeric(s, errors='coerce').copy()
    
    # Validation gate for empty or all-NaN series
    if s.dropna().empty:
        return s

    # 1. Outlier Mitigation (Winsorization)
    # Mitigates the influence of extreme values on the mean and standard deviation
    if winsorize_pct and winsorize_pct > 0:
        lower_limit = s.quantile(winsorize_pct)
        upper_limit = s.quantile(1 - winsorize_pct)
        s = s.clip(lower=lower_limit, upper=upper_limit)

    # 2. Directional Alignment
    # Inverts the signal if a lower value is quantitatively superior
    if not higher_is_better:
        if method == 'reciprocal':
            # Guard against division by zero
            s = s.replace(0, np.nan)
            s = 1.0 / s
        else:
            s = -s

    # 3. Statistical Standardization
    if zscore:
        mu = s.mean()
        sigma = s.std(ddof=1) # Use sample standard deviation
        
        # Guard against zero-variance to prevent division by zero
        if sigma > 1e-12:
            s = (s - mu) / sigma
        else:
            s = s - mu

    return s