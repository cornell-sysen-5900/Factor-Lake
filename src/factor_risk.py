"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/factor_risk.py
PURPOSE: How much of a WRDS run's return is exposure to common factors: a Fama-French five-factor
         plus momentum regression of its monthly excess returns.
VERSION: 1.0.0

The run's monthly returns after trading costs (src/wrds_engine.py, step 6) minus the
risk-free rate are regressed by OLS, with an intercept, on the six factors of
market_monthly.parquet: mktrf, smb, hml, rmw, cma and umd (pipeline/DATA_DICTIONARY.md).

- Alpha is the intercept, annualized x 12. Exposures are the slopes.
- t-statistics use Newey-West standard errors (Bartlett weights, with the n / (n - k)
  small-sample factor, as statsmodels' HAC with use_correction). The lag is
  floor(4 (n / 100) ^ (2/9)) months (Newey and West, 1994): 3 for 36 to 99 months, 4 for 100 to 272,
  5 from 273 (the full monthly range, 290 months).
- Below MIN_MONTHS months there is no result: seven coefficients need several years of
  months to be estimated with any precision.

Pure numpy and pandas (the Lambda bundles nothing else), no Streamlit.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# The factors, in the order of the regression and of the Results table
FACTOR_COLUMNS = ('mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd')
FACTOR_LABELS = {'mktrf': 'Market', 'smb': 'Size (SMB)', 'hml': 'Value (HML)', 'rmw': 'Profitability (RMW)',
                 'cma': 'Investment (CMA)', 'umd': 'Momentum (UMD)'}
# Fewest months with a regression: three years (29 degrees of freedom for 7 coefficients)
MIN_MONTHS = 36
# |t| at or above this is significant (about 5%, two-sided)
SIGNIFICANT_T = 2.0
# R-squared at or above this: the factors explain "most" of the month-to-month return
MOSTLY_R_SQUARED = 0.8
MONTHS_PER_YEAR = 12
# The plain-language line names at most this many tilts, the strongest (it stays short)
MAX_TILTS = 2
# Each factor's tilt in words, by the sign of its exposure. The market is described by
# R-squared. Size is left out: the factors are built on the whole US market, so every
# Russell 2000 strategy has a positive size exposure and it would lead every line.
TILT_WORDS = {'hml': ('value', 'growth'),
              'rmw': ('high-profitability', 'low-profitability'),
              'cma': ('conservative-investment', 'aggressive-investment'), 'umd': ('momentum', 'anti-momentum')}


def newey_west_lag(n: int) -> int:
    """The Newey-West lag for n months: floor(4 (n / 100) ^ (2/9))."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_cov(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    """
    Newey-West covariance of OLS coefficients: (X'X)^-1 S (X'X)^-1 x n / (n - k), with
    S = sum_t u_t^2 x_t x_t' + sum_{l=1..lag} (1 - l / (lag + 1)) sum_t u_t u_{t-l} (x_t x_{t-l}' + x_{t-l} x_t').
    """
    n, k = x.shape
    scores = x * residuals[:, None]
    s = scores.T @ scores
    for lag_l in range(1, lag + 1):
        gamma = scores[lag_l:].T @ scores[:-lag_l]
        s += (1.0 - lag_l / (lag + 1.0)) * (gamma + gamma.T)
    bread = np.linalg.inv(x.T @ x)
    return bread @ s @ bread * (n / (n - k))


def factor_regression(excess_returns: np.ndarray, factors: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    OLS of monthly excess returns (decimals, n) on factor returns (n x len(FACTOR_COLUMNS))
    with an intercept. None below MIN_MONTHS months.

    Returns months, newey_west_lag, alpha_monthly and alpha_annual (x 12, decimals),
    alpha_t, r_squared, and per factor (FACTOR_COLUMNS order) {'factor', 'exposure', 't_stat'}.
    """
    y = np.asarray(excess_returns, dtype=float)
    f = np.asarray(factors, dtype=float)
    n = len(y)
    if n < MIN_MONTHS:
        return None
    x = np.column_stack([np.ones(n), f])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coef
    lag = newey_west_lag(n)
    se = np.sqrt(np.diag(newey_west_cov(x, residuals, lag)))
    t = np.divide(coef, se, out=np.full_like(coef, np.nan), where=se > 0)
    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float((residuals ** 2).sum()) / total if total > 0 else float('nan')
    return {
        'months': n,
        'newey_west_lag': lag,
        'alpha_monthly': float(coef[0]),
        'alpha_annual': float(coef[0]) * MONTHS_PER_YEAR,
        'alpha_t': float(t[0]),
        'r_squared': r_squared,
        'factors': [{'factor': name, 'exposure': float(b), 't_stat': float(tb)}
                    for name, b, tb in zip(FACTOR_COLUMNS, coef[1:], t[1:])],
    }


def wrds_factor_risk(results: Dict[str, Any], market: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    The factor regression of a WRDS backtest: its monthly returns after costs
    (results['monthly_dates'] and ['monthly_returns'], in percent) minus rf, on the
    factors of market (market_monthly's columns, indexed by month end). rf is the
    engine's: a month without rf uses tbill_3m / 12. Months missing a factor (or both
    rates) are left out. None below MIN_MONTHS months, or for results without monthly
    returns.

    Returns factor_regression's fields plus first_month and last_month ('YYYY-MM').
    """
    if not results.get('monthly_dates'):
        return None
    dates = pd.DatetimeIndex(pd.to_datetime(results['monthly_dates']))
    returns = pd.Series(np.asarray(results['monthly_returns'], dtype=float) / 100.0, index=dates)
    rows = market.reindex(dates)[list(FACTOR_COLUMNS) + ['rf']].astype(float)
    if 'tbill_3m' in market:                               # as src/wrds_engine._risk_free fills it
        rows['rf'] = rows['rf'].fillna(market['tbill_3m'].reindex(dates).astype(float) / 12.0)
    known = rows.notna().all(axis=1).to_numpy()
    rows, returns = rows[known], returns[known]
    fit = factor_regression((returns - rows['rf']).to_numpy(), rows[list(FACTOR_COLUMNS)].to_numpy())
    if fit is None:
        return None
    return {**fit, 'first_month': rows.index[0].strftime('%Y-%m'), 'last_month': rows.index[-1].strftime('%Y-%m')}


def _join(words: List[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def describe_factor_risk(risk: Dict[str, Any]) -> str:
    """
    One plain-language line from simple rules, e.g. "Mostly a value and momentum tilt;
    alpha not significant.":

    - Tilts: the factors other than market and size with |t| >= 2, the MAX_TILTS strongest
      (by |t|), by the sign of the exposure (TILT_WORDS).
    - "Mostly" when R-squared >= 0.8 (the factors explain most of the monthly return),
      else "Partly". Without a tilt: "Mostly market exposure" or "No clear factor tilt".
    - Alpha: significant (|t| >= 2) with its annual size and sign, or "not significant".
    """
    mostly = risk['r_squared'] >= MOSTLY_R_SQUARED
    tilts = sorted((f for f in risk['factors'] if f['factor'] in TILT_WORDS and abs(f['t_stat']) >= SIGNIFICANT_T),
                   key=lambda f: -abs(f['t_stat']))[:MAX_TILTS]
    if tilts:
        words = _join([TILT_WORDS[f['factor']][0 if f['exposure'] > 0 else 1] for f in tilts])
        article = 'an' if words[0] in 'aeiou' else 'a'
        lead = f"{'Mostly' if mostly else 'Partly'} {article} {words} tilt"
    else:
        lead = 'Mostly market exposure' if mostly else 'No clear factor tilt'
    if abs(risk['alpha_t']) >= SIGNIFICANT_T:
        sign = 'positive' if risk['alpha_annual'] > 0 else 'negative'
        alpha = f"significant {sign} alpha of {risk['alpha_annual']:+.1%} a year"
    else:
        alpha = 'alpha not significant'
    return f"{lead}; {alpha}."
