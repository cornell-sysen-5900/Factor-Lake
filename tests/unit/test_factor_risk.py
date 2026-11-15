"""
Tests for the factor regression (src/factor_risk.py): OLS with Newey-West t-statistics
against statsmodels and a hand-computed case, alpha annualized x 12, the minimum number
of months, the WRDS adapter's alignment and rf, and the plain-language line.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.factor_risk import (
    FACTOR_COLUMNS,
    MIN_MONTHS,
    describe_factor_risk,
    factor_regression,
    newey_west_cov,
    newey_west_lag,
    wrds_factor_risk,
)

RTOL = 1e-9


def _sample(n, seed=0):
    """Six factor series and an excess return with known exposures, alpha and autocorrelated noise."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.005, 0.04, (n, len(FACTOR_COLUMNS)))
    noise = rng.normal(0, 0.02, n)
    noise[1:] += 0.4 * noise[:-1]                                     # autocorrelated, so the lags matter
    excess = 0.002 + factors @ np.array([1.1, 0.6, 0.3, -0.2, 0.1, -0.15]) + noise
    return excess, factors


@pytest.mark.parametrize('n', [MIN_MONTHS, 120, 290])
def test_matches_statsmodels_ols_with_newey_west(n):
    excess, factors = _sample(n, seed=n)
    fit = factor_regression(excess, factors)
    lag = newey_west_lag(n)
    reference = sm.OLS(excess, sm.add_constant(factors)).fit(
        cov_type='HAC', cov_kwds={'maxlags': lag, 'kernel': 'bartlett', 'use_correction': True})
    assert fit['months'] == n and fit['newey_west_lag'] == lag
    assert fit['alpha_monthly'] == pytest.approx(reference.params[0], rel=RTOL)
    assert fit['alpha_t'] == pytest.approx(reference.tvalues[0], rel=RTOL)
    assert fit['r_squared'] == pytest.approx(reference.rsquared, rel=RTOL)
    assert [f['factor'] for f in fit['factors']] == list(FACTOR_COLUMNS)
    np.testing.assert_allclose([f['exposure'] for f in fit['factors']], reference.params[1:], rtol=RTOL)
    np.testing.assert_allclose([f['t_stat'] for f in fit['factors']], reference.tvalues[1:], rtol=RTOL)


def test_alpha_is_annualized_times_twelve():
    excess, factors = _sample(60)
    fit = factor_regression(excess, factors)
    assert fit['alpha_annual'] == 12 * fit['alpha_monthly']


@pytest.mark.parametrize('n, lag', [(36, 3), (99, 3), (100, 4), (272, 4), (273, 5), (290, 5)])
def test_newey_west_lag(n, lag):
    assert newey_west_lag(n) == lag


def test_newey_west_by_hand():
    """A 4-month, one-regressor case with lag 1, summed term by term."""
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 4.0]])
    u = np.array([0.5, -1.0, 1.5, -1.0])
    # S = sum u_t^2 x_t x_t' + 1/2 sum_{t>=1} u_t u_{t-1} (x_t x_{t-1}' + x_{t-1} x_t')
    s = sum(u[t] ** 2 * np.outer(x[t], x[t]) for t in range(4))
    s += 0.5 * sum(u[t] * u[t - 1] * (np.outer(x[t], x[t - 1]) + np.outer(x[t - 1], x[t])) for t in range(1, 4))
    xtx_inv = np.linalg.inv(x.T @ x)
    expected = xtx_inv @ s @ xtx_inv * 4 / (4 - 2)
    np.testing.assert_allclose(newey_west_cov(x, u, 1), expected, rtol=1e-12)


def test_an_exact_fit_recovers_alpha_and_exposures():
    rng = np.random.default_rng(1)
    factors = rng.normal(0, 0.05, (48, 6))
    betas = np.array([1.0, 0.5, 0.4, 0.0, -0.3, 0.2])
    fit = factor_regression(0.001 + factors @ betas, factors)
    assert fit['alpha_monthly'] == pytest.approx(0.001, abs=1e-12)
    assert fit['alpha_annual'] == pytest.approx(0.012, abs=1e-12)
    np.testing.assert_allclose([f['exposure'] for f in fit['factors']], betas, atol=1e-12)
    assert fit['r_squared'] == pytest.approx(1.0, abs=1e-12)


def test_fewer_than_the_minimum_months_gives_nothing():
    excess, factors = _sample(MIN_MONTHS - 1)
    assert factor_regression(excess, factors) is None
    assert factor_regression(*_sample(MIN_MONTHS)) is not None


# ---- The WRDS adapter -------------------------------------------------------------------

def _market(dates, factors, rf):
    return pd.DataFrame(factors, index=dates, columns=list(FACTOR_COLUMNS)).assign(rf=rf, iwm_total_ret=0.0)


def test_wrds_regresses_returns_after_costs_minus_rf_on_the_same_months():
    n = 48
    excess, factors = _sample(n + 5, seed=7)
    dates = pd.date_range('2010-01-01', periods=n + 5, freq='BME')
    rf = np.linspace(0.0001, 0.003, n + 5)
    market = _market(dates, factors, rf)
    # The run covers months 3 to n + 2 only; market has more months, which must be ignored
    run = slice(3, n + 3)
    results = {'monthly_dates': [d.strftime('%Y-%m-%d') for d in dates[run]],
               'monthly_returns': list((excess[run] + rf[run]) * 100)}
    risk = wrds_factor_risk(results, market)
    expected = factor_regression(excess[run], factors[run])
    assert risk['alpha_annual'] == pytest.approx(expected['alpha_annual'], rel=1e-10)
    assert risk['r_squared'] == pytest.approx(expected['r_squared'], rel=1e-10)
    assert (risk['first_month'], risk['last_month']) == (dates[3].strftime('%Y-%m'), dates[n + 2].strftime('%Y-%m'))


def test_wrds_leaves_out_months_without_factors_and_needs_enough_months():
    dates = pd.date_range('2010-01-01', periods=MIN_MONTHS + 1, freq='BME')
    excess, factors = _sample(MIN_MONTHS + 1)
    market = _market(dates, factors, 0.001)
    results = {'monthly_dates': [d.strftime('%Y-%m-%d') for d in dates], 'monthly_returns': list(excess * 100)}
    assert wrds_factor_risk(results, market)['months'] == MIN_MONTHS + 1
    market.loc[dates[5], 'umd'] = np.nan
    assert wrds_factor_risk(results, market)['months'] == MIN_MONTHS
    # A month without rf uses the 3-month T-bill / 12, as the engine does
    market['tbill_3m'] = 0.012
    market.loc[dates[6], 'rf'] = np.nan
    assert wrds_factor_risk(results, market)['months'] == MIN_MONTHS
    market.loc[dates[6], 'tbill_3m'] = np.nan
    assert wrds_factor_risk(results, market) is None
    # Results saved before monthly returns have none
    assert wrds_factor_risk({'yearly_returns': [1.0]}, market) is None


# ---- The plain-language line -------------------------------------------------------------

def _risk(r_squared=0.9, alpha=0.01, alpha_t=0.5, **t_stats):
    """t_stats: factor -> (exposure, t)."""
    return {'r_squared': r_squared, 'alpha_annual': alpha, 'alpha_t': alpha_t,
            'factors': [{'factor': f, 'exposure': t_stats.get(f, (0.1, 0.5))[0],
                         't_stat': t_stats.get(f, (0.1, 0.5))[1]} for f in FACTOR_COLUMNS]}


@pytest.mark.parametrize('risk, line', [
    (_risk(hml=(0.3, 2.5), umd=(0.2, 3.0)), "Mostly a momentum and value tilt; alpha not significant."),
    # The two strongest only; the market and size never count as tilts, however strong
    (_risk(0.5, mktrf=(1.0, 20.0), smb=(0.9, 22.0), hml=(-0.3, -2.0), umd=(0.2, 4.0), rmw=(-0.2, -3.0)),
     "Partly a momentum and low-profitability tilt; alpha not significant."),
    (_risk(smb=(-0.4, -6.0)), "Mostly market exposure; alpha not significant."),
    (_risk(cma=(-0.4, -2.2)), "Mostly an aggressive-investment tilt; alpha not significant."),
    (_risk(0.85, 0.031, 2.4), "Mostly market exposure; significant positive alpha of +3.1% a year."),
    (_risk(0.3, -0.025, -2.0, rmw=(0.5, 1.99)), "No clear factor tilt; significant negative alpha of -2.5% a year."),
])
def test_plain_language_line(risk, line):
    assert describe_factor_risk(risk) == line
