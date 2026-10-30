"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/performance_metrics.py
PURPOSE: Mathematical core with comprehensive index coverage.
VERSION: 1.3.0
"""

import numpy as np
from typing import Dict, List, Any

def compute_comprehensive_metrics(p_ret: np.ndarray, b_ret: np.ndarray, g_ret: np.ndarray,
                                 v_ret: np.ndarray, rf: np.ndarray,
                                 p_vals: List[float], start: int, end: int,
                                 periods_per_year: int = 1) -> Dict[str, Any]:
    """
    Calculates risk/return metrics across all benchmarks.

    The returns are per period. periods_per_year (k) annualizes the volatilities,
    Sharpe ratios and information ratios by sqrt(k). With the default k = 1 (annual
    periods, as in Legacy) they are left exactly as computed.
    """

    def annualize(value):
        return value * np.sqrt(periods_per_year) if periods_per_year != 1 else value

    # Excess Returns
    excess_p, excess_b = p_ret - rf, b_ret - rf
    excess_g, excess_v = g_ret - rf, v_ret - rf
    
    def get_vols(rets):
        return np.std(rets, ddof=1) if len(rets) > 1 else 0

    # Sharpe logic (Mean Excess / Vol Excess)
    def get_sharpe(exc_rets, exc_vol):
        return np.mean(exc_rets) / exc_vol if exc_vol > 0 else 0

    def get_ir(port_ret, bench_ret):
        active = port_ret - bench_ret
        te = np.std(active, ddof=1)
        return np.mean(active) / te if te > 0 else 0

    def get_beta(p_exc, b_exc):
        return np.polyfit(b_exc, p_exc, 1)[0] if len(p_exc) > 1 else 1.0

    def get_mdd(values):
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak if len(peak) > 0 and np.all(peak > 0) else np.zeros_like(values)
        return np.min(dd) if len(dd) > 0 else 0

    def build_wealth(rets):
        vals = [p_vals[0]]
        for r in rets: vals.append(vals[-1] * (1 + r))
        return vals

    # Year-by-Year Comparisons
    comparisons = []
    for i in range(len(p_ret)):
        comparisons.append({
            'year': int(start + i),
            'p_ret': p_ret[i] * 100,
            'b_ret': b_ret[i] * 100,
            'b_win': bool(p_ret[i] > b_ret[i]),
            'g_ret': g_ret[i] * 100,
            'g_win': bool(p_ret[i] > g_ret[i]),
            'v_ret': v_ret[i] * 100,
            'v_win': bool(p_ret[i] > v_ret[i])
        })

    return {
        'final_value': p_vals[-1],
        'portfolio_values': p_vals,
        'yearly_returns': (p_ret * 100).tolist(),
        'benchmark_returns': (b_ret * 100).tolist(),
        'growth_benchmark_returns': (g_ret * 100).tolist(),
        'value_benchmark_returns': (v_ret * 100).tolist(),
        'years': list(range(int(start), int(end) + 1)),
        'portfolio_beta': get_beta(excess_p, excess_b),
        'portfolio_beta_growth': get_beta(excess_p, excess_g),
        'portfolio_beta_value': get_beta(excess_p, excess_v),
        'max_drawdown_portfolio': get_mdd(p_vals),
        'max_drawdown_benchmark': get_mdd(build_wealth(b_ret)),
        'max_drawdown_growth': get_mdd(build_wealth(g_ret)),
        'max_drawdown_value': get_mdd(build_wealth(v_ret)),
        'sharpe_portfolio': annualize(get_sharpe(excess_p, get_vols(excess_p))),
        'sharpe_benchmark': annualize(get_sharpe(excess_b, get_vols(excess_b))),
        'sharpe_growth': annualize(get_sharpe(excess_g, get_vols(excess_g))),
        'sharpe_value': annualize(get_sharpe(excess_v, get_vols(excess_v))),
        'vol_raw_portfolio': annualize(get_vols(p_ret)),
        'vol_raw_benchmark': annualize(get_vols(b_ret)),
        'vol_raw_growth': annualize(get_vols(g_ret)),
        'vol_raw_value': annualize(get_vols(v_ret)),
        'vol_excess_portfolio': annualize(get_vols(excess_p)),
        'vol_excess_benchmark': annualize(get_vols(excess_b)),
        'vol_excess_growth': annualize(get_vols(excess_g)),
        'vol_excess_value': annualize(get_vols(excess_v)),
        'yearly_comparisons': comparisons,
        'information_ratio': annualize(get_ir(p_ret, b_ret)),
        'information_ratio_growth': annualize(get_ir(p_ret, g_ret)),
        'information_ratio_value': annualize(get_ir(p_ret, v_ret)),
        'win_rate': float(np.mean(p_ret > b_ret)) if len(p_ret) > 0 else 0.0,
        'win_rate_growth': float(np.mean(p_ret > g_ret)) if len(p_ret) > 0 else 0.0,
        'win_rate_value': float(np.mean(p_ret > v_ret)) if len(p_ret) > 0 else 0.0,
        'risk_free_rate_source': "FRED 4 Week T-Bill (Oct 1)"
    }