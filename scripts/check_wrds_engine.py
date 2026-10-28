"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/check_wrds_engine.py
PURPOSE: Checks the WRDS engine on the real WRDS files: tracking vs IWM, run time and memory.
VERSION: 1.0.0

Usage (from the project root, with the pipeline's processed/final folder):
    uv run python scripts/check_wrds_engine.py "C:/path/to/WRDS FA26/processed/final"

1. Loads the three files the engine reads (the same column selection as the Lambda).
2. Runs a monthly, market-cap weighted portfolio of the whole universe and compares it
   with IWM: it passes when the correlation of monthly returns is at least
   MIN_CORRELATION and the annualized tracking error at most MAX_TRACKING_ERROR.
3. Times monthly Mixing runs with 3 and 10 factors, and reports the process's peak memory.

The results are printed only. The data is licensed, so its statistics are never
written to the repo (pipeline/README.md).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.wrds_data import WRDS_FILES, read_local_wrds_data  # noqa: E402
from src.wrds_engine import run_wrds_backtest  # noqa: E402
from src.wrds_factors import WRDS_FACTOR_COLUMNS  # noqa: E402

MIN_CORRELATION = 0.99
MAX_TRACKING_ERROR = 0.03
TEN_FACTORS = ['mom_12_1', 'f_book_to_price', 'f_earnings_to_price', 'f_roe', 'f_roa', 'f_gross_profitability',
               'f_accruals_to_assets', 'f_asset_growth_1yr', 'vol_close_12m', 'f_sue']


def peak_memory_mb() -> float:
    """Peak resident memory of this process (working set on Windows, max RSS elsewhere)."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 2**20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == 'darwin' else peak / 2**10


def run_case(folder: str, case: str) -> dict:
    """
    One case in a fresh process, like one Lambda invocation: load the data, run the
    backtest, and report the times and the process's peak memory.
    """
    started = time.perf_counter()
    data = read_local_wrds_data(folder)
    loaded = time.perf_counter()
    if case == 'universe':
        results, holdings = run_wrds_backtest(
            data, ['mkt_cap_musd'], {'mkt_cap_musd': 'top'}, 'monthly', 2008, 2026, 1000.0, top_pct=100.0,
            use_market_cap_weight=True, factor_combination='integrating')
    else:
        factors = TEN_FACTORS[:int(case)]
        assert set(factors) <= set(WRDS_FACTOR_COLUMNS)
        results, holdings = run_wrds_backtest(data, factors, {f: 'top' for f in factors}, 'monthly', 2008, 2026, 1000.0)
    return {'load_s': loaded - started, 'run_s': time.perf_counter() - loaded, 'peak_mb': peak_memory_mb(),
            'holdings_mb': holdings.memory_usage(deep=True).sum() / 2**20,
            'portfolio': results['yearly_returns'], 'iwm': results['benchmark_returns']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('folder', help="The WRDS pull's processed/final folder")
    args = parser.parse_args(argv)
    folder = Path(args.folder)
    sizes = ', '.join(f"{f} {(folder / f).stat().st_size / 2**20:.1f} MB" for f in WRDS_FILES)
    print(f"Files: {sizes}")

    import multiprocessing
    context = multiprocessing.get_context('spawn')
    cases = {'universe': "Monthly, cap weighted, whole universe",
             '3': "Monthly Mixing, 3 factors, equal weight",
             '10': "Monthly Mixing, 10 factors, equal weight"}
    outcomes = {}
    for case, label in cases.items():
        with context.Pool(1) as pool:
            outcomes[case] = out = pool.apply(run_case, (str(folder), case))
        print(f"{label:<44} load {out['load_s']:5.2f} s   run {out['run_s']:5.2f} s   "
              f"peak memory {out['peak_mb']:5.0f} MB   holdings {out['holdings_mb']:5.1f} MB")

    portfolio, iwm = np.array(outcomes['universe']['portfolio']), np.array(outcomes['universe']['iwm'])
    correlation = float(np.corrcoef(portfolio, iwm)[0, 1])
    tracking_error = float(np.std((portfolio - iwm) / 100.0, ddof=1) * np.sqrt(12))
    tracks = correlation >= MIN_CORRELATION and tracking_error <= MAX_TRACKING_ERROR
    print(f"Whole universe vs IWM: correlation {correlation:.4f} (>= {MIN_CORRELATION}), "
          f"tracking error {tracking_error:.2%}/yr (<= {MAX_TRACKING_ERROR:.0%}): {'PASS' if tracks else 'FAIL'}")
    return 0 if tracks else 1


if __name__ == '__main__':
    sys.exit(main())
