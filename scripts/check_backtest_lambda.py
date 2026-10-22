"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/check_backtest_lambda.py
PURPOSE: Checks the deployed backtest Lambda against the app's local backtest, and its result cache.
VERSION: 1.0.0

Usage (from the project root, after scripts/deploy_backtest_lambda.py):
    uv run python scripts/check_backtest_lambda.py

For a few sample backtests (both factor combinations, every delisting strategy, a
cohort comparison) it:
1. runs the backtest locally on the S3 data, exactly as the app does without Lambda;
2. runs it in Lambda and compares every number (relative difference at most 1e-9:
   Lambda runs the same library versions on Linux x86, so the last binary digit can differ);
3. runs it in Lambda again and checks the result came from the cache, unchanged.

Holdings are compared stock by stock, without Rank and Score (see comparable).
Stocks whose scores tie exactly (winsorizing gives the top 1% of each factor the
same score) are sorted in an order that depends on the machine. That changes
their row order and Rank, and for a stock bought by several Mixing sleeves which
sleeve's Rank and Score the table shows. Which stocks are held, their shares,
prices, weights, returns and contributions must still match.
"""

import argparse
import math
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_jobs import build_job, describe_job, run_job  # noqa: E402
from src.lambda_client import DEFAULT_FUNCTION_NAME, BacktestLambdaClient  # noqa: E402
from src.s3_client import S3DataManager  # noqa: E402

TOLERANCE = 1e-9
# Holdings columns that depend on the order of exactly tied stocks (see comparable)
TIE_ORDER_COLUMNS = ('Rank', 'Score')
LABELS = {'ROE_using_9-30_Data': 'ROE using 9/30 Data', 'ROA': 'ROA %', '6-Mo_Momentum': '6-Mo Momentum %'}
BASE = {'start_year': 2008, 'end_year': 2024, 'initial_aum': 1_000_000.0, 'use_market_cap_weight': False,
        'restrict_fossil_fuels': False, 'selected_sectors': None}


def sample_jobs(unique: bool) -> List[dict]:
    """The checked backtests. unique=True changes the AUM so nothing comes from an earlier run's cache."""
    # Random cents (up to $10,000): a clock-based offset repeated on clocks with coarse ticks
    aum = BASE['initial_aum'] + (secrets.randbelow(1_000_000) / 100 if unique else 0)
    two = ['ROE_using_9-30_Data', 'ROA']
    three = two + ['6-Mo_Momentum']
    tops = {f: 'top' for f in three}
    return [
        build_job('backtest', two, tops, dict(BASE, initial_aum=aum, factor_combination='mixing',
                                               delisting_strategy='zero_return'), LABELS),
        build_job('backtest', three, dict(tops, ROA='bottom'),
                  dict(BASE, initial_aum=aum, factor_combination='integrating', delisting_strategy='hold_cash',
                       use_market_cap_weight=True), LABELS),
        build_job('backtest', three, tops, dict(BASE, initial_aum=aum, factor_combination='mixing',
                                                 delisting_strategy='reinvest', restrict_fossil_fuels=True), LABELS),
        build_job('cohort', two, tops, dict(BASE, initial_aum=aum, factor_combination='integrating',
                                             delisting_strategy='zero_return'), LABELS, cohort_pct=20),
    ]


def comparable(result: dict) -> dict:
    """
    The result with each year's holdings keyed by ticker and without Rank and Score,
    so the comparison does not depend on the order of exactly tied stocks (see the
    module docstring).
    """
    if 'holdings_by_year' not in result:
        return result
    holdings = {year: {row['Ticker']: {k: v for k, v in row.items() if k not in TIE_ORDER_COLUMNS}
                       for row in rows}
                for year, rows in result['holdings_by_year'].items()}
    return dict(result, holdings_by_year=holdings)


def max_difference(a: Any, b: Any, path: str = '') -> Tuple[float, str]:
    """Largest relative difference between two results, and where it is. inf when they differ in shape or text."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(map(str, a)) != set(map(str, b)):
            return math.inf, f"{path} keys"
        b_by_key = {str(k): v for k, v in b.items()}
        return max((max_difference(v, b_by_key[str(k)], f"{path}.{k}") for k, v in a.items()), default=(0.0, path))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf, f"{path} length"
        return max((max_difference(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))), default=(0.0, path))
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return (0.0, path) if a == b else (math.inf, path)
    if math.isnan(a) and math.isnan(b):
        return 0.0, path
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale, path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the backtest Lambda with local backtests.")
    parser.add_argument('--function-name', help="Default: FACTOR_LAKE_BACKTEST_LAMBDA (.env), else factor-lake-backtest")
    parser.add_argument('--reuse-cache', action='store_true',
                        help="Use the same inputs every time (the first Lambda call may then be a cache hit)")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')

    function_name = args.function_name or os.environ.get('FACTOR_LAKE_BACKTEST_LAMBDA') or DEFAULT_FUNCTION_NAME
    print(f"Checking Lambda {function_name}. Loading the market data from S3 for the local runs...")
    universe = S3DataManager().fetch_all_data()
    client = BacktestLambdaClient(function_name)
    failures = 0

    for job in sample_jobs(unique=not args.reuse_cache):
        print(f"\n{describe_job(job)}")
        started = time.perf_counter()
        local = run_job(universe, job)
        print(f"  local:  {time.perf_counter() - started:6.2f} s")

        remote, first = client.run(job)
        print(f"  lambda: {first['round_trip_seconds']:6.2f} s  cached={first['cached']}  "
              f"{first['delivery']}  {first['size_bytes'] / 1e6:.2f} MB  result {first['hash'][:12]}")
        diff, where = max_difference(comparable(local), comparable(remote))
        match = diff <= TOLERANCE
        print(f"  match:  {'yes' if match else 'NO'} (largest relative difference {diff:.1e}"
              f"{'' if match else ' at ' + where})")

        again, second = client.run(job)
        same = max_difference(remote, again)[0] == 0.0
        cached = second['cached'] and second['hash'] == first['hash']
        print(f"  repeat: {second['round_trip_seconds']:6.2f} s  cached={second['cached']}  "
              f"identical={'yes' if same else 'NO'}")
        failures += (not match) + (not cached) + (not same)

    print(f"\n{'All checks passed.' if not failures else f'{failures} check(s) failed.'}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
