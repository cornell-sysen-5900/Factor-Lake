"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/benchmarks.py
PURPOSE: Reference data for annual index returns and risk-free rates.
VERSION: 2.1.0
"""

from typing import List, Dict

# Annual Returns (%) for benchmarks; Decimal for RISK_FREE_RATES.
# Data is captured for 12-month periods ending September 30 (Q3).
benchmarks = {
    "Russell 2000": {
        2002: -9.30, 2003: 36.50, 2004: 18.76, 2005: 17.95, 2006: 9.92,
        2007: 12.34, 2008: -14.49, 2009: -9.54, 2010: 13.35, 2011: -3.54,
        2012: 31.91, 2013: 30.04, 2014: 3.93, 2015: 1.25, 2016: 15.46,
        2017: 20.74, 2018: 15.24, 2019: -8.89, 2020: 0.40, 2021: 47.67,
        2022: -23.51, 2023: 8.94, 2024: 26.76
    },
    "Russell 2000 Growth": {
        2002: -18.16, 2003: 41.73, 2004: 11.92, 2005: 17.96, 2006: 5.88,
        2007: 18.95, 2008: -17.08, 2009: -6.32, 2010: 14.78, 2011: -1.12,
        2012: 31.18, 2013: 33.07, 2014: 3.78, 2015: 4.05, 2016: 12.12,
        2017: 20.99, 2018: 21.06, 2019: -9.63, 2020: 15.72, 2021: 33.28,
        2022: -29.27, 2023: 9.58, 2024: 27.66
    },
    "Russell 2000 Value": {
        2002: -1.46, 2003: 31.65, 2004: 25.67, 2005: 17.75, 2006: 14.01,
        2007: 6.08, 2008: -12.27, 2009: -12.61, 2010: 11.84, 2011: -5.99,
        2012: 32.63, 2013: 27.03, 2014: 4.12, 2015: -1.60, 2016: 18.82,
        2017: 20.55, 2018: 9.32, 2019: -8.24, 2020: -14.87, 2021: 63.93,
        2022: -17.69, 2023: 7.84, 2024: 25.89
    },
    
    # FRED 4-Week Treasury Bill Secondary Market Rate (Annualized)
    "RISK_FREE_RATES": {
        2002: 0.0156, 2003: 0.0102, 2004: 0.0182, 2005: 0.0349, 2006: 0.0473,
        2007: 0.0462, 2008: 0.0148, 2009: 0.0014, 2010: 0.0014, 2011: 0.0010,
        2012: 0.0015, 2013: 0.0005, 2014: 0.0002, 2015: 0.0001, 2016: 0.0033,
        2017: 0.0110, 2018: 0.0220, 2019: 0.0179, 2020: 0.0009, 2021: 0.0003,
        2022: 0.0306, 2023: 0.0531, 2024: 0.0475
    }
}

def get_benchmark_list(index: int, start_year: int, end_year: int) -> List[float]:
    """
    Retrieves time-series data for benchmarks or risk-free rates.

    Args:
        index (int): 1: Russell 2000, 2: R2000 Growth, 3: R2000 Value, 4: Risk-Free.
        start_year (int): Initial year (inclusive).
        end_year (int): Terminal year (exclusive).

    Returns:
        List[float]: Annual returns. Indices 1-3 are in %, Index 4 is decimal.
    """
    benchmark_lookup = {
        1: "Russell 2000",
        2: "Russell 2000 Growth",
        3: "Russell 2000 Value",
        4: "RISK_FREE_RATES"
    }
    
    name = benchmark_lookup.get(index)
    if name is None:
        raise ValueError(f"Invalid benchmark index: {index}. Choose from 1, 2, 3, or 4.")

    data_dict = benchmarks[name]
    
    # Year-range selection with safe defaulting
    return [data_dict.get(yr, 0.0) for yr in range(start_year, end_year)]