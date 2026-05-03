import pandas as pd
from src.portfolio_filters import apply_sector_filters


def test_apply_sector_filter_basic():
    df = pd.DataFrame({
        "Ticker": ["A", "B", "C", "D"],
        "Scotts_Sector_5": ["Consumer", "Technology", "Financials", "Industrials"],
        "Ending_Price": [10, 20, 30, 40],
    })
    out = apply_sector_filters(df, ["Consumer", "Financials"])
    assert set(out["Ticker"]) == {"A", "C"}


def test_apply_sector_filter_missing_col():
    df = pd.DataFrame({
        "Ticker": ["A"],
        "Ending_Price": [10],
    })
    # Should return unchanged if sector column missing
    out = apply_sector_filters(df, ["Consumer"])
    assert len(out) == 1
