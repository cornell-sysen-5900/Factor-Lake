"""Deterministic synthetic stocks for opt-in demonstrations; never a production fallback."""
import numpy as np
import pandas as pd
from app.streamlit_config import FACTOR_METADATA, SECTOR_OPTIONS


def build_demo_universe() -> pd.DataFrame:
    """Return all UI factors for 80 invented stocks, 2002–2024, without network access."""
    rng = np.random.default_rng(5900)
    rows = []
    for year in range(2002, 2025):
        for i in range(80):
            sector = SECTOR_OPTIONS[i % len(SECTOR_OPTIONS)]
            price = float(rng.uniform(10, 150))
            factors = {meta['column']: float(rng.normal()) for meta in FACTOR_METADATA.values()}
            ret = float(6 + 4 * factors['ROA'] + 3 * factors['12-Mo_Momentum'] + rng.normal(0, 18))
            delisted = (i + year) % 67 == 0
            rows.append({**factors, 'Ticker': f'DEMO{i:03d}', 'Security_Name': f'Synthetic Company {i:03d}',
                         'Year': year, 'Date': pd.Timestamp(year, 9, 30), 'Ending_Price': price,
                         'Next-Years_Return': np.nan if delisted else ret,
                         'Market_Capitalization': float(rng.uniform(1e8, 5e9)),
                         'Scotts_Sector_5': sector,
                         'FactSet_Industry': 'Coal' if sector == 'Energy' else f'Synthetic {sector}',
                         'Delist_Date': pd.Timestamp(year + 1, 6, 30) if delisted else pd.NaT,
                         'Delist_Price': price * 0.7 if delisted else np.nan})
    return pd.DataFrame(rows)
