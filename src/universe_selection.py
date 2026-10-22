"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/universe_selection.py
PURPOSE: Applies the sidebar's year, sector and fossil fuel filters to the market data.
VERSION: 1.0.0

Shared by the Streamlit app (Load Market Data) and the backtest Lambda, so both
backtest exactly the same rows.
"""

from typing import Iterable, Optional

import pandas as pd

# FactSet industries excluded by "Restrict Fossil Fuel Companies", compared lowercased with
# everything but letters and digits removed
FOSSIL_FUEL_INDUSTRIES = {
    "integratedoil", "oilfieldservicesequipment", "oilgasproduction", "coal", "oilrefiningmarketing",
}


def select_universe(raw: pd.DataFrame, start_year: int, end_year: int,
                    selected_sectors: Optional[Iterable[str]] = None,
                    restrict_fossil_fuels: bool = False) -> pd.DataFrame:
    """
    Returns the rows of `raw` inside [start_year, end_year] that pass the sector and
    fossil fuel filters. `raw` is never modified and is not copied whole: all filters
    form one row mask and only the kept rows are copied. When nothing is filtered out,
    the result is a shallow copy that shares raw's columns, so treat it as read-only.
    """
    keep = (raw['Year'] >= int(start_year)) & (raw['Year'] <= int(end_year))

    sectors = list(selected_sectors or [])
    if sectors:
        keep &= raw['Scotts_Sector_5'].isin(sectors)

    if restrict_fossil_fuels:
        industry_col = 'FactSet_Industry' if 'FactSet_Industry' in raw.columns else 'FactSet Industry'
        if industry_col in raw.columns:
            clean_ind = raw[industry_col].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
            keep &= ~clean_ind.isin(FOSSIL_FUEL_INDUSTRIES)

    if keep.all():
        # Nothing filtered out: share raw's columns instead of copying them. The shallow
        # copy is still a new DataFrame, so adding or dropping columns on it never touches
        # raw, and each call gives a new object (the Data Viewer uses that to notice reloads).
        return raw.copy(deep=False)
    return raw[keep.to_numpy()]   # copies only the kept rows
