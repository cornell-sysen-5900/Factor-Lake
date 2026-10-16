"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/data_viewer.py
PURPOSE: Data Viewer tab: filter, view and export the market data loaded in the Analysis tab.
VERSION: 1.1.0
"""

import gc
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.viewer_data import (
    DATE_COL,
    INDUSTRY_COL,
    SECTOR_COL,
    VIEWER_STATE_KEY,
    filter_market_data,
    industries_in_sectors,
    keep_valid_selections,
    to_csv_bytes,
)


def clear_viewer_data() -> None:
    """
    Drops the viewer table. Streamlit releases the CSV download that still refers to it
    when it cleans up at the end of the same rerun.
    """
    holder = st.session_state.pop(VIEWER_STATE_KEY, None)
    if holder:
        holder.clear()
    gc.collect()


def _options(df: pd.DataFrame, column: str) -> List[str]:
    if column not in df.columns:
        return []
    return sorted(str(v) for v in df[column].dropna().unique())


def render_data_viewer_tab() -> None:
    """
    Lets users filter the market data already loaded in Analysis by date range and
    sector or industry, view it, and download it as CSV. Never fetches data itself.
    """
    st.header("Data Viewer")
    data: Optional[pd.DataFrame] = st.session_state.get('rdata')
    holder: Optional[Dict[str, Any]] = st.session_state.get(VIEWER_STATE_KEY)

    if data is None or data.empty or DATE_COL not in data.columns:
        if holder:
            clear_viewer_data()
        st.info("Load market data in the Analysis tab first. The viewer reuses that data "
                "and does not download anything itself.")
        return

    # Market data was reloaded since the table was built: drop the stale table
    if holder and holder.get('source') is not data:
        clear_viewer_data()

    st.caption("Shows the market data loaded in the Analysis tab (after the sidebar filters). "
               "Changing a filter clears the table; click Load data to see it again.")

    dates = pd.to_datetime(data[DATE_COL], errors='coerce').dropna()
    first_day, last_day = dates.min().date(), dates.max().date()
    sector_options = _options(data, SECTOR_COL)
    industry_options = _options(data, INDUSTRY_COL)
    # New keys whenever the loaded data changes, so old selections never fall outside the new options.
    # Built from the full option lists, so narrowing the industries by sector keeps every key.
    suffix = abs(hash((len(data), first_day, last_day, tuple(sector_options), tuple(industry_options))))

    # Wide bounds: the browser silently ignores a typed date outside min/max (keeping the old
    # range), so any date is accepted here and the filter simply clips to the data's dates.
    picked = st.date_input("Date range", value=(first_day, last_day), min_value=date(1900, 1, 1),
                           max_value=max(last_day, date.today()), key=f"viewer_dates_{suffix}",
                           on_change=clear_viewer_data)
    st.caption(f"Data available from {first_day.isoformat()} to {last_day.isoformat()}.")
    col1, col2 = st.columns(2)
    with col1:
        sectors = st.multiselect("Sector", sector_options, key=f"viewer_sectors_{suffix}",
                                 placeholder="All sectors", on_change=clear_viewer_data)
    with col2:
        # Only industries in the chosen sectors; drop chosen ones a removed sector took away
        industries_key = f"viewer_industries_{suffix}"
        industry_choices = industries_in_sectors(data, sectors)
        chosen = st.session_state.get(industries_key, [])
        kept = keep_valid_selections(chosen, industry_choices)
        if kept != list(chosen):
            st.session_state[industries_key] = kept
        industries = st.multiselect("Industry", industry_choices, key=industries_key,
                                    placeholder="All industries", on_change=clear_viewer_data)

    range_complete = isinstance(picked, (tuple, list)) and len(picked) == 2
    load_col, clear_col = st.columns(2)
    with load_col:
        load = st.button("Load data", type="primary", width="stretch", disabled=not range_complete,
                         key="viewer_load")
    with clear_col:
        st.button("Clear", width="stretch", key="viewer_clear", on_click=clear_viewer_data)
    if not range_complete:
        st.caption("Pick both a start and an end date.")

    if load and range_complete:
        clear_viewer_data()
        holder = {'df': filter_market_data(data, picked[0], picked[1], sectors, industries), 'source': data,
                  'start': picked[0], 'end': picked[1]}
        st.session_state[VIEWER_STATE_KEY] = holder

    holder = st.session_state.get(VIEWER_STATE_KEY)
    if not holder:
        return
    table: pd.DataFrame = holder['df']
    if table.empty:
        st.info("No rows match these filters.")
        return

    # The applied range is shown because the date box can display a range the browser never sent
    start_txt, end_txt = holder['start'].isoformat(), holder['end'].isoformat()
    st.caption(f"{len(table):,} rows × {table.shape[1]} columns · {start_txt} to {end_txt}")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download CSV",
        # Built only when clicked, from the same table shown above. It keeps its own reference so a
        # click that lands while a Clear or filter change is rerunning still gets the full file.
        data=lambda df=table: to_csv_bytes(df),
        file_name=f"factor_lake_data_{start_txt}_{end_txt}.csv",
        mime="text/csv",
        key="viewer_download",
        on_click="ignore",
    )
