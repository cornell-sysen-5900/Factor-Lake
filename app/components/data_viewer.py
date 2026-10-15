"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/data_viewer.py
PURPOSE: Data Viewer tab: browse and export the raw market data loaded in the
         Analysis tab, filtered by year range, sector and industry.
VERSION: 1.0.0

This tab never downloads data. It only reads st.session_state.raw_data (the
shared copy loaded by the Load Market Data button) and never modifies it.
Every tab runs on every rerun, so nothing is built until "Load data" is clicked,
and the built slice is dropped as soon as a filter changes or "Clear" is clicked.
"""

import gc
from typing import Any, Dict

import streamlit as st

from app.data_viewer_filters import (
    SECTOR_COLUMN,
    category_options,
    csv_file_name,
    filter_raw_data,
    industry_column,
    raw_data_to_csv,
    year_options,
)

# session_state key holding the loaded slice: {'filters': ..., 'df': DataFrame}
SLICE_KEY = 'data_viewer_slice'


def clear_data_viewer_slice() -> None:
    """Drops the loaded slice and frees its memory."""
    if SLICE_KEY in st.session_state:
        del st.session_state[SLICE_KEY]
        gc.collect()


def render_data_viewer() -> None:
    raw = st.session_state.get('raw_data')
    if raw is None:
        st.info("Load market data in the Analysis tab first, then come back here to view it.")
        return

    st.caption("Filters apply to the full loaded dataset; the sidebar filters are ignored here.")
    filters = _render_filters(raw)

    # A slice built with other filters no longer matches what is shown
    stored = st.session_state.get(SLICE_KEY)
    if stored is not None and stored['filters'] != filters:
        clear_data_viewer_slice()

    col_load, col_clear = st.columns(2)
    if col_load.button("Load data", type="primary", width="stretch", key='dv_load'):
        clear_data_viewer_slice()
        st.session_state[SLICE_KEY] = {
            'filters': filters,
            'df': filter_raw_data(raw, filters['years'], filters['sectors'], filters['industries']),
        }
    col_clear.button("Clear", width="stretch", key='dv_clear', on_click=clear_data_viewer_slice)

    stored = st.session_state.get(SLICE_KEY)
    if stored is None:
        st.caption("Choose filters and click **Load data** to view the rows.")
        return

    df = stored['df']
    st.caption(f"**Rows:** {len(df):,}")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        data=lambda: raw_data_to_csv(df),   # built only when clicked
        file_name=csv_file_name(filters['years']),
        mime="text/csv",
        key='dv_download',
        on_click="ignore",
    )


def _render_filters(raw) -> Dict[str, Any]:
    years = year_options(raw)
    year_range = st.select_slider(
        "Years", options=years, value=(years[0], years[-1]), key='dv_years'
    )
    sectors = st.multiselect(
        "Sectors", category_options(raw, SECTOR_COLUMN),
        key='dv_sectors', placeholder="All sectors"
    )
    ind_col = industry_column(raw)
    industries = st.multiselect(
        "Industries (optional)", category_options(raw, ind_col),
        key='dv_industries', placeholder="All industries"
    ) if ind_col else []
    return {
        'years': (int(year_range[0]), int(year_range[1])),
        'sectors': tuple(sectors),
        'industries': tuple(industries),
    }
