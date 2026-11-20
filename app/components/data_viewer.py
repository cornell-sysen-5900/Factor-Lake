"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/data_viewer.py
PURPOSE: Data Viewer tab: filter, view and export the market data loaded in the Analysis tab.
VERSION: 1.2.0
"""

import gc
import os
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.viewer_data import (
    DATE_COL,
    DISPLAY_ROW_LIMIT,
    INDUSTRY_COL,
    SECTOR_COL,
    VIEWER_STATE_KEY,
    filter_market_data,
    hidden_rows_caption,
    industries_in_sectors,
    keep_valid_selections,
    to_csv_bytes,
)
from src.wrds_gics import GICS_LEVELS, codes_under, level_names, make_gics_filter, names_to_codes
from src.wrds_preview import APPROVAL_ENV, DISABLED_MESSAGE, PREVIEW_COLUMNS, PREVIEW_ROWS

PREVIEW_STATE_KEY = 'wrds_preview'


def clear_viewer_data() -> None:
    """
    Drops the viewer table. Streamlit releases the CSV download that still refers to it
    when it cleans up at the end of the same rerun.
    """
    holder = st.session_state.pop(VIEWER_STATE_KEY, None)
    st.session_state.pop(PREVIEW_STATE_KEY, None)
    if holder:
        holder.clear()
    gc.collect()


def _options(df: pd.DataFrame, column: str) -> List[str]:
    if column not in df.columns:
        return []
    return sorted(str(v) for v in df[column].dropna().unique())


def render_legacy_data_viewer_tab() -> None:
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
    hidden_caption = hidden_rows_caption(len(table))
    if hidden_caption:
        st.caption(hidden_caption)
    # Large tables are slow to send to the browser, so only the first rows are shown
    st.dataframe(table.head(DISPLAY_ROW_LIMIT), hide_index=True, width="stretch")
    st.download_button(
        "Download CSV",
        # Built only when clicked, from the full table (every filtered row). It keeps its own reference so a
        # click that lands while a Clear or filter change is rerunning still gets the full file.
        data=lambda df=table: to_csv_bytes(df),
        file_name=f"factor_lake_data_{start_txt}_{end_txt}.csv",
        mime="text/csv",
        key="viewer_download",
        on_click="ignore",
    )


def _preview_approved():
    value = os.environ.get(APPROVAL_ENV)
    if value is None:
        try:
            value = st.secrets.get(APPROVAL_ENV, False)
        except (FileNotFoundError, KeyError):
            value = False
    return str(value).strip().lower() == 'true'


def _preview_gics_filter():
    """Four cascading lists. The deepest chosen level defines the filter."""
    parent_level, parents, chosen = None, [], None
    for level, (label, _, _) in GICS_LEVELS.items():
        names = level_names(level)
        codes = codes_under(level, parents, parent_level) if parents else sorted(names)
        options = [names[c] for c in codes]
        key = f'preview_gics_{level}'
        if key in st.session_state:
            st.session_state[key] = [v for v in st.session_state[key] if v in options]
        selected = st.multiselect(label, options, key=key, placeholder=f'All {label.lower()} choices',
                                  on_change=clear_viewer_data)
        if selected:
            parents = names_to_codes(level, selected)
            parent_level = level
            chosen = make_gics_filter(level, parents)
    return chosen


def render_data_viewer_tab() -> None:
    """Only bounded WRDS sample rows enter app memory; no data-download control."""
    from app.streamlit_utils import get_backtest_lambda_name, get_lambda_client
    st.header('Data Viewer')
    st.caption(f'Preview up to {PREVIEW_ROWS} WRDS rows, each a different company, filtered before sampling: '
               'each company’s most recent row, newest first. These examples describe the fields; '
               'they are not a representative portfolio.')
    if not _preview_approved():
        clear_viewer_data()
        st.info(DISABLED_MESSAGE)
        return
    function_name = get_backtest_lambda_name()
    if not function_name:
        clear_viewer_data()
        st.info('Configure the backtest Lambda to load a WRDS preview.')
        return
    gics = _preview_gics_filter()
    st.caption('These preview filters are independent of the Analysis sidebar. GICS uses the company’s current '
               'classification across its history; Sector includes SIC fallback, while finer levels require GICS.')
    request = {'gics_filter': gics}
    holder = st.session_state.get(PREVIEW_STATE_KEY)
    if holder and (holder['request'] != request or holder['function'] != function_name):
        clear_viewer_data()
    load_col, clear_col = st.columns(2)
    with load_col:
        load = st.button('Load preview', type='primary', key='preview_load', width='stretch')
    with clear_col:
        st.button('Clear preview', key='preview_clear', on_click=clear_viewer_data, width='stretch')
    if load:
        clear_viewer_data()
        try:
            with st.spinner('Loading WRDS sample rows...'):
                response = get_lambda_client(function_name).preview(request)
            if response['status'] == 'ok':
                st.session_state[PREVIEW_STATE_KEY] = {'request': request, 'function': function_name,
                                                      'preview': response['preview']}
            else:
                st.info(response['message'])
        except Exception:
            st.error('The WRDS preview could not be loaded. Check the Lambda deployment and try again.')
    holder = st.session_state.get(PREVIEW_STATE_KEY)
    if holder is None:
        return
    preview = holder['preview']
    if not preview['rows']:
        st.info('No WRDS rows match these GICS filters.')
        return
    table = pd.DataFrame(preview['rows'], columns=list(PREVIEW_COLUMNS))
    st.caption(f'{len(table)} sample rows · one per company, up to {PREVIEW_ROWS}. Missing values appear as —.')
    # Static, escaped HTML has no dataframe toolbar, CSV export or hidden full dataset.
    st.html('<div style="overflow:auto;max-height:600px" aria-label="WRDS data preview">' +
            table.fillna('—').to_html(index=False, escape=True, na_rep='—', border=0) + '</div>')
