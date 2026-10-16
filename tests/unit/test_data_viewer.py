"""
Tests for the Data Viewer tab (app/components/data_viewer.py).

Pure tests cover filtering and the CSV export. AppTest tests drive the real app
with a small synthetic universe, so no S3 or Supabase access is needed.
"""
import gc
import io
import weakref
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.viewer_data import (
    VIEWER_STATE_KEY,
    filter_market_data,
    industries_in_sectors,
    keep_valid_selections,
    to_csv_bytes,
)
from src.s3_client import S3DataManager

APP_DIR = Path(__file__).resolve().parents[2] / 'app'
SECTORS = ['Technology', 'Healthcare', 'Energy']
INDUSTRIES = {'Technology': 'Software', 'Healthcare': 'Biotech', 'Energy': 'Oil Gas Production'}


def _universe() -> pd.DataFrame:
    """12 tickers x 2002-2024 yearly snapshots across three sectors/industries."""
    rng = np.random.RandomState(1)
    rows = []
    for year in range(2002, 2025):
        for i in range(12):
            sector = SECTORS[i % 3]
            rows.append({
                'Ticker-Region': f'T{i:02d}-US',
                'Security_Name': f'Company {i}',
                'Date': f'{year}-09-30',
                'Ending_Price': rng.uniform(10, 100),
                'Next-Years_Return': rng.normal(8, 20),
                'ROE_using_9-30_Data': rng.normal(0.1, 0.05),
                'Market_Capitalization': rng.uniform(1e8, 5e9),
                'Scotts_Sector_5': sector,
                'FactSet_Industry': INDUSTRIES[sector],
                'Ticker': f'T{i:02d}',
                'Year': year,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_filter_by_date_range_is_inclusive():
    out = filter_market_data(_universe(), date(2010, 9, 30), date(2012, 9, 30), [], [])
    assert sorted(out['Date'].unique()) == ['2010-09-30', '2011-09-30', '2012-09-30']
    assert len(out) == 3 * 12


def test_filter_by_sector_and_industry():
    df = _universe()
    by_sector = filter_market_data(df, date(2002, 1, 1), date(2024, 12, 31), ['Energy'], [])
    assert set(by_sector['Scotts_Sector_5']) == {'Energy'}
    by_industry = filter_market_data(df, date(2002, 1, 1), date(2024, 12, 31), [], ['Software', 'Biotech'])
    assert set(by_industry['FactSet_Industry']) == {'Software', 'Biotech'}
    none = filter_market_data(df, date(2002, 1, 1), date(2024, 12, 31), ['Energy'], ['Software'])
    assert none.empty


def test_filter_puts_identifiers_first_and_returns_an_independent_copy():
    df = _universe()
    out = filter_market_data(df, date(2002, 1, 1), date(2024, 12, 31), [], [])
    assert list(out.columns[:3]) == ['Ticker', 'Security_Name', 'Date']
    assert set(out.columns) == set(df.columns)
    out.loc[0, 'Ending_Price'] = -1.0
    assert (df['Ending_Price'] >= 0).all()  # the loaded market data is untouched


def test_csv_round_trips_to_the_same_table():
    table = filter_market_data(_universe(), date(2005, 1, 1), date(2008, 12, 31), ['Healthcare'], [])
    back = pd.read_csv(io.BytesIO(to_csv_bytes(table)))
    assert list(back.columns) == list(table.columns)
    pd.testing.assert_frame_equal(back, table, check_dtype=False)


def _sector_industry_frame() -> pd.DataFrame:
    return pd.DataFrame({
        'Scotts_Sector_5': ['Technology', 'Technology', 'Energy', 'Energy', 'Healthcare', None],
        'FactSet_Industry': ['Software', 'Semiconductors', 'Coal', 'Oil Gas Production', 'Biotech', 'Unknown'],
    })


def test_all_industries_when_no_sector_is_chosen():
    assert industries_in_sectors(_sector_industry_frame(), []) == [
        'Biotech', 'Coal', 'Oil Gas Production', 'Semiconductors', 'Software', 'Unknown']


@pytest.mark.parametrize('sectors, expected', [
    (['Technology'], ['Semiconductors', 'Software']),
    (['Energy'], ['Coal', 'Oil Gas Production']),
    (['Energy', 'Healthcare'], ['Biotech', 'Coal', 'Oil Gas Production']),
])
def test_only_industries_in_the_chosen_sectors(sectors, expected):
    assert industries_in_sectors(_sector_industry_frame(), sectors) == expected


def test_industry_options_without_an_industry_column():
    assert industries_in_sectors(_sector_industry_frame().drop(columns=['FactSet_Industry']), ['Energy']) == []


def test_keep_valid_selections_drops_only_what_is_no_longer_offered():
    assert keep_valid_selections(['Software', 'Coal', 'Biotech'], ['Coal', 'Oil Gas Production']) == ['Coal']
    assert keep_valid_selections(['Software', 'Semiconductors'], ['Semiconductors', 'Software']) == [
        'Software', 'Semiconductors']                                  # order kept
    assert keep_valid_selections([], ['Coal']) == []


# ---------------------------------------------------------------------------
# AppTest: the real app
# ---------------------------------------------------------------------------

@pytest.fixture
def at(monkeypatch):
    # `streamlit run` puts app/ on sys.path; AppTest does not.
    monkeypatch.syspath_prepend(str(APP_DIR))
    app = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    app.session_state['raw_data'] = _universe()
    app.run()
    return app


def _load_market_data(at):
    from app.streamlit_config import FACTOR_METADATA
    # Load Market Data stays disabled until at least one factor is added
    at.button(key=f"factor_add_{FACTOR_METADATA['ROE using 9/30 Data']['key']}").click().run()
    next(b for b in at.button if b.label == 'Load Market Data').click().run()


def _widget(collection, label):
    return next(w for w in collection if w.label == label)


def _load_viewer(at, start=None, end=None, sectors=None):
    if start is not None:
        _widget(at.date_input, 'Date range').set_value((start, end)).run()
    if sectors is not None:
        _widget(at.multiselect, 'Sector').set_value(sectors).run()
    at.button(key='viewer_load').click().run()


def _assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_data_viewer_tab_sits_between_results_and_about(at):
    assert [t.label for t in at.tabs][:4] == ['Analysis', 'Results', 'Data Viewer', 'About']


def test_viewer_asks_for_market_data_first(at):
    _assert_clean(at)
    assert any('Load market data in the Analysis tab first' in i.value for i in at.info)
    assert not any(b.key == 'viewer_load' for b in at.button)
    assert len(at.dataframe) == 0


def test_load_shows_the_filtered_table_and_the_csv_matches_it(at):
    _load_market_data(at)
    _load_viewer(at, date(2010, 9, 30), date(2015, 9, 30), ['Technology'])
    _assert_clean(at)

    table = at.session_state[VIEWER_STATE_KEY]['df']
    expected = filter_market_data(at.session_state['rdata'], date(2010, 9, 30), date(2015, 9, 30),
                                  ['Technology'], [])
    pd.testing.assert_frame_equal(table, expected)
    assert len(table) == 6 * 4  # 6 years x 4 Technology tickers

    shown = at.dataframe[0].value
    assert list(shown.columns) == list(table.columns)
    # The CSV is built from the stored table; it must equal the table on screen
    assert to_csv_bytes(shown.reset_index(drop=True)) == to_csv_bytes(table)
    assert len(at.get('download_button')) == 1


def test_changing_any_filter_clears_the_table(at):
    _load_market_data(at)
    for change in (
        lambda: _widget(at.multiselect, 'Sector').set_value(['Energy']).run(),
        lambda: _widget(at.multiselect, 'Industry').set_value(['Oil Gas Production']).run(),  # still matches Energy
        lambda: _widget(at.date_input, 'Date range').set_value((date(2003, 1, 1), date(2020, 1, 1))).run(),
    ):
        at.button(key='viewer_load').click().run()
        assert VIEWER_STATE_KEY in at.session_state
        assert len(at.dataframe) == 1
        change()
        _assert_clean(at)
        assert VIEWER_STATE_KEY not in at.session_state
        assert len(at.dataframe) == 0


def test_clear_frees_the_table(at):
    _load_market_data(at)
    _load_viewer(at)
    holder = at.session_state[VIEWER_STATE_KEY]
    table_ref = weakref.ref(holder['df'])

    at.button(key='viewer_clear').click().run()
    gc.collect()

    _assert_clean(at)
    assert VIEWER_STATE_KEY not in at.session_state
    assert holder == {}            # the deferred CSV download can no longer reach the table
    del holder
    gc.collect()
    assert table_ref() is None     # nothing keeps the filtered table in memory
    assert len(at.dataframe) == 0


def test_reloading_market_data_clears_the_table(at):
    _load_market_data(at)
    _load_viewer(at)
    next(b for b in at.button if b.label == 'Load Market Data').click().run()
    _assert_clean(at)
    assert VIEWER_STATE_KEY not in at.session_state


def test_no_rows_message(at):
    _load_market_data(at)
    # A sector/industry mismatch can no longer be picked (industries follow sectors),
    # so use a date range before the data starts
    _widget(at.date_input, 'Date range').set_value((date(1990, 1, 1), date(1995, 12, 31))).run()
    at.button(key='viewer_load').click().run()
    _assert_clean(at)
    assert any('No rows match' in i.value for i in at.info)
    assert len(at.get('download_button')) == 0


def test_viewer_makes_no_extra_data_source_requests(monkeypatch):
    calls = []

    def counting_fetch(self):
        calls.append(1)
        return _universe()

    monkeypatch.setenv('FACTOR_LAKE_DATA_SOURCE', 's3')
    monkeypatch.setattr(S3DataManager, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(S3DataManager, 'fetch_all_data', counting_fetch)
    monkeypatch.syspath_prepend(str(APP_DIR))
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
        at.run()
        _load_market_data(at)
        assert len(calls) == 1                     # the one download done by Load Market Data
        st.cache_resource.clear()                  # so any loader call from the viewer would download again

        _load_viewer(at, date(2004, 9, 30), date(2009, 9, 30), ['Healthcare'])
        _widget(at.multiselect, 'Industry').set_value(['Biotech']).run()
        at.button(key='viewer_load').click().run()
        at.button(key='viewer_clear').click().run()
        _assert_clean(at)
        assert len(calls) == 1                     # the viewer added no S3 requests
    finally:
        st.cache_resource.clear()


@pytest.fixture
def downloads(monkeypatch):
    """Records the data callable of every st.download_button, then renders the real button."""
    captured = []
    real = st.download_button

    def spy(label, data, *args, **kwargs):
        captured.append(data)
        return real(label, data, *args, **kwargs)

    monkeypatch.setattr(st, 'download_button', spy)
    return captured


def test_download_serves_exactly_the_table_on_screen(at, downloads):
    _load_market_data(at)
    _load_viewer(at, date(2005, 9, 30), date(2012, 9, 30), ['Healthcare'])
    _assert_clean(at)

    payload = downloads[-1]()                       # the bytes the button would send
    shown = at.dataframe[0].value.reset_index(drop=True)
    assert payload == to_csv_bytes(shown)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(payload)), shown, check_dtype=False)


def test_download_clicked_during_a_clear_still_gets_the_whole_file(at, downloads):
    _load_market_data(at)
    _load_viewer(at)
    callable_before_clear = downloads[-1]
    expected = callable_before_clear()

    at.button(key='viewer_clear').click().run()
    assert VIEWER_STATE_KEY not in at.session_state
    assert callable_before_clear() == expected and len(expected) > 0   # never a silent empty file


def test_typed_range_starting_before_the_data_is_accepted_and_shown(at):
    _load_market_data(at)
    _load_viewer(at, date(2002, 1, 1), date(2010, 12, 31))
    _assert_clean(at)
    table = at.session_state[VIEWER_STATE_KEY]['df']
    assert table['Date'].min() == '2002-09-30' and table['Date'].max() == '2010-09-30'
    assert any(c.value.endswith('· 2002-01-01 to 2010-12-31') for c in at.caption)
    assert any(c.value == 'Data available from 2002-09-30 to 2024-09-30.' for c in at.caption)


def test_viewer_uses_the_sidebar_filtered_analysis_data(at):
    at.number_input(key='start_year_input').set_value(2015).run()
    _load_market_data(at)
    rdata = at.session_state['rdata']
    assert rdata['Year'].min() == 2015 and len(rdata) < len(at.session_state['raw_data'])

    _load_viewer(at)
    table = at.session_state[VIEWER_STATE_KEY]['df']
    assert table['Date'].min() == '2015-09-30'
    assert len(table) == len(rdata)
    assert any(c.value == 'Data available from 2015-09-30 to 2024-09-30.' for c in at.caption)


def test_table_is_dropped_when_reloaded_data_is_empty(at):
    _load_market_data(at)
    _load_viewer(at)
    assert VIEWER_STATE_KEY in at.session_state

    _widget(at.sidebar.checkbox, 'Enable Sector Filter').check().run()
    _widget(at.sidebar.multiselect, 'Include following sectors:').set_value(['Utilities']).run()  # no such rows
    next(b for b in at.button if b.label == 'Load Market Data').click().run()

    assert at.session_state['rdata'].empty
    assert VIEWER_STATE_KEY not in at.session_state
    assert any('Load market data in the Analysis tab first' in i.value for i in at.info)


def test_industry_options_follow_the_sector_selection(at):
    _load_market_data(at)
    assert _widget(at.multiselect, 'Industry').options == ['Biotech', 'Oil Gas Production', 'Software']

    _widget(at.multiselect, 'Sector').set_value(['Energy', 'Healthcare']).run()
    assert _widget(at.multiselect, 'Industry').options == ['Biotech', 'Oil Gas Production']

    _widget(at.multiselect, 'Sector').set_value([]).run()
    assert len(_widget(at.multiselect, 'Industry').options) == 3


def test_removing_a_sector_drops_its_selected_industries(at):
    _load_market_data(at)
    _widget(at.multiselect, 'Sector').set_value(['Technology', 'Energy']).run()
    _widget(at.multiselect, 'Industry').set_value(['Software', 'Oil Gas Production']).run()

    _widget(at.multiselect, 'Sector').set_value(['Energy']).run()   # Technology removed

    _assert_clean(at)
    industry = _widget(at.multiselect, 'Industry')
    assert industry.value == ['Oil Gas Production']
    assert at.session_state[industry.key] == ['Oil Gas Production']   # no hidden selection left
    _load_viewer(at)
    assert set(at.session_state[VIEWER_STATE_KEY]['df']['FactSet_Industry']) == {'Oil Gas Production'}


def test_changing_sectors_keeps_the_date_range_and_sector_selections(at):
    _load_market_data(at)
    _widget(at.date_input, 'Date range').set_value((date(2005, 9, 30), date(2010, 9, 30))).run()
    _widget(at.multiselect, 'Sector').set_value(['Technology']).run()
    keys_before = (_widget(at.date_input, 'Date range').key, _widget(at.multiselect, 'Sector').key,
                   _widget(at.multiselect, 'Industry').key)

    _widget(at.multiselect, 'Sector').set_value(['Technology', 'Energy']).run()

    _assert_clean(at)
    assert (_widget(at.date_input, 'Date range').key, _widget(at.multiselect, 'Sector').key,
            _widget(at.multiselect, 'Industry').key) == keys_before        # same widgets, not new ones
    assert _widget(at.date_input, 'Date range').value == (date(2005, 9, 30), date(2010, 9, 30))
    assert _widget(at.multiselect, 'Sector').value == ['Technology', 'Energy']
