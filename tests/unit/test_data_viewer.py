"""
Tests for the Data Viewer tab.

Pure tests cover app/data_viewer_filters.py. AppTest tests drive the real app
(app/streamlit_app.py) with S3 and Supabase patched out, so they also prove the
viewer never downloads anything.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.testing.v1 import AppTest

from app.data_viewer_filters import (
    DISPLAY_ROW_LIMIT,
    category_options,
    csv_file_name,
    filter_raw_data,
    industry_column,
    industry_options,
    keep_valid_selections,
    raw_data_to_csv,
    rows_caption,
    year_options,
)
from src.s3_client import S3DataManager
from src.supabase_client import SupabaseManager
from tests.unit.test_saved_runs import _click_main_tab, browser_like_widget_states

APP_DIR = Path(__file__).resolve().parents[2] / 'app'
SLICE_KEY = 'data_viewer_slice'

SECTOR_INDUSTRIES = {
    'Technology': ['Software', 'Semiconductors'],
    'Energy': ['Oil & Gas Production', 'Coal'],
    'Health Care': ['Biotechnology'],
}


def _universe(industry_col='FactSet_Industry', per_industry=3):
    """Tickers x 2002-2006 across three sectors and five industries."""
    rng = np.random.RandomState(0)
    rows = []
    for year in range(2002, 2007):
        i = 0
        for sector, industries in SECTOR_INDUSTRIES.items():
            for industry in industries:
                for _ in range(per_industry):
                    rows.append({
                        'Ticker': f'T{i:03d}',
                        'Year': year,
                        'Ending_Price': rng.uniform(10, 100),
                        'Next-Years_Return': rng.normal(8, 20),
                        'ROE_using_9-30_Data': rng.normal(0.1, 0.05),
                        'Market_Capitalization': rng.uniform(1e8, 5e9),
                        'Scotts_Sector_5': sector,
                        industry_col: industry,
                    })
                    i += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestFilterRawData:
    def test_year_range_is_inclusive(self):
        out = filter_raw_data(_universe(), (2003, 2005))
        assert sorted(out['Year'].unique()) == [2003, 2004, 2005]
        assert len(out) == 3 * 15

    def test_sector_filter(self):
        out = filter_raw_data(_universe(), (2002, 2006), sectors=['Energy'])
        assert set(out['Scotts_Sector_5']) == {'Energy'}
        assert len(out) == 5 * 6

    def test_industry_filter(self):
        out = filter_raw_data(_universe(), (2002, 2006), industries=['Software', 'Coal'])
        assert set(out['FactSet_Industry']) == {'Software', 'Coal'}
        assert len(out) == 5 * 6

    def test_all_filters_combined(self):
        out = filter_raw_data(_universe(), (2004, 2004), ['Technology'], ['Software'])
        assert len(out) == 3
        assert set(zip(out['Year'], out['Scotts_Sector_5'], out['FactSet_Industry'])) == {
            (2004, 'Technology', 'Software')}

    def test_empty_sector_and_industry_lists_mean_all(self):
        assert len(filter_raw_data(_universe(), (2002, 2006))) == len(_universe())

    def test_industry_column_with_space_in_name(self):
        df = _universe(industry_col='FactSet Industry')
        assert industry_column(df) == 'FactSet Industry'
        out = filter_raw_data(df, (2002, 2006), industries=['Coal'])
        assert set(out['FactSet Industry']) == {'Coal'}

    def test_raw_data_is_not_modified(self):
        raw = _universe()
        before = raw.copy()
        out = filter_raw_data(raw, (2003, 2004), ['Energy'], ['Coal'])
        out.loc[:, 'Ending_Price'] = 0.0      # changing the slice must not reach raw
        pd.testing.assert_frame_equal(raw, before)

    def test_options_come_from_the_data(self):
        df = _universe()
        assert year_options(df) == [2002, 2003, 2004, 2005, 2006]
        assert category_options(df, 'Scotts_Sector_5') == ['Energy', 'Health Care', 'Technology']
        assert category_options(df, industry_column(df)) == [
            'Biotechnology', 'Coal', 'Oil & Gas Production', 'Semiconductors', 'Software']
        assert category_options(df, None) == []

    def test_csv_round_trips_the_table(self):
        df = filter_raw_data(_universe(), (2003, 2004), ['Technology'])
        exported = pd.read_csv(io.BytesIO(raw_data_to_csv(df)), float_precision='round_trip')
        pd.testing.assert_frame_equal(exported, df, check_exact=True)

    def test_rows_caption(self):
        assert rows_caption(6) == '**Rows:** 6'
        assert rows_caption(DISPLAY_ROW_LIMIT) == '**Rows:** 1,000'
        assert rows_caption(1600) == 'Showing 1,000 of 1,600 rows. Download CSV for all rows.'

    def test_csv_file_name(self):
        assert csv_file_name((2003, 2005)) == 'raw_data_2003_2005.csv'


class TestIndustryOptions:
    def test_all_industries_when_no_sector_is_chosen(self):
        assert industry_options(_universe()) == [
            'Biotechnology', 'Coal', 'Oil & Gas Production', 'Semiconductors', 'Software']

    @pytest.mark.parametrize('sectors, expected', [
        (['Technology'], ['Semiconductors', 'Software']),
        (['Energy'], ['Coal', 'Oil & Gas Production']),
        (['Health Care'], ['Biotechnology']),
        (['Energy', 'Health Care'], ['Biotechnology', 'Coal', 'Oil & Gas Production']),
    ])
    def test_only_industries_in_the_chosen_sectors(self, sectors, expected):
        assert industry_options(_universe(), sectors) == expected

    def test_industry_column_with_space_in_name(self):
        assert industry_options(_universe(industry_col='FactSet Industry'), ['Energy']) == [
            'Coal', 'Oil & Gas Production']

    def test_no_industry_column(self):
        assert industry_options(_universe().drop(columns=['FactSet_Industry']), ['Energy']) == []

    def test_raw_data_is_not_modified(self):
        raw = _universe()
        before = raw.copy()
        industry_options(raw, ['Energy'])
        pd.testing.assert_frame_equal(raw, before)


class TestKeepValidSelections:
    def test_drops_selections_no_longer_offered(self):
        assert keep_valid_selections(['Software', 'Coal', 'Biotechnology'], ['Coal', 'Oil & Gas Production']) == ['Coal']

    def test_keeps_order_and_everything_still_offered(self):
        assert keep_valid_selections(['Software', 'Semiconductors'], ['Semiconductors', 'Software']) == [
            'Software', 'Semiconductors']

    def test_nothing_selected(self):
        assert keep_valid_selections([], ['Coal']) == []


# ---------------------------------------------------------------------------
# AppTest: the real Data Viewer tab
# ---------------------------------------------------------------------------

@pytest.fixture
def data_calls(monkeypatch):
    """Patches S3 and Supabase; records every attempt to reach either."""
    calls = []

    def fake_s3_fetch(self):
        calls.append('s3')
        return _universe()

    def no_supabase(self, *a, **k):
        calls.append('supabase')
        raise AssertionError("Supabase must not be called")

    monkeypatch.setenv('FACTOR_LAKE_DATA_SOURCE', 's3')
    monkeypatch.setattr(S3DataManager, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(S3DataManager, 'fetch_all_data', fake_s3_fetch)
    monkeypatch.setattr(S3DataManager, 'read_parquet', lambda self, *a, **k: calls.append('s3') or pd.DataFrame())
    monkeypatch.setattr(SupabaseManager, '__init__', no_supabase)
    monkeypatch.setattr(SupabaseManager, 'fetch_all_data', no_supabase)
    st.cache_resource.clear()
    yield calls
    st.cache_resource.clear()


@pytest.fixture
def deferred_downloads(monkeypatch):
    """Records the callable behind each deferred download button, by file id."""
    callables = {}
    original = MediaFileManager.add_deferred

    def record(self, data_callable, mimetype, coordinates, file_name=None):
        file_id = original(self, data_callable, mimetype, coordinates, file_name)
        callables[file_id] = (data_callable, file_name)
        return file_id

    monkeypatch.setattr(MediaFileManager, 'add_deferred', record)
    return callables


@pytest.fixture
def at(monkeypatch, data_calls):
    """The app before any market data is loaded (data comes from patched S3)."""
    monkeypatch.syspath_prepend(str(APP_DIR))
    browser_like_widget_states(monkeypatch)
    app = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    app.run()
    return app


def _load_market_data(at):
    """Loads market data in the Analysis tab, then opens the Data Viewer."""
    from app.streamlit_config import FACTOR_METADATA
    at.button(key=f"factor_add_{FACTOR_METADATA['ROE using 9/30 Data']['key']}").click().run()
    next(b for b in at.button if b.label == 'Load Market Data').click().run()
    _click_main_tab(at, 'Data Viewer')


def _viewer_tab(at):
    return next(t for t in at.tabs if t.label == 'Data Viewer')


def _assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_tabs_keep_results_second(at):
    assert [t.label for t in at.tabs][:4] == ['Analysis', 'Results', 'Data Viewer', 'About']


def test_info_message_until_market_data_is_loaded(at, data_calls):
    _click_main_tab(at, 'Data Viewer')
    viewer = _viewer_tab(at)
    assert any('Load market data in the Analysis tab first' in i.value for i in viewer.info)
    assert not viewer.select_slider and not viewer.multiselect and not viewer.button
    assert data_calls == []


def test_viewer_never_reaches_s3_or_supabase(at, data_calls, deferred_downloads):
    _load_market_data(at)
    assert data_calls == ['s3']            # the Load Market Data button, once

    at.select_slider(key='dv_years').set_value((2003, 2005)).run()
    at.multiselect(key='dv_sectors').set_value(['Energy']).run()
    at.button(key='dv_load').click().run()
    callable_, _ = deferred_downloads[at.get('download_button')[0].proto.deferred_file_id]
    callable_()
    at.button(key='dv_clear').click().run()
    at.button(key='dv_load').click().run()

    _assert_clean(at)
    assert data_calls == ['s3']            # nothing more from the viewer


def test_load_data_shows_filtered_rows_from_raw_data(at):
    # The sidebar keeps only 2005+, but the viewer filters the full raw data
    at.number_input(key='start_year_input').set_value(2005).run()
    _load_market_data(at)
    assert at.session_state['rdata']['Year'].min() == 2005
    st_rdata_rows = len(at.session_state['rdata'])
    raw_before = at.session_state['raw_data'].copy()

    at.select_slider(key='dv_years').set_value((2003, 2004)).run()
    at.multiselect(key='dv_sectors').set_value(['Technology']).run()
    at.multiselect(key='dv_industries').set_value(['Software']).run()
    assert SLICE_KEY not in at.session_state            # nothing built before Load data
    at.button(key='dv_load').click().run()

    _assert_clean(at)
    shown = _viewer_tab(at).dataframe[0].value
    assert len(shown) == 2 * 3
    assert set(shown['Year']) == {2003, 2004}
    assert set(shown['Scotts_Sector_5']) == {'Technology'}
    assert set(shown['FactSet_Industry']) == {'Software'}
    assert any('**Rows:** 6' in c.value for c in _viewer_tab(at).caption)
    pd.testing.assert_frame_equal(at.session_state['raw_data'], raw_before)
    assert len(at.session_state['rdata']) == st_rdata_rows


def test_nothing_is_built_without_load_data(at):
    _load_market_data(at)
    at.run()
    at.multiselect(key='dv_sectors').set_value(['Energy']).run()
    assert SLICE_KEY not in at.session_state
    assert not _viewer_tab(at).dataframe


@pytest.mark.parametrize('change', [
    lambda at: at.select_slider(key='dv_years').set_value((2002, 2003)),
    lambda at: at.multiselect(key='dv_sectors').set_value(['Energy']),
    lambda at: at.multiselect(key='dv_industries').set_value(['Coal']),
], ids=['years', 'sectors', 'industries'])
def test_changing_a_filter_clears_the_slice(at, change):
    _load_market_data(at)
    at.button(key='dv_load').click().run()
    assert SLICE_KEY in at.session_state

    change(at).run()

    _assert_clean(at)
    assert SLICE_KEY not in at.session_state
    assert not _viewer_tab(at).dataframe


def test_clear_button_clears_the_slice(at):
    _load_market_data(at)
    at.button(key='dv_load').click().run()
    assert SLICE_KEY in at.session_state

    at.button(key='dv_clear').click().run()

    _assert_clean(at)
    assert SLICE_KEY not in at.session_state
    assert not _viewer_tab(at).dataframe


def test_screen_shows_first_1000_rows_and_csv_has_all(at, deferred_downloads, monkeypatch):
    big = _universe(per_industry=80)                      # 400 tickers x 5 years
    monkeypatch.setattr(S3DataManager, 'fetch_all_data', lambda self: big)
    _load_market_data(at)
    at.select_slider(key='dv_years').set_value((2003, 2006)).run()
    at.button(key='dv_load').click().run()

    _assert_clean(at)
    viewer = _viewer_tab(at)
    shown = viewer.dataframe[0].value
    full = at.session_state[SLICE_KEY]['df']
    assert len(full) == 1600
    assert len(shown) == DISPLAY_ROW_LIMIT
    assert any(c.value == 'Showing 1,000 of 1,600 rows. Download CSV for all rows.' for c in viewer.caption)

    callable_, file_name = deferred_downloads[viewer.get('download_button')[0].proto.deferred_file_id]
    assert file_name == 'raw_data_2003_2006.csv'
    exported = pd.read_csv(io.BytesIO(callable_()), float_precision='round_trip')
    assert exported.columns.tolist() == shown.columns.tolist()
    pd.testing.assert_frame_equal(exported, full, check_exact=True)                  # every filtered row
    pd.testing.assert_frame_equal(exported.head(DISPLAY_ROW_LIMIT), shown, check_exact=True)


def test_small_result_shows_every_row_and_csv_matches(at, deferred_downloads):
    _load_market_data(at)
    at.select_slider(key='dv_years').set_value((2004, 2006)).run()
    at.multiselect(key='dv_sectors').set_value(['Health Care', 'Energy']).run()
    at.button(key='dv_load').click().run()

    viewer = _viewer_tab(at)
    callable_, _ = deferred_downloads[viewer.get('download_button')[0].proto.deferred_file_id]
    exported = pd.read_csv(io.BytesIO(callable_()), float_precision='round_trip')
    assert len(exported) == 3 * 9
    pd.testing.assert_frame_equal(exported, viewer.dataframe[0].value, check_exact=True)


def test_industry_options_follow_the_sector_selection(at):
    _load_market_data(at)
    assert at.multiselect(key='dv_industries').options == [
        'Biotechnology', 'Coal', 'Oil & Gas Production', 'Semiconductors', 'Software']

    at.multiselect(key='dv_sectors').set_value(['Energy']).run()
    assert at.multiselect(key='dv_industries').options == ['Coal', 'Oil & Gas Production']

    at.multiselect(key='dv_sectors').set_value([]).run()
    assert len(at.multiselect(key='dv_industries').options) == 5


def test_removing_a_sector_drops_its_selected_industries(at):
    _load_market_data(at)
    at.multiselect(key='dv_sectors').set_value(['Technology', 'Energy']).run()
    at.multiselect(key='dv_industries').set_value(['Software', 'Coal']).run()

    at.multiselect(key='dv_sectors').set_value(['Energy']).run()   # Technology removed

    _assert_clean(at)
    assert at.multiselect(key='dv_industries').value == ['Coal']
    assert at.session_state['dv_industries'] == ['Coal']            # no hidden selection left
    at.button(key='dv_load').click().run()
    shown = _viewer_tab(at).dataframe[0].value
    assert set(shown['FactSet_Industry']) == {'Coal'}


def test_industries_still_in_the_sectors_are_kept(at):
    _load_market_data(at)
    at.multiselect(key='dv_industries').set_value(['Software', 'Coal']).run()   # no sector yet

    at.multiselect(key='dv_sectors').set_value(['Technology']).run()

    assert at.multiselect(key='dv_industries').value == ['Software']
    assert at.session_state['dv_industries'] == ['Software']


@pytest.mark.parametrize('other_tab', ['Analysis', 'Results', 'About'])
def test_leaving_the_data_viewer_clears_the_slice(at, other_tab):
    _load_market_data(at)
    at.button(key='dv_load').click().run()
    assert SLICE_KEY in at.session_state

    _click_main_tab(at, other_tab)

    _assert_clean(at)
    assert at.session_state['main_tabs'] == other_tab
    assert SLICE_KEY not in at.session_state
    assert not _viewer_tab(at).dataframe                # the viewer is not drawn at all

    _click_main_tab(at, 'Data Viewer')                  # coming back starts empty
    assert not _viewer_tab(at).dataframe
    assert any('click **Load data**' in c.value for c in _viewer_tab(at).caption)


def test_staying_on_the_data_viewer_keeps_the_slice(at):
    _load_market_data(at)
    at.button(key='dv_load').click().run()
    shown = _viewer_tab(at).dataframe[0].value

    at.run()                                            # any rerun while the viewer is open
    _click_main_tab(at, 'Data Viewer')                  # clicking the open tab again

    _assert_clean(at)
    assert SLICE_KEY in at.session_state
    pd.testing.assert_frame_equal(_viewer_tab(at).dataframe[0].value, shown)


def test_only_the_open_tab_is_drawn(at):
    _load_market_data(at)                               # Data Viewer open
    assert at.select_slider(key='dv_years') is not None
    assert not any(b.label == 'Load Market Data' for b in at.button)   # Analysis not drawn
    assert not any('About Factor-Lake' in h.value for h in at.header)  # About not drawn

    _click_main_tab(at, 'About')
    assert any('About Factor-Lake' in h.value for h in at.header)
    assert not at.select_slider
