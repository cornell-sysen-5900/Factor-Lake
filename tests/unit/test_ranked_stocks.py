"""
Tests for the per-year ranked stocks table.

Engine tests cover build_ranked_stocks_table / build_ranked_stocks_by_year in
src/backtest_engine.py. Helper tests cover the pure display/CSV/default-year
functions in app/ranked_stocks.py. AppTest tests drive the real app
with the synthetic universe from test_saved_runs.
"""
import io
import math

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import build_ranked_stocks_table, build_ranked_stocks_by_year
from src.factor_utils import normalize_series
from app.ranked_stocks import (
    build_ranked_display_table,
    default_ranked_year,
    ranked_table_to_csv,
)
from tests.unit.test_saved_runs import RUN, _add_factor, _assert_clean, _click, _open_tab, at  # noqa: F401

FACTORS = ['ROE_using_9-30_Data', 'ROA']
DIRECTIONS = {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom'}
LABELS = {'ROE_using_9-30_Data': 'ROE using 9/30 Data', 'ROA': 'ROA %'}


def _universe(years=(2020, 2021, 2022), n=20, seed=1):
    rng = np.random.RandomState(seed)
    rows = []
    for year in years:
        for i in range(n):
            rows.append({
                'Ticker': f'T{i:02d}',
                'Year': year,
                'ROE_using_9-30_Data': rng.normal(0.1, 0.05),
                'ROA': rng.normal(0.05, 0.03),
                'Next-Years_Return': rng.normal(8, 20),
                'Scotts_Sector_5': 'Technology',
            })
    return pd.DataFrame(rows)


def _expected_ranking(data, year, top_pct):
    """The composite ranking, computed independently of build_ranked_stocks_table."""
    df_year = data[data['Year'] == year].set_index('Ticker')
    scores = pd.concat([
        normalize_series(df_year[f], higher_is_better=DIRECTIONS[f] == 'top') for f in FACTORS
    ], axis=1).mean(axis=1).dropna().sort_values(ascending=False)
    return scores.head(max(1, math.floor(len(scores) * top_pct / 100.0)))


# ---------------------------------------------------------------------------
# Engine: one year
# ---------------------------------------------------------------------------

class TestBuildRankedStocksTable:
    def test_one_column_per_factor_after_composite_score(self):
        table = build_ranked_stocks_table(_universe(), FACTORS, DIRECTIONS, 2021, 25.0, LABELS)
        assert table.columns.tolist() == [
            'Rank', 'Ticker', 'Composite Score', 'ROE using 9/30 Data', 'ROA %',
            'Next-Year Return %', 'Sector',
        ]

    def test_factor_columns_hold_each_tickers_raw_value(self):
        data = _universe()
        table = build_ranked_stocks_table(data, FACTORS, DIRECTIONS, 2021, 25.0, LABELS)
        raw = data[data['Year'] == 2021].set_index('Ticker')
        for _, row in table.iterrows():
            assert row['ROE using 9/30 Data'] == raw.loc[row['Ticker'], 'ROE_using_9-30_Data']
            assert row['ROA %'] == raw.loc[row['Ticker'], 'ROA']

    def test_ranking_and_scores_unchanged(self):
        data = _universe()
        table = build_ranked_stocks_table(data, FACTORS, DIRECTIONS, 2021, 25.0, LABELS)
        expected = _expected_ranking(data, 2021, 25.0)
        assert table['Ticker'].tolist() == expected.index.tolist()
        assert table['Composite Score'].tolist() == pytest.approx(expected.values.tolist())
        assert table['Rank'].tolist() == list(range(1, len(expected) + 1))

    def test_column_names_used_when_no_labels_given(self):
        table = build_ranked_stocks_table(_universe(), FACTORS, DIRECTIONS, 2021, 25.0)
        assert table.columns.tolist()[3:5] == FACTORS

    def test_factor_missing_from_data_is_a_blank_column(self):
        data = _universe().drop(columns=['ROA'])
        table = build_ranked_stocks_table(data, FACTORS, DIRECTIONS, 2021, 25.0, LABELS)
        assert 'ROA %' in table.columns
        assert table['ROA %'].isna().all()
        assert table['ROE using 9/30 Data'].notna().all()


# ---------------------------------------------------------------------------
# Engine: every rebalance year
# ---------------------------------------------------------------------------

class TestBuildRankedStocksByYear:
    def test_one_table_per_rebalance_year(self):
        # range(start_year, end_year): the end year is not a rebalance year
        by_year = build_ranked_stocks_by_year(_universe(), FACTORS, DIRECTIONS, 2020, 2022, 25.0, LABELS)
        assert sorted(by_year) == [2020, 2021]

    def test_each_year_matches_the_single_year_table(self):
        data = _universe()
        by_year = build_ranked_stocks_by_year(data, FACTORS, DIRECTIONS, 2020, 2023, 25.0, LABELS)
        for year, rows in by_year.items():
            single = build_ranked_stocks_table(data, FACTORS, DIRECTIONS, year, 25.0, LABELS)
            assert rows == single.to_dict('records')

    def test_rankings_change_between_years(self):
        by_year = build_ranked_stocks_by_year(_universe(), FACTORS, DIRECTIONS, 2020, 2023, 25.0, LABELS)
        tickers = [[r['Ticker'] for r in rows] for rows in by_year.values()]
        assert tickers[0] != tickers[1]

    def test_empty_years_are_skipped(self):
        data = _universe(years=(2020, 2022))
        by_year = build_ranked_stocks_by_year(data, FACTORS, DIRECTIONS, 2020, 2023, 25.0, LABELS)
        assert sorted(by_year) == [2020, 2022]

    def test_no_factors_gives_no_tables(self):
        assert build_ranked_stocks_by_year(_universe(), [], {}, 2020, 2023) == {}


# ---------------------------------------------------------------------------
# Results tab helpers
# ---------------------------------------------------------------------------

class TestRankedTableHelpers:
    def test_newest_year_is_default(self):
        assert default_ranked_year([2005, 2023, 2010]) == 2023

    def test_no_default_without_years(self):
        assert default_ranked_year([]) is None

    def test_display_table_converts_and_rounds(self):
        rows = [{'Rank': 1, 'Ticker': 'A', 'Composite Score': '1.234567',
                 'ROA %': '0.0512', 'Next-Year Return %': '12.3456', 'Sector': 'Tech'}]
        df = build_ranked_display_table(rows, ['ROA %'])
        assert df.loc[0, 'Composite Score'] == 1.2346
        assert df.loc[0, 'ROA %'] == 0.0512
        assert df.loc[0, 'Next-Year Return %'] == 12.35

    def test_csv_has_same_rows_and_columns_as_table(self):
        by_year = build_ranked_stocks_by_year(_universe(), FACTORS, DIRECTIONS, 2020, 2023, 25.0, LABELS)
        df = build_ranked_display_table(by_year[2021], list(LABELS.values()))
        exported = pd.read_csv(io.BytesIO(ranked_table_to_csv(df)))
        pd.testing.assert_frame_equal(exported, df)


# ---------------------------------------------------------------------------
# AppTest: the real Results tab
# ---------------------------------------------------------------------------

def _run_two_factor_backtest(at):
    _add_factor(at, 'ROE using 9/30 Data')
    _add_factor(at, 'ROA %')
    _click(at, 'Load Market Data')
    _click(at, RUN)


@pytest.fixture
def served_files(monkeypatch):
    """Records every file AppTest stores for a download, by file id."""
    from streamlit.runtime.memory_media_file_storage import MemoryFile, MemoryMediaFileStorage
    files = {}
    original = MemoryMediaFileStorage.load_and_get_id

    def record(self, path_or_data, mimetype, kind, filename=None):
        file_id = original(self, path_or_data, mimetype, kind, filename)
        files[file_id] = MemoryFile(content=path_or_data, mimetype=mimetype, kind=kind, filename=filename)
        return file_id

    monkeypatch.setattr(MemoryMediaFileStorage, 'load_and_get_id', record)
    return files


def _downloaded_csv(at, served_files, key):
    """The exact file the Download button with this key hands the browser."""
    button = next(b for b in at.get('download_button') if b.proto.id.endswith(key))
    file_id = button.proto.url.rsplit('/', 1)[1].split('.')[0]
    return served_files[file_id]


def test_year_selector_defaults_to_newest_year(at):
    _run_two_factor_backtest(at)
    _assert_clean(at)
    selector = at.selectbox(key='ranked_year_1')
    years = sorted(at.session_state['saved_runs'][0]['results']['ranked_stocks_by_year'])
    assert years == list(range(2002, 2024))   # synthetic data 2002-2024, end year excluded
    assert selector.value == 2023


def test_table_shows_factor_values_and_updates_per_year(at):
    _run_two_factor_backtest(at)
    first = at.dataframe[0].value
    assert first.columns.tolist()[3:5] == ['ROE using 9/30 Data', 'ROA %']

    at.selectbox(key='ranked_year_1').set_value(2010).run()
    _assert_clean(at)
    shown = at.dataframe[0].value
    data = at.session_state['saved_runs'][0]['data']
    raw = data[data['Year'] == 2010].set_index('Ticker')
    assert shown['Ticker'].tolist() != first['Ticker'].tolist()
    for _, row in shown.iterrows():
        assert row['ROA %'] == raw.loc[row['Ticker'], 'ROA']


def test_downloaded_csv_matches_table_on_screen(at, served_files):
    _run_two_factor_backtest(at)
    at.selectbox(key='ranked_year_1').set_value(2010).run()
    file = _downloaded_csv(at, served_files, 'ranked_csv_1')
    assert file.filename == 'ranked_stocks_2010.csv'
    assert file.mimetype == 'text/csv'
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(file.content)), at.dataframe[0].value)


def test_saved_runs_have_independent_year_selectors(at, served_files):
    _run_two_factor_backtest(at)          # Run 1
    _click(at, RUN)                       # Run 2, same factors
    run_1 = 'Run 1: ROE using 9/30 Data +1'

    _open_tab(at, run_1)
    at.run()
    _open_tab(at, run_1)
    at.selectbox(key='ranked_year_1').set_value(2010).run()
    assert at.selectbox(key='ranked_year_1').value == 2010

    _open_tab(at, 'Run 2: ROE using 9/30 Data +1')
    at.run()
    _assert_clean(at)
    assert at.selectbox(key='ranked_year_2').value == 2023
    assert _downloaded_csv(at, served_files, 'ranked_csv_2').filename == 'ranked_stocks_2023.csv'

    # Run 1 keeps its chosen year after switching away and back
    _open_tab(at, run_1)
    at.run()
    assert at.selectbox(key='ranked_year_1').value == 2010
    assert _downloaded_csv(at, served_files, 'ranked_csv_1').filename == 'ranked_stocks_2010.csv'
