"""
Tests for Load Market Data (load_and_process_data in app/streamlit_utils.py).

raw_data is the shared copy of the market data (st.cache_resource), so it must
never change. rdata, the sidebar-filtered data, must be exactly what the old
copy-then-filter code produced, without copying raw_data whole.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.backtest_engine as backtest_engine

APP_DIR = Path(__file__).resolve().parents[2] / 'app'
FOSSIL = {"integratedoil", "oilfieldservicesequipment", "oilgasproduction", "coal", "oilrefiningmarketing"}
SECTOR_INDUSTRIES = {
    'Technology': ['Software', 'Semiconductors'],
    'Energy': ['Oil & Gas Production', 'Coal', 'Integrated Oil'],
    'Healthcare': ['Biotechnology'],
    'Financials': ['Banks'],
}


def _universe() -> pd.DataFrame:
    """8 tickers per industry x 2002-2024, including fossil fuel industries."""
    rng = np.random.RandomState(7)
    rows = []
    for year in range(2002, 2025):
        i = 0
        for sector, industries in SECTOR_INDUSTRIES.items():
            for industry in industries:
                for _ in range(8):
                    rows.append({
                        'Ticker': f'T{i:03d}', 'Year': year, 'Date': f'{year}-09-30',
                        'Ending_Price': rng.uniform(10, 100), 'Next-Years_Return': rng.normal(8, 20),
                        'ROE_using_9-30_Data': rng.normal(0.1, 0.05), 'ROA': rng.normal(0.05, 0.03),
                        'Market_Capitalization': rng.uniform(1e8, 5e9),
                        'Scotts_Sector_5': sector, 'FactSet_Industry': industry,
                    })
                    i += 1
    return pd.DataFrame(rows)


def _old_filter(raw: pd.DataFrame, start: int, end: int, sectors, fossil: bool) -> pd.DataFrame:
    """The previous load_and_process_data: copy everything, then filter step by step."""
    df = raw.copy()
    if sectors:
        df = df[df['Scotts_Sector_5'].isin(sectors)]
    if fossil:
        clean = df['FactSet_Industry'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        df = df[~clean.isin(FOSSIL)]
    return df[(df['Year'] >= start) & (df['Year'] <= end)]


SETUPS = {
    'defaults': {},
    'start_year': {'start': 2010},
    'sectors': {'sectors': ['Technology', 'Energy']},
    'fossil': {'fossil': True},
    'all_filters': {'start': 2008, 'fossil': True, 'sectors': ['Energy', 'Healthcare']},
}


def _load(monkeypatch, setup):
    """Loads market data in the real app with the given sidebar settings."""
    monkeypatch.syspath_prepend(str(APP_DIR))
    at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    raw = _universe()
    at.session_state['raw_data'] = raw
    at.run()
    at.button(key='factor_add_roe').click().run()
    if setup.get('start'):
        at.number_input(key='start_year_input').set_value(setup['start']).run()
    if setup.get('fossil'):
        next(c for c in at.sidebar.checkbox if c.label == 'Restrict Fossil Fuel Companies').check().run()
    if setup.get('sectors'):
        next(c for c in at.sidebar.checkbox if c.label == 'Enable Sector Filter').check().run()
        next(m for m in at.sidebar.multiselect
             if m.label == 'Include following sectors:').set_value(setup['sectors']).run()
    next(b for b in at.button if b.label == 'Load Market Data').click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at, raw


@pytest.mark.parametrize('name', SETUPS)
def test_rdata_matches_the_old_copy_then_filter(monkeypatch, name):
    setup = SETUPS[name]
    at, raw = _load(monkeypatch, setup)
    expected = _old_filter(_universe(), setup.get('start', 2002), 2024,
                           setup.get('sectors'), setup.get('fossil', False))
    pd.testing.assert_frame_equal(at.session_state['rdata'], expected, check_exact=True)  # same rows, index, dtypes


@pytest.mark.parametrize('name', SETUPS)
def test_raw_data_is_unchanged_by_loading_and_running(monkeypatch, name):
    at, raw = _load(monkeypatch, SETUPS[name])
    next(b for b in at.button if b.label == 'Run Portfolio Analysis').click().run()
    assert not at.exception, [e.value for e in at.exception]

    assert at.session_state['raw_data'] is raw
    pd.testing.assert_frame_equal(raw, _universe(), check_exact=True)


@pytest.mark.parametrize('name', SETUPS)
def test_backtest_results_match_the_old_data(monkeypatch, name):
    setup = SETUPS[name]
    at, _ = _load(monkeypatch, setup)
    next(b for b in at.button if b.label == 'Run Portfolio Analysis').click().run()
    run = at.session_state['saved_runs'][0]

    old_data = _old_filter(_universe(), setup.get('start', 2002), 2024,
                           setup.get('sectors'), setup.get('fossil', False))
    s = run['settings']
    expected = backtest_engine.rebalance_portfolio(
        old_data, run['factors'], factor_directions=run['factor_directions'],
        start_year=int(s['start_year']), end_year=int(s['end_year']), initial_aum=s['initial_aum'],
        benchmark_index=s.get('benchmark_index', 1), top_pct=s.get('top_pct', 10.0),
        use_market_cap_weight=s.get('use_market_cap_weight', False),
        delisting_strategy=s.get('delisting_strategy', 'zero_return'),
        factor_combination=s.get('factor_combination', 'mixing'),
        factor_labels=dict(zip(run['factors'], run['factor_labels'])),
    )
    for key, value in expected.items():
        assert repr(run['results'][key]) == repr(value), key


def test_no_filters_share_raw_data_without_copying_it(monkeypatch):
    at, raw = _load(monkeypatch, {})
    rdata = at.session_state['rdata']

    assert rdata is not raw                                         # its own DataFrame...
    assert np.shares_memory(rdata['Ending_Price'].to_numpy(), raw['Ending_Price'].to_numpy())  # ...same data

    rdata['Extra'] = 1.0                                            # adding a column stays local
    assert 'Extra' not in raw.columns


def test_filtered_rdata_is_a_separate_copy(monkeypatch):
    at, raw = _load(monkeypatch, {'start': 2010})
    rdata = at.session_state['rdata']
    assert not np.shares_memory(rdata['Ending_Price'].to_numpy(), raw['Ending_Price'].to_numpy())


def test_each_load_gives_a_new_rdata_object(monkeypatch):
    # The Data Viewer clears its table when rdata is a different object (reloaded data)
    at, _ = _load(monkeypatch, {})
    first = at.session_state['rdata']
    next(b for b in at.button if b.label == 'Load Market Data').click().run()
    assert at.session_state['rdata'] is not first
