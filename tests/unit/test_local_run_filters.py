"""A local run uses the sidebar's filters at Run time, not the ones from Load Market Data."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st

import app.streamlit_utils as utils


class _State(dict):
    """st.session_state outside a Streamlit run: a dict with attribute access."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _universe():
    """Two sectors, 2002-2006, with the columns the loader and engine read (ROA and ROE factors)."""
    rng = np.random.RandomState(1)
    return pd.DataFrame([{
        'Ticker': f'{sector[0]}{i:02d}', 'Year': year, 'Ending_Price': rng.uniform(10, 100),
        'Next-Years_Return': rng.normal(8, 20), 'ROA': rng.normal(0.05, 0.03),
        'ROE_using_9-30_Data': rng.normal(0.1, 0.05),
        'Market_Capitalization': rng.uniform(1e8, 5e9), 'Scotts_Sector_5': sector, 'FactSet_Industry': 'Software',
    } for year in range(2002, 2007) for sector in ('Technology', 'Health') for i in range(10)])


@pytest.fixture
def state(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'app'))   # as `streamlit run` does
    monkeypatch.delenv(utils.BACKTEST_LAMBDA_ENV, raising=False)
    session = _State(raw_data=_universe(), rdata=None, loaded_filters=None, data_loaded=False,
                     next_run_id=1, saved_runs=[], analysis_history=[])
    monkeypatch.setattr(st, 'session_state', session)
    return session


def test_filters_changed_after_loading_apply_to_the_run(state):
    settings = {'start_year': 2002, 'end_year': 2004, 'initial_aum': 1000.0, 'use_lambda': False,
                'selected_sectors': [], 'restrict_fossil_fuels': False}
    utils.load_and_process_data(settings)
    assert set(state.rdata['Scotts_Sector_5']) == {'Technology', 'Health'}
    # After loading, the user narrows the sectors and extends the years, then runs without reloading
    settings = {**settings, 'selected_sectors': ['Technology'], 'end_year': 2006}
    run = utils.run_backtest_logic(settings, ['ROA %'], {'ROA %': 'top'})
    # Before the fix the run used the loaded data: both sectors, and no returns after 2003,
    # while its settings, captions and formula said Technology through 2006
    assert set(run['data']['Scotts_Sector_5']) == {'Technology'}
    assert run['data']['Year'].max() == 2006
    assert len(run['results']['yearly_returns']) == 4               # 2002-2003 ... 2005-2006
    assert run['universe']['selected_sectors'] == ['Technology']
    assert 'sectors = Technology;' in run['calculation_preview']['formula']
    # The session's data now matches the sidebar, so the next run does not filter again
    assert state.rdata is run['data'] and state.loaded_period == (2002, 2006)
