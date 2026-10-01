"""
Tests for the factor builder on the Analysis tab (app/components/factor_selection.py).

AppTest drives app/streamlit_app.py with a small synthetic universe in
session_state['raw_data'], so no data source access is needed.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.streamlit_config import FACTOR_METADATA, FACTOR_GROUPS
from tests.unit.test_saved_runs import _synthetic_universe

APP_DIR = Path(__file__).resolve().parents[2] / 'app'
HIGHER = 'Higher is better'
LOWER = 'Lower is better'

# Defaults agreed for the factor builder (UI label -> direction when added)
EXPECTED_DEFAULTS = {
    '12-Mo Momentum %': HIGHER,
    '6-Mo Momentum %': HIGHER,
    '1-Mo Momentum %': HIGHER,
    'Price to Book Using 9/30 Data': LOWER,
    'Book/Price': HIGHER,
    'Next FY Earns/P': HIGHER,
    'ROE using 9/30 Data': HIGHER,
    'ROA using 9/30 Data': HIGHER,
    'ROA %': HIGHER,
    'Accruals/Assets': LOWER,
    '1-Yr Price Vol %': LOWER,
    '1-Yr Asset Growth %': LOWER,
    '1-Yr CapEX Growth %': LOWER,
}


@pytest.fixture
def at(monkeypatch):
    # `streamlit run` puts app/ on sys.path; AppTest does not.
    monkeypatch.syspath_prepend(str(APP_DIR))
    app = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=60)
    app.session_state['raw_data'] = _synthetic_universe()
    app.run()
    return app


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _pills(at, group):
    return at.button_group(key=f"factor_pick_{FACTOR_GROUPS.index(group)}")


def _add(at, *names):
    # One click on a factor adds it
    for name in names:
        _pills(at, FACTOR_METADATA[name]['group']).set_value(name).run()




def _direction(at, name):
    return at.button_group(key=f"factor_dir_{FACTOR_METADATA[name]['key']}")


def _assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_config_defaults_and_groups():
    assert {n: (HIGHER if m['higher_is_better'] else LOWER) for n, m in FACTOR_METADATA.items()} \
        == EXPECTED_DEFAULTS
    assert all(m['group'] in FACTOR_GROUPS and m['tooltip'] for m in FACTOR_METADATA.values())


def test_initial_state(at):
    _assert_clean(at)
    assert all(_pills(at, g).value is None for g in FACTOR_GROUPS)
    assert not any(b.key == 'factor_add' for b in at.button)   # no Add button: a click adds
    assert _button(at, 'Load Market Data').disabled
    assert 'No factors added yet. Click one above.' in [c.value for c in at.caption]
    assert [w.value for w in at.warning] == ['Please select at least one factor to run the analysis']
    assert not at.success


def test_options_are_grouped_under_category_titles(at):
    assert [(_pills(at, g).label, _pills(at, g).options) for g in FACTOR_GROUPS] == [
        ('**Momentum**', ['12-Mo Momentum %', '6-Mo Momentum %', '1-Mo Momentum %']),
        ('**Value**', ['Price to Book Using 9/30 Data', 'Book/Price', 'Next FY Earns/P']),
        ('**Profitability**', ['ROE using 9/30 Data', 'ROA using 9/30 Data', 'ROA %']),
        ('**Quality**', ['Accruals/Assets', '1-Yr Price Vol %']),
        ('**Growth**', ['1-Yr Asset Growth %', '1-Yr CapEX Growth %']),
    ]


def test_click_adds_with_default_direction_and_removes_option(at):
    _pills(at, 'Quality').set_value('1-Yr Price Vol %').run()

    _assert_clean(at)
    assert _direction(at, '1-Yr Price Vol %').value == LOWER
    assert _pills(at, 'Quality').options == ['Accruals/Assets']
    assert _pills(at, 'Quality').value is None                 # the pick is cleared
    assert not _button(at, 'Load Market Data').disabled
    assert 'No factors added yet. Click one above.' not in [c.value for c in at.caption]
    assert [s.value for s in at.success] == ['Selected 1 factor(s): 1-Yr Price Vol % (Lower is better)']


def test_factors_are_listed_in_click_order(at):
    _add(at, '12-Mo Momentum %', 'Price to Book Using 9/30 Data', '1-Yr Asset Growth %')
    _assert_clean(at)
    assert list(at.session_state['factor_builder_selected']) == [
        '12-Mo Momentum %', 'Price to Book Using 9/30 Data', '1-Yr Asset Growth %']
    assert [_direction(at, n).value for n in at.session_state['factor_builder_selected']] == [
        HIGHER, LOWER, LOWER]


def test_every_factor_gets_its_default(at):
    _add(at, *EXPECTED_DEFAULTS)
    _assert_clean(at)
    assert {n: _direction(at, n).value for n in EXPECTED_DEFAULTS} == EXPECTED_DEFAULTS
    assert all(_pills(at, g).options == [] and _pills(at, g).disabled for g in FACTOR_GROUPS)
    assert [c.value for c in at.caption].count('All added') == len(FACTOR_GROUPS)


def test_direction_change_summary_and_deselect(at):
    _add(at, '12-Mo Momentum %')
    _add(at, '1-Yr Price Vol %')
    _direction(at, '12-Mo Momentum %').set_value(LOWER).run()
    assert [s.value for s in at.success] == [
        'Selected 2 factor(s): 12-Mo Momentum % (Lower is better), 1-Yr Price Vol % (Lower is better)']

    # Clicking the active option deselects it; the previous direction is kept
    _direction(at, '12-Mo Momentum %').set_value(None).run()
    _assert_clean(at)
    assert _direction(at, '12-Mo Momentum %').value == LOWER
    assert at.session_state['factor_builder_selected']['12-Mo Momentum %'] == 'bottom'


def test_remove_returns_factor_and_readd_resets_default(at):
    _add(at, '1-Yr Price Vol %')
    _direction(at, '1-Yr Price Vol %').set_value(HIGHER).run()
    at.button(key='factor_remove_vol').click().run()

    _assert_clean(at)
    assert _pills(at, 'Quality').options == ['Accruals/Assets', '1-Yr Price Vol %']
    assert _button(at, 'Load Market Data').disabled
    assert 'No factors added yet. Click one above.' in [c.value for c in at.caption]

    _add(at, '1-Yr Price Vol %')
    assert _direction(at, '1-Yr Price Vol %').value == LOWER


def test_backtest_receives_old_directions_and_grid_order(at):
    # Added out of grid order (the synthetic universe only has the ROE and ROA % columns)
    _add(at, 'ROA %')
    _direction(at, 'ROA %').set_value(LOWER).run()
    _add(at, 'ROE using 9/30 Data')
    _button(at, 'Load Market Data').click().run()
    _button(at, 'Run Portfolio Analysis').click().run()

    _assert_clean(at)
    run = at.session_state['saved_runs'][0]
    # Same order and 'top'/'bottom' values the old checkbox grid produced
    assert run['factor_labels'] == ['ROE using 9/30 Data', 'ROA %']
    assert run['factor_directions'] == {'ROE_using_9-30_Data': 'top', 'ROA': 'bottom'}


def test_selection_survives_reruns(at):
    _add(at, 'Book/Price')
    _direction(at, 'Book/Price').set_value(LOWER).run()
    at.run()
    at.run()
    assert _direction(at, 'Book/Price').value == LOWER
    assert at.session_state['factor_builder_selected'] == {'Book/Price': 'bottom'}
