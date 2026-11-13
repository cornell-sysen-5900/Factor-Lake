"""Crosswalk semantics, explicit defaults, UI, and hand-calculated forward spreads."""
import hashlib
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.streamlit_config import FACTOR_METADATA, WRDS_FACTOR_METADATA
from app.wrds_factor_review import REASONS, factor_rows, legacy_rows, matches_for
from scripts.wrds_factor_report import main, render_report
from src.wrds_data import PANEL_FILE, RETURNS_FILE
from src.wrds_factor_evidence import monthly_spreads
from src.wrds_factors import WRDS_FACTOR_COLUMNS
from tests.unit.test_backtest_lambda import APP_DIR


def frames():
    # January signal increases 1..10; the following February return increases 1%..10%.
    panel = pd.DataFrame({'date': pd.to_datetime(['2020-01-31'] * 10), 'permno': range(1, 11),
                          'prc': 10., 'mkt_cap_musd': 100., 'f_roa': range(1, 11)})
    returns = pd.DataFrame({'month_end': pd.to_datetime(['2020-02-28'] * 10), 'permno': range(1, 11),
                            'ret_m': np.arange(1, 11) / 100., 'delisted_in_month': False})
    return panel, returns


def test_catalog_covers_every_wrds_and_legacy_factor_once():
    assert set(REASONS) == set(WRDS_FACTOR_COLUMNS)
    assert [r['column'] for r in factor_rows()] == list(WRDS_FACTOR_COLUMNS)
    assert len(legacy_rows()) == len(FACTOR_METADATA) == 13
    assert len({r['legacy_column'] for r in legacy_rows()}) == 13
    assert all(r['reason'] and r['default_direction'] in ('Higher', 'Lower') for r in factor_rows())


def test_related_is_never_misrepresented_as_a_definition_match():
    assert [m.column for m in matches_for('f_roa')] == ['ROA']
    assert not matches_for('mom_12_1')             # skip-month momentum is not 12-month momentum
    assert not matches_for('mom_12m')             # CRSP total returns vs Legacy price returns
    assert not matches_for('f_book_to_price')     # inverse P/B and unspecified B/P price date
    assert not matches_for('f_forward_ep')        # FY1 is not next fiscal year
    assert not matches_for('f_forward_ep_fy2')    # Legacy 3-analyst coverage is not guaranteed
    assert all(r['wrds_match'] == 'none' for r in legacy_rows() if r['status'] == 'related')


def test_picker_highlights_matches_and_preserves_user_direction(monkeypatch):
    monkeypatch.syspath_prepend(str(APP_DIR))
    at = AppTest.from_string('''
import streamlit as st
from components.factor_selection import render_factor_selection
render_factor_selection('wrds')
''').run()
    assert not at.exception
    # Matches are tinted through CSS, not type="primary" (which looks like Run Portfolio Analysis)
    assert at.button(key='factor_add_w_roa').proto.type == 'secondary'
    assert at.button(key='factor_add_w_mom_12_1').proto.type == 'secondary'
    tint = ''.join(h.proto.body for h in at.get('html'))
    assert '.st-key-factor_add_w_roa button' in tint
    assert 'factor_add_w_mom_12_1' not in tint
    assert 'ROA %' in at.button(key='factor_add_w_roa').help
    assert any('Colored cards' in c.value for c in at.caption)
    at.button(key='factor_add_w_roa').click().run()
    assert at.session_state['factor_builder_selected']['ROA'] == 'top'
    at.button_group(key='factor_dir_w_roa').set_value('Lower is better').run()
    at.run()
    assert at.session_state['factor_builder_selected']['ROA'] == 'bottom'


def test_hand_calculated_forward_returns_and_lower_direction():
    panel, returns = frames()
    before_panel, before_returns = panel.copy(deep=True), returns.copy(deep=True)
    top, monthly = monthly_spreads(panel, returns, {'f_roa': True})
    bottom, _ = monthly_spreads(panel, returns, {'f_roa': False})
    assert monthly[0]['signal_date'] == '2020-01-31' and monthly[0]['return_month'] == '2020-02'
    assert monthly[0]['long_return'] == pytest.approx(.1)
    assert monthly[0]['short_return'] == pytest.approx(.01)
    assert top['f_roa']['annualized_spread'] == pytest.approx(12 * .09)
    assert bottom['f_roa']['annualized_spread'] == pytest.approx(-12 * .09)
    assert bottom['f_roa']['evidence_status'] == 'team review'
    pd.testing.assert_frame_equal(panel, before_panel)
    pd.testing.assert_frame_equal(returns, before_returns)


def test_current_month_and_far_future_returns_do_not_enter_next_month_spread():
    panel, returns = frames()
    fake = pd.concat([returns.assign(month_end=pd.Timestamp('2020-01-31'), ret_m=100),
                      returns, returns.assign(month_end=pd.Timestamp('2021-01-29'), ret_m=200)])
    baseline = monthly_spreads(panel, returns, {'f_roa': True})
    assert monthly_spreads(panel, fake, {'f_roa': True}) == baseline


def test_delisting_loss_is_included_and_missing_return_keeps_weight():
    panel, returns = frames()
    returns.loc[returns.permno == 10, ['ret_m', 'delisted_in_month']] = [-1., True]
    results, _ = monthly_spreads(panel, returns, {'f_roa': True})
    assert results['f_roa']['mean_monthly_spread'] == pytest.approx(-1.01)
    returns.loc[returns.permno == 10, 'ret_m'] = np.nan
    results, series = monthly_spreads(panel, returns, {'f_roa': True}, tail_pct=20)
    # Long weights remain 1/2 each: (9% + 0%)/2, not 9% after survivor reweighting.
    assert series[0]['long_return'] == pytest.approx(.045)
    assert results['f_roa']['missing_returns'] == 1
    assert results['f_roa']['coverage_status'] == 'review coverage'


def test_small_universe_nonfinite_constant_and_missing_month_handled_explicitly():
    panel, returns = frames()
    panel = panel.iloc[:2].copy()
    result, rows = monthly_spreads(panel, returns, {'f_roa': True})
    assert rows[0]['stocks_per_leg'] == 1
    panel['f_roa'] = [np.inf, 1]
    result, rows = monthly_spreads(panel, returns, {'f_roa': True})
    assert not rows and result['f_roa']['annualized_spread'] is None
    panel['f_roa'] = 1
    assert not monthly_spreads(panel, returns, {'f_roa': True})[1]
    panel['f_roa'] = [1, 2]
    returns['month_end'] = pd.Timestamp('2020-03-31')
    result, rows = monthly_spreads(panel, returns, {'f_roa': True})
    assert not rows and result['f_roa']['skipped_months'] == 1


def test_ties_break_deterministically_and_inputs_cannot_overlap_or_duplicate():
    panel, returns = frames()
    panel['f_roa'] = [1] * 5 + [2] * 5
    result = monthly_spreads(panel, returns, {'f_roa': True}, tail_pct=50)
    shuffled = monthly_spreads(panel.sample(frac=1, random_state=1), returns.sample(frac=1, random_state=2),
                               {'f_roa': True}, tail_pct=50)
    assert result == shuffled
    with pytest.raises(ValueError, match='disjoint'):
        monthly_spreads(panel, returns, {'f_roa': True}, tail_pct=51)
    with pytest.raises(ValueError, match='Duplicate'):
        monthly_spreads(pd.concat([panel, panel.iloc[:1]]), returns, {'f_roa': True})
    with pytest.raises(ValueError, match='Duplicate'):
        monthly_spreads(panel, pd.concat([returns, returns.iloc[:1]]), {'f_roa': True})


def test_empty_buyable_universe_and_entirely_missing_returns_are_not_positive_evidence():
    panel, returns = frames()
    panel['prc'] = 0
    summary, _ = monthly_spreads(panel, returns, {'f_roa': True})
    assert summary['f_roa']['skipped_months'] == 1
    panel['prc'] = 10
    returns['ret_m'] = np.nan
    summary, _ = monthly_spreads(panel, returns, {'f_roa': True})
    assert summary['f_roa']['annualized_spread'] is None
    assert summary['f_roa']['evidence_status'] == 'insufficient data'


def test_report_flags_disagreement_without_changing_defaults():
    defaults = deepcopy(WRDS_FACTOR_METADATA)
    panel, returns = frames()
    returns['ret_m'] *= -1
    evidence, _ = monthly_spreads(panel, returns, {'f_roa': True})
    report = render_report(evidence, source='Synthetic test data')
    assert 'team review' in report and '-108.000 pp/year' in report
    assert WRDS_FACTOR_METADATA == defaults
    assert 'Empirical spreads pending' in render_report()
    assert all(f'`{column}`' in render_report() for column in WRDS_FACTOR_COLUMNS)


def test_catalog_cli_and_real_data_output_guard(tmp_path):
    output = tmp_path / 'catalog.md'
    assert main(['--catalog-only', '--output', str(output)]) == 0
    assert 'Empirical spreads pending' in output.read_text()
    with pytest.raises(SystemExit):
        main(['--s3', '--output', 'DOCS/never_write_licensed_results.md'])


def test_private_report_cli_reads_parquet_and_writes_traceable_evidence(tmp_path):
    panel, returns = frames()
    for column in WRDS_FACTOR_COLUMNS:
        panel[column] = np.arange(1, 11, dtype=float)
    panel.to_parquet(tmp_path / PANEL_FILE)
    returns.to_parquet(tmp_path / RETURNS_FILE)
    original_hashes = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                       for name in (PANEL_FILE, RETURNS_FILE)}
    output = tmp_path / 'review.md'
    assert main(['--data-dir', str(tmp_path), '--output', str(output)]) == 0
    payload = json.loads(output.with_suffix('.json').read_text())
    assert set(payload['factors']) == set(WRDS_FACTOR_COLUMNS)
    assert payload['factors']['f_roa']['annualized_spread'] == pytest.approx(1.08)
    assert payload['factors']['f_asset_growth_1yr']['annualized_spread'] == pytest.approx(-1.08)
    assert payload['factors']['f_asset_growth_1yr']['evidence_status'] == 'team review'
    for name, digest in original_hashes.items():
        assert payload['inputs'][name]['sha256'] == digest
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert len(pd.read_csv(output.with_suffix('.csv'))) == len(WRDS_FACTOR_COLUMNS)
    assert 'pending data' not in output.read_text()
    assert payload['requested_years'] == [2002, 2026]
