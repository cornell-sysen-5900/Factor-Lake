"""Crosswalk semantics, explicit defaults, UI, and hand-calculated forward spreads."""
import hashlib
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import strategy_config as cfg
from app.streamlit_config import FACTOR_METADATA, WRDS_FACTOR_METADATA
from app.wrds_factor_review import REASONS, factor_rows, legacy_rows, replaces
from scripts.wrds_factor_report import main, render_report
from src.wrds_data import PANEL_FILE, RETURNS_FILE
from src.wrds_factor_evidence import CONSISTENT, THEORY, THEORY_LOST, choose_default, monthly_spreads
from src.wrds_factors import WRDS_FACTOR_COLUMNS
from tests.unit.test_backtest_lambda import APP_DIR
from tests.unit.test_wrds_mode_switch import LEGACY_DEFAULT_SETTINGS


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


def test_every_legacy_factor_has_one_replacement_with_a_level_and_one_line_difference():
    rows = {r['legacy_factor']: r for r in legacy_rows()}
    assert set(rows) == set(FACTOR_METADATA)
    assert all(r['level'] in ('Same', 'Similar') and r['difference'] and '\n' not in r['difference']
               for r in rows.values())
    assert {n for n, r in rows.items() if r['level'] == 'Same'} == {
        'ROA %', '1-Yr Asset Growth %', '1-Yr CapEX Growth %'}
    # Reciprocal: the direction flips
    pb = rows['Price to Book Using 9/30 Data']
    assert (pb['wrds_factor'], pb['level'], pb['legacy_default'], pb['wrds_default']) == (
        'Book/Price', 'Similar', 'Lower', 'Higher')
    mo = rows['1-Mo Momentum %']
    assert (mo['wrds_factor'], mo['level']) == ('1-Month Return (Reversal)', 'Similar')
    assert 'Legacy defaults Higher, WRDS Lower' in mo['difference']
    assert (mo['legacy_default'], mo['wrds_default']) == ('Higher', 'Lower')
    # Data mixed: no flip from the theory defaults. Consistent ones agree with theory in-sample.
    basis = {m['column']: m['default_basis'] for m in WRDS_FACTOR_METADATA.values()}
    assert set(basis.values()) == {CONSISTENT, THEORY, THEORY_LOST}
    assert {c for c, b in basis.items() if b == CONSISTENT} == {
        'f_fcf_yield', 'f_sales_to_ev', 'f_forward_sales_to_ev', 'f_net_share_issuance', 'f_eps_dispersion'}
    assert {c for c, b in basis.items() if b == THEORY_LOST} == {
        'f_dividend_yield', 'f_sales_growth_1yr', 'beta_252d', 'beta_60m', 'f_ltg'}
    assert all(m['higher_is_better'] == m['theory_higher_is_better'] for m in WRDS_FACTOR_METADATA.values())
    assert [n for n, _ in replaces('f_roa')] == ['ROA using 9/30 Data', 'ROA %']
    assert not replaces('mom_12_1')               # skip-month momentum replaces no Legacy factor


def test_picker_tints_replacements_and_preserves_user_direction(monkeypatch):
    monkeypatch.syspath_prepend(str(APP_DIR))
    at = AppTest.from_string('''
import streamlit as st
from components.factor_selection import render_factor_selection
render_factor_selection('wrds')
''').run()
    assert not at.exception
    # Replacements are tinted through CSS, not type="primary" (which looks like Run Portfolio Analysis)
    assert all(b.proto.type == 'secondary' for b in at.button if b.key.startswith('factor_add_'))
    tint = ''.join(h.proto.body for h in at.get('html'))
    replacing = {m['key'] for m in WRDS_FACTOR_METADATA.values() if replaces(m['column'])}
    assert len(replacing) == 11
    assert all(f'.st-key-factor_add_{key} button' in tint for key in replacing)
    assert 'factor_add_w_mom_12_1' not in tint
    assert 'Replaces Legacy ROA % (Same)' in at.button(key='factor_add_w_roa').help
    # A direction flip leads the tooltip; no flip, no flip line
    assert at.button(key='factor_add_w_mom_1m').help.startswith(
        'Legacy 1-Mo Momentum % defaults Higher; this WRDS factor defaults Lower (reversal).')
    assert at.button(key='factor_add_w_btp').help.startswith(
        'Legacy Price to Book Using 9/30 Data defaults Lower; this WRDS factor defaults Higher (reciprocal).')
    assert at.button(key='factor_add_w_accruals').help.startswith('Net income minus operating cash flow')
    # The rule's label sits under the direction reason
    assert at.button(key='factor_add_w_fcf').help.endswith(
        f"Default rationale: {REASONS['f_fcf_yield']}\n\nConsistent: favored in most years since 2002")
    assert at.button(key='factor_add_w_roa').help.endswith('\n\nTheory default; data mixed')
    assert any(c.value.startswith(':material/link: Tinted cards replace a Legacy factor') for c in at.caption)
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
    assert (top['f_roa']['positive_years'], top['f_roa']['years'], top['f_roa']['share_years_positive']) == (1, 1, 1)
    assert bottom['f_roa']['share_years_positive'] == 0
    pd.testing.assert_frame_equal(panel, before_panel)
    pd.testing.assert_frame_equal(returns, before_returns)


def test_share_of_years_positive_sums_each_calendar_year():
    panel, returns = frames()
    # 2020: +9% in February. 2021: a January signal with reversed February returns, -9%.
    panel = pd.concat([panel, panel.assign(date=pd.Timestamp('2021-01-29'))])
    reversed_feb = returns.assign(month_end=pd.Timestamp('2021-02-26'), ret_m=returns['ret_m'][::-1].values)
    returns = pd.concat([returns, reversed_feb])
    summary, _ = monthly_spreads(panel, returns, {'f_roa': True})
    assert summary['f_roa']['annualized_spread'] == pytest.approx(0)
    assert (summary['f_roa']['positive_years'], summary['f_roa']['years']) == (1, 2)
    assert summary['f_roa']['share_years_positive'] == pytest.approx(.5)
    assert summary['f_roa']['negative_years'] == 1


def _summary(positive, negative, years):
    return {'annualized_spread': 0.01, 'positive_years': positive, 'negative_years': negative, 'years': years}


def _meta(theory, current=None):
    return {'theory_higher_is_better': theory, 'higher_is_better': theory if current is None else current}


@pytest.mark.parametrize('summary,higher,expected', [
    (_summary(7, 3, 10), True, (True, CONSISTENT)),     # exactly 70% of years: consistent
    (_summary(7, 3, 10), False, (False, CONSISTENT)),
    (_summary(3, 7, 10), True, (False, CONSISTENT)),    # the other direction wins 70%: flip
    (_summary(6, 4, 10), True, (True, THEORY)),         # below 70%: the theory default stands
    (_summary(5, 5, 10), False, (False, THEORY)),       # exactly half: still "data mixed"
    (_summary(4, 6, 10), True, (True, THEORY_LOST)),    # under half: it stands, labelled as such
    (_summary(4, 5, 10), True, (True, THEORY_LOST)),    # a zero-spread year counts for neither side
])
def test_default_rule(summary, higher, expected):
    assert choose_default(summary, _meta(higher)) == expected
    assert choose_default(dict(summary, annualized_spread=None), _meta(higher)) is None


def test_rule_falls_back_to_the_stored_theory_default_not_the_current_one():
    # A refresh flipped this factor to Lower; its theory default is still Higher. The summary is
    # oriented by the current default, so Lower won 6 of 10 years and Higher won 4.
    flipped = _meta(True, current=False)
    assert choose_default(_summary(6, 4, 10), flipped) == (True, THEORY_LOST)
    # Higher wins 6 of 10 years: mixed data, back to the theory default Higher
    assert choose_default(_summary(4, 6, 10), flipped) == (True, THEORY)
    # Lower still wins 7 of 10: the data keeps it Lower
    assert choose_default(_summary(7, 3, 10), flipped) == (False, CONSISTENT)


def test_every_wrds_factor_stores_its_theory_default():
    assert all(type(m['theory_higher_is_better']) is bool for m in WRDS_FACTOR_METADATA.values())


def test_saved_strategy_keeps_its_direction_when_the_default_differs():
    # Opposite of every current default: an import restores what was saved, not the default
    saved = {m['column']: 'bottom' if m['higher_is_better'] else 'top' for m in WRDS_FACTOR_METADATA.values()}
    settings = {k: v for k, v in LEGACY_DEFAULT_SETTINGS.items() if k != 'delisting_strategy'}
    settings.update(data_mode='wrds', frequency='monthly', start_year=2010, end_year=2025)
    values = cfg.widget_values(cfg.loads(cfg.dumps(cfg.build(settings, list(saved), saved, mode='wrds')).encode()))
    by_column = {m['column']: name for name, m in WRDS_FACTOR_METADATA.items()}
    assert values['factor_builder_selected'] == {by_column[c]: d for c, d in saved.items()}


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
    assert all(f'`{column}`' in render_report() for column in WRDS_FACTOR_COLUMNS)


def test_catalog_cli_writes_the_public_doc_and_guards_real_data_output(tmp_path):
    output = tmp_path / 'catalog.md'
    assert main(['--catalog-only', '--output', str(output)]) == 0
    text = output.read_text(encoding='utf-8')
    # DOCS/WRDS_FACTOR_REVIEW.md is this page: regenerate it when the mapping or defaults change
    assert text == (APP_DIR.parent / 'DOCS' / 'WRDS_FACTOR_REVIEW.md').read_text(encoding='utf-8')
    assert all(f'| {name} |' in text for name in FACTOR_METADATA)
    assert all(f'| {name} | {"Higher" if m["higher_is_better"] else "Lower"} |' in text
               for name, m in WRDS_FACTOR_METADATA.items())
    assert 'pending' not in text and 'pp/year' not in text            # no statistics in public docs
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
    payload = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
    assert set(payload['factors']) == set(WRDS_FACTOR_COLUMNS)
    assert payload['factors']['f_roa']['annualized_spread'] == pytest.approx(1.08)
    assert payload['factors']['f_asset_growth_1yr']['annualized_spread'] == pytest.approx(-1.08)
    assert payload['factors']['f_asset_growth_1yr']['evidence_status'] == 'team review'
    for name, digest in original_hashes.items():
        assert payload['inputs'][name]['sha256'] == digest
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert len(pd.read_csv(output.with_suffix('.csv'))) == len(WRDS_FACTOR_COLUMNS)
    assert 'pending data' not in output.read_text(encoding='utf-8')
    assert payload['requested_years'] == [2002, 2026]
