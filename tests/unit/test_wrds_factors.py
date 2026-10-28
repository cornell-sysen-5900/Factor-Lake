"""
Tests for the WRDS factor list: the app's WRDS_FACTOR_METADATA (names, groups,
default directions, tooltips) and the Lambda's allowlist (src/wrds_factors.py).
"""
from app.streamlit_config import (FACTOR_METADATA, WRDS_FACTOR_GROUPS, WRDS_FACTOR_METADATA,
                                  WRDS_FACTOR_OPTIONS)
from src.wrds_factors import UNCLASSIFIED_SECTOR, WRDS_FACTOR_COLUMNS, WRDS_SECTORS

# Columns about the month after t: never a factor (look-ahead)
FORWARD_COLUMNS = {'ret_next_1m', 'delisted_next_month'}


def test_app_list_and_lambda_allowlist_name_the_same_columns():
    columns = [m['column'] for m in WRDS_FACTOR_METADATA.values()]
    assert columns == list(WRDS_FACTOR_COLUMNS)
    assert len(set(columns)) == len(columns)
    assert not FORWARD_COLUMNS & set(columns)


def test_every_factor_has_a_known_group_a_tooltip_and_a_unique_wrds_key():
    keys = [m['key'] for m in WRDS_FACTOR_METADATA.values()]
    assert len(set(keys)) == len(keys) and all(k.startswith('w_') for k in keys)
    assert not set(keys) & {m['key'] for m in FACTOR_METADATA.values()}
    assert all(m['group'] in WRDS_FACTOR_GROUPS and m['tooltip'] for m in WRDS_FACTOR_METADATA.values())
    # Listed in group order, every group used
    order = [WRDS_FACTOR_GROUPS.index(m['group']) for m in WRDS_FACTOR_METADATA.values()]
    assert order == sorted(order) and set(order) == set(range(len(WRDS_FACTOR_GROUPS)))
    assert WRDS_FACTOR_OPTIONS == list(WRDS_FACTOR_METADATA)


def test_every_lower_default_says_why_and_higher_ones_do_not_claim_lower():
    for name, meta in WRDS_FACTOR_METADATA.items():
        says_lower = 'Lower is better: ' in meta['tooltip']
        assert says_lower == (not meta['higher_is_better']), name


def test_decided_defaults_and_names():
    lower = {n for n, m in WRDS_FACTOR_METADATA.items() if not m['higher_is_better']}
    assert {'1-Month Return (Reversal)', 'Long-Term Growth Forecast', 'Market Cap',
            'Beta (1-Yr Daily)', 'Beta (5-Yr Monthly)'} <= lower
    assert WRDS_FACTOR_METADATA['1-Month Return (Reversal)']['column'] == 'mom_1m'


def test_eleven_gics_sectors_without_unclassified():
    assert len(WRDS_SECTORS) == 11 and UNCLASSIFIED_SECTOR not in WRDS_SECTORS
