"""
Versioned, inputs-only strategy files. Validation never mutates application state.

WRDS settings may also have gics_filter ({'level', 'codes'}, src/wrds_gics.py), the GICS
filter of the sidebar's Filter level; with it, selected_sectors is []. Files without it
(older ones, with selected_sectors) open at the Sector level.
"""
import json
import math
from copy import deepcopy

from app import streamlit_config as config
from src.trading_costs import MAX_TRADING_COST_BPS
from src.wrds_calendar import FIRST_YEAR, LAST_YEAR
from src.wrds_factors import WRDS_SECTORS
from src.wrds_gics import GICS_LEVELS, level_names, parent_code, validate_gics_filter

MAX_FILE_BYTES = 64 * 1024
COMMON = {'start_year', 'end_year', 'initial_aum', 'restrict_fossil_fuels',
          'use_market_cap_weight', 'factor_combination', 'selected_sectors',
          'sector_filter_enabled', 'trading_cost_bps'}
# WRDS settings that are there only when set
WRDS_OPTIONAL = {'gics_filter'}
# Sidebar widgets of the WRDS GICS filter (app/components/sidebar.py): the level picker and
# each level's list
GICS_LEVEL_INPUT = 'wrds_gics_level_input'
GICS_LEVEL_INPUTS = {'sector': 'wrds_sectors_input', 'industry_group': 'wrds_industry_groups_input',
                     'industry': 'wrds_industries_input', 'sub_industry': 'wrds_sub_industries_input'}


def catalog(mode):
    return config.WRDS_FACTOR_METADATA if mode == 'wrds' else config.FACTOR_METADATA


def validate(document):
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require(isinstance(document, dict), 'The JSON must contain a strategy object.')
    require(set(document) == {'schema_version', 'data_mode', 'name', 'note', 'settings', 'factors'},
            'Expected schema_version, data_mode, name, note, settings and factors only.')
    require(type(document['schema_version']) is int and document['schema_version'] == 1,
            'Unsupported schema version. Export a new file from this app.')
    mode = document['data_mode']
    require(mode in ('legacy', 'wrds'), 'Data mode must be legacy or wrds.')
    for field, limit in [('name', 100), ('note', 1000)]:
        require(isinstance(document[field], str) and len(document[field]) <= limit,
                f'{field.capitalize()} must be text of at most {limit} characters.')
        try:
            document[field].encode('utf-8')
        except UnicodeError as exc:
            raise ValueError(f'{field.capitalize()} contains invalid Unicode text.') from exc
    s = document['settings']
    fields = COMMON | ({'frequency'} if mode == 'wrds' else {'delisting_strategy'})
    optional = WRDS_OPTIONAL if mode == 'wrds' else set()
    require(isinstance(s, dict) and fields <= set(s) <= fields | optional,
            'Settings contain missing or unsupported fields.')
    for key in ('restrict_fossil_fuels', 'use_market_cap_weight', 'sector_filter_enabled'):
        require(type(s[key]) is bool, f'{key} must be true or false.')
    lo, hi = (FIRST_YEAR, LAST_YEAR) if mode == 'wrds' else (config.LEGACY_FIRST_YEAR, config.LEGACY_LAST_YEAR)
    for key in ('start_year', 'end_year'):
        limit = config.LEGACY_LAST_START_YEAR if mode == 'legacy' and key == 'start_year' else hi
        require(type(s[key]) is int and lo <= s[key] <= limit, f'{key} must be between {lo} and {limit}.')
    require(s['start_year'] <= s['end_year'], 'Start year must not exceed end year.')
    aum = s['initial_aum']
    require(type(aum) in (int, float) and 0 < aum <= 1e9 and math.isfinite(aum),
            'Initial investment must be greater than 0 and at most 1 billion.')
    cost = s['trading_cost_bps']
    require(type(cost) in (int, float) and 0 <= cost <= MAX_TRADING_COST_BPS and math.isfinite(cost)
            and cost == int(cost), 'Trading cost must be a whole number within the supported range.')
    require(s['factor_combination'] in ('mixing', 'integrating'), 'Unsupported factor combination.')
    if mode == 'wrds':
        require(s['frequency'] in config.WRDS_FREQUENCY_OPTIONS.values(), 'Unsupported rebalancing frequency.')
    else:
        require(s['delisting_strategy'] in ('zero_return', 'hold_cash', 'reinvest'), 'Unsupported delisting strategy.')
    sectors = s['selected_sectors']
    allowed_sectors = WRDS_SECTORS if mode == 'wrds' else config.SECTOR_OPTIONS
    if s['sector_filter_enabled']:
        require(isinstance(sectors, list) and all(isinstance(v, str) and v in allowed_sectors for v in sectors),
                'Selected sectors must be a list of supported sectors for this data mode.')
        require(len(set(sectors)) == len(sectors), 'Selected sectors contain duplicates.')
    else:
        require(sectors is None, 'Selected sectors must be null when sector filtering is off.')
    if 'gics_filter' in s:
        try:
            validate_gics_filter(s['gics_filter'])
        except ValueError as exc:
            raise ValueError(f'Invalid GICS filter: {exc}') from exc
        require(s['sector_filter_enabled'] and sectors == [],
                'A GICS filter needs sector filtering on and an empty selected_sectors.')
    factors = document['factors']
    columns = {m['column'] for m in catalog(mode).values()}
    require(isinstance(factors, list) and 0 < len(factors) <= len(columns), 'Select at least one supported factor.')
    seen = set()
    for f in factors:
        require(isinstance(f, dict) and set(f) == {'column', 'direction'}, 'Each factor needs column and direction.')
        require(isinstance(f['column'], str) and f['column'] in columns, 'Unknown factor for this data mode.')
        require(f['column'] not in seen, 'Duplicate factors are not allowed.')
        require(f['direction'] in ('top', 'bottom'), 'Factor direction must be top or bottom.')
        seen.add(f['column'])
    return deepcopy(document)


def loads(raw):
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('Strategy files must be at most 64 KB.')
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON field: {key}.')
            result[key] = value
        return result
    try:
        return validate(json.loads(raw, object_pairs_hook=unique_object))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Invalid JSON. Upload a UTF-8 strategy file exported by this app.') from exc


def build(settings, factors, directions, name='', note='', mode=None):
    mode = mode or settings.get('data_mode', 'legacy')
    # Refuse to silently discard engine settings that the current controls cannot restore.
    if settings.get('top_pct', 10.0) != 10.0 or settings.get('benchmark_index', 1) != 1:
        raise ValueError('This run uses a custom portfolio percentage or benchmark '
                         'not supported by the current controls.')
    fields = COMMON | ({'frequency'} if mode == 'wrds' else {'delisting_strategy'})
    s = {k: deepcopy(settings[k]) for k in fields if k in settings}
    if mode == 'wrds' and settings.get('gics_filter'):
        s['gics_filter'] = deepcopy(settings['gics_filter'])
    s.setdefault('trading_cost_bps', 0)
    s.setdefault('sector_filter_enabled', bool(s.get('selected_sectors')))
    if not s['sector_filter_enabled']:
        s['selected_sectors'] = None
    return validate({'schema_version': 1, 'data_mode': mode, 'name': name, 'note': note,
                     'settings': s, 'factors': [{'column': f, 'direction': directions.get(f, 'top')} for f in factors]})


def dumps(document):
    return json.dumps(validate(document), indent=2, ensure_ascii=False, allow_nan=False)


def gics_widget_values(settings):
    """
    The WRDS GICS filter's widgets for these settings: the level, its choices, and above
    them the sectors, groups and industries they are in. Older settings (selected_sectors) are
    the Sector level.
    """
    gics = settings.get('gics_filter')
    if not gics:
        return {GICS_LEVEL_INPUT: GICS_LEVELS['sector'][0], GICS_LEVEL_INPUTS['sector']: settings['selected_sectors']}
    values = {GICS_LEVEL_INPUT: GICS_LEVELS[gics['level']][0]}
    levels = list(GICS_LEVELS)
    for level in levels[:levels.index(gics['level']) + 1]:
        codes = sorted({parent_code(c, level) for c in gics['codes']})
        values[GICS_LEVEL_INPUTS[level]] = sorted(level_names(level)[c] for c in codes)
    return values


def widget_values(document):
    """Create a complete update before touching widgets (including hidden mode controls)."""
    d = validate(document)
    s, mode = d['settings'], d['data_mode']
    values = {'data_mode_input': 'WRDS' if mode == 'wrds' else 'Legacy',
              'restrict_fossil_fuels_input': s['restrict_fossil_fuels'],
              'weighting_method_input': 'Market Cap Weight' if s['use_market_cap_weight'] else 'Equal Weight',
              'factor_combination_input': s['factor_combination'].title(),
              'sector_filter_enabled_input': s['sector_filter_enabled'],
              'initial_aum_input': float(s['initial_aum']),
              'trading_cost_bps_input': int(s['trading_cost_bps'])}
    prefix = 'wrds_' if mode == 'wrds' else ''
    values[prefix + 'start_year_input'] = s['start_year']
    values[prefix + 'end_year_input'] = s['end_year']
    if s['sector_filter_enabled'] and mode == 'wrds':
        values.update(gics_widget_values(s))
    elif s['sector_filter_enabled']:
        values['selected_sectors_input'] = s['selected_sectors']
    if mode == 'wrds':
        values['wrds_frequency_input'] = next(k for k, v in config.WRDS_FREQUENCY_OPTIONS.items()
                                              if v == s['frequency'])
    else:
        values['delisting_strategy_input'] = {
            'zero_return': 'Zero Return', 'hold_cash': 'Hold Cash', 'reinvest': 'Reinvest',
        }[s['delisting_strategy']]
    by_column = {m['column']: (label, m) for label, m in catalog(mode).items()}
    selected = {}
    for f in d['factors']:
        label, meta = by_column[f['column']]
        selected[label] = f['direction']
        values[f"factor_dir_{meta['key']}"] = 'Higher is better' if f['direction'] == 'top' else 'Lower is better'
    values['factor_builder_selected'] = selected
    return values
