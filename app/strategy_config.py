"""Versioned, inputs-only strategy files. Validation never mutates application state."""
import json
import math
from copy import deepcopy
from app import streamlit_config as config
from src.trading_costs import MAX_TRADING_COST_BPS
from src.wrds_calendar import FIRST_YEAR, LAST_YEAR
from src.wrds_factors import WRDS_SECTORS

MAX_FILE_BYTES = 64 * 1024
COMMON = {'start_year', 'end_year', 'initial_aum', 'restrict_fossil_fuels',
          'use_market_cap_weight', 'factor_combination', 'selected_sectors',
          'sector_filter_enabled', 'trading_cost_bps'}


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
    require(isinstance(s, dict) and set(s) == fields, 'Settings contain missing or unsupported fields.')
    for key in ('restrict_fossil_fuels', 'use_market_cap_weight', 'sector_filter_enabled'):
        require(type(s[key]) is bool, f'{key} must be true or false.')
    lo, hi = (FIRST_YEAR, LAST_YEAR) if mode == 'wrds' else (2002, 2024)
    for key in ('start_year', 'end_year'):
        limit = 2023 if mode == 'legacy' and key == 'start_year' else hi
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
        raise ValueError('This run uses a custom portfolio percentage or benchmark not supported by the current controls.')
    fields = COMMON | ({'frequency'} if mode == 'wrds' else {'delisting_strategy'})
    s = {k: deepcopy(settings[k]) for k in fields if k in settings}
    s.setdefault('trading_cost_bps', 0)
    s.setdefault('sector_filter_enabled', bool(s.get('selected_sectors')))
    if not s['sector_filter_enabled']:
        s['selected_sectors'] = None
    return validate({'schema_version': 1, 'data_mode': mode, 'name': name, 'note': note,
                     'settings': s, 'factors': [{'column': f, 'direction': directions.get(f, 'top')} for f in factors]})


def dumps(document):
    return json.dumps(validate(document), indent=2, ensure_ascii=False, allow_nan=False)


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
              'trading_cost_bps_input': int(s['trading_cost_bps']),
              'strategy_name': d['name'], 'strategy_note': d['note'],
              'strategy_name_input': d['name'], 'strategy_note_input': d['note']}
    prefix = 'wrds_' if mode == 'wrds' else ''
    values[prefix + 'start_year_input'] = s['start_year']
    values[prefix + 'end_year_input'] = s['end_year']
    if s['sector_filter_enabled']:
        values['wrds_sectors_input' if mode == 'wrds' else 'selected_sectors_input'] = s['selected_sectors']
    if mode == 'wrds':
        values['wrds_frequency_input'] = next(k for k, v in config.WRDS_FREQUENCY_OPTIONS.items() if v == s['frequency'])
    else:
        values['delisting_strategy_input'] = {'zero_return': 'Zero Return', 'hold_cash': 'Hold Cash', 'reinvest': 'Reinvest'}[s['delisting_strategy']]
    by_column = {m['column']: (label, m) for label, m in catalog(mode).items()}
    selected = {}
    for f in d['factors']:
        label, meta = by_column[f['column']]
        selected[label] = f['direction']
        values[f"factor_dir_{meta['key']}"] = 'Higher is better' if f['direction'] == 'top' else 'Lower is better'
    values['factor_builder_selected'] = selected
    return values
