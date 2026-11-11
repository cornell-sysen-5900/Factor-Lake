"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_gics.py
PURPOSE: The GICS hierarchy of the WRDS panel and the GICS filter of a WRDS backtest.
VERSION: 1.0.0

The panel's GICS is each company's current (or last) classification applied to all of
its history (gics_source = current_header, pipeline/DATA_DICTIONARY.md). The codes nest:
the first 2, 4 and 6 digits of an 8-digit sub-industry (gsubind) are its sector,
industry group and industry. The tables below hold every code in the panel with its
name, as pipeline/config.py names them (tests/unit/test_wrds_gics.py checks both).

A GICS filter is {'level': one of GICS_LEVELS, 'codes': sorted codes of that level}:
- 'sector' keeps the stocks whose sector_name is one of the sectors. sector_name fills a
  missing GICS code from the SIC code, so this is exactly the older selected_sectors
  filter (only stocks without any sector, 'Unclassified', are left out).
- 'industry_group', 'industry' and 'sub_industry' keep the stocks whose gsubind starts
  with one of the codes. Stocks without GICS are left out.
"""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

GICS_SECTORS = {
    10: 'Energy', 15: 'Materials', 20: 'Industrials', 25: 'Consumer Discretionary', 30: 'Consumer Staples',
    35: 'Health Care', 40: 'Financials', 45: 'Information Technology', 50: 'Communication Services', 55: 'Utilities',
    60: 'Real Estate',
}
GICS_INDUSTRY_GROUPS = {
    1010: 'Energy', 1510: 'Materials', 2010: 'Capital Goods', 2020: 'Commercial & Professional Services',
    2030: 'Transportation', 2510: 'Automobiles & Components', 2520: 'Consumer Durables & Apparel',
    2530: 'Consumer Services', 2540: 'Media (discontinued 2018)',
    2550: 'Consumer Discretionary Distribution & Retail', 3010: 'Consumer Staples Distribution & Retail',
    3020: 'Food, Beverage & Tobacco', 3030: 'Household & Personal Products', 3510: 'Health Care Equipment & Services',
    3520: 'Pharmaceuticals, Biotechnology & Life Sciences', 4010: 'Banks', 4020: 'Financial Services',
    4030: 'Insurance', 4040: 'Real Estate (discontinued 2016)', 4510: 'Software & Services',
    4520: 'Technology Hardware & Equipment', 4530: 'Semiconductors & Semiconductor Equipment',
    5010: 'Telecommunication Services', 5020: 'Media & Entertainment', 5510: 'Utilities',
    6010: 'Equity Real Estate Investment Trusts (REITs)', 6020: 'Real Estate Management & Development',
}
GICS_INDUSTRIES = {
    101010: 'Energy Equipment & Services', 101020: 'Oil, Gas & Consumable Fuels', 151010: 'Chemicals',
    151020: 'Construction Materials', 151030: 'Containers & Packaging', 151040: 'Metals & Mining',
    151050: 'Paper & Forest Products', 201010: 'Aerospace & Defense', 201020: 'Building Products',
    201030: 'Construction & Engineering', 201040: 'Electrical Equipment', 201050: 'Industrial Conglomerates',
    201060: 'Machinery', 201070: 'Trading Companies & Distributors', 202010: 'Commercial Services & Supplies',
    202020: 'Professional Services', 203010: 'Air Freight & Logistics', 203020: 'Passenger Airlines',
    203030: 'Marine Transportation', 203040: 'Ground Transportation', 203050: 'Transportation Infrastructure',
    251010: 'Automobile Components', 251020: 'Automobiles', 252010: 'Household Durables', 252020: 'Leisure Products',
    252030: 'Textiles, Apparel & Luxury Goods', 253010: 'Hotels, Restaurants & Leisure',
    253020: 'Diversified Consumer Services', 254010: 'Media (discontinued 2018)', 255010: 'Distributors',
    255020: 'Internet & Direct Marketing Retail (discontinued 2023)', 255030: 'Broadline Retail',
    255040: 'Specialty Retail', 301010: 'Consumer Staples Distribution & Retail', 302010: 'Beverages',
    302020: 'Food Products', 302030: 'Tobacco', 303010: 'Household Products', 303020: 'Personal Care Products',
    351010: 'Health Care Equipment & Supplies', 351020: 'Health Care Providers & Services',
    351030: 'Health Care Technology', 352010: 'Biotechnology', 352020: 'Pharmaceuticals',
    352030: 'Life Sciences Tools & Services', 401010: 'Banks',
    401020: 'Thrifts & Mortgage Finance (discontinued 2023)', 402010: 'Financial Services',
    402020: 'Consumer Finance', 402030: 'Capital Markets', 402040: 'Mortgage Real Estate Investment Trusts (REITs)',
    403010: 'Insurance', 404010: 'Real Estate (discontinued 2006)',
    404020: 'Real Estate Investment Trusts (discontinued 2016)',
    404030: 'Real Estate Management & Development (discontinued 2016)',
    451010: 'Internet Software & Services (discontinued 2018)', 451020: 'IT Services', 451030: 'Software',
    452010: 'Communications Equipment', 452020: 'Technology Hardware, Storage & Peripherals',
    452030: 'Electronic Equipment, Instruments & Components',
    452050: 'Semiconductor Equipment & Products (discontinued 2002)',
    453010: 'Semiconductors & Semiconductor Equipment', 501010: 'Diversified Telecommunication Services',
    501020: 'Wireless Telecommunication Services', 502010: 'Media', 502020: 'Entertainment',
    502030: 'Interactive Media & Services', 551010: 'Electric Utilities', 551020: 'Gas Utilities',
    551030: 'Multi-Utilities', 551040: 'Water Utilities',
    551050: 'Independent Power and Renewable Electricity Producers', 601010: 'Diversified REITs',
    601020: 'Real Estate Management & Development (2016-2023 code, renumbered 602010)', 601025: 'Industrial REITs',
    601030: 'Hotel & Resort REITs', 601040: 'Office REITs', 601050: 'Health Care REITs', 601060: 'Residential REITs',
    601070: 'Retail REITs', 601080: 'Specialized REITs', 602010: 'Real Estate Management & Development',
}
GICS_SUB_INDUSTRIES = {
    10101010: 'Oil & Gas Drilling', 10101020: 'Oil & Gas Equipment & Services', 10102010: 'Integrated Oil & Gas',
    10102020: 'Oil & Gas Exploration & Production', 10102030: 'Oil & Gas Refining & Marketing',
    10102040: 'Oil & Gas Storage & Transportation', 10102050: 'Coal & Consumable Fuels',
    15101010: 'Commodity Chemicals', 15101020: 'Diversified Chemicals',
    15101030: 'Fertilizers & Agricultural Chemicals', 15101040: 'Industrial Gases', 15101050: 'Specialty Chemicals',
    15102010: 'Construction Materials', 15103010: 'Metal, Glass & Plastic Containers',
    15103020: 'Paper & Plastic Packaging Products & Materials', 15104010: 'Aluminum',
    15104020: 'Diversified Metals & Mining', 15104025: 'Copper', 15104030: 'Gold',
    15104040: 'Precious Metals & Minerals', 15104045: 'Silver', 15104050: 'Steel', 15105010: 'Forest Products',
    15105020: 'Paper Products', 20101010: 'Aerospace & Defense', 20102010: 'Building Products',
    20103010: 'Construction & Engineering', 20104010: 'Electrical Components & Equipment',
    20104020: 'Heavy Electrical Equipment', 20105010: 'Industrial Conglomerates',
    20106010: 'Construction Machinery & Heavy Transportation Equipment', 20106015: 'Agricultural & Farm Machinery',
    20106020: 'Industrial Machinery & Supplies & Components', 20107010: 'Trading Companies & Distributors',
    20201010: 'Commercial Printing', 20201030: 'Diversified Commercial & Professional Services (discontinued 2008)',
    20201040: 'Human Resource & Employment Services (discontinued 2008)',
    20201050: 'Environmental & Facilities Services', 20201060: 'Office Services & Supplies',
    20201070: 'Diversified Support Services', 20201080: 'Security & Alarm Services',
    20202010: 'Human Resource & Employment Services', 20202020: 'Research & Consulting Services',
    20202030: 'Data Processing & Outsourced Services', 20301010: 'Air Freight & Logistics',
    20302010: 'Passenger Airlines', 20303010: 'Marine Transportation', 20304010: 'Rail Transportation',
    20304020: 'Trucking (discontinued 2023)', 20304030: 'Cargo Ground Transportation',
    20304040: 'Passenger Ground Transportation', 20305010: 'Airport Services', 20305020: 'Highways & Railtracks',
    20305030: 'Marine Ports & Services', 25101010: 'Automotive Parts & Equipment', 25101020: 'Tires & Rubber',
    25102010: 'Automobile Manufacturers', 25102020: 'Motorcycle Manufacturers', 25201010: 'Consumer Electronics',
    25201020: 'Home Furnishings', 25201030: 'Homebuilding', 25201040: 'Household Appliances',
    25201050: 'Housewares & Specialties', 25202010: 'Leisure Products',
    25202020: 'Photographic Products (discontinued 2014)', 25203010: 'Apparel, Accessories & Luxury Goods',
    25203020: 'Footwear', 25203030: 'Textiles', 25301010: 'Casinos & Gaming',
    25301020: 'Hotels, Resorts & Cruise Lines', 25301030: 'Leisure Facilities', 25301040: 'Restaurants',
    25302010: 'Education Services', 25302020: 'Specialized Consumer Services',
    25401010: 'Advertising (discontinued 2018)', 25401020: 'Broadcasting (discontinued 2018)',
    25401025: 'Cable & Satellite (discontinued 2018)', 25401030: 'Movies & Entertainment (discontinued 2018)',
    25401040: 'Publishing (discontinued 2018)', 25501010: 'Distributors',
    25502010: 'Catalog Retail (discontinued 2018)',
    25502020: 'Internet & Direct Marketing Retail (discontinued 2023)',
    25503010: 'Department Stores (discontinued 2023)', 25503020: 'General Merchandise Stores (discontinued 2023)',
    25503030: 'Broadline Retail', 25504010: 'Apparel Retail', 25504020: 'Computer & Electronics Retail',
    25504030: 'Home Improvement Retail', 25504040: 'Other Specialty Retail', 25504050: 'Automotive Retail',
    25504060: 'Homefurnishing Retail', 30101010: 'Drug Retail', 30101020: 'Food Distributors',
    30101030: 'Food Retail', 30101040: 'Consumer Staples Merchandise Retail', 30201010: 'Brewers',
    30201020: 'Distillers & Vintners', 30201030: 'Soft Drinks & Non-alcoholic Beverages',
    30202010: 'Agricultural Products & Services', 30202030: 'Packaged Foods & Meats', 30203010: 'Tobacco',
    30301010: 'Household Products', 30302010: 'Personal Care Products', 35101010: 'Health Care Equipment',
    35101020: 'Health Care Supplies', 35102010: 'Health Care Distributors', 35102015: 'Health Care Services',
    35102020: 'Health Care Facilities', 35102030: 'Managed Health Care', 35103010: 'Health Care Technology',
    35201010: 'Biotechnology', 35202010: 'Pharmaceuticals', 35203010: 'Life Sciences Tools & Services',
    40101010: 'Diversified Banks', 40101015: 'Regional Banks',
    40102010: 'Thrifts & Mortgage Finance (discontinued 2023)', 40201020: 'Diversified Financial Services',
    40201030: 'Multi-Sector Holdings', 40201040: 'Specialized Finance',
    40201050: 'Commercial & Residential Mortgage Finance', 40201060: 'Transaction & Payment Processing Services',
    40202010: 'Consumer Finance', 40203010: 'Asset Management & Custody Banks',
    40203020: 'Investment Banking & Brokerage', 40203030: 'Diversified Capital Markets',
    40203040: 'Financial Exchanges & Data', 40204010: 'Mortgage REITs', 40301010: 'Insurance Brokers',
    40301020: 'Life & Health Insurance', 40301030: 'Multi-line Insurance', 40301040: 'Property & Casualty Insurance',
    40301050: 'Reinsurance', 40401010: 'Real Estate Investment Trusts (discontinued 2006)',
    40401020: 'Real Estate Management & Development (discontinued 2006)',
    40402010: 'Diversified REITs (discontinued 2016)', 40402030: 'Mortgage REITs (discontinued 2016)',
    40402035: 'Hotel & Resort REITs (discontinued 2016)', 40402040: 'Office REITs (discontinued 2016)',
    40402045: 'Health Care REITs (discontinued 2016)', 40402050: 'Residential REITs (discontinued 2016)',
    40402060: 'Retail REITs (discontinued 2016)', 40402070: 'Specialized REITs (discontinued 2016)',
    40403010: 'Diversified Real Estate Activities (discontinued 2016)',
    40403020: 'Real Estate Operating Companies (discontinued 2016)',
    40403030: 'Real Estate Development (discontinued 2016)', 40403040: 'Real Estate Services (discontinued 2016)',
    45101010: 'Internet Software & Services (discontinued 2018)', 45102010: 'IT Consulting & Other Services',
    45102020: 'Data Processing & Outsourced Services (discontinued 2023)',
    45102030: 'Internet Services & Infrastructure', 45103010: 'Application Software', 45103020: 'Systems Software',
    45103030: 'Home Entertainment Software (discontinued 2018)', 45201020: 'Communications Equipment',
    45202010: 'Computer Hardware (discontinued 2014)', 45202020: 'Computer Storage & Peripherals (discontinued 2014)',
    45202030: 'Technology Hardware, Storage & Peripherals', 45203010: 'Electronic Equipment & Instruments',
    45203015: 'Electronic Components', 45203020: 'Electronic Manufacturing Services',
    45203030: 'Technology Distributors', 45205010: 'Semiconductor Equipment (discontinued 2002)',
    45301010: 'Semiconductor Materials & Equipment', 45301020: 'Semiconductors', 50101010: 'Alternative Carriers',
    50101020: 'Integrated Telecommunication Services', 50102010: 'Wireless Telecommunication Services',
    50201010: 'Advertising', 50201020: 'Broadcasting', 50201030: 'Cable & Satellite', 50201040: 'Publishing',
    50202010: 'Movies & Entertainment', 50202020: 'Interactive Home Entertainment',
    50203010: 'Interactive Media & Services', 55101010: 'Electric Utilities', 55102010: 'Gas Utilities',
    55103010: 'Multi-Utilities', 55104010: 'Water Utilities',
    55105010: 'Independent Power Producers & Energy Traders', 55105020: 'Renewable Electricity',
    60101010: 'Diversified REITs', 60101020: 'Industrial REITs (2016-2023 code)',
    60101030: 'Hotel & Resort REITs (2016-2023 code)', 60101040: 'Office REITs (2016-2023 code)',
    60101050: 'Health Care REITs (2016-2023 code)', 60101060: 'Residential REITs (2016-2023 code)',
    60101070: 'Retail REITs (2016-2023 code)', 60101080: 'Specialized REITs (2016-2023 code)',
    60102020: 'Real Estate Operating Companies (2016-2023 code)', 60102040: 'Real Estate Services (2016-2023 code)',
    60102510: 'Industrial REITs', 60103010: 'Hotel & Resort REITs', 60104010: 'Office REITs',
    60105010: 'Health Care REITs', 60106010: 'Multi-Family Residential REITs',
    60106020: 'Single-Family Residential REITs', 60107010: 'Retail REITs', 60108010: 'Other Specialized REITs',
    60108020: 'Self-Storage REITs', 60108030: 'Telecom Tower REITs', 60108040: 'Timber REITs',
    60108050: 'Data Center REITs', 60201010: 'Diversified Real Estate Activities',
    60201020: 'Real Estate Operating Companies', 60201030: 'Real Estate Development',
    60201040: 'Real Estate Services',
}

# level -> (label, digits, code -> name), from the widest to the narrowest
GICS_LEVELS = {
    'sector': ('Sector', 2, GICS_SECTORS),
    'industry_group': ('Industry group', 4, GICS_INDUSTRY_GROUPS),
    'industry': ('Industry', 6, GICS_INDUSTRIES),
    'sub_industry': ('Sub-industry', 8, GICS_SUB_INDUSTRIES),
}
# The panel's name column of each level
GICS_NAME_COLUMNS = {'sector': 'gsector_name', 'industry_group': 'ggroup_name', 'industry': 'gind_name',
                     'sub_industry': 'gsubind_name'}


def level_label(level: str) -> str:
    return GICS_LEVELS[level][0]


def level_names(level: str) -> Dict[int, str]:
    """code -> name of one level."""
    return GICS_LEVELS[level][2]


def parent_code(code: int, level: str) -> int:
    """
    A code's code at the same or a wider level, e.g. 45103010 -> 45 or 451030 -> 4510 for
    'industry_group' (GICS codes never start with 0, so their length is their level).
    """
    return int(code) // 10 ** (len(str(int(code))) - GICS_LEVELS[level][1])


def codes_under(level: str, parents: Iterable[int], parent_level: str) -> List[int]:
    """The level's codes inside the parent level's codes, sorted."""
    wanted = {int(p) for p in parents}
    shift = 10 ** (GICS_LEVELS[level][1] - GICS_LEVELS[parent_level][1])
    return sorted(c for c in level_names(level) if c // shift in wanted)


def names_to_codes(level: str, names: Iterable[str]) -> List[int]:
    """The codes of a level's names, sorted. Raises ValueError for an unknown name."""
    by_name = {name: code for code, name in level_names(level).items()}
    names = list(names)
    if bad := [n for n in names if n not in by_name]:
        raise ValueError(f"Unknown GICS {level_label(level).lower()} names: {bad}.")
    return sorted(by_name[n] for n in names)


def sector_codes(sector_names: Iterable[str]) -> List[int]:
    """The sector codes of sector names (an older job's selected_sectors), sorted."""
    by_name = {name: code for code, name in GICS_SECTORS.items()}
    sector_names = list(sector_names)
    if bad := [s for s in sector_names if s not in by_name]:
        raise ValueError(f"Unknown GICS sectors: {bad}.")
    return sorted(by_name[s] for s in sector_names)


def validate_gics_filter(gics_filter: Any) -> None:
    """Raises ValueError with a readable message unless gics_filter is a valid GICS filter."""
    if not isinstance(gics_filter, dict) or set(gics_filter) != {'level', 'codes'}:
        raise ValueError("gics_filter must be an object with level and codes.")
    level, codes = gics_filter['level'], gics_filter['codes']
    if level not in GICS_LEVELS:
        raise ValueError(f"gics_filter level must be one of {tuple(GICS_LEVELS)}, not '{level}'.")
    if not isinstance(codes, list) or not codes:
        raise ValueError("gics_filter codes must be a list of at least one GICS code.")
    if any(not isinstance(c, int) or isinstance(c, bool) for c in codes) or codes != sorted(set(codes)):
        raise ValueError("gics_filter codes must be whole numbers, sorted, without repeats.")
    if bad := [c for c in codes if c not in level_names(level)]:
        raise ValueError(f"Unknown GICS {level_label(level).lower()} codes: {bad}.")


def make_gics_filter(level: str, codes: Iterable[int]) -> Optional[Dict[str, Any]]:
    """A validated GICS filter (codes sorted, without repeats), or None when there are no codes."""
    codes = sorted({int(c) for c in codes})
    if not codes:
        return None
    gics_filter = {'level': level, 'codes': codes}
    validate_gics_filter(gics_filter)
    return gics_filter


def job_gics_filter(gics_filter: Optional[Dict[str, Any]],
                    selected_sectors: Optional[Iterable[str]]) -> Optional[Dict[str, Any]]:
    """
    The GICS filter a job applies: its gics_filter, or for an older job its
    selected_sectors as a sector filter. None when it has neither.
    """
    sectors = list(selected_sectors or [])
    if gics_filter and sectors:
        raise ValueError("A job has either gics_filter or selected_sectors, not both.")
    if gics_filter:
        validate_gics_filter(gics_filter)
        return gics_filter
    return make_gics_filter('sector', sector_codes(sectors)) if sectors else None


def gics_mask(panel: pd.DataFrame, gics_filter: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    The panel rows the GICS filter keeps, as a boolean array (all of them without a
    filter). The sector level reads sector_name, the others gsubind (see the module docstring).
    """
    if not gics_filter:
        return np.ones(len(panel), dtype=bool)
    level, codes = gics_filter['level'], gics_filter['codes']
    if level == 'sector':
        names = [GICS_SECTORS[c] for c in codes]
        return panel['sector_name'].isin(names).to_numpy(dtype=bool)
    prefix = panel['gsubind'] // 10 ** (8 - GICS_LEVELS[level][1])
    return prefix.isin(codes).to_numpy(dtype=bool, na_value=False)


def describe_gics_filter(gics_filter: Optional[Dict[str, Any]], limit: Optional[int] = None) -> str:
    """
    The filter for captions: the names in alphabetical order, as the sector filter always
    showed them ('Energy, Financials'), and below the sector level after the level
    ('Industry: Banks, Insurance'). With limit, at most limit names ('... and 3 more').
    'All sectors' without a filter.
    """
    if not gics_filter:
        return 'All sectors'
    level = gics_filter['level']
    names = sorted(level_names(level).get(c, str(c)) for c in gics_filter['codes'])
    shown = ', '.join(names[:limit])
    if limit is not None and len(names) > limit:
        shown += f" and {len(names) - limit} more"
    return shown if level == 'sector' else f"{level_label(level)}: {shown}"
