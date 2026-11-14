"""Legacy-to-WRDS replacements and the theory rationale for each WRDS default direction.

Every Legacy factor maps to its closest WRDS factor: "Same" when the documented definition
agrees (units may differ, which does not change ranks), "Similar" when the idea is the same
but the lookback, return convention or orientation differs. Mappings are definition-based,
not fitted to data; WRDS_FACTOR_METADATA remains the source of default directions.
"""
from dataclasses import dataclass

from app.streamlit_config import FACTOR_METADATA, WRDS_FACTOR_METADATA

SAME, SIMILAR = 'Same', 'Similar'


@dataclass(frozen=True)
class Replacement:
    legacy_column: str
    wrds_column: str
    level: str
    difference: str


REPLACEMENTS = (
    Replacement('12-Mo_Momentum', 'mom_12m', SIMILAR,
                'Legacy uses price returns; WRDS compounds 252 trading days of CRSP total returns.'),
    Replacement('6-Mo_Momentum', 'mom_6m', SIMILAR,
                'Legacy uses price returns; WRDS compounds 126 trading days of CRSP total returns.'),
    Replacement('1-Mo_Momentum', 'mom_1m', SIMILAR,
                'WRDS uses 21 trading days of CRSP total returns; Legacy defaults Higher, WRDS Lower (reversal).'),
    Replacement('Price_to_Book_Using_9-30_Data', 'f_book_to_price', SIMILAR,
                'Reciprocal, so the direction flips: Lower P/B in Legacy is Higher Book/Price in WRDS.'),
    Replacement('Book-Price', 'f_book_to_price', SIMILAR,
                "WRDS prices book equity at the panel date's market cap; Legacy's price date is unspecified."),
    Replacement('Next_FY_Earns-P', 'f_forward_ep_fy2', SIMILAR,
                'WRDS uses the FY2 (next fiscal year) consensus without the Legacy 3+ analyst minimum.'),
    Replacement('ROE_using_9-30_Data', 'f_roe', SIMILAR,
                'WRDS uses TTM income over positive Fama-French book equity; Legacy does not specify either.'),
    Replacement('ROA_using_9-30_Data', 'f_roa', SIMILAR,
                'WRDS uses TTM income; the Legacy 9/30 snapshot does not specify the income lookback.'),
    Replacement('ROA', 'f_roa', SAME,
                'Same TTM net income over total assets; Legacy is in percent, WRDS in decimals.'),
    Replacement('Accruals-Assets', 'f_accruals_to_assets', SIMILAR,
                'WRDS uses TTM income and operating cash flow; the Legacy lookback is unspecified.'),
    Replacement('1-Yr_Price_Vol', 'vol_close_12m', SIMILAR,
                'WRDS annualizes 252 daily log CRSP returns with 80% coverage; Legacy specifies neither rule.'),
    Replacement('1-Yr_Asset_Growth', 'f_asset_growth_1yr', SAME,
                'Same four-quarter change in total assets; Legacy is in percent, WRDS in decimals.'),
    Replacement('1-Yr_CapEX_Growth', 'f_capex_growth_1yr', SAME,
                'Same 12-month capex versus the prior year; WRDS skips prior-year capex below $1M.'),
)

# Each reason is a theory-based rationale for the existing default, not a promise of
# outperformance. Mixed or context-dependent cases are explicitly described as conventions.
REASONS = {
    'mom_12_1': 'Intermediate-horizon return continuation; skipping the latest month reduces reversal exposure.',
    'mom_12m': 'Intermediate-horizon return continuation; including the latest month can add reversal exposure.',
    'mom_6m': 'Intermediate-horizon return continuation motivates favoring higher six-month returns.',
    'mom_1m': 'Short-term reversal motivates favoring last month\'s losers; this is not a momentum default.',
    'f_book_to_price': 'More book equity per dollar paid provides a value tilt; use Higher for B/P, not Lower for P/B.',
    'f_earnings_to_price': 'More earnings per dollar paid provides a value tilt; E/P is the inverse '
        'orientation of P/E.',
    'f_sales_to_price': 'More revenue per dollar paid is a value tilt, although margins and business models differ.',
    'f_fcf_yield': 'More free cash flow per dollar paid supports cash-flow valuation.',
    'f_dividend_yield': 'More dividends per dollar paid supports an income/value tilt; distressed yields need caution.',
    'f_ebitda_to_ev': 'More EBITDA per dollar of enterprise value favors cheaper operating cash-earnings capacity.',
    'f_ebit_to_ev': 'More operating earnings per dollar of enterprise value supports a value tilt.',
    'f_sales_to_ev': 'More sales per dollar of enterprise value supports a value tilt; profitability still matters.',
    'f_forward_ep': 'More expected FY1 earnings per dollar paid supports forward value, subject to forecast error.',
    'f_forward_ep_fy2': 'More expected FY2 earnings per dollar paid supports forward value, subject to forecast error.',
    'f_forward_cfp': 'More expected cash flow per dollar paid supports forward cash-flow value.',
    'f_forward_sales_to_ev': 'More forecast sales per dollar of enterprise value supports forward value.',
    'f_forward_ebitda_to_ev': 'More forecast EBITDA per dollar of enterprise value supports forward value.',
    'f_roe': 'Higher income relative to equity indicates profitability; leverage can also raise ROE.',
    'f_roa': 'Higher income relative to assets indicates more productive use of the asset base.',
    'f_gross_profitability': 'Higher gross profit relative to assets is a profitability signal.',
    'f_accruals_to_assets': 'Lower accruals favor earnings backed by operating cash flow rather than '
        'accounting accruals.',
    'f_leverage': 'Lower debt relative to assets favors balance-sheet safety; the return relation is '
        'context-dependent.',
    'f_current_ratio': 'Higher short-term liquidity supports a safety tilt; very high ratios can imply idle assets.',
    'f_net_share_issuance': 'Lower issuance favors payout and limits dilution; issuance can also fund '
        'productive projects.',
    'f_asset_growth_1yr': 'The investment effect favors conservative asset growth over aggressive expansion.',
    'f_sales_growth_1yr': 'Revenue growth supports operating strength; this project convention is valuation-dependent.',
    'f_capex_growth_1yr': 'The investment effect motivates lower investment growth; productive capex '
        'can still add value.',
    'f_rd_to_sales': 'R&D intensity can proxy intangible investment; this project convention is sector-dependent.',
    'vol_close_12m': 'A defensive tilt favors lower realized volatility; it does not guarantee higher raw returns.',
    'vol_close_6m': 'A defensive tilt favors lower medium-horizon realized volatility.',
    'vol_close_3m': 'A defensive tilt favors lower recent realized volatility; shorter estimates are noisier.',
    'vol_gk_1m': 'A defensive tilt favors lower intramonth range volatility; this short-horizon proxy is noisy.',
    'beta_252d': 'Leverage constraints motivate a low-beta tilt; this unlevered sort does not replicate '
        'the BAB factor.',
    'beta_60m': 'Leverage constraints motivate a low-beta tilt; a five-year beta reacts slowly to changing exposures.',
    'f_sue': 'Underreaction to positive earnings news motivates favoring larger standardized earnings increases.',
    'f_analyst_surprise': 'Underreaction to earnings announcements motivates favoring positive consensus surprises.',
    'f_eps_revision': 'Upward analyst revisions indicate improving earnings expectations; favor net upward revisions.',
    'f_eps_dispersion': 'Lower disagreement limits optimistic pricing under short-sale constraints; evidence is mixed.',
    'f_ltg': 'Lower long-term forecasts avoid excessive extrapolation of optimistic growth '
        'expectations; evidence is mixed.',
    'mkt_cap_musd': 'A size tilt favors smaller firms; the premium is unstable and trading costs can be higher.',
}


_LEGACY_NAMES = {meta['column']: name for name, meta in FACTOR_METADATA.items()}
_WRDS_NAMES = {meta['column']: name for name, meta in WRDS_FACTOR_METADATA.items()}


def _direction(meta) -> str:
    return 'Higher' if meta['higher_is_better'] else 'Lower'


def replaces(column: str):
    """The Legacy factors a WRDS column replaces, as (Legacy name, Replacement) pairs."""
    return tuple((_LEGACY_NAMES[r.legacy_column], r) for r in REPLACEMENTS if r.wrds_column == column)


def legacy_names(column: str):
    return tuple(name for name, _ in replaces(column))


def factor_tooltip(name: str) -> str:
    meta = WRDS_FACTOR_METADATA[name]
    lines = [f"Replaces Legacy {legacy} ({r.level}): {r.difference}" for legacy, r in replaces(meta['column'])]
    return '\n\n'.join([meta['tooltip'], *lines, f"Default rationale: {REASONS[meta['column']]}"])


def factor_rows():
    """All WRDS factors, including those that replace no Legacy factor, in picker order."""
    return [{'factor': name, 'column': m['column'], 'replaces': '; '.join(legacy_names(m['column'])) or '—',
             'default_direction': _direction(m), 'reason': REASONS[m['column']]}
            for name, m in WRDS_FACTOR_METADATA.items()]


def legacy_rows():
    """Every Legacy factor exactly once, with its WRDS replacement."""
    by_legacy = {r.legacy_column: r for r in REPLACEMENTS}
    rows = []
    for name, meta in FACTOR_METADATA.items():
        r = by_legacy[meta['column']]
        wrds = _WRDS_NAMES[r.wrds_column]
        rows.append({'legacy_factor': name, 'legacy_column': r.legacy_column, 'wrds_factor': wrds,
                     'wrds_column': r.wrds_column, 'level': r.level, 'difference': r.difference,
                     'legacy_default': _direction(meta),
                     'wrds_default': _direction(WRDS_FACTOR_METADATA[wrds])})
    return rows
