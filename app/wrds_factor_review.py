"""Definition-based Legacy crosswalk and theory rationale, without empirical sign fitting.

A match means the documented formula and lookback agree, not identical vendor values.
Unknown Legacy conventions remain related candidates; they do not get match styling.
The app's existing WRDS_FACTOR_METADATA remains the source of default directions.
"""
from dataclasses import dataclass

from app.streamlit_config import FACTOR_METADATA, WRDS_FACTOR_METADATA


@dataclass(frozen=True)
class LegacyMatch:
    column: str
    wrds_column: str
    status: str
    note: str


LEGACY_MATCHES = (
    LegacyMatch('12-Mo_Momentum', 'mom_12m', 'related',
                'Legacy excludes dividends; WRDS compounds CRSP returns over 252 trading days. '
                'Return convention and calendar lookback differ.'),
    LegacyMatch('6-Mo_Momentum', 'mom_6m', 'related',
                'Legacy excludes dividends; WRDS compounds CRSP returns over 126 trading days.'),
    LegacyMatch('1-Mo_Momentum', 'mom_1m', 'related',
                'Legacy excludes dividends; WRDS uses 21 trading-day CRSP returns. '
                'WRDS also defaults to reversal (Lower), while Legacy defaults to Higher.'),
    LegacyMatch('Price_to_Book_Using_9-30_Data', 'f_book_to_price', 'related',
                'Price/book is the reciprocal of book/price, not the same factor definition. '
                'No reciprocal or automatic direction conversion is performed.'),
    LegacyMatch('Book-Price', 'f_book_to_price', 'related',
                'Legacy says latest-quarter book/price; its price date is not specified. '
                'WRDS uses company market cap at the panel date. Verify Legacy price timing first.'),
    LegacyMatch('Next_FY_Earns-P', 'f_forward_ep_fy2', 'related',
                'FY2 is the next fiscal year candidate, not FY1. Legacy requires 3+ analysts; '
                'that coverage rule is not part of this WRDS factor. No exact match confirmed.'),
    LegacyMatch('ROE_using_9-30_Data', 'f_roe', 'related',
                'Legacy does not specify the income lookback or equity adjustments. '
                'WRDS uses TTM income and positive Fama-French-style book equity.'),
    LegacyMatch('ROA_using_9-30_Data', 'f_roa', 'related',
                'The Legacy 9/30 snapshot does not specify the income lookback; WRDS explicitly uses TTM income.'),
    LegacyMatch('ROA', 'f_roa', 'match',
                'Both documented definitions are trailing-12-month net income divided by total assets. '
                'Legacy percent and WRDS decimal units differ; vendor values and observation dates may differ.'),
    LegacyMatch('Accruals-Assets', 'f_accruals_to_assets', 'related',
                'Both use income minus operating cash flow over assets, but the Legacy income/cash-flow '
                'lookback is unspecified. WRDS uses TTM flows.'),
    LegacyMatch('1-Yr_Price_Vol', 'vol_close_12m', 'related',
                'Both describe annualized daily volatility; WRDS uses log CRSP returns, 252 observations '
                'and an 80% coverage rule. Legacy return convention and minimum coverage are unspecified.'),
    LegacyMatch('1-Yr_Asset_Growth', 'f_asset_growth_1yr', 'match',
                'Both measure the percentage change in total assets over one year (four fiscal quarters). '
                'Percent/decimal units and point-in-time observation dates differ.'),
    LegacyMatch('1-Yr_CapEX_Growth', 'f_capex_growth_1yr', 'match',
                'Both compare 12-month capex with the preceding year. WRDS additionally excludes a '
                'prior-year denominator below $1M; this coverage difference is retained, not rebuilt.'),
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


def matches_for(column: str, *, confirmed_only: bool = True):
    return tuple(m for m in LEGACY_MATCHES if m.wrds_column == column
                 and (not confirmed_only or m.status == 'match'))


def legacy_names(column: str):
    names = {meta['column']: name for name, meta in FACTOR_METADATA.items()}
    return tuple(names[m.column] for m in matches_for(column))


def factor_tooltip(name: str) -> str:
    meta = WRDS_FACTOR_METADATA[name]
    column = meta['column']
    names = legacy_names(column)
    match_note = 'Legacy definition match: ' + ', '.join(names) if names else 'No confirmed Legacy definition match.'
    details = ' '.join(m.note for m in matches_for(column, confirmed_only=False))
    return f"{meta['tooltip']}\n\n{match_note}\n{details}\n\nDefault rationale: {REASONS[column]}"


def factor_rows():
    """All WRDS factors, including those without a Legacy match, in picker order."""
    return [{'factor': name, 'column': m['column'], 'legacy_match': '; '.join(legacy_names(m['column'])) or 'none',
             'default_direction': 'Higher' if m['higher_is_better'] else 'Lower',
             'reason': REASONS[m['column']]}
            for name, m in WRDS_FACTOR_METADATA.items()]


def legacy_rows():
    """Every Legacy factor exactly once; related candidates never become matches."""
    names = {m['column']: name for name, m in WRDS_FACTOR_METADATA.items()}
    crosswalk = {m.column: m for m in LEGACY_MATCHES}
    rows = []
    for name, meta in FACTOR_METADATA.items():
        m = crosswalk[meta['column']]
        rows.append({'legacy_factor': name, 'legacy_column': m.column,
                     'wrds_match': names[m.wrds_column] if m.status == 'match' else 'none',
                     'related_candidate': names[m.wrds_column] if m.status == 'related' else '',
                     'status': m.status, 'note': m.note})
    return rows
