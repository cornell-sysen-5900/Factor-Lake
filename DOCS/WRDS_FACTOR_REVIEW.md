# WRDS factor matching and default directions

## Review status

**Empirical spreads pending: this catalog contains no real-data results.** Run the accompanying script against the authorized WRDS data to complete the evidence.

Matches are based on documented formula and lookback, not equal vendor values. Units, publication timing and coverage can differ. Similar names, reciprocals and undocumented lookbacks are related candidates only. Unmatched factors are listed, not rebuilt.

Requested return window: **2002–2026**. Tail size: **10%**.
Source: No data loaded (catalog only)

## WRDS factors

| WRDS factor | Column | Legacy match | Default | Reason | Long–short spread | Coverage | Review |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12-1 Mo Momentum | `mom_12_1` | none | Higher | Intermediate-horizon return continuation; skipping the latest month reduces reversal exposure. | pending | not computed | pending data |
| 12-Mo Momentum | `mom_12m` | none | Higher | Intermediate-horizon return continuation; including the latest month can add reversal exposure. | pending | not computed | pending data |
| 6-Mo Momentum | `mom_6m` | none | Higher | Intermediate-horizon return continuation motivates favoring higher six-month returns. | pending | not computed | pending data |
| 1-Month Return (Reversal) | `mom_1m` | none | Lower | Short-term reversal motivates favoring last month's losers; this is not a momentum default. | pending | not computed | pending data |
| Book/Price | `f_book_to_price` | none | Higher | More book equity per dollar paid provides a value tilt; use Higher for B/P, not Lower for P/B. | pending | not computed | pending data |
| Earnings/Price | `f_earnings_to_price` | none | Higher | More earnings per dollar paid provides a value tilt; E/P is the inverse orientation of P/E. | pending | not computed | pending data |
| Sales/Price | `f_sales_to_price` | none | Higher | More revenue per dollar paid is a value tilt, although margins and business models differ. | pending | not computed | pending data |
| FCF Yield | `f_fcf_yield` | none | Higher | More free cash flow per dollar paid supports cash-flow valuation. | pending | not computed | pending data |
| Dividend Yield | `f_dividend_yield` | none | Higher | More dividends per dollar paid supports an income/value tilt; distressed yields need caution. | pending | not computed | pending data |
| EBITDA/EV | `f_ebitda_to_ev` | none | Higher | More EBITDA per dollar of enterprise value favors cheaper operating cash-earnings capacity. | pending | not computed | pending data |
| EBIT/EV | `f_ebit_to_ev` | none | Higher | More operating earnings per dollar of enterprise value supports a value tilt. | pending | not computed | pending data |
| Sales/EV | `f_sales_to_ev` | none | Higher | More sales per dollar of enterprise value supports a value tilt; profitability still matters. | pending | not computed | pending data |
| Forward E/P (FY1) | `f_forward_ep` | none | Higher | More expected FY1 earnings per dollar paid supports forward value, subject to forecast error. | pending | not computed | pending data |
| Forward E/P (FY2) | `f_forward_ep_fy2` | none | Higher | More expected FY2 earnings per dollar paid supports forward value, subject to forecast error. | pending | not computed | pending data |
| Forward Cash Flow/P | `f_forward_cfp` | none | Higher | More expected cash flow per dollar paid supports forward cash-flow value. | pending | not computed | pending data |
| Forward Sales/EV | `f_forward_sales_to_ev` | none | Higher | More forecast sales per dollar of enterprise value supports forward value. | pending | not computed | pending data |
| Forward EBITDA/EV | `f_forward_ebitda_to_ev` | none | Higher | More forecast EBITDA per dollar of enterprise value supports forward value. | pending | not computed | pending data |
| ROE | `f_roe` | none | Higher | Higher income relative to equity indicates profitability; leverage can also raise ROE. | pending | not computed | pending data |
| ROA | `f_roa` | ROA % | Higher | Higher income relative to assets indicates more productive use of the asset base. | pending | not computed | pending data |
| Gross Profitability | `f_gross_profitability` | none | Higher | Higher gross profit relative to assets is a profitability signal. | pending | not computed | pending data |
| Accruals/Assets | `f_accruals_to_assets` | none | Lower | Lower accruals favor earnings backed by operating cash flow rather than accounting accruals. | pending | not computed | pending data |
| Debt/Assets | `f_leverage` | none | Lower | Lower debt relative to assets favors balance-sheet safety; the return relation is context-dependent. | pending | not computed | pending data |
| Current Ratio | `f_current_ratio` | none | Higher | Higher short-term liquidity supports a safety tilt; very high ratios can imply idle assets. | pending | not computed | pending data |
| Net Share Issuance | `f_net_share_issuance` | none | Lower | Lower issuance favors payout and limits dilution; issuance can also fund productive projects. | pending | not computed | pending data |
| 1-Yr Asset Growth | `f_asset_growth_1yr` | 1-Yr Asset Growth % | Lower | The investment effect favors conservative asset growth over aggressive expansion. | pending | not computed | pending data |
| 1-Yr Sales Growth | `f_sales_growth_1yr` | none | Higher | Revenue growth supports operating strength; this project convention is valuation-dependent. | pending | not computed | pending data |
| 1-Yr CapEx Growth | `f_capex_growth_1yr` | 1-Yr CapEX Growth % | Lower | The investment effect motivates lower investment growth; productive capex can still add value. | pending | not computed | pending data |
| R&D/Sales | `f_rd_to_sales` | none | Higher | R&D intensity can proxy intangible investment; this project convention is sector-dependent. | pending | not computed | pending data |
| 12-Mo Volatility | `vol_close_12m` | none | Lower | A defensive tilt favors lower realized volatility; it does not guarantee higher raw returns. | pending | not computed | pending data |
| 6-Mo Volatility | `vol_close_6m` | none | Lower | A defensive tilt favors lower medium-horizon realized volatility. | pending | not computed | pending data |
| 3-Mo Volatility | `vol_close_3m` | none | Lower | A defensive tilt favors lower recent realized volatility; shorter estimates are noisier. | pending | not computed | pending data |
| 1-Mo Volatility (High-Low) | `vol_gk_1m` | none | Lower | A defensive tilt favors lower intramonth range volatility; this short-horizon proxy is noisy. | pending | not computed | pending data |
| Beta (1-Yr Daily) | `beta_252d` | none | Lower | Leverage constraints motivate a low-beta tilt; this unlevered sort does not replicate the BAB factor. | pending | not computed | pending data |
| Beta (5-Yr Monthly) | `beta_60m` | none | Lower | Leverage constraints motivate a low-beta tilt; a five-year beta reacts slowly to changing exposures. | pending | not computed | pending data |
| Earnings Surprise (SUE) | `f_sue` | none | Higher | Underreaction to positive earnings news motivates favoring larger standardized earnings increases. | pending | not computed | pending data |
| Analyst EPS Surprise | `f_analyst_surprise` | none | Higher | Underreaction to earnings announcements motivates favoring positive consensus surprises. | pending | not computed | pending data |
| EPS Revisions | `f_eps_revision` | none | Higher | Upward analyst revisions indicate improving earnings expectations; favor net upward revisions. | pending | not computed | pending data |
| Analyst Disagreement | `f_eps_dispersion` | none | Lower | Lower disagreement limits optimistic pricing under short-sale constraints; evidence is mixed. | pending | not computed | pending data |
| Long-Term Growth Forecast | `f_ltg` | none | Lower | Lower long-term forecasts avoid excessive extrapolation of optimistic growth expectations; evidence is mixed. | pending | not computed | pending data |
| Market Cap | `mkt_cap_musd` | none | Lower | A size tilt favors smaller firms; the premium is unstable and trading costs can be higher. | pending | not computed | pending data |

## Every Legacy factor

| Legacy factor | Confirmed WRDS match | Related candidate only | Definition / limitation |
| --- | --- | --- | --- |
| 12-Mo Momentum % | none | 12-Mo Momentum | Legacy excludes dividends; WRDS compounds CRSP returns over 252 trading days. Return convention and calendar lookback differ. |
| 6-Mo Momentum % | none | 6-Mo Momentum | Legacy excludes dividends; WRDS compounds CRSP returns over 126 trading days. |
| 1-Mo Momentum % | none | 1-Month Return (Reversal) | Legacy excludes dividends; WRDS uses 21 trading-day CRSP returns. WRDS also defaults to reversal (Lower), while Legacy defaults to Higher. |
| Price to Book Using 9/30 Data | none | Book/Price | Price/book is the reciprocal of book/price, not the same factor definition. No reciprocal or automatic direction conversion is performed. |
| Book/Price | none | Book/Price | Legacy says latest-quarter book/price; its price date is not specified. WRDS uses company market cap at the panel date. Verify Legacy price timing first. |
| Next FY Earns/P | none | Forward E/P (FY2) | FY2 is the next fiscal year candidate, not FY1. Legacy requires 3+ analysts; that coverage rule is not part of this WRDS factor. No exact match confirmed. |
| ROE using 9/30 Data | none | ROE | Legacy does not specify the income lookback or equity adjustments. WRDS uses TTM income and positive Fama-French-style book equity. |
| ROA using 9/30 Data | none | ROA | The Legacy 9/30 snapshot does not specify the income lookback; WRDS explicitly uses TTM income. |
| ROA % | ROA | — | Both documented definitions are trailing-12-month net income divided by total assets. Legacy percent and WRDS decimal units differ; vendor values and observation dates may differ. |
| Accruals/Assets | none | Accruals/Assets | Both use income minus operating cash flow over assets, but the Legacy income/cash-flow lookback is unspecified. WRDS uses TTM flows. |
| 1-Yr Price Vol % | none | 12-Mo Volatility | Both describe annualized daily volatility; WRDS uses log CRSP returns, 252 observations and an 80% coverage rule. Legacy return convention and minimum coverage are unspecified. |
| 1-Yr Asset Growth % | 1-Yr Asset Growth | — | Both measure the percentage change in total assets over one year (four fiscal quarters). Percent/decimal units and point-in-time observation dates differ. |
| 1-Yr CapEX Growth % | 1-Yr CapEx Growth | — | Both compare 12-month capex with the preceding year. WRDS additionally excludes a prior-year denominator below $1M; this coverage difference is retained, not rebuilt. |

## Evidence methodology

Monthly equal-weight tails, 10% by default, with at least one stock per leg and no overlap.
Sort finite raw factor values at month end t; use returns in the next calendar month (t+1), including
CRSP delisting returns. Break ties by PERMNO. Use the existing buyability filter (positive price and market cap),
without a sector or fossil-fuel restriction. No forward-return column is used in selection.
For Higher defaults: long high, short low. For Lower defaults: long low, short high.
Reported spread is 12 times the arithmetic mean of monthly long-minus-short returns, in percentage points
per year; it is NOT a difference of CAGRs and is NOT a compounded trading strategy.
Each leg has $1 notional (gross exposure $2); borrowing fees, financing, transaction costs and taxes are excluded.
Missing individual returns retain their weight and are filled with 0% as in the app, with a count and coverage flag.
Months with fewer than two finite signals, a constant signal, or no next-month return-file coverage are skipped
and counted. Actual factor coverage is shown; partial 2002 and 2026 years are not presented as full years.
This diagnostic uses raw single-factor ranks, not the app's winsorized multi-factor scoring.
Negative default-oriented spreads are marked for team review; no default is flipped automatically.
Positive spreads are descriptive evidence, not proof of significance or out-of-sample performance.

## Sources and interpretation

- Internal definitions: `app/streamlit_config.py`, `src/factor_registry.py`,
  `pipeline/DATA_DICTIONARY.md`, and `pipeline/stages/s02_prices.py`.
- Existing default decisions: `DOCS/WRDS_MODE_DESIGN.md`, decision 4 and section 2.1.
- Kenneth French, [Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):
  value, size, profitability, investment, momentum and reversal constructions. These are conceptual references,
  not claims that this report replicates the published factor portfolios.
- French, [Momentum](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_mom_factor.html).
- Asness, Frazzini and Pedersen, [Quality Minus Junk](https://www.aqr.com/insights/research/working-paper/quality-minus-junk).
- Frazzini and Pedersen, [Betting Against Beta](https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta).

Rationales retain the project's existing theory-based choices. Liquidity, sales growth, R&D intensity,
analyst disagreement and long-term forecasts are explicitly context-dependent conventions; none is asserted
to be universally optimal. The evidence is for team review, not automatic sign selection.

## Generate the private evidence report

Run from the repository root, using your existing local AWS configuration:

```bash
uv run python scripts/wrds_factor_report.py --s3 --output ../private_reports/wrds_factor_review.md
```

Or replace `--s3` with `--data-dir "/path/to/processed/final"`. The script also writes a monthly CSV and a JSON provenance/summary file beside the report. Keep all real-data outputs outside this public repository. No defaults or data files are modified.
