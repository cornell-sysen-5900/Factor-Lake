# WRDS factor review

Which WRDS factor replaces each Legacy factor, and why each WRDS factor defaults to Higher or Lower. In the factor picker, tinted cards with a link icon replace a Legacy factor.

## Legacy to WRDS

**Same**: same definition (units may differ, which does not change the ranking). **Similar**: same idea, with the difference listed.

| Legacy factor | WRDS replacement | Level | Default (Legacy → WRDS) | Difference |
| --- | --- | --- | --- | --- |
| 12-Mo Momentum % | 12-Mo Momentum | Similar | Higher → Higher | Legacy uses price returns; WRDS compounds 252 trading days of CRSP total returns. |
| 6-Mo Momentum % | 6-Mo Momentum | Similar | Higher → Higher | Legacy uses price returns; WRDS compounds 126 trading days of CRSP total returns. |
| 1-Mo Momentum % | 1-Month Return (Reversal) | Similar | Higher → Lower | WRDS uses 21 trading days of CRSP total returns; Legacy defaults Higher, WRDS Lower (reversal). |
| Price to Book Using 9/30 Data | Book/Price | Similar | Lower → Higher | Reciprocal, so the direction flips: Lower P/B in Legacy is Higher Book/Price in WRDS. |
| Book/Price | Book/Price | Similar | Higher → Higher | WRDS prices book equity at the panel date's market cap; Legacy's price date is unspecified. |
| Next FY Earns/P | Forward E/P (FY2) | Similar | Higher → Higher | WRDS uses the FY2 (next fiscal year) consensus without the Legacy 3+ analyst minimum. |
| ROE using 9/30 Data | ROE | Similar | Higher → Higher | WRDS uses TTM income over positive Fama-French book equity; Legacy does not specify either. |
| ROA using 9/30 Data | ROA | Similar | Higher → Higher | WRDS uses TTM income; the Legacy 9/30 snapshot does not specify the income lookback. |
| ROA % | ROA | Same | Higher → Higher | Same TTM net income over total assets; Legacy is in percent, WRDS in decimals. |
| Accruals/Assets | Accruals/Assets | Similar | Lower → Lower | WRDS uses TTM income and operating cash flow; the Legacy lookback is unspecified. |
| 1-Yr Price Vol % | 12-Mo Volatility | Similar | Lower → Lower | WRDS annualizes 252 daily log CRSP returns with 80% coverage; Legacy specifies neither rule. |
| 1-Yr Asset Growth % | 1-Yr Asset Growth | Same | Lower → Lower | Same four-quarter change in total assets; Legacy is in percent, WRDS in decimals. |
| 1-Yr CapEX Growth % | 1-Yr CapEx Growth | Same | Lower → Lower | Same 12-month capex versus the prior year; WRDS skips prior-year capex below $1M. |

## Default directions

Each default is checked against the factor's long-short spread in each calendar year, 2002–2026. A direction that wins in at least 70% of years is the default; otherwise the theory default stands, and the basis says whether it won at least 50% of years. Saved analyses and strategy files keep the directions they were saved with.

| WRDS factor | Default | Basis | Reason |
| --- | --- | --- | --- |
| 12-1 Mo Momentum | Higher | Theory default; data mixed | Intermediate-horizon return continuation; skipping the latest month reduces reversal exposure. |
| 12-Mo Momentum | Higher | Theory default; data mixed | Intermediate-horizon return continuation; including the latest month can add reversal exposure. |
| 6-Mo Momentum | Higher | Theory default; data mixed | Intermediate-horizon return continuation motivates favoring higher six-month returns. |
| 1-Month Return (Reversal) | Lower | Theory default; data mixed | Short-term reversal motivates favoring last month's losers; this is not a momentum default. |
| Book/Price | Higher | Theory default; data mixed | More book equity per dollar paid provides a value tilt; use Higher for B/P, not Lower for P/B. |
| Earnings/Price | Higher | Theory default; data mixed | More earnings per dollar paid provides a value tilt; E/P is the inverse orientation of P/E. |
| Sales/Price | Higher | Theory default; data mixed | More revenue per dollar paid is a value tilt, although margins and business models differ. |
| FCF Yield | Higher | Consistent: favored in most years since 2002 | More free cash flow per dollar paid supports cash-flow valuation. |
| Dividend Yield | Higher | Theory default; the other direction won more years since 2002 | More dividends per dollar paid supports an income/value tilt; distressed yields need caution. |
| EBITDA/EV | Higher | Theory default; data mixed | More EBITDA per dollar of enterprise value favors cheaper operating cash-earnings capacity. |
| EBIT/EV | Higher | Theory default; data mixed | More operating earnings per dollar of enterprise value supports a value tilt. |
| Sales/EV | Higher | Consistent: favored in most years since 2002 | More sales per dollar of enterprise value supports a value tilt; profitability still matters. |
| Forward E/P (FY1) | Higher | Theory default; data mixed | More expected FY1 earnings per dollar paid supports forward value, subject to forecast error. |
| Forward E/P (FY2) | Higher | Theory default; data mixed | More expected FY2 earnings per dollar paid supports forward value, subject to forecast error. |
| Forward Cash Flow/P | Higher | Theory default; data mixed | More expected cash flow per dollar paid supports forward cash-flow value. |
| Forward Sales/EV | Higher | Consistent: favored in most years since 2002 | More forecast sales per dollar of enterprise value supports forward value. |
| Forward EBITDA/EV | Higher | Theory default; data mixed | More forecast EBITDA per dollar of enterprise value supports forward value. |
| ROE | Higher | Theory default; data mixed | Higher income relative to equity indicates profitability; leverage can also raise ROE. |
| ROA | Higher | Theory default; data mixed | Higher income relative to assets indicates more productive use of the asset base. |
| Gross Profitability | Higher | Theory default; data mixed | Higher gross profit relative to assets is a profitability signal. |
| Accruals/Assets | Lower | Theory default; data mixed | Lower accruals favor earnings backed by operating cash flow rather than accounting accruals. |
| Debt/Assets | Lower | Theory default; data mixed | Lower debt relative to assets favors balance-sheet safety; the return relation is context-dependent. |
| Current Ratio | Higher | Theory default; data mixed | Higher short-term liquidity supports a safety tilt; very high ratios can imply idle assets. |
| Net Share Issuance | Lower | Consistent: favored in most years since 2002 | Lower issuance favors payout and limits dilution; issuance can also fund productive projects. |
| 1-Yr Asset Growth | Lower | Theory default; data mixed | The investment effect favors conservative asset growth over aggressive expansion. |
| 1-Yr Sales Growth | Higher | Theory default; the other direction won more years since 2002 | Revenue growth supports operating strength; this project convention is valuation-dependent. |
| 1-Yr CapEx Growth | Lower | Theory default; data mixed | The investment effect motivates lower investment growth; productive capex can still add value. |
| R&D/Sales | Higher | Theory default; data mixed | R&D intensity can proxy intangible investment; this project convention is sector-dependent. |
| 12-Mo Volatility | Lower | Theory default; data mixed | A defensive tilt favors lower realized volatility; it does not guarantee higher raw returns. |
| 6-Mo Volatility | Lower | Theory default; data mixed | A defensive tilt favors lower medium-horizon realized volatility. |
| 3-Mo Volatility | Lower | Theory default; data mixed | A defensive tilt favors lower recent realized volatility; shorter estimates are noisier. |
| 1-Mo Volatility (High-Low) | Lower | Theory default; data mixed | A defensive tilt favors lower intramonth range volatility; this short-horizon proxy is noisy. |
| Beta (1-Yr Daily) | Lower | Theory default; the other direction won more years since 2002 | Leverage constraints motivate a low-beta tilt; this unlevered sort does not replicate the BAB factor. |
| Beta (5-Yr Monthly) | Lower | Theory default; the other direction won more years since 2002 | Leverage constraints motivate a low-beta tilt; a five-year beta reacts slowly to changing exposures. |
| Earnings Surprise (SUE) | Higher | Theory default; data mixed | Underreaction to positive earnings news motivates favoring larger standardized earnings increases. |
| Analyst EPS Surprise | Higher | Theory default; data mixed | Underreaction to earnings announcements motivates favoring positive consensus surprises. |
| EPS Revisions | Higher | Theory default; data mixed | Upward analyst revisions indicate improving earnings expectations; favor net upward revisions. |
| Analyst Disagreement | Lower | Consistent: favored in most years since 2002 | Lower disagreement limits optimistic pricing under short-sale constraints; evidence is mixed. |
| Long-Term Growth Forecast | Lower | Theory default; the other direction won more years since 2002 | Lower long-term forecasts avoid excessive extrapolation of optimistic growth expectations; evidence is mixed. |
| Market Cap | Lower | Theory default; data mixed | A size tilt favors smaller firms; the premium is unstable and trading costs can be higher. |

## Private report

The spreads stay private. From the repository root:

```bash
uv run python scripts/wrds_factor_report.py --data-dir "/path/to/processed/final" --output ../private_reports/wrds_factor_review.md
```

Use `--s3` instead of `--data-dir` to read the data from S3. The output must be outside this repository. It lists each factor's spread and the share of years its default wins, and flags defaults that differ from the rule. `--catalog-only --output DOCS/WRDS_FACTOR_REVIEW.md` regenerates this page.
