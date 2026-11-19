# Legacy-to-WRDS definition matches

New runs use WRDS. Saved Legacy analyses remain viewable by ID in Results and Compare.

This report distinguishes a documented **Same** definition from a **Similar** alternative in the
existing reviewed catalog. `none` means there is no confirmed definition match; a related factor
may still be available. Similar factors are not rebuilt or silently substituted. Units, dates,
vendor values and coverage can differ even for a Same match.

| Legacy factor | WRDS definition match | Related WRDS factor | Notes |
|---|---|---|---|
| 12-Mo Momentum % | none | 12-Mo Momentum | Legacy uses price returns; WRDS compounds 252 trading days of CRSP total returns. |
| 6-Mo Momentum % | none | 6-Mo Momentum | Legacy uses price returns; WRDS compounds 126 trading days of CRSP total returns. |
| 1-Mo Momentum % | none | 1-Month Return (Reversal) | WRDS uses 21 trading days of CRSP total returns; Legacy defaults Higher, WRDS Lower (reversal). |
| Price to Book Using 9/30 Data | none | Book/Price | Reciprocal, so the direction flips: Lower P/B in Legacy is Higher Book/Price in WRDS. |
| Book/Price | none | Book/Price | WRDS prices book equity at the panel date's market cap; Legacy's price date is unspecified. |
| Next FY Earns/P | none | Forward E/P (FY2) | WRDS uses the FY2 (next fiscal year) consensus without the Legacy 3+ analyst minimum. |
| ROE using 9/30 Data | none | ROE | WRDS uses TTM income over positive Fama-French book equity; Legacy does not specify either. |
| ROA using 9/30 Data | none | ROA | WRDS uses TTM income; the Legacy 9/30 snapshot does not specify the income lookback. |
| ROA % | ROA | — | Same TTM net income over total assets; Legacy is in percent, WRDS in decimals. |
| Accruals/Assets | none | Accruals/Assets | WRDS uses TTM income and operating cash flow; the Legacy lookback is unspecified. |
| 1-Yr Price Vol % | none | 12-Mo Volatility | WRDS annualizes 252 daily log CRSP returns with 80% coverage; Legacy specifies neither rule. |
| 1-Yr Asset Growth % | 1-Yr Asset Growth | — | Same four-quarter change in total assets; Legacy is in percent, WRDS in decimals. |
| 1-Yr CapEX Growth % | 1-Yr CapEx Growth | — | Same 12-month capex versus the prior year; WRDS skips prior-year capex below $1M. |
