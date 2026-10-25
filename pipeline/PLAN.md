# WRDS pipeline plan (phase 1, for review)

This is the plan for stages s01 to s08. No cleaning code is written yet. Please review the rules, and above all
the [open questions](#open-questions) and the [problems in the old notebook](#problems-in-the-old-notebook),
before phase 2 starts.

**Sources.** `archive/WRDS_Cleaning.ipynb` (code, markdown and printed outputs of its last run, 2026-10-22) and
the s00 inventory of `raw/` (2026-10-25). `archive/data_audit/report.md` is **not in the data folder**, so the
audit decisions below (A5, A7, A8, B2, D4, G2, H2) are taken from the notebook's comments and markdown, which
cite them. "Last run" numbers are the notebook's; they become the expected values in each stage's checks.

## Conventions for every stage

- **Keys:** `permno` (int32) plus a date. Company-level data also keeps `gvkey` (text) and `permco` (int32).
- **Dates:** tz-naive `datetime64[ns]`. `month_end` = last CRSP trading day of the month (from s02's calendar).
- **Units:** returns and yields are decimals. `mkt_cap_musd` is in $ millions. Compustat amounts are $ millions,
  Compustat shares are millions, CRSP `shrout` is thousands. The daily prices keep `mkt_cap` in $ thousands
  because the app's daily reader expects it.
- **Point in time:** a value may be used at date `t` only if it was public by `t`: snapshot `eff_dt`,
  fundamentals `report_date`, IBES `statpers`, surprise `anndats`. s07 asserts this.
- **Checks:** FAIL = the output is wrong and the run stops. WARN = unusual, reported, run continues. Checks that
  compare to the last run use a tolerance (usually ±2% on counts), so a refresh with more months passes.
- **Text ids:** CUSIPs, tickers and gvkeys are read as text with `keep_default_na=False`, because `NA` is a real
  IBES ticker.

## What s00 found in raw/

All 15 files are present with every expected column. Status FAIL for one reason: the IBES file has no FPI 4.

| file | rows | dates | note |
|---|---|---|---|
| iwm_holdings_raw.csv | 432,497 | 2008-02-29 to 2026-06-30 | 5,690 lines without permno (cash, futures, unlinked) |
| r2000_price_crsp_ciz.csv | 15,311,195 | 2006-12-01 to 2026-09-30 | 5,917 permnos; 1,801 exact duplicate rows; 3,625 flag rows = 3,186 unique |
| benchmark_etfs.csv | 14,961 | 2006-12-01 to 2026-09-30 | IWM, IWN, IWO, 4,987 days each |
| delistings.csv | 3,186 | 2007-02-14 to 2026-09-29 | delret missing for 49 |
| index_close.csv | 15,012 | 2006-12-01 to 2026-10-21 | 3 indexes, 5,004 days each; file is not sorted |
| crsp_ibes_link.csv | 10,015 | sdate from 1976 | 112 of our permnos have no link; 197 linked permnos are not ours |
| ff_daily.csv / ff_monthly.csv | 4,966 / 237 | to 2026-08-31 | already decimals |
| fundamentals_raw.csv | 257,528 | 2005-01-31 to 2026-09-30 | 67 columns, 5,881 permnos; **402 rows in CAD**; rdq missing 729 |
| fundamentals_annual_raw.csv | 65,360 | 2004-06-30 to 2026-09-30 | 68 columns, 5,856 permnos; all USD; **dp missing on 3,450 rows (5%)**; no rdq column |
| forward_estimates.csv | 10,056,427 | 2005-01-20 to 2026-08-20 | 7 measures, **FPI 0-3 only (no FPI 4)**; 27,133 non-USD rows; 1,796 duplicate ticker-statpers-measure-fpi rows |
| fred_DGS10/DGS3MO/DBAA/DAAA.csv | 5,428 each | 2006-01-03 to 2026-10-22 | 222 blank days each (holidays) |

Where each source ends: holdings 2026-06, prices 2026-09-30, fundamentals 2026-09-30, IBES 2026-08-20,
Fama-French 2026-08-31, index levels 2026-10-21, FRED 2026-10-22.

---

## s01_membership

**Inputs:** `raw/iwm_holdings_raw.csv`, `raw/query_inputs/permno_list.txt` (only to compare).

**Outputs** (`processed/s01_membership/`):

| file | one row per | columns |
|---|---|---|
| snapshots.parquet | report_dt, permno | report_dt, eff_dt, permno, weight, ticker, security_name, cusip, n_lines, permno_from_cusip |
| periods.parquet | membership period | permno, index_effective, index_through, first_known, n_snapshots, ticker, security_name |
| known_gaps.parquet, known_gaps.md | security without a permno | cusip, security_name, ticker, first, last, snapshots, max_weight |
| permno_list.txt | permno | the upload list for queries 2, 3, 6, 9, 11 |

**Rules** (ported):
1. A holdings line without a permno gets the permno of the same CUSIP on other report dates (the most common one)
   (audit A8). Last run: 1,569 lines, 130 securities, 0 CUSIPs linked to two permnos.
2. Lines that still have no permno, have a CUSIP and a ticker, and whose name does not match the non-stock pattern
   (cash, futures, swaps, rights, CVRs, notes, ...; the regex is ported unchanged into config) are the known gaps.
   No extra WRDS pull (A8). Last run: 72 securities.
3. One row per stock per snapshot: share-class lines are summed (`weight` = sum of percent_tna), `eff_dt` = the
   earliest line. **Change:** ticker, security_name and cusip come from the line with the largest weight instead of
   the last line in file order, so the result does not depend on row order.
4. Membership periods: a stock missing from 2 or more consecutive snapshots starts a new period; one missing
   snapshot is bridged (A7). `index_effective` and `index_through` are report_dt values; `first_known` = the
   period's first eff_dt, when that report became public (A5).
5. permno_list.txt = every permno in the snapshots, sorted. It is written to processed/; copying it to
   raw/query_inputs/ is a manual refresh step (the pipeline never writes raw/).

**Checks:** no duplicate (report_dt, permno) (FAIL); eff_dt >= report_dt (FAIL); no CUSIP maps to two permnos
(WARN, list them); stocks per snapshot 1,800-2,150 (WARN; last run 1,831-2,108); weight sum per snapshot 90-110%
(WARN; 2014-09-30 is 108%, 2014-10-31 93.5%, known); missing months are exactly 2009-04, 2009-05, 2009-06 (WARN
on any new one); stocks missing from a snapshot but in both neighbours, max (WARN above 20; last run 14);
permno_list.txt equals raw/query_inputs/permno_list.txt (WARN: the permno-based queries must be rerun). Last run:
218 snapshots, 428,354 rows, 5,918 permnos, 7,576 periods.

**Tests:** CUSIP fill (including a CUSIP with no permno anywhere -> gap); non-stock names are not gaps; share
classes summed; ticker from the largest line regardless of row order; a one-snapshot absence is bridged and a
two-snapshot absence splits the period; first_known = min eff_dt of the period; reruns give identical files.

## s02_prices

**Inputs:** `raw/r2000_price_crsp_ciz.csv` (columns permno, permco, dlycaldt, ticker, siccd, dlydelflg,
delreasontype, dlyprc, dlyret, dlyvol, shrout, dlyopen, dlyhigh, dlylow, dlycumfacpr), `raw/delistings.csv`.

**Outputs** (`processed/s02_prices/`):

| file | one row per | columns |
|---|---|---|
| daily/year=YYYY/data.parquet | permno, trading day | date, permno, permco, ticker, siccd, prc, prc_adj, cfacpr, shrout, mkt_cap, mkt_cap_musd, ret, vol, is_delisting_row, delist_ret_filled, mom_1m, mom_6m, mom_12m, vol_gk_{1m,3m,6m,12m}, vol_close_{1m,3m,6m,12m} |
| monthly.parquet | permno, month | month_end, permno, permco, ticker, siccd, last_trade_date, prc, shrout, mkt_cap_musd, mom_*, vol_*, ret_m, trading_days, delisted_in_month, delist_ret_filled |
| calendar.parquet | month | month, month_end, trading_days |
| delistings.parquet | delisting | permno, delisting_date, flagged_row_date, delactiontype, delreasontype, delpaymenttype, delret, delret_used |

**Rules** (ported):
1. Read the needed columns with pyarrow (about 2 GB in memory). Rolling windows run per permno, so the stage
   processes permnos in chunks if memory is short.
2. Drop exact duplicate rows (B2; last run 1,801, 439 of them flag rows). If two rows share permno and date but
   differ, FAIL (none today).
3. `prc` = |DlyPrc| (negative = bid/ask midpoint); 0 = missing. `ret` < -1 = missing code. `vol` < 0 and
   `shrout` <= 0 = missing. `mkt_cap` = prc x shrout ($ thousands), `prc_adj` = prc / cfacpr.
4. **Delisting returns (G2):** the flagged row (DlyDelFlg = Y, the day after the last trade, always the permno's
   last row) already holds the delisting return in DlyRet. It is never added again. If it is missing, fill -30%
   when delactiontype = GDR, else 0% (`config.DELISTING_FILL`); last run 49 fills (36 at -30%, 13 at 0%).
5. Momentum: compounded from log returns over 21 / 126 / 252 rows, NaN unless 80% of the rows have a return.
   Using returns (not prices) makes it split safe.
6. Volatility, annualized with 252 days: close-to-close std of log returns, and Garman-Klass from
   open/high/low/close, windows 21 / 63 / 126 / 252 rows, 80% rule.
7. Monthly return = compounded daily returns of the month including the flag row (`min_count=1`). A flag row in the
   month after the last trade gives a month with one row whose return is the delisting return.
8. **Change:** month-end market data (prc, shrout, mkt_cap_musd, ticker, mom, vol) come from the last row with
   DlyDelFlg = N in the month, not the flag row ([problem 3](#problems-in-the-old-notebook)). `siccd` = last
   non-zero SIC on or before the month end.
9. Calendar: month_end = the latest trading date of any permno in the month.

**Checks:** unique (permno, date) after dedup (FAIL); duplicate count vs last run (WARN); one flag row per
delisted permno, and it is the permno's last row (FAIL); flag rows = delistings.csv rows (FAIL); every delisting
falls in exactly one month (FAIL); fills = blank delret in delistings.csv (FAIL); ret_m equals the compounded daily
returns for 25 random permnos (FAIL); daily returns above +1000% or prices above $100,000 (WARN, list);
permnos vs permno_list.txt (WARN; today 79728 has no CIZ rows); no gaps in the calendar (FAIL).
Last run: 15,309,394 daily rows, 733,698 monthly rows.

**Tests:** duplicates dropped, conflicting duplicates FAIL; negative and zero prices; -99 return code; a flag row
with a return is not changed; missing flag-row return filled -30% for GDR and 0% for MER; momentum and
volatility against hand-computed values, including the 80% rule; a split (cfacpr change) does not move momentum;
monthly compounding; month-end price from the last trading row when the flag row is in the same month; flag row
in the next month.

## s03_fundamentals_q

**Inputs:** `raw/fundamentals_raw.csv` (all 67 columns, 127 MB).

**Output** (`processed/s03_fundamentals_q/fundamentals_q.parquet`), one row per permno per fiscal quarter:

- **Ids and timing:** permno, gvkey, period_end (datadate), report_date, report_date_source (`rdq` / `proxy`),
  fyearq, fqtr, curcdq, ajexq, gsector, ggroup, gind, gsubind, naics.
- **Quarter-end market data:** price_qe (prccq), mkt_cap_qe_musd (prccq x cshoq), cshoq, cshfdq.
- **Balance sheet (as reported):** atq, ltq, actq, lctq, cheq, invtq, rectq, apq, ppentq, gdwlq, intanq, dlcq,
  dlttq, pstkq, mibtq, seqq, ceqq, txditcq.
- **RV and DCF building blocks:**

| column | rule |
|---|---|
| book_equity | seqq (else ceqq + pstkq, else atq - ltq) + txditcq - pstkq; missing parts 0 (ported) |
| total_debt, net_debt | dlttq + dlcq; total_debt - cheq (missing parts 0) |
| ev_claims | net_debt + pstkq + mibtq (missing 0); EV = market cap + ev_claims |
| nwc | (actq - cheq) - (lctq - dlcq) (operating working capital) |
| d_nwc_4q | nwc - nwc exactly 4 quarters earlier |
| ebitda_ttm, ebit_ttm, da_ttm | oibdpq, oiadpq, dpq summed over 4 quarters |
| pretax_ttm, tax_ttm, interest_ttm | piq, txtq, xintq summed |
| tax_rate_ttm | tax_ttm / pretax_ttm, NaN when pretax_ttm <= 0 or the result is outside 0-1 |
| sale_ttm, revt_ttm, cogs_ttm, xsga_ttm, xrd_ttm, ni_ttm, ib_ttm | summed |
| oancf_ttm, capx_ttm, dv_ttm, sstk_ttm, prstkc_ttm, dltis_ttm, dltr_ttm, aqc_ttm | quarterly values from the year-to-date items (Q1 = YTD, else YTD - previous quarter of the same fiscal year), then summed |
| fcf_ttm | oancf_ttm - capx_ttm |
| epsfx_ttm_adj, dvpsx_ttm_adj | per-share sums on a split-adjusted basis (each quarter / ajexq), restated in this quarter's shares (x ajexq) (D4) |
| cshfd_adj, csho_adj | cshfdq x ajexq, cshoq x ajexq (comparable across splits) |

- **The 17 factors at quarter end** (ported unchanged): f_book_to_price, f_earnings_to_price, f_sales_to_price,
  f_ebitda_to_ev, f_dividend_yield, f_roe, f_roa, f_gross_profitability, f_accruals_to_assets, f_leverage,
  f_current_ratio, f_rd_to_sales (R&D missing = 0), f_asset_growth_1yr, f_sales_growth_1yr, f_capex_growth_1yr,
  f_net_share_issuance, f_sue. The price-scaled five are recomputed at month end in s07.

**Rules:**
1. Keep rows with permno, datadate, fyearq and fqtr. One row per (permno, fyearq, fqtr): keep the most complete
   row, then the latest rdq (last run removed 81; they come from fiscal-year changes).
2. `report_date` = rdq; if missing (0.3%), datadate + 45 days for fiscal Q1-Q3 and + 90 days for Q4.
3. **TTM = exactly 4 consecutive fiscal quarters** (quarter index fyearq x 4 + fqtr); one missing quarter makes
   the sum NaN. Growth and changes use exactly 4 quarters earlier.
4. Impossible values to NaN: sales or capex growth below -100%, negative R&D ratio; inf to NaN.

**Checks:** unique (permno, fyearq, fqtr) (FAIL); report_date >= period_end (WARN with count); rdq proxy share
(WARN above 2%); non-USD rows counted (WARN; 402 today); ajexq > 0 (FAIL); factor coverage for 2015+ within 5
points of the last run (WARN; e.g. roa 95%, book_to_price 93%, sue 85%); TTM equals the hand sum for 20 random rows
(FAIL). Last run: 256,935 rows.

**Tests:** a missing quarter makes 4 TTM windows NaN; YTD to quarterly including Q1 and a fiscal-year change; a
2-for-1 split (ajexq 1 -> 2) keeps EPS TTM in current shares; duplicate quarters keep the most complete row; rdq
proxy +45 / +90; book equity fallback chain; growth exactly 4 quarters back; tax rate NaN rules; NWC and EV claims
with missing parts.

## s04_fundamentals_a

**Inputs:** `raw/fundamentals_annual_raw.csv`, `processed/s03_fundamentals_q/fundamentals_q.parquet` (Q4 rdq).

**Output** (`processed/s04_fundamentals_a/fundamentals_a.parquet`), one row per permno per fiscal year:
permno, gvkey, fyear, fyr, period_end, report_date, report_date_source, curcd, ajex, price_fye (prcc_f), csho,
cshfd, emp, sich, gsector, ggroup, gind, gsubind, naics; the statement items as reported (sale, revt, cogs, xsga,
xrd, oibdp, dp, oiadp, xint, xintd, pi, txt, ni, ib, epsfx, epspx, dvpsx_f; at, lt, act, lct, che, invt, rect, ap,
ppent, gdwl, intan, dlc, dltt, pstk, mib, seq, ceq, txditc; oancf, capx, dv, dvc, sstk, prstkc, dltis, dltr, aqc);
and the same building blocks as s03 on an annual basis (book_equity, total_debt, net_debt, ev_claims, nwc, d_nwc,
fcf, tax_rate, epsfx_adj = epsfx / ajex, growth of sale, ebit, capex vs the previous fiscal year).

**Rules:**
1. One row per (permno, fyear): keep the most complete row (1 duplicate today).
2. **report_date** (the annual file has no rdq): the rdq of the quarterly row with the same gvkey and datadate and
   fqtr = 4 (matches 96.4% of rows today), else datadate + 90 days.
3. Previous-year values only from exactly fyear - 1; a fiscal-year-end change gives NaN changes.
4. No TTM: each row is a fiscal year. dp missing stays missing (5% of rows); EBITDA uses oibdp as reported.

**Checks:** unique key (FAIL); report_date >= period_end (WARN); share of proxy report dates (WARN above 10%);
annual sale and oiadp vs the sum of the 4 quarters in s03 within 1% for at least 95% of matched rows (WARN);
missing dp, oiadp counted.

**Tests:** Q4 rdq borrowed, proxy otherwise; fiscal-year change gives NaN growth; dedup; building blocks with
missing parts.

## s05_ibes

**Inputs:** `raw/forward_estimates.csv` (1.1 GB, needed columns only), `raw/crsp_ibes_link.csv`,
`processed/s01_membership/snapshots.parquet` (our permnos), `processed/s02_prices/daily/` (date, permno, prc,
cfacpr).

**Outputs** (`processed/s05_ibes/`):

| file | one row per | columns |
|---|---|---|
| link.parquet | link | ibes_ticker, permno, ncusip, sdate, edate, edate_used, score |
| ibes_tickers.txt | IBES ticker | the upload list for query 10 |
| consensus.parquet | permno, statpers, measure, fpi | permno, ibes_ticker, statpers, measure, fpi, fiscalp, fpedats, numest, numup, numdown, meanest, medest, stdev, highest, lowest, price, price_ibes_basis |
| actuals.parquet | permno, measure, fpedats | permno, ibes_ticker, measure, fpedats, actual, anndats_act (future information at statpers: never for selection) |
| factors.parquet | permno, statpers | permno, ibes_ticker, statpers, fpedats, numest, f_forward_ep, f_eps_revision, f_eps_dispersion, f_analyst_surprise, surprise_anndats, surprise_fpedats, f_ltg |

**Rules:**
1. Link (ported): scores 1-2, our permnos only, each permno's latest link treated as open-ended because WRDS ends
   active links at the table's last update (2025-12-31) (H2). The link step runs first and writes
   ibes_tickers.txt even when forward_estimates.csv is missing (then the stage FAILs with "run query 10").
2. Keep CURCODE = USD. Drop duplicate (ticker, statpers, measure, fpi) rows (H2), keeping the row with the most
   estimates, then the first (1,796 today). If IBES gives two different actuals for one (ticker, measure,
   fpedats), the actual is not used.
3. Attach permno when sdate <= statpers <= edate_used. A permno-statpers with two tickers is dropped.
4. Price = CRSP price on statpers or the last trading day before, at most 7 days earlier.
5. **Share basis:** IBES adjusted values are restated in today's shares; CRSP prices are not.
   `price_ibes_basis = price x cfacpr_last / cfacpr_t` (ported; 18.5% of EPS rows restated last run). This applies
   to the per-share measures (EPS, CPS, DPS). SAL, EBT, EBI, CPX are totals ($ millions; FY1 medians today:
   SAL 408, EBT 97, EBI 55, CPX 25) and need no share adjustment.
6. FPI 0 (fiscalp LTG) is a long-term growth rate in percent (EPS median 13.8); stored as a decimal (/100).
7. Factors (EPS FY1, ported): f_forward_ep = meanest / price_ibes_basis; f_eps_revision = (numup - numdown) /
   numest; f_eps_dispersion = stdev / |meanest| (NaN if meanest = 0); f_analyst_surprise = (actual - last
   consensus before the announcement) / price at that consensus, from the latest announcement on or before
   statpers, never an older one as fallback, later fiscal year wins on the same day. Added: f_ltg = EPS FPI 0
   meanest / 100.

**Checks:** rows by measure and fpi after filters; FPI 4 absent (WARN until re-pulled); every measure in
config present (FAIL); permnos linked vs ours (last run 5,751 of 5,918 for EPS FY1); price found for 90%+ of EPS
FY1 rows (WARN; 91.6% last run); surprise_anndats <= statpers (FAIL); no actual or anndats_act column in
consensus or factors (FAIL); ibes_tickers.txt vs raw/query_inputs/ibes_tickers.txt (WARN).

**Tests:** link window and open-ended latest link; two tickers on one permno-date dropped; `NA` read as a ticker;
duplicate rows; conflicting actuals give no surprise; a split after statpers restates the price; price older than 7
days is NaN; surprise appears only after the announcement and never falls back; same-day announcements; LTG
percent to decimal; totals are not share-adjusted.

## s06_rates_benchmarks

**Inputs:** `raw/fred_*.csv`, `raw/ff_daily.csv`, `raw/ff_monthly.csv`, `raw/index_close.csv`,
`raw/benchmark_etfs.csv`, `processed/s02_prices/calendar.parquet`.

**Outputs** (`processed/s06_rates_benchmarks/`):

| file | one row per | columns |
|---|---|---|
| rates_daily.parquet | FRED date | date, ust_10y, tbill_3m, moodys_baa, moodys_aaa |
| rates_monthly.parquet | month | month, month_end, the same 4 (last value on or before month_end, at most 7 days old) |
| benchmarks_daily.parquet | date | date, r2000_price_ret, r2000_growth_price_ret, r2000_value_price_ret, iwm_total_ret, iwn_total_ret, iwo_total_ret, mktrf, smb, hml, rmw, cma, umd, rf |
| benchmarks_monthly.parquet | month | month, month_end, the same columns |

**Rules:** FRED percent / 100, blank days stay NaN (never filled in the daily table). Fama-French are already
decimals. Index levels: drop 2012-10-29 and 2012-10-30 (Hurricane Sandy, no CRSP rows), price return from
consecutive levels; monthly = last level on or before month_end. ETF monthly returns compound the daily returns.
Monthly Fama-French = the published monthly factors matched by month (dateff). The first month (no prior level) is
dropped. Each monthly row is that month's return; the panel's ret_next_1m at month t lines up with month t+1.

**Checks:** yields within -1% to 25% (FAIL); no blank run longer than 5 business days (WARN); FF daily mean rf
below 0.001 (FAIL: percent units); IWM total minus Russell 2000 price return 0.5-2.5% a year (WARN; 1.34%
last run); source end dates reported; benchmark dates covered by the calendar (WARN).

**Tests:** percent to decimal; month-end as-of with a blank month end; Sandy days dropped; ETF compounding; FF
matched by month when dateff is not the CRSP month end.

## s07_panel

**Inputs:** s01 snapshots, s02 monthly + calendar + daily cfacpr, s03 fundamentals_q, s05 factors.

**Outputs** (`processed/s07_panel/`):
- **panel.parquet**, one row per index stock per month_end: date, permno, permco, ticker, security_name, sector,
  siccd, is_fossil_fuel, index_weight, snapshot_report_dt, snapshot_eff_dt, new_entrant, has_price, price_date,
  prc, shrout, mkt_cap_musd, company_mkt_cap_musd, mom_1m, mom_6m, mom_12m, mom_12_1, vol_*, fund_period_end,
  fund_report_date, the 17 f_* fundamentals (price-scaled five at month end, `_qe` versions kept), the RV
  building blocks used by the app (book_equity, sale_ttm, ebitda_ttm, ebit_ttm, net_debt, ev_claims, fcf_ttm,
  epsfx_ttm_adj), ibes_statpers, numest, f_forward_ep, f_eps_revision, f_eps_dispersion, f_analyst_surprise,
  surprise_anndats, f_ltg, ret_next_1m, delisted_next_month. (61 columns last run, plus the additions.)
- **monthly_returns.parquet**: every permno in the snapshots, every month with prices, also after leaving the index:
  month_end, permno, ret_m, delisted_in_month, delisting_fill_applied, trading_days. Use it for holding returns.

**Rules** (ported):
1. Rebalance dates = calendar month_ends from the first month with an effective snapshot (2008-04) to the last
   month with prices.
2. Universe at t = the latest snapshot whose every line has eff_dt <= t (2009-04 uses 2009-02, 2009-05 and 2009-06
   use 2009-03). index_weight = snapshot weight; new_entrant = not in the universe at the previous month end.
3. Market data from s02 monthly for month t (`price_date` = last trading day); mom_12_1 = (1 + mom_12m) /
   (1 + mom_1m) - 1; ret_next_1m = ret_m of month t+1 (includes any delisting return).
4. Fundamentals: as-of merge on report_date <= t. **Change:** among quarters reported by t, take the latest
   period_end, so a late-filed older quarter never replaces a newer one ([problem 4](#problems-in-the-old-notebook)).
5. Price-scaled factors at t: book / mkt cap, sales / mkt cap, EBITDA / (mkt cap + net debt) with NaN if EV <= 0,
   and the per-share TTM EPS and dividends restated to month-end shares with cfacpr(t) / cfacpr(quarter end)
   (1 when the quarter end predates the CRSP history), divided by prc. Market cap is the company's
   ([open question 3](#open-questions)).
6. IBES: latest statpers <= t, at most 60 days old.
7. Sector from siccd as of t with `config.SECTOR_RULES` in order, else Industrials ([open question
   4](#open-questions)); is_fossil_fuel for SIC 1200-1399, 2900-2999, 4920-4929, 5170-5179.

**Checks (asserts, FAIL):** no fundamentals with report_date > t; no IBES with statpers > t or older than 60 days;
no surprise announced after t; no snapshot with eff_dt > t; no price_date > t; no duplicate (date, permno).
**WARN:** stocks per month outside 1,800-2,100 (last run 1,831-2,043); coverage by year of price, roa,
book-to-price, forward E/P, next-month return below the last run by more than 3 points. **Smoke test (printed):**
top-decile ROA, rebalanced each June and held 12 months, vs IWM (last run 12.5% vs 12.1% a year over 18 years).
Last run: 435,860 rows, 222 month ends 2008-04-30 to 2026-09-30.

**Tests:** a snapshot is used only when its last line is effective; the 2009 gap uses the older snapshot;
fundamentals reported after t are not used; a late-filed older quarter does not replace a newer one; IBES 61 days
old is NaN; a split between quarter end and month end restates EPS; every sector rule with a boundary SIC (1311
Energy, 4911 Utilities, 9995 Other, missing Other, 3571 Technology, 2834 Healthcare, 6021 Financials, 5411
Consumer, 3711 Industrials); fossil flag; ret_next_1m includes a delisting return in month t+1.

## s08_export

**Inputs:** the outputs of s01-s07, and `processed/s00_inventory/inventory.json` (raw file checksums, so the 4 GB
is not hashed twice).

**Outputs** (`processed/s08_export/final/`): r2000_monthly_panel.parquet, r2000_monthly_returns.parquet,
daily_prices/year=YYYY/data.parquet + daily_prices/_manifest.json, daily_last_prices.parquet,
iwm_membership_snapshots.parquet, r2000_constituents_periods.parquet, fundamental_factors.parquet (quarterly),
fundamentals_annual.parquet, ibes_consensus.parquet, ibes_factors.parquet, rates_daily.parquet,
rates_monthly.parquet, benchmarks_daily.parquet, benchmarks_monthly.parquet, delistings.parquet, and
manifest.json, DATA_DICTIONARY.md, UPLOAD.md.

**Rules** (ported): the daily files keep the layout, column names and cleaning of the repo's
`scripts/publish_daily_data_to_s3.py` so `src/s3_client.py` can read them, plus the flag rows (`is_delisting_row`,
which that script would drop). manifest.json: data_version (build date), created_at, pipeline git commit, raw
sha256 from s00, per dataset rows / MB / date range / permnos / column types, units, sector mapping, fossil flag,
delisting rule, month-end convention, known gaps, where each source ends. DATA_DICTIONARY.md documents every
column (meaning, unit, source); the stage FAILs on an undocumented column. UPLOAD.md explains the dry run, upload
and check against `s3://sysen-5900-factor-lake/factor-lake-data/wrds_YYYY_MM/` for a teammate with the AWS keys.
**Nothing is uploaded by the pipeline.**

**Checks:** each file reads back with its source's row count (FAIL); permno int32 and dates datetime64[ns] (FAIL);
every column documented (FAIL); sha256 in the manifest equals s00 (FAIL); the repo's daily reader loads one year
(FAIL); total size reported (last run 1,987 MB).

**Tests:** an undocumented column FAILs; types are cast; the manifest lists every file; daily files round-trip
through `src/s3_client.py`'s reader.

---

## Problems in the old notebook

Ordered by how much they can move results. Each has a proposed fix in the stage above unless it is an open
question.

1. **Canadian-dollar fundamentals mixed with USD prices.** 402 quarterly rows (19 permnos) have curcdq = CAD and
   are used unconverted in book-to-price, sales-to-price, EBITDA/EV, E/P and dividend yield. Ratios that do not
   involve price (ROA, leverage, growth) are fine. See open question 2.
2. **Share classes.** Compustat amounts are per company (gvkey), but market cap is per CRSP security. 1,123
   quarterly rows repeat a gvkey and quarter for a second permno, so each class's book-to-price, sales-to-price and EBITDA/EV
   divide company totals by one class's market cap. See open question 3.
3. **Month-end data can come from the delisting row.** The monthly table takes the month's last row. For 774 of
   the 3,186 flag rows CIZ gives a post-delisting price and no shares, so that month's prc is the delisting price
   and mkt_cap is missing (the data dictionary's "prc is empty on delisting rows" is true only for the other
   2,412). Fix in s02 rule 8.
4. **A late-filed older quarter can replace a newer one.** The panel's as-of merge picks the row with the latest
   report_date <= t; if a company files an old quarter (or restatement) after a newer one, the older period wins.
   Fix in s07 rule 4. Size unknown until phase 2 counts it.
5. **Row-order dependence.** Membership ticker/name/cusip take the "last" line, and IBES duplicates keep the
   "first" row, both in file order. Fix: largest weight; most estimates.
6. **Sector map.** SIC major group 28 (all chemicals, e.g. paints and fertilizers) is Healthcare, while medical
   devices (SIC 3841-3851) fall in Technology through major group 38. See open question 4.
7. **SUE uses basic EPS (epspxq), E/P uses diluted (epsfxq).** Probably harmless, but unclear whether intended.
8. **IBES restatement uses CRSP's last cfacpr (2026-09-30)**, not the IBES adjustment date; a split between the
   CRSP end and the IBES pull would be missed. Tiny today (both pulls are from October 2026); pull them together.
9. **The audit files are not in the archive.** `data_audit/report.md`, `verify_data.py`, `progress.md` and
   `known_gaps.md` are missing, so their checks cannot be compared. Their checks that the notebook prints are
   ported into the stage checks above.

## Open questions

1. **IBES FPI 4.** The file has FPI 0-3 only. Re-pull query 10 with FPI 4, or drop FY4 from the plan?
2. **Non-USD fundamentals (402 rows).** (a) Set the price-scaled factors and EV to NaN for non-USD rows (my
   recommendation; no FX data is pulled), (b) pull Compustat exchange rates and convert, or (c) drop the rows.
3. **Company market cap for share classes.** Use `company_mkt_cap_musd` = sum over the permco's permnos on that
   date for the company-level ratios (recommended), or keep the per-class market cap as the notebook did?
4. **Sectors.** (a) Keep the 8 SIC sectors exactly as the notebook (matches the old app), (b) fix SIC 28 (only
   2830-2836 to Healthcare) and medical devices (3841-3851 to Healthcare), or (c) map Compustat GICS (gsector, now
   in the files) to the 8 sectors. GICS in this file is the current classification, not historical.
5. **Extra analyst factors.** Beyond the 4 ported ones and f_ltg: forward sales, EBITDA or cash-flow yields
   (FY1 SAL / company market cap, ...) and FY2 E/P? Or keep only consensus.parquet and leave factors to the app?
6. **DCF data in the panel.** Keep the annual statements in their own table (recommended; the DCF code can as-of
   merge on report_date), or also merge the latest fiscal year into the monthly panel?
7. **Upload lists.** The pipeline writes permno_list.txt and ibes_tickers.txt to processed/ and you copy them to
   raw/query_inputs/ before a refresh. OK, or should s01 and s05 write them straight into raw/query_inputs/ (an
   exception to "never write raw/")?
8. **Data folder clean-up.** The data folder also has empty `cleaned/`, `parquests/` and `scripts/` folders. The
   pipeline ignores them; delete them?

## Phase 2 order

s01 -> s02 -> s06 (needs only the s02 calendar) -> s03 -> s04 -> s05 -> s07 -> s08, one PR per one or two stages,
each with its tests and a run on the real data whose summary.json is compared with the last-run numbers above.
