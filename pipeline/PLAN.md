# WRDS pipeline plan (phase 1, for review)

This is the plan for stages s01 to s08. No cleaning code is written yet. The first review's answers are in
[Decisions](#decisions-2026-10-25) and are written into the stage rules below. What is still open is in
[Open questions](#open-questions).

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

## Decisions (2026-10-25)

| # | decision | where |
|---|---|---|
| 1 | IBES FY4 is dropped. Use FPI 0 (long-term growth) and FPI 1-3. | s05; `config.IBES_FPI` |
| 2 | Non-USD fundamentals (402 quarterly rows, 19 permnos): price-based ratios (B/P, E/P, S/P, EV multiples, dividend yield) are missing for those rows. Ratios within the statements (ROA, ROE, margins, growth, leverage) are kept. | s03 rule 5, s07 rule 5 |
| 3 | Share classes: market cap is summed across share classes by permco for company-level ratios. Per-share ratios stay per share. | s07 rule 5 |
| 4 | Sectors from Compustat GICS: gsector (11 sectors) for the sector filter, gind (industry) for RV peer groups. The SIC mapping is only a fallback where GICS is missing, and it is fixed (SIC 28 is not all Healthcare; medical devices are Health Care). GICS here is the **current** classification, not historical. The app's sector list must change to the GICS names. | s07 rule 7 |
| 5 | Analyst data: keep the full consensus table, and add forward sales yield, forward EBITDA/EV, cash flow yield, FY2 E/P and long-term growth to the panel. | s05, s07 rule 8 |
| 6 | DCF data: the annual statements stay their own table and are not merged into the monthly panel. | s04, s07 |
| 7 | Upload lists: the pipeline writes permno_list.txt and ibes_tickers.txt to processed/; they are copied to raw/query_inputs/ by hand. | s01 rule 5, s05 rule 1 |
| 8 | The empty folders in the data folder (`cleaned/`, `parquests/`, `scripts/`) are deleted by hand; the pipeline never touches them. | - |
| 9 | Delisting rows (DlyDelFlg = Y) never provide a month-end price or market cap. Those come from the last real trading day; the delisting row only contributes its return. | s02 rule 8 |
| 10 | Annual report date = the rdq of the matching quarterly Q4 row (same gvkey and fiscal year), otherwise datadate + 90 days. | s04 rule 2 |

## What s00 found in raw/

All 15 files are present with every expected column. With FY4 dropped (decision 1) s00 expects FPI 0-3, and its
status is PASS (50 checks).

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

**Units and odd values** (a throwaway inspection of the raw files, 2026-10-25; it feeds the checks below):
- Compustat is in $ millions (median quarterly sales 85, total assets 773; annual sales 329) and shares in
  millions. GICS gsector has the 11 sector codes, missing on 0.3% of rows; gind has 81 industries.
- IBES totals (SAL, EBT, EBI, CPX) are $ millions (FY1 medians 390, 97, 56, 25); per-share measures are dollars
  (EPS 0.75, CPS 1.84, DPS 0.04-0.06; 48% of DPS FY1 estimates are 0); FPI 0 is percent (EPS median 13.8).
- FRED yields are percent (DGS10 0.52 to 5.31; DGS3MO 0 to 5.63; BAA 3.11 to 9.54; AAA 2.01 to 6.55).
- Odd values: 0.53% of per-share IBES rows have |mean estimate| above $1,000 (up to $3 bn, 127 tickers such as
  Eagle Bulk and Pacific Drilling). These are IBES restating old estimates for later reverse splits, and they
  cancel against `price_ibes_basis`. 3 Air Transport Services SAL rows (2023) are about $525 bn against about
  $2 bn normally, which looks like a units glitch. Compustat has 389 quarterly rows with negative sales, 35 with
  negative D&A, 23 annual rows with negative capex, 24 quarterly prices below $0.01, and one row each with zero
  shares or zero assets. xintq (interest) is missing on 21% of quarterly rows, including 44,102 rows of firms
  with debt; xint is missing on 17% of annual rows. IBES stdev is missing on 29% of rows (one analyst).
- **Non-index stocks in the holdings.** IWM held Royal Bank of Canada on 2008-02-29 (1 line) and Exxon Mobil on
  2023-11-30 and 2023-12-31, probably shares received in stock-for-stock acquisitions of index members. They
  are real holdings lines, so they are in permno_list.txt and their full CRSP, Compustat and IBES histories were
  pulled. RBC is 87 of the 402 CAD rows (total assets up to $2.5 trillion). See open question 1.

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
5. permno_list.txt = every permno in the snapshots, sorted. It is written to processed/ only; you copy it to
   raw/query_inputs/ before the permno-based queries (decision 7; the pipeline never writes raw/).

**Checks:** no duplicate (report_dt, permno) (FAIL); eff_dt >= report_dt (FAIL); no CUSIP maps to two permnos
(WARN, list them); stocks per snapshot 1,800-2,150 (WARN; last run 1,831-2,108); weight sum per snapshot 90-110%
(WARN; 2014-09-30 is 108%, 2014-10-31 93.5%, known); missing months are exactly 2009-04, 2009-05, 2009-06 (WARN
on any new one); stocks missing from a snapshot but in both neighbours, max (WARN above 20; last run 14);
permno_list.txt equals raw/query_inputs/permno_list.txt (WARN: the permno-based queries must be rerun); stocks in
only one or two snapshots with a weight below 0.01% (WARN, list them: candidates for merger-consideration
holdings such as RBC and Exxon). Last run:
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
8. **Delisting rows never provide a month-end price or market cap** (decision 9). Month-end market data (prc,
   shrout, mkt_cap_musd, ticker, mom, vol, `last_trade_date`) come from the last real trading day in the month
   (DlyDelFlg = N). The delisting row only contributes its return to `ret_m`. A month that holds only the
   delisting row has a return and no month-end price ([problem 3](#problems-in-the-old-notebook)). `siccd` = last
   non-zero SIC on or before the month end.
9. Calendar: month_end = the latest trading date of any permno in the month.

**Checks:** unique (permno, date) after dedup (FAIL); duplicate count vs last run (WARN); one flag row per
delisted permno, and it is the permno's last row (FAIL); flag rows = delistings.csv rows (FAIL); every delisting
falls in exactly one month (FAIL); fills = blank delret in delistings.csv (FAIL); ret_m equals the compounded daily
returns for 25 random permnos (FAIL); no month-end prc or mkt_cap taken from a delisting row (FAIL); daily returns
above +1000% or prices above $100,000 (WARN, list);
permnos vs permno_list.txt (WARN; today 79728 has no CIZ rows); no gaps in the calendar (FAIL).
Last run: 15,309,394 daily rows, 733,698 monthly rows.

**Tests:** duplicates dropped, conflicting duplicates FAIL; negative and zero prices; -99 return code; a flag row
with a return is not changed; missing flag-row return filled -30% for GDR and 0% for MER; momentum and
volatility against hand-computed values, including the 80% rule; a split (cfacpr change) does not move momentum;
monthly compounding; month-end price and market cap from the last trading row when the flag row is in the same
month, including a flag row that carries a price; flag row alone in the next month (return, no price).

## s03_fundamentals_q

**Inputs:** `raw/fundamentals_raw.csv` (all 67 columns, 127 MB).

**Output** (`processed/s03_fundamentals_q/fundamentals_q.parquet`), one row per permno per fiscal quarter:

- **Ids and timing:** permno, gvkey, period_end (datadate), report_date, report_date_source (`rdq` / `proxy`),
  fyearq, fqtr, curcdq, is_usd, ajexq.
- **Industry (current GICS, see decision 4):** gsector, gsector_name, ggroup, gind, gind_name, gsubind, naics.
  The names come from a GICS code table in config.
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
| pretax_ttm, tax_ttm, interest_ttm | piq, txtq, xintq summed (xintq is missing on 21% of rows, so interest_ttm is often missing; the annual table has xint) |
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
4. Impossible values to NaN: sales or capex growth below -100%, negative R&D ratio; inf to NaN. Other values stay
   as reported (negative quarterly sales and D&A are restatement adjustments) and are counted in the checks.
5. **Non-USD rows** (curcdq != USD; decision 2): `is_usd` = False, and the price-based ratios are NaN:
   f_book_to_price, f_earnings_to_price, f_sales_to_price, f_ebitda_to_ev, f_dividend_yield (also at quarter
   end), and mkt_cap_qe_musd. Ratios within the statements (ROA, ROE, margins, growth, leverage, current ratio,
   accruals, SUE) are kept. The amounts stay in their own currency, so s07 also leaves the price-based ratios
   missing for these rows.

**Checks:** unique (permno, fyearq, fqtr) (FAIL); report_date >= period_end (WARN with count); rdq proxy share
(WARN above 2%); non-USD rows counted, and no price-based ratio on a non-USD row (FAIL); negative saleq and dpq,
cshoq or atq of 0, prccq below $0.01 counted (WARN; 389, 35, 1, 1, 24 today); ajexq > 0 (FAIL); factor coverage
for 2015+ within 5
points of the last run (WARN; e.g. roa 95%, book_to_price 93%, sue 85%); TTM equals the hand sum for 20 random rows
(FAIL). Last run: 256,935 rows.

**Tests:** a missing quarter makes 4 TTM windows NaN; YTD to quarterly including Q1 and a fiscal-year change; a
2-for-1 split (ajexq 1 -> 2) keeps EPS TTM in current shares; duplicate quarters keep the most complete row; rdq
proxy +45 / +90; book equity fallback chain; growth exactly 4 quarters back; tax rate NaN rules; NWC and EV claims
with missing parts; a CAD row keeps ROA and leverage but has no B/P, E/P, S/P, EBITDA/EV or dividend yield.

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
2. **report_date** (the annual file has no rdq; decision 10): the rdq of the matching quarterly Q4 row, i.e. the
   same gvkey, fyearq = fyear and fqtr = 4 (96.4% of annual rows match today; in 2 of them the Q4 datadate
   differs from the annual datadate, which is WARNed), otherwise datadate + 90 days.
3. Previous-year values only from exactly fyear - 1; a fiscal-year-end change gives NaN changes.
4. No TTM: each row is a fiscal year. dp missing stays missing (5% of rows); EBITDA uses oibdp as reported.
5. **Own table** (decision 6): fundamentals_a is not merged into the monthly panel. DCF code joins it by permno
   with report_date <= the valuation date.

**Checks:** unique key (FAIL); report_date >= period_end (WARN); share of proxy report dates (WARN above 10%);
annual sale and oiadp vs the sum of the 4 quarters in s03 within 1% for at least 95% of matched rows (WARN);
missing dp, oiadp, xint counted; negative capx or sale counted (WARN; 23 and 61 today).

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
| consensus.parquet | permno, statpers, measure, fpi | the full consensus table (decision 5): all 7 measures, FPI 0-3. permno, ibes_ticker, statpers, measure, fpi, fiscalp, fpedats, numest, numup, numdown, meanest, medest, stdev, highest, lowest, price, price_ibes_basis, suspect_total |
| actuals.parquet | permno, measure, fpedats | permno, ibes_ticker, measure, fpedats, actual, anndats_act (future information at statpers: never for selection) |
| factors.parquet | permno, statpers | permno, ibes_ticker, statpers, fpedats, numest, f_forward_ep, f_eps_revision, f_eps_dispersion, f_analyst_surprise, surprise_anndats, surprise_fpedats, f_ltg |

**Rules:**
1. Link (ported): scores 1-2, our permnos only, each permno's latest link treated as open-ended because WRDS ends
   active links at the table's last update (2025-12-31) (H2). The link step runs first and writes
   ibes_tickers.txt to processed/ only (decision 7), even when forward_estimates.csv is missing (then the stage
   FAILs with "run query 10").
2. Keep FPI 0-3 (FY4 is not used, decision 1) and CURCODE = USD. Drop duplicate (ticker, statpers, measure, fpi)
   rows (H2), keeping the row with the most
   estimates, then the first (1,796 today). If IBES gives two different actuals for one (ticker, measure,
   fpedats), the actual is not used.
3. Attach permno when sdate <= statpers <= edate_used. A permno-statpers with two tickers is dropped.
4. Price = CRSP price on statpers or the last trading day before, at most 7 days earlier.
5. **Share basis:** IBES adjusted values are restated in today's shares; CRSP prices are not.
   `price_ibes_basis = price x cfacpr_last / cfacpr_t` (ported; 18.5% of EPS rows restated last run). This applies
   to the per-share measures (EPS, CPS, DPS). SAL, EBT, EBI, CPX are totals ($ millions; FY1 medians today:
   SAL 408, EBT 97, EBI 55, CPX 25) and need no share adjustment. Per-share values in the thousands of dollars
   (0.5% of rows) are old estimates restated for later reverse splits. They are kept, because `price_ibes_basis`
   is restated the same way.
6. FPI 0 (fiscalp LTG) is a long-term growth rate in percent (EPS median 13.8); stored as a decimal (/100).
   Values beyond ±100% (4,360 rows) are kept and counted.
7. **Suspect totals:** a SAL, EBT, EBI or CPX mean estimate more than 100 times the median of the same ticker,
   measure and FPI is set to NaN and flagged `suspect_total` (the 3 Air Transport Services SAL rows of 2023).
8. Factors at statpers (EPS FY1, ported): f_forward_ep = meanest / price_ibes_basis; f_eps_revision = (numup -
   numdown) / numest; f_eps_dispersion = stdev / |meanest| (NaN if meanest = 0); f_analyst_surprise = (actual -
   last consensus before the announcement) / price at that consensus, from the latest announcement on or before
   statpers, never an older one as fallback, later fiscal year wins on the same day. Added: f_ltg = EPS FPI 0
   meanest / 100. The other analyst yields need month-end market cap and EV, so s07 computes them from
   consensus.parquet (s07 rule 8).

**Checks:** rows by measure and fpi after filters; only FPI 0-3 (FAIL on any other); every measure in config
present (FAIL); suspect totals counted (WARN; 3 today); per-share |meanest| above $1,000 counted (WARN; 0.5%);
permnos linked vs ours (last run 5,751 of 5,918 for EPS FY1); price found for 90%+ of EPS
FY1 rows (WARN; 91.6% last run); surprise_anndats <= statpers (FAIL); no actual or anndats_act column in
consensus or factors (FAIL); ibes_tickers.txt vs raw/query_inputs/ibes_tickers.txt (WARN).

**Tests:** link window and open-ended latest link; two tickers on one permno-date dropped; `NA` read as a ticker;
duplicate rows; conflicting actuals give no surprise; a split after statpers restates the price; price older than 7
days is NaN; surprise appears only after the announcement and never falls back; same-day announcements; LTG
percent to decimal; totals are not share-adjusted; FPI 4 rows are dropped; a total 100x the ticker's median is
NaN and flagged.

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

**Inputs:** s01 snapshots, s02 monthly + calendar + daily cfacpr, s03 fundamentals_q, s05 factors and consensus.
Not s04: the annual statements stay their own table (decision 6).

**Outputs** (`processed/s07_panel/`):
- **panel.parquet**, one row per index stock per month_end: date, permno, permco, ticker, security_name, sector
  (GICS sector name), gsector, gind, gind_name, sector_source (`gics` / `sic`), siccd, is_fossil_fuel,
  index_weight, snapshot_report_dt, snapshot_eff_dt, new_entrant, has_price, price_date, prc, shrout,
  mkt_cap_musd, company_mkt_cap_musd, n_share_classes, mom_1m, mom_6m, mom_12m, mom_12_1, vol_*, fund_period_end,
  fund_report_date, fund_is_usd, the 17 f_* fundamentals (price-scaled five at month end, `_qe` versions kept),
  the RV building blocks used by the app (book_equity, sale_ttm, ebitda_ttm, ebit_ttm, net_debt, ev_claims,
  fcf_ttm, epsfx_ttm_adj), ibes_statpers, numest, f_forward_ep, f_eps_revision, f_eps_dispersion,
  f_analyst_surprise, surprise_anndats, the new analyst yields f_forward_sales_yield, f_forward_ebitda_ev,
  f_cash_flow_yield, f_fy2_ep, f_ltg, ret_next_1m, delisted_next_month. (61 columns last run, plus the additions.)
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
5. Price-scaled factors at t (decisions 2 and 3):
   - **Company-level ratios use the company's market cap.** `company_mkt_cap_musd` = sum of the month-end market
     cap of all permnos with the same permco on that date (`n_share_classes` = how many). Book / company mkt cap,
     sales / company mkt cap, EBITDA / (company mkt cap + net debt) with NaN if EV <= 0.
   - **Per-share ratios stay per share**, with the security's own price: the per-share TTM EPS and dividends
     restated to month-end shares with cfacpr(t) / cfacpr(quarter end) (1 when the quarter end predates the CRSP
     history), divided by prc.
   - **Non-USD fundamentals** (fund_is_usd = False): all of these are NaN. Statement ratios are kept.
6. IBES: latest statpers <= t, at most 60 days old.
7. **Sector and industry from Compustat GICS** (decision 4). `gsector` (11 sectors) is the sector filter, named
   with the GICS sector names. `gind` (industry, 81 in the data) is the peer group for relative value. They come
   from the fundamentals row used at t, else the permno's latest row with GICS. 22 permnos change gsector across
   rows because they move to another gvkey. **GICS here is the current classification, not historical**: a
   company that changed sector carries its current sector back in time.
   - **Fallback** only where GICS is missing (40 permnos today: 37 without Compustat rows, 3 without gsector): the
     SIC code as of t is mapped to a GICS sector name (`sector_source` = `sic`). The mapping is fixed so that SIC
     28 is not all Health Care. Rules are checked in this order; the first match wins, so specific ranges come
     first:

     | # | GICS sector | SIC |
     |---|---|---|
     | 1 | Unclassified | missing, 0, or 9900-9999 |
     | 2 | Health Care | 2830-2836 (drugs, biologics), 3841-3851 (medical devices and instruments), 8000-8099, 8731 |
     | 3 | Consumer Staples | 0100-0999, 2000-2199, 2840-2844 (soap, cosmetics), 5140-5149, 5400-5499, 5912 |
     | 4 | Consumer Discretionary | 2200-2399, 2500-2599, 3630-3639, 3651, 3710-3716, 3750-3751, 3940-3949, 5200-5999, 7000-7099, 7200-7299, 7500-7599, 7900-7999, 8200-8299 |
     | 5 | Energy | 1200-1399, 2900-2999 |
     | 6 | Utilities | 4900-4949, 4960-4999 |
     | 7 | Real Estate | 6500-6599, 6798 (REITs) |
     | 8 | Financials | 6000-6799 |
     | 9 | Communication Services | 2710-2799 (publishing), 4800-4899, 7810-7899 |
     | 10 | Information Technology | 3570-3579, 3600-3699, 3820-3829, 7370-7379 |
     | 11 | Materials | 1000-1199, 1400-1499, 2400-2499, 2600-2699, 2800-2899 (the rest of chemicals), 3200-3399 |
     | 12 | Industrials | every other SIC code |

     This replaces `config.SECTOR_RULES` (the notebook's 8 SIC sectors) in phase 2.

   - **The app needs updating** (separate PR): `app/streamlit_config.py` `SECTOR_OPTIONS` lists Consumer,
     Technology, Financials, Industrials, Healthcare, Energy, Materials, Utilities. That is already different from
     the notebook's 8 (which had Other, not Materials), and it must become the 11 GICS names.
   - is_fossil_fuel stays SIC-based: SIC 1200-1399, 2900-2999, 4920-4929, 5170-5179.
8. **Analyst yields at month end** (decision 5), from the consensus with the latest statpers <= t, at most 60 days
   old, all USD. price_ibes_basis(t) = prc(t) x cfacpr_last / cfacpr(t):

   | column | definition |
   |---|---|
   | f_forward_sales_yield | SAL FY1 mean / company_mkt_cap_musd |
   | f_forward_ebitda_ev | EBT FY1 mean / (company_mkt_cap_musd + ev_claims), NaN if EV <= 0 or the fundamentals are not USD |
   | f_cash_flow_yield | CPS FY1 mean / price_ibes_basis(t) |
   | f_fy2_ep | EPS FY2 mean / price_ibes_basis(t) |
   | f_ltg | EPS FPI 0 mean / 100 (not price-scaled) |

   The ported f_forward_ep keeps its notebook definition (price at statpers).

**Checks (asserts, FAIL):** no fundamentals with report_date > t; no IBES with statpers > t or older than 60 days;
no surprise announced after t; no snapshot with eff_dt > t; no price_date > t; no duplicate (date, permno).
**FAIL:** no price-based ratio on a non-USD fundamentals row; company_mkt_cap_musd >= mkt_cap_musd; every row has a
sector. **WARN:** share of rows on the SIC fallback above 2%; stocks per month outside 1,800-2,100 (last run 1,831-2,043); coverage by year of price, roa,
book-to-price, forward E/P, next-month return below the last run by more than 3 points. **Smoke test (printed):**
top-decile ROA, rebalanced each June and held 12 months, vs IWM (last run 12.5% vs 12.1% a year over 18 years).
Last run: 435,860 rows, 222 month ends 2008-04-30 to 2026-09-30.

**Tests:** a snapshot is used only when its last line is effective; the 2009 gap uses the older snapshot;
fundamentals reported after t are not used; a late-filed older quarter does not replace a newer one; IBES 61 days
old is NaN; a split between quarter end and month end restates EPS; GICS used when present, from the gvkey in use
at t; every SIC fallback rule with a boundary SIC (2834 and 3845 Health Care, 2821 and 2851 Materials, 2844
Consumer Staples, 1311 Energy, 4911 Utilities, 6798 Real Estate, 6021 Financials, 4813 Communication Services,
3674 and 7372 Information Technology, 3711 Consumer Discretionary, 3531 Industrials, missing and 9995
Unclassified); fossil flag; two share classes of one permco: company ratios use the summed market cap and per-share
ratios each class's own price; a CAD fundamentals row has no price ratios but keeps ROA; each analyst yield
against a hand value; ret_next_1m includes a delisting return in month t+1.

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
which that script would drop). fundamentals_annual.parquet is exported as its own table (decision 6), and
ibes_consensus.parquet is the full consensus table (decision 5). manifest.json: data_version (build date),
created_at, pipeline git commit, raw sha256 from s00, per dataset rows / MB / date range / permnos / column
types, units, sector source (GICS, current classification) and the SIC fallback mapping, fossil flag,
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

Ordered by how much they can move results. Each now has a fix in the stage rules above.

1. **Canadian-dollar fundamentals mixed with USD prices.** 402 quarterly rows (19 permnos) have curcdq = CAD and
   are used unconverted in book-to-price, sales-to-price, EBITDA/EV, E/P and dividend yield. Ratios that do not
   involve price (ROA, leverage, growth) are fine. Fixed by decision 2 (s03 rule 5).
2. **Share classes.** Compustat amounts are per company (gvkey), but market cap is per CRSP security. 1,123
   quarterly rows repeat a gvkey and quarter for a second permno, so each class's book-to-price, sales-to-price
   and EBITDA/EV divide company totals by one class's market cap. Fixed by decision 3 (s07 rule 5).
3. **Month-end data can come from the delisting row.** The monthly table takes the month's last row. For 774 of
   the 3,186 flag rows CIZ gives a post-delisting price and no shares, so that month's prc is the delisting price
   and mkt_cap is missing (the data dictionary's "prc is empty on delisting rows" is true only for the other
   2,412). Fixed by decision 9 (s02 rule 8).
4. **A late-filed older quarter can replace a newer one.** The panel's as-of merge picks the row with the latest
   report_date <= t; if a company files an old quarter (or restatement) after a newer one, the older period wins.
   Fix in s07 rule 4. Size unknown until phase 2 counts it.
5. **Row-order dependence.** Membership ticker/name/cusip take the "last" line, and IBES duplicates keep the
   "first" row, both in file order. Fix: largest weight; most estimates.
6. **Sector map.** SIC major group 28 (all chemicals, e.g. paints and fertilizers) is Healthcare, while medical
   devices (SIC 3841-3851) fall in Technology through major group 38. Fixed by decision 4: GICS, with a corrected
   SIC fallback (s07 rule 7).
7. **SUE uses basic EPS (epspxq), E/P uses diluted (epsfxq).** Probably harmless, but unclear whether intended.
8. **IBES restatement uses CRSP's last cfacpr (2026-09-30)**, not the IBES adjustment date; a split between the
   CRSP end and the IBES pull would be missed. Tiny today (both pulls are from October 2026); pull them together.
9. **The audit files are not in the archive.** `data_audit/report.md`, `verify_data.py`, `progress.md` and
   `known_gaps.md` are missing, so their checks cannot be compared. Their checks that the notebook prints are
   ported into the stage checks above.

## Open questions

The first round is answered in [Decisions](#decisions-2026-10-25). Raised by the raw data inspection:

1. **Non-index stocks received in mergers** (Royal Bank of Canada 2008-02-29, Exxon Mobil 2023-11 and 2023-12).
   They are real IWM lines, so the notebook put them in the universe for those months. Keep them (the holdings
   are the truth), or drop snapshot lines whose market cap is far above the index's largest members (for example
   above 5x the snapshot's 99th percentile)? s01 lists the candidates either way.
2. **Share classes outside our CRSP pull.** company_mkt_cap_musd can only sum the permnos we downloaded, and
   the CIZ query is by permno_list.txt. A class that IWM never held would be missing from the sum. s07 will count
   the permcos that look affected (fundamentals shares well above the summed CRSP shares). If there are many, add
   a CIZ pull by PERMCO?
3. **Missing quarterly interest expense** (xintq missing on 44,102 rows of firms with debt). Leave interest_ttm
   missing and let the DCF use the annual xint (recommended), or estimate it?

## Phase 2 order

s01 -> s02 -> s06 (needs only the s02 calendar) -> s03 -> s04 -> s05 -> s07 -> s08, one PR per one or two stages,
each with its tests and a run on the real data whose summary.json is compared with the last-run numbers above.
