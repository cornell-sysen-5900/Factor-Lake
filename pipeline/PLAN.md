# WRDS pipeline plan

The plan for stages s01 to s08. s00 to s06 are implemented (phases 2a to 2c); s07 and s08 are not written yet.
[Final outputs](#final-outputs) is what the pipeline must deliver; every stage builds toward it. The review
answers are in [Decisions](#decisions) and are written into the stage rules below.

## Final outputs

s08_export writes these files to `processed/final/` (s08's stage folder; Parquet, pyarrow, zstd). Every factor
and multiple is a raw value; the app z-scores or ranks at each rebalance.

| file | one row per | key columns | built in | used by |
|---|---|---|---|---|
| **Factors** | | | | |
| factor_panel_monthly.parquet | index member, month end (point in time) | date, permno, ticker, security_name, gsector, gind, gsubind, sector_name, mkt_cap_musd, company_mkt_cap_musd, all fundamental factors (f_*), price factors (mom_1m, mom_6m, mom_12m, mom_12_1, vol_*), analyst factors (f_forward_ep, f_eps_revision, f_eps_dispersion, f_analyst_surprise, f_forward_sales_yield, f_forward_ebitda_ev, f_cash_flow_yield, f_fy2_ep, f_ltg), beta_252d, beta_60m, ret_next_1m | s07 | factor selection and backtest ranking, sector filter (gsector), Data Viewer |
| universe_periods.parquet | stock, membership period | permno, index_effective, index_through, first_known, n_snapshots, ticker, security_name, excluded_non_member | s01, finalized in s07 | universe selection, Data Viewer |
| **Prices and returns** | | | | |
| returns_monthly.parquet | stock, month, for every permno in permno_list.txt, also after it leaves the index | month_end, permno, ret_m (includes delisting returns), delisted_in_month, delist_ret_filled, trading_days | s02 | backtest holding returns, delisting handling |
| prices_daily/year=YYYY/data.parquet | stock, trading day | date, permno, ticker, prc, prc_adj, cfacpr, ret, vol, shrout, mkt_cap, mkt_cap_musd, is_delisting_row | s02 | daily data loader (`src/s3_client.py`), sub-annual rebalancing, Data Viewer |
| **Market** | | | | |
| market_daily.parquet, market_monthly.parquet | trading day / month | r2000_price_ret, r2000_growth_price_ret, r2000_value_price_ret, iwm_total_ret, iwo_total_ret, iwn_total_ret, mktrf, smb, hml, rmw, cma, umd, rf, ust_10y, tbill_3m, moodys_baa, moodys_aaa (decimals), term_spread, credit_spread, was_filled (daily) | s06 | benchmarks, performance metrics (risk-free rate), betas, DCF discount rates |
| **Relative value** | | | | |
| rv_multiples_monthly.parquet | index member, month end | date, permno, gind, prc, mkt_cap_musd, ev_musd, pe_ttm, pb, ps, ev_ebitda, ev_ebit, ev_sales, fcf_yield, dividend_yield, pe_fy1, pe_fy2, ev_ebitda_fy1, ev_sales_fy1, ltg, peg, and `<multiple>_pct_gind` (percentile rank within the GICS industry that month) | s07 | RV screen |
| rv_peer_stats_monthly.parquet | gind, month end | date, gind, gind_name, n_peers, `<multiple>_p25`, `<multiple>_p50`, `<multiple>_p75` | s07 | RV peer comparison |
| **DCF** | | | | |
| dcf_inputs_annual.parquet | company, fiscal year | permno, gvkey, fyear, period_end, report_date (Q4 rdq rule), revenue, ebitda, ebit, da, capex, oancf, d_nwc, tax_rate, tax_rate_fcff, statutory_tax_rate, interest_expense, interest_source, total_debt, cash, net_debt, preferred, minority_interest, diluted_shares, basic_shares, book_equity, dividends, buybacks, share_issuance, fcff, revenue_growth, ebitda_growth, GICS codes, dcf_applicable | s04 | DCF valuation |
| estimates_consensus.parquet | permno, statpers, measure, fpi | permno, ibes_ticker, statpers, measure (EPS, SAL, EBT, EBI, CPX, CPS, DPS), fpi (0-3), fpedats, numest, numup, numdown, meanest, medest, stdev, highest, lowest, actual, anndats_act (USD only), price_statpers, cfacpr_statpers, cfacpr_last, crsp_basis_factor, suspect_total | s05 | DCF forecasts, forward multiples |
| **Reference** | | | | |
| fundamentals_quarterly.parquet | permno, fiscal quarter | full quarterly statements, building blocks and the quarter-end factors | s03 | Data Viewer, factor research |
| security_master.parquet | permno | permno, permco, gvkey, ibes_ticker, ticker, security_name (latest), gsector, gsector_name, ggroup, gind, gind_name, gsubind, siccd, first_date, last_date | s07 | ticker and name lookup, holdings table |
| manifest.json, DATA_DICTIONARY.md | - | data_version, source sha256s, per-file rows, dates, columns, units, known gaps; every column with meaning, unit and source | s08 | data loading checks, documentation |

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

## Decisions

Decisions 1-10 are from the first review, 11-15 from the second, 16-20 from the fourth (all 2026-10-25).

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
| 10 | Annual report date = the rdq of the matching quarterly Q4 row (same gvkey and fiscal year, and, added after the first run, the same period end), otherwise datadate + 90 days. | s04 rule 2 |
| 11 | Non-member stocks from mergers (revised in the third review: the first version also excluded real members): a stock-snapshot is excluded from the universe if (a) its month-end market cap is more than 3x the largest among the other stocks in that snapshot **and** its weight is below 0.05%, or (b) the stock appears in 3 or fewer snapshots in total, all with weight 0.00%, **and** its market cap is above that snapshot's 95th percentile. Their data is kept; they are only excluded from the universe, and listed in processed/s01_membership/excluded_non_members.parquet and the s01 summary. | s01 rule 6 |
| 12 | Missing share classes: s07 compares the company market cap from our permnos with all permnos sharing a permco, by year, and flags a gap above 1% of total universe market cap in any year. No new pull now. | s07 checks |
| 13 | Interest expense is never filled with 0. The DCF uses annual xint, else xintd; where both are missing, the cost of debt falls back to the FRED Baa yield. | s03, s04 rule 6 |
| 14 | Prices below $0.01: price-based ratios are missing for those rows, and the rows are counted. | s03 rule 5, s07 rule 5 |
| 15 | IBES forward E/P uses the price on the same restated basis as the estimate; its 0.1% and 99.9% values are reported as a check. | s05 rule 7, s07 rule 8 |
| 16 | GICS industry 601020 = Real Estate Management & Development (code used 2016-2023, renumbered 602010 in 2023), sector 60. | config.GICS_INDUSTRIES |
| 17 | FCFF uses tax_rate_fcff: the effective tax rate, or 0% when pretax income <= 0 (no tax on a loss); tax_rate stays the reported effective rate. s04 adds statutory_tax_rate (35% up to fiscal 2017, 21% from 2018) and dcf_applicable (False for GICS sectors 40 Financials and 60 Real Estate). | s04 rules 5-6 |
| 18 | Near-zero denominators in s03: f_sales_growth_1yr needs prior TTM sales >= $1M, f_rd_to_sales current TTM sales >= $1M, f_capex_growth_1yr prior TTM capex >= $1M, f_sue a std of EPS changes >= $0.01; otherwise NaN. | s03 rule 7 |
| 19 | The s04 annual vs quarterly revenue check is informational, threshold 90%. | s04 checks |
| 20 | s05 outputs estimates_consensus with the CRSP basis factor (the analyst factors move to s07); s06 is renamed s06_market and adds term and credit spreads and holiday-filled yields. | s05, s06 |

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
  pulled. RBC is 87 of the 402 CAD rows (total assets up to $2.5 trillion). Handled by decision 11.

---

## s01_membership

**Inputs:** `raw/iwm_holdings_raw.csv`, `raw/r2000_price_crsp_ciz.csv` (permno, date, flag, price, shares only,
for rule 6), `raw/query_inputs/permno_list.txt` (only to compare).

**Outputs** (`processed/s01_membership/`):

| file | one row per | columns |
|---|---|---|
| snapshots.parquet | report_dt, permno | report_dt, eff_dt, permno, weight, ticker, security_name, cusip, n_lines, permno_from_cusip, mkt_cap_musd, max_other_mkt_cap_musd, n_snapshots_total, exclusion_rule, in_universe |
| periods.parquet | membership period | permno, index_effective, index_through, first_known, n_snapshots, n_excluded, excluded_non_member, ticker, security_name |
| excluded_non_members.parquet | excluded stock-snapshot | report_dt, permno, ticker, security_name, mkt_cap_musd, max_other_mkt_cap_musd, weight, n_snapshots_total, rule (`a`, `b` or `a+b`) |
| known_gaps.parquet, known_gaps.md | security without a permno | cusip, security_name, ticker, first, last, snapshots, max_weight |
| permno_list.txt | permno | the upload list for queries 2, 3, 6, 9, 11 (every snapshot permno, excluded ones too, since their data is kept) |

**Rules** (ported):
1. A holdings line without a permno gets the permno of the same CUSIP on other report dates (the most common one)
   (audit A8). Last run: 1,569 lines, 130 securities, 0 CUSIPs linked to two permnos.
2. Lines that still have no permno, have a CUSIP and a ticker, and whose name does not match the non-stock pattern
   (cash, futures, swaps, rights, CVRs, notes, ...; the regex is ported unchanged into config) are the known gaps.
   No extra WRDS pull (A8). Last run: 72 securities.
3. One row per stock per snapshot: share-class lines are summed (`weight` = sum of percent_tna), `eff_dt` = the
   earliest line. **Change:** ticker, security_name and cusip come from the line with the largest weight, then the
   most shares, instead of the last line in file order, so the result does not depend on row order. (Equal weights
   happen when a SPAC and its new name are both listed, e.g. V G Acquisition / 23andMe on 2022-08-31; the share
   tie-break picks the new name, as the notebook did.)
4. Membership periods: a stock missing from 2 or more consecutive snapshots starts a new period; one missing
   snapshot is bridged (A7). `index_effective` and `index_through` are report_dt values; `first_known` = the
   period's first eff_dt, when that report became public (A5).
5. permno_list.txt = every permno in the snapshots, sorted. It is written to processed/ only; you copy it to
   raw/query_inputs/ before the permno-based queries (decision 7; the pipeline never writes raw/).
6. **Non-member stocks from mergers** (decision 11, revised). A stock-snapshot gets `in_universe` = False if
   (a) its month-end market cap is more than 3x the largest market cap among the other stocks in that snapshot and
   its weight is below 0.05%, or
   (b) the stock is in 3 or fewer snapshots in total, its weight is 0.00% in all of them (rounded to 2 decimals, as
   IWM reports it), and its market cap is above that snapshot's 95th percentile (`p95_mkt_cap_musd`).
   The data of these stocks is kept everywhere; only the universe drops them. The weight condition in (a) keeps a
   real member that grew large (Super Micro Computer, 2024-02, weight 1.6%); the size condition in (b) keeps small
   members that were in the index briefly or are in the last snapshots of the data. Today: 11 stock-snapshots,
   10 stocks (RBC, NRG, ASML, LinnCo, Martin Marietta, Orbital ATK, Alexion, Salesforce, Exxon, Kimco).
   - **Where rule (a) runs.** The month-end market cap is s02's: the last real trading day of the report_dt's
     month (DlyDelFlg = N), |DlyPrc| x ShrOut. s01 cannot read s02's output, because s01 must run first: its
     permno_list.txt is the upload list for the CIZ query that s02 cleans. So s01 reads the four needed CIZ
     columns from raw/ and calls the same function s02 uses (`s02_prices.month_end_market_caps`), so both stages
     give identical market caps. During a refresh, the first s01 run (before query 2 is re-pulled) cannot price
     new stocks. Those snapshots are counted (WARN), and the `--stage all` run after the pull applies the rule
     fully.
   - Rule (a) compares each stock with the others in the same snapshot, so only a snapshot's largest stock can
     trip it. A stock without a CIZ market cap is never excluded by rule (a).
   - `periods.parquet` keeps the excluded stocks' periods with `excluded_non_member` = True when every snapshot of
     the period is excluded; s07 builds the final universe_periods from the in-universe snapshots.

**Checks:** no duplicate (report_dt, permno) (FAIL); eff_dt >= report_dt (FAIL); no CUSIP maps to two permnos
(WARN, list them); in-universe stocks per snapshot 1,800-2,150 (WARN; last run 1,831-2,108); weight sum per
snapshot 90-110% (WARN; 2014-09-30 is 108%, 2014-10-31 93.5%, known); missing months are exactly 2009-04, 2009-05,
2009-06 (WARN on any new one); stocks missing from a snapshot but in both neighbours, max (WARN above 20; last run
14); permno_list.txt equals raw/query_inputs/permno_list.txt (WARN: the permno-based queries must be rerun);
snapshot stocks without a CIZ market cap (WARN if more than 1%); excluded stock-snapshots listed in the summary
(name, ticker, dates, market cap, weight, rule); counts within 2% of the last run (WARN). Last run: 218
snapshots, 428,354 rows, 5,918 permnos, 7,576 periods.

**Tests:** CUSIP fill (including a CUSIP with no permno anywhere -> gap); non-stock names are not gaps; share
classes summed; ticker from the largest line regardless of row order; a one-snapshot absence is bridged and a
two-snapshot absence splits the period; first_known = min eff_dt of the period; rule (a) excludes a stock at
3.1x the next largest with weight 0.04%, but not at 2.9x or with weight 0.05%; rule (b) excludes a large stock in
3 zero-weight snapshots, but not a small one, one with 4 snapshots, or one with a non-zero weight; ties in weight
broken on shares; reruns give identical files.

## s02_prices

**Inputs:** `raw/r2000_price_crsp_ciz.csv` (columns permno, permco, dlycaldt, ticker, siccd, dlydelflg,
delreasontype, dlyprc, dlyret, dlyvol, shrout, dlyopen, dlyhigh, dlylow, dlycumfacpr), `raw/delistings.csv`.

**Outputs** (`processed/s02_prices/`):

| file | one row per | columns |
|---|---|---|
| daily/year=YYYY/data.parquet | permno, trading day | date, permno, permco, ticker, siccd, prc, prc_adj, cfacpr, shrout, mkt_cap, mkt_cap_musd, ret, vol, is_delisting_row, delist_ret_filled, mom_1m, mom_6m, mom_12m, vol_gk_{1m,3m,6m,12m}, vol_close_{1m,3m,6m,12m} |
| monthly.parquet | permno, month | month_end, permno, permco, ticker, siccd, last_trade_date, prc, shrout, mkt_cap_musd, mom_*, vol_*, ret_m, trading_days, delisted_in_month, delist_ret_filled (s08 exports returns_monthly from it) |
| calendar.parquet | month | month, month_end, trading_days |
| delistings.parquet | delisting | permno, delisting_date, flagged_row_date, flagged_row_prc, delactiontype, delreasontype, delpaymenttype, delret, delret_used |

**Rules** (ported):
1. Read the needed columns with pyarrow (about 2 GB in memory). Rolling windows run per permno, so the stage
   processes permnos in chunks if memory is short.
2. Drop exact duplicate rows (B2; last run 1,801, 439 of them flag rows). If two rows share permno and date but
   differ, FAIL (none today).
3. `prc` = |DlyPrc| (negative = bid/ask midpoint); 0 = missing. `ret` < -1 = missing code. `vol` < 0 and
   `shrout` <= 0 = missing. `mkt_cap` = prc x shrout ($ thousands), `prc_adj` = prc / cfacpr. On delisting rows
   prc, prc_adj and mkt_cap are set missing in the daily file too (decision 9), so no reader can take a price from
   them; the 774 post-delisting prices CIZ gives are kept in delistings.parquet as `flagged_row_prc`.
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
falls in exactly one month (FAIL); fills = blank delret in delistings.csv (FAIL); the flag row's return equals
delistings.csv delret where both exist (WARN below 99%); ret_m equals the compounded daily
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

**Output** (`processed/s03_fundamentals_q/fundamentals_quarterly.parquet`, exported as fundamentals_quarterly), one
row per permno per fiscal quarter:

- **Ids and timing:** permno, gvkey, tic, conm, period_end (datadate), report_date, report_date_source (`rdq` /
  `proxy`), rdq, fyearq, fqtr, curcdq, is_usd, ajexq.
- **Industry (decision 4):** gsector, gsector_name, ggroup, gind, gind_name, gsubind, naics. Names come from the
  GICS tables in config. Compustat's GICS is each company's latest classification, so inactive companies keep
  discontinued codes (e.g. 254010 Media before 2018, 404020 REITs before 2016, 601020 Real Estate Management &
  Development 2016-2023); they keep their last name.
- **Quarter-end market data:** price_qe (prccq), cshoq, mkt_cap_qe_musd (prccq x cshoq; missing for non-USD rows).
- **As reported:** balance sheet (atq, ltq, actq, lctq, cheq, invtq, rectq, apq, ppentq, gdwlq, intanq, dlcq,
  dlttq, pstkq, mibtq, seqq, ceqq, txditcq), income (saleq, revtq, cogsq, xsgaq, xrdq, oibdpq, dpq, oiadpq, xintq,
  piq, txtq, niq, ibq, epsfxq, epspxq, dvpsxq) and the quarterly cash flows from the YTD items (oancf_q, capx_q,
  dv_q, sstk_q, prstkc_q, dltis_q, dltr_q, aqc_q).
- **RV and DCF building blocks:**

| column | rule |
|---|---|
| revenue_ttm, revt_ttm, cogs_ttm, xsga_ttm, xrd_ttm | saleq, revtq, cogsq, xsgaq, xrdq over 4 quarters |
| ebitda_ttm, ebit_ttm, da_ttm | oibdpq, oiadpq, dpq over 4 quarters |
| interest_ttm | xintq over 4 quarters, never 0-filled (decision 13): missing when any of the 4 is missing (21% of xintq rows are) |
| pretax_ttm, tax_ttm, ni_ttm, ib_ttm | piq, txtq, niq, ibq over 4 quarters |
| tax_rate_ttm | tax_ttm / pretax_ttm, only when pretax_ttm > 0, clipped to 0-60% |
| oancf_ttm, capx_ttm, dv_ttm, sstk_ttm, prstkc_ttm, dltis_ttm, dltr_ttm, aqc_ttm | the quarterly cash flows over 4 quarters |
| fcf_ttm | oancf_ttm - capx_ttm |
| book_equity | seqq (else ceqq + pstkq, else atq - ltq) + txditcq - pstkq (Fama-French; missing parts 0) |
| total_debt, cash, net_debt | dlttq + dlcq (missing parts 0); cheq; total_debt - cheq |
| preferred, minority_interest, ev_claims | pstkq; mibtq; net_debt + preferred + minority interest (missing parts 0). EV = market cap + ev_claims |
| nwc, d_nwc_4q | (actq - cheq) - (lctq - dlcq); its change vs exactly 4 quarters earlier |
| epsfxq_ttm_adj, dvpsxq_ttm_adj | diluted EPS and dividends per share: each quarter / ajexq, summed, x this quarter's ajexq (D4) |
| diluted_shares, csho_adj, cshfd_adj | cshfdq; cshoq x ajexq; cshfdq x ajexq (comparable across splits) |

- **The 17 factors at quarter end** (ported unchanged from the notebook): f_book_to_price, f_earnings_to_price,
  f_sales_to_price, f_ebitda_to_ev (EV = market cap + net debt, as the notebook), f_dividend_yield, f_roe, f_roa,
  f_gross_profitability, f_accruals_to_assets, f_leverage, f_current_ratio, f_rd_to_sales (R&D missing = 0),
  f_asset_growth_1yr, f_sales_growth_1yr, f_capex_growth_1yr, f_net_share_issuance, f_sue. s07 recomputes the
  price-scaled five at month end.

**Rules:**
1. Keep rows with permno, datadate, fyearq and fqtr. One row per (permno, fyearq, fqtr): the most complete row,
   then the latest rdq (81 removed; they come from fiscal-year changes).
2. `report_date` = rdq; if missing (725 rows, 0.3%), datadate + 45 days for fiscal Q1-Q3 and + 90 days for Q4.
3. YTD cash flows to quarterly: Q1 = YTD; later quarters = YTD - the previous quarter's YTD of the same fiscal
   year, else missing. **TTM = exactly 4 consecutive fiscal quarters** (index fyearq x 4 + fqtr); one missing
   quarter makes the sum missing. Growth and changes use exactly 4 quarters earlier.
4. Split adjustment with ajexq for EPS, dividends per share and share counts (D4). SUE = (split-adjusted basic EPS
   minus the same 4 quarters earlier) / std of the previous 8 such changes, at least 6 of them.
5. Impossible values to NaN: sales or capex growth below -100%, negative R&D ratio (183, 625, 18 rows); inf to
   NaN. Other values stay as reported and are counted (negative saleq 389, negative dpq 35, zero cshoq 1, zero
   atq 1).
6. **Non-USD rows** (402 rows, 19 permnos; decision 2) and **prices below $0.01** (24 rows; decision 14): the 5
   price-based factors are missing; non-USD rows also have no mkt_cap_qe_musd. Statement ratios are kept.
7. **Near-zero denominators** (decision 18): f_sales_growth_1yr needs prior TTM sales >= $1M, f_rd_to_sales
   current TTM sales >= $1M, f_capex_growth_1yr prior TTM capex >= $1M, f_sue a std of the previous 8 EPS
   changes >= $0.01. Before -> after (all rows): sales growth mean 1.31 -> 0.20 (p99 4.18 -> 2.64), R&D to sales
   10.4 -> 0.48 (p99 46.1 -> 9.40), capex growth 4.3e12 -> 0.26 (p99 10.9 -> 4.53), SUE -9.6e10 -> -1,096
   (p1/p99 unchanged at -7.4/4.7). Capex growth coverage 2015+ drops from 78% to 66%. SUE keeps a few huge
   values (min -2.2e8) from huge EPS changes, not from the denominator.

**Checks:** unique (permno, fyearq, fqtr) (FAIL); duplicates and rows vs the notebook rerun (WARN); report_date
on or after period_end (WARN); rdq proxy at most 2% (WARN); no price-based ratio on non-USD or sub-$0.01 rows
(FAIL); ajexq > 0 (FAIL); tax rate within 0-60% (FAIL); interest_ttm present exactly where all 4 xintq are
(FAIL); revenue_ttm equals an independent sum of 4 fiscal quarters for 200 random rows (FAIL); factor coverage
2015+ within 5 points of the notebook (WARN; capex growth expected at 66% since decision 18); every gind has a name (WARN). The summary has count / missing % /
mean / p1 / p50 / p99 for every factor and building block, and coverage overall and for 2015+.

**Tests:** a missing quarter makes every TTM window that needs it missing; interest never 0-filled; YTD to
quarterly including Q1 and a missing previous quarter; growth vs exactly 4 quarters earlier (missing when that
quarter is missing); a 2-for-1 split keeps EPS and dividends TTM in current shares and share counts comparable;
tax rate rule (normal, clipped at 60%, negative taxes to 0, pretax <= 0 missing); non-USD and sub-$0.01 rows lose
only the price ratios; duplicate quarters keep the most complete row; rdq proxy +45 / +90; book equity fallback;
SUE against a hand value; an end-to-end run.

## s04_fundamentals_a

**Inputs:** `raw/fundamentals_annual_raw.csv`, `processed/s03_fundamentals_q/fundamentals_quarterly.parquet` (Q4
rdq, and TTM revenue for a cross-check).

**Output** (`processed/s04_fundamentals_a/dcf_inputs_annual.parquet`, exported as dcf_inputs_annual), one row per
permno per fiscal year:

- **Ids and timing:** permno, gvkey, tic, conm, fyear, fyr, period_end, report_date, report_date_source
  (`q4_rdq` / `proxy`), curcd, is_usd, ajex, price_fye (prcc_f), epsfx, emp, sich, GICS codes and names.
- **DCF inputs:** revenue (sale), ebitda (oibdp), ebit (oiadp), da (dp), capex (capx), oancf, pretax_income (pi),
  taxes (txt), tax_rate, interest_expense and interest_source, net_income (ni), nwc and d_nwc, total_debt, cash
  (che), net_debt, preferred (pstk), minority_interest (mib), diluted_shares (cshfd), basic_shares (csho),
  book_equity, dividends (dvc, common) and dividends_total (dv), buybacks (prstkc), share_issuance (sstk), fcff,
  revenue_growth, ebitda_growth.

**Rules:**
1. One row per (permno, fyear): the most complete row, then the latest datadate (1 duplicate today).
2. **report_date** (decision 10): the rdq of the quarterly Q4 row with the same gvkey and fiscal year **that ends
   on the same date as the annual row**, otherwise datadate + 90 days. The same-date condition was added after the
   first run: after a fiscal-year-end change (TuHURA Biosciences 2024, Grubb & Ellis 2007, June to December) the
   Q4 of the same fiscal year is the old June quarter, and its rdq predated the annual period end, a look-ahead.
   Today 63,034 rows use the Q4 rdq and 2,325 the proxy (2 rejected for that reason).
3. Changes and growth only vs exactly fyear - 1; a missing or changed year gives missing values. Revenue and EBITDA
   growth need a positive previous value.
4. **Interest expense** (decision 13): xint, else xintd, never 0-filled. `interest_source` = xint (54,069) /
   xintd (2,019) / missing (9,271). Where it is missing, the DCF's cost of debt falls back to the FRED Baa yield.
5. tax_rate = txt / pi, only when pi > 0, clipped to 0-60% (as s03; the reported effective rate).
   **tax_rate_fcff** = tax_rate, or 0% when pi <= 0 (no tax on a loss; decision 17). statutory_tax_rate = 35%
   for fiscal years up to 2017, 21% from 2018. nwc = (act - che) - (lct - dlc); total debt, cash, net debt and
   book equity as in s03.
6. **fcff** = ebit x (1 - tax_rate_fcff) + da - capex - d_nwc, missing if any part is missing.
   **dcf_applicable** = False for GICS sectors 40 (Financials) and 60 (Real Estate), else True (50,063 of 65,359
   years). FCFF coverage: 40% of all years with the reported tax rate, now 70% of all years and 89% of
   DCF-applicable years; the rest misses d_nwc (first years and gaps) or D&A.
7. **Own table** (decision 6): not merged into the monthly panel. DCF code joins it by permno with report_date <=
   the valuation date.

**Checks:** unique (permno, fyear) (FAIL); report_date on or after period_end (WARN); proxy report dates at most
10% (WARN); tax rate within 0-60% (FAIL); interest never filled where xint and xintd are missing (FAIL); FCFF
present exactly when all its parts are (FAIL); annual revenue within 1% of the quarterly TTM at the matching Q4
for at least 90% of years (informational, decision 19; 94.8% today, 96.9% within 5%: mostly restatements, since
Compustat restates annual figures and keeps quarters as first reported). The summary has the count / missing % / mean / p1 / p50 /
p99 of every column, the report date and interest source counts, and coverage.

**Tests:** report date from the matching Q4 rdq, proxy when the Q4 ends on another date or is missing; FCFF against
a hand value and missing when a part is missing; 0% tax in a loss year; statutory rate and dcf_applicable; changes only vs exactly the previous fiscal year; interest xint,
then xintd, never 0; tax rate rule; duplicate years keep the most complete row; an end-to-end run.

## s05_ibes

**Inputs:** `raw/forward_estimates.csv` (1.1 GB, needed columns only), `raw/crsp_ibes_link.csv`,
`processed/s01_membership/snapshots.parquet` (our permnos), `processed/s02_prices/daily/` (date, permno, prc,
cfacpr), `raw/query_inputs/ibes_tickers.txt` (only to compare).

**Outputs** (`processed/s05_ibes/`):

| file | one row per | columns |
|---|---|---|
| estimates_consensus.parquet | permno, statpers, measure, fpi | permno, ibes_ticker, statpers, measure, fpi, fpedats, numest, numup, numdown, meanest, medest, stdev, highest, lowest, actual, anndats_act, price_statpers, cfacpr_statpers, cfacpr_last, crsp_basis_factor, suspect_total |
| link.parquet | link | ticker, permno, ncusip, sdate, edate, edate_used, score |
| ibes_tickers.txt | IBES ticker | the upload list for query 10 (decision 7) |

`actual` and `anndats_act` are future information at statpers: s07 may use an actual only when anndats_act <= t.

**Rules:**
1. Keep CURCODE = USD (27,133 rows dropped) and FPI 0-3 (decision 1).
2. If IBES gives two different actuals for one (ticker, measure, fpedats), the actual is blanked (not trusted for
   surprises; ported).
3. Duplicate (ticker, statpers, measure, fpi) rows: keep the one with more analysts (numest), then the first in the
   file (634 dropped).
4. **Link** (ported): crsp_ibes_link.csv with scores 1-2, our permnos only; a row gets a permno when sdate <=
   statpers <= edate. Each permno's latest link (latest sdate) is open-ended past its edate, because WRDS ends active
   links at the table's last update (2025-12-31) (H2). If two tickers give the same permno, statpers, measure and
   fpi, the better link (score 1) wins, then more analysts (0 cases today). One IBES ticker can link to two
   permnos (share classes), so there are slightly more linked rows (10,061,304) than raw rows.
5. **Units:** SAL, EBT, EBI, CPX are totals in $ millions; EPS, CPS, DPS are $ per share; FPI 0 (long-term
   growth) is converted from percent to decimal (meanest, medest, stdev, highest, lowest / 100).
6. **100x rule:** a SAL, EBT, EBI or CPX mean estimate (FPI 1-3) above 100 times the ticker's own median for that
   measure is set to NaN (all 5 values) and flagged `suspect_total`. See open question 1: today it flags 6,494
   rows, mostly real pre-revenue companies.
7. **Share basis.** CRSP's cumulative price factor `cfacpr` falls at each split (GameStop's 4-for-1 on 2022-07-22:
   4.05 before, 1.01 after) and is 1.0 at the end of the data, so `prc / cfacpr` is a price in end-of-data shares.
   IBES summary values are restated to the latest share basis too (GameStop's FY1 EPS on 2022-07-14, before the
   split, is -0.92, a post-split number). So:
   - `cfacpr_statpers` = CRSP cfacpr on the last trading day with a price on or before statpers (at most 7 days
     earlier), `price_statpers` = that day's price; `cfacpr_last` = the permno's last cfacpr in the data.
   - `crsp_basis_factor` = cfacpr_statpers / cfacpr_last. **An IBES per-share value x crsp_basis_factor is in
     statpers shares**, the same basis as price_statpers. Equivalently, price_statpers / crsp_basis_factor is the
     price on the IBES basis (the notebook's `price_ibes_basis`).
   - At a month end t in s07: per-share estimate on the basis of prc(t) = meanest x cfacpr(t) / cfacpr_last.
     Forward E/P = meanest x cfacpr(t) / cfacpr_last / prc(t). Totals need no factor.
   - Caveat: cfacpr_last is CRSP's last day (2026-09-30), IBES's basis is its pull date. A split between the two
     would be missed; pull both together.
8. The analyst factors (forward E/P, revision, dispersion, surprise, and the yields of decision 5) are computed in
   s07 from this table, at the month end, so they use the month-end price and market cap (decision 15 included).

**Checks:** only USD and FPI 0-3 (FAIL); unique (permno, statpers, measure, fpi) (FAIL); every measure present
(FAIL); every row inside one of its links' date windows (FAIL); long-term growth in decimals (EPS FPI 0 median
0.03-0.30) (FAIL); suspect totals under 0.01% (WARN); CRSP price on statpers for 90%+ of EPS FY1 rows (WARN);
crsp_basis_factor > 0 (FAIL); EPS FY1 rows and permnos vs the notebook rerun (WARN); ibes_tickers.txt vs
raw/query_inputs (WARN). The summary has rows and coverage by measure and FPI, linked rows and permnos by year, and
count / missing % / mean / p1 / p50 / p99 of meanest for every measure and FPI.

**Tests:** USD and FPI filters; duplicates keep more analysts; conflicting actuals blanked; link date windows,
score filter, our permnos only, the latest link open-ended; two tickers for one permno keep the better link;
long-term growth to decimals; the 100x rule; crsp_basis_factor across a 4-for-1 split and the 7-day price rule.

## s06_market

**Inputs:** `raw/index_close.csv`, `raw/benchmark_etfs.csv`, `raw/ff_daily.csv`, `raw/ff_monthly.csv`,
`raw/fred_DGS10.csv`, `raw/fred_DGS3MO.csv`, `raw/fred_DBAA.csv`, `raw/fred_DAAA.csv`,
`processed/s02_prices/calendar.parquet` (month ends).

**Outputs** (`processed/s06_market/`, exported as market_daily and market_monthly):

| file | one row per | columns |
|---|---|---|
| market_daily.parquet | date (trading day or FRED business day, from 2006-12-04) | date, r2000_price_ret, r2000_growth_price_ret, r2000_value_price_ret, iwm_total_ret, iwn_total_ret, iwo_total_ret, mktrf, smb, hml, rmw, cma, umd, rf, ust_10y, tbill_3m, moodys_baa, moodys_aaa, term_spread, credit_spread, was_filled |
| market_monthly.parquet | month of the CRSP calendar (2007-01 on) | month, month_end, the same return, yield and spread columns |

**Rules:**
1. Index levels (Compustat I0124 Russell 2000, I0120 Growth, I0122 Value): 2012-10-29 and 2012-10-30 dropped
   (Hurricane Sandy, no CRSP rows), so the 2012-10-31 return spans 10-26 to 10-31. Price return from consecutive
   levels; monthly = the level on the last day <= month_end vs the previous month's.
2. ETFs (CRSP IWM 88222, IWN 88399, IWO 88401): daily total returns as CRSP gives them; monthly = compounded.
3. Fama-French: daily as given (decimals); monthly = the published monthly factors, matched by the month of dateff.
4. FRED: percent / 100. Blank days (bond-market holidays such as Columbus and Veterans Day, when stocks trade) are
   forward-filled, and `was_filled` = True on those days (219 today). term_spread = ust_10y - tbill_3m,
   credit_spread = moodys_baa - moodys_aaa. Monthly = the value on the last day <= month_end.
5. The first month (no previous index level) is dropped. Each monthly row is that month's return; the panel's
   ret_next_1m at month t lines up with month t+1.

**Checks:** market-closed days have no index return (FAIL); yields in range: ust_10y 0-6%, tbill_3m -0.5-7%,
moodys_baa 2-10%, moodys_aaa 1-9% (FAIL); Fama-French daily rf mean below 0.001 (FAIL: units); IWM total minus
Russell 2000 price return 0.5-2.0% a year (WARN; 1.34% today); published monthly Fama-French within 50 bp of the
compounded daily factors (WARN; see the phase 2c results: true for mktrf and rf, not for the long-short factors);
no yield filled for more than 5 days in a row (WARN); every month present (FAIL). The summary has the end date of
every source, coverage, and count / missing % / mean / p1 / p50 / p99 of every column, daily and monthly.

**Tests:** Sandy days dropped and the next return spans them; percent to decimal; a holiday forward-filled and
flagged; spreads; first month dropped; ETF and Fama-French monthly compounding; the index monthly return from
month-end levels; yields at the month end.

## s07_panel

**Inputs:** s01 snapshots and periods, s02 monthly + calendar + daily (cfacpr, ret), s03 fundamentals_quarterly,
s05 estimates_consensus, s06 market_daily and market_monthly (Fama-French market for betas). Not s04: the annual
statements stay their own table (decision 6).

**Outputs** (`processed/s07_panel/`):
- **panel.parquet**, one row per index stock per month_end: date, permno, permco, ticker, security_name, sector
  (GICS sector name), gsector, gind, gind_name, sector_source (`gics` / `sic`), siccd, is_fossil_fuel,
  index_weight, snapshot_report_dt, snapshot_eff_dt, new_entrant, has_price, price_date, prc, shrout,
  mkt_cap_musd, company_mkt_cap_musd, n_share_classes, mom_1m, mom_6m, mom_12m, mom_12_1, vol_*, fund_period_end,
  fund_report_date, fund_is_usd, the 17 f_* fundamentals (price-scaled five at month end, `_qe` versions kept),
  the RV building blocks used by the app (book_equity, revenue_ttm, ebitda_ttm, ebit_ttm, net_debt, ev_claims,
  fcf_ttm, epsfx_ttm_adj), ibes_statpers, numest, f_forward_ep, f_eps_revision, f_eps_dispersion,
  f_analyst_surprise, surprise_anndats, the new analyst yields f_forward_sales_yield, f_forward_ebitda_ev,
  f_cash_flow_yield, f_fy2_ep, f_ltg, beta_252d, beta_60m, ret_next_1m, delisted_next_month. (61 columns last
  run, plus the additions.) Exported as factor_panel_monthly.
- **rv_multiples.parquet** and **rv_peer_stats.parquet** (rules 10-11), exported as rv_multiples_monthly and
  rv_peer_stats_monthly.
- **universe_periods.parquet**: s01's periods rebuilt from the in-universe snapshots, plus the excluded stocks'
  periods with excluded_non_member = True.
- **security_master.parquet**: one row per permno (rule 12).
- The monthly returns table is s02's monthly.parquet; s08 exports it as returns_monthly.

**Rules** (ported):
1. Rebalance dates = calendar month_ends from the first month with an effective snapshot (2008-04) to the last
   month with prices.
2. Universe at t = the latest snapshot whose every line has eff_dt <= t (2009-04 uses 2009-02, 2009-05 and 2009-06
   use 2009-03), keeping only lines with in_universe = True (decision 11). index_weight = snapshot weight;
   new_entrant = not in the universe at the previous month end.
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
   - **Month-end price below $0.01** (decision 14): all price-based ratios and multiples are NaN; counted.
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
   old, all USD. Per-share estimates are put on the basis of prc(t) with cfacpr(t) / cfacpr_last (s05 rule 7);
   price_ibes_basis(t) = prc(t) x cfacpr_last / cfacpr(t) is the same thing seen from the price side:

   | column | definition |
   |---|---|
   | f_forward_sales_yield | SAL FY1 mean / company_mkt_cap_musd |
   | f_forward_ebitda_ev | EBT FY1 mean / (company_mkt_cap_musd + ev_claims), NaN if EV <= 0 or the fundamentals are not USD |
   | f_cash_flow_yield | CPS FY1 mean / price_ibes_basis(t) |
   | f_fy2_ep | EPS FY2 mean / price_ibes_basis(t) |
   | f_ltg | EPS FPI 0 mean / 100 (not price-scaled) |

   The ported analyst factors are computed here from estimates_consensus: f_forward_ep = EPS FY1 / price on the
   IBES basis (at statpers, as the notebook), with its 0.1% and 99.9% values in the summary (decision 15);
   f_eps_revision, f_eps_dispersion and f_analyst_surprise as in the notebook, using actual only when
   anndats_act <= t.
9. **Betas** vs the Fama-French market (mktrf + rf, excess returns on both sides): `beta_252d` from the last 252
   daily returns up to t, `beta_60m` from the last 60 monthly returns up to t. Each needs 80% of the observations
   (202 days, 48 months), else NaN. Delisting returns are included as they occur.
10. **RV multiples at t** (rv_multiples): prc, company_mkt_cap_musd, ev_musd = company mkt cap + total debt +
    preferred + minority interest - cash (from the fundamentals used at t); trailing pe_ttm (price / TTM diluted
    EPS restated to month-end shares), pb, ps, ev_ebitda, ev_ebit, ev_sales, fcf_yield (fcf_ttm / company mkt cap),
    dividend_yield; forward pe_fy1 and pe_fy2 (price_ibes_basis / EPS FY1, FY2), ev_ebitda_fy1 (EV / EBT FY1),
    ev_sales_fy1 (EV / SAL FY1); ltg (EPS FPI 0 / 100) and peg = pe_fy1 / (ltg x 100). A multiple with a
    denominator <= 0 is NaN (a negative P/E is not ranked). Non-USD fundamentals and prices below $0.01 give NaN.
    `<multiple>_pct_gind` = percentile rank (0-1) of the multiple among the stocks of the same gind that month, NaN
    when the industry has fewer than 5 values.
11. **RV peer stats** (rv_peer_stats): per gind and month, n_peers and the 25th, 50th and 75th percentile of each
    multiple over the stocks with a value.
12. **security_master**: one row per permno in s02: permco, gvkey (latest), ibes_ticker (latest link), latest
    ticker and security_name, GICS codes and names, latest siccd, first_date and last_date in the daily data.

**Checks (asserts, FAIL):** no fundamentals with report_date > t; no IBES with statpers > t or older than 60 days;
no surprise announced after t; no snapshot with eff_dt > t; no price_date > t; no duplicate (date, permno).
**FAIL:** no price-based ratio on a non-USD fundamentals row or a price below $0.01; company_mkt_cap_musd >=
mkt_cap_musd; every row has a sector; excluded non-member stock-snapshots are not in the panel.
**Missing share classes** (decision 12): the CIZ file only holds our permnos, so the permnos we did not pull are
not visible in it. The check uses Compustat's company market cap instead (prccq x cshoq, all common shares of the
company), at each quarter end: the gap = Compustat company market cap - the summed CRSP market cap of our permnos
of that permco on the same date, where positive and above 5% of that company (smaller gaps are timing and share
count noise). By year, the sum of the gaps over the universe vs the total universe market cap; WARN above 1% in
any year, listing the largest companies. No new pull now. **WARN:** share of rows on the SIC fallback above 2%; percentiles of each multiple reported, and
any multiple whose median moves more than 50% from one year to the next; stocks per month outside 1,800-2,100
(last run 1,831-2,043); coverage by year of price, roa,
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
against a hand value; ret_next_1m includes a delisting return in month t+1; beta on a synthetic series with a
known beta, NaN below 80% of observations; EV with each part missing; negative-earnings P/E is NaN; gind percentile
ranks and peer percentiles on a hand-checked industry; an excluded non-member is not in the panel but is in
universe_periods with the flag.

## s08_export

**Inputs:** the outputs of s01-s07, and `processed/s00_inventory/inventory.json` (raw file checksums, so the 4 GB
is not hashed twice).

**Outputs** (`processed/final/`, s08's stage folder): exactly the files in [Final outputs](#final-outputs):
factor_panel_monthly, universe_periods, returns_monthly, prices_daily/year=YYYY/data.parquet (+ _manifest.json),
market_daily, market_monthly, rv_multiples_monthly, rv_peer_stats_monthly, dcf_inputs_annual,
estimates_consensus, fundamentals_quarterly, security_master, manifest.json, DATA_DICTIONARY.md, plus UPLOAD.md.

**Rules** (ported): the daily files keep the layout, column names and cleaning of the repo's
`scripts/publish_daily_data_to_s3.py` so `src/s3_client.py` can read them, plus the flag rows (`is_delisting_row`,
which that script would drop). dcf_inputs_annual is exported as its own table (decision 6), and
estimates_consensus is the full consensus table (decision 5). manifest.json: data_version (build date),
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

## Phase 2a results (s01, s02 on the 2026-10-25 data)

s01 PASS (15 s), s02 WARN (90 s; 6 daily returns above +1000%, all real penny-stock spikes such as Tempest
Therapeutics on 2023-10-11; permno 79728 has no CIZ rows). The archive has no notebook outputs, so the notebook's
MEMBERSHIP and PRICES cells were rerun unchanged on the same raw files (paths redirected only) to compare:

| compared | result |
|---|---|
| snapshots (428,354 rows, 218 dates) | identical keys, eff_dt, weight, ticker, name, CUSIP |
| periods (7,576), permno_list.txt (5,918), known gaps (72) | identical |
| daily rows (15,309,394); 300 random permnos (780,850 rows) | identical ret, mom_1m/6m/12m, vol_gk and vol_close 1m/12m, mkt_cap; prc differs on the 46 delisting rows where CIZ gives a post-delisting price (now missing, decision 9) |
| monthly rows (733,698) | identical ret_m, delisting flags, fills, SIC on every row; prc, mkt cap, momentum and volatility identical in every month without a delisting row. They differ only in the 3,186 delisting months, which now use the last real trading day (decision 9): 2,760 of them now have a price (the notebook had 774 post-delisting prices and no market cap); 426 hold only the delisting row and have none |

The first version of decision 11 excluded 72 stock-snapshots (48 stocks), including Super Micro Computer and 21
new stocks in the 2026-06-30 snapshot; the revised rule excludes the 10 merger stocks only.

## Phase 2b results (s03, s04)

s03 WARN (9 s; one GICS code without a name), s04 WARN (2 s; revenue agreement 94.8% vs 95%). The notebook's
FUNDAMENTALS cell was rerun unchanged on the same raw file. Rows: 257,447 in both, and the same source row was
kept for every fiscal quarter. The 12 statement factors are identical on all 257,447 rows. The 5 price-based
factors differ on 1,735 factor-rows, all of them a notebook value that is now missing for one of two decisions:

| factor | differ | non-USD (decision 2) | price below $0.01 (decision 14) | other |
|---|---|---|---|---|
| f_book_to_price | 372 | 372 | 0 | 0 |
| f_earnings_to_price | 351 | 328 | 23 | 0 |
| f_sales_to_price | 351 | 328 | 23 | 0 |
| f_ebitda_to_ev | 302 | 291 | 11 | 0 |
| f_dividend_yield | 359 | 335 | 24 | 0 |
| the other 12 | 0 | | | |

(Split-adjusted E/P and dividend yield, the impossible-value rules and the SUE definition were already in the
archived notebook, so they cause no differences.)

## Phase 2c results (s03, s04 fixes; s05, s06)

s03 PASS, s04 PASS (FCFF for 70% of years, 89% of DCF-applicable years), s05 WARN (59 s; the 100x rule flags
6,494 rows, see open question 1), s06 WARN (1 s; long-short Fama-French factors vs compounded daily).

- **s05 vs the notebook's IBES FACTORS cell** (rerun unchanged on the same raw files; the notebook keeps EPS FY1
  only): 683,325 rows and 5,751 permnos in both; every (permno, statpers) key matches, and meanest, medest,
  stdev, numest, numup, numdown and the ticker are identical on all rows. The USD filter and the open-ended
  latest link are the same rules in both. The 63 EPS FY1 duplicate groups (126 rows) differ only in `actual`,
  so keeping the row with more analysts instead of the first gives the same values.
- **s06 Fama-French check:** mktrf (max 8 bp) and rf (max 11 bp) match their compounded daily values every
  month. The long-short factors do not always: smb 4 months over 50 bp, hml 11, rmw 5, cma 3, umd 126 (max 759
  bp). The months line up (same-month correlation 0.96 for umd, 0.997-0.998 for smb and hml, near 0 shifted
  one month), so this is how Ken French builds the daily factors (daily momentum portfolios are formed
  differently from the monthly ones), not a data error. The published monthly factors are what market_monthly
  uses.

## Open questions

1. **The 100x rule flags real companies.** "A total above 100x the ticker's own median for that measure" flags
   6,494 rows (CPX 3,650, SAL 2,133, EBI 529, EBT 182) for 456 tickers, mostly pre-revenue companies whose sales
   estimates grow from near zero (Ocugen: median $0.02M, later estimates up to $954M; Endocyte, ImmunityBio) and
   real capex ramps (Plug Power). Comparing each estimate with the neighbouring estimates of the same ticker,
   measure and FPI (100x both the previous and next statpers) would flag 204 rows, which look like real
   glitches. Keep the rule, or switch?

## Phase 2 order

s01 -> s02 -> s06 (needs only the s02 calendar) -> s03 -> s04 -> s05 -> s07 -> s08, one PR per one or two stages,
each with its tests and a run on the real data whose summary.json is compared with the last-run numbers above.
