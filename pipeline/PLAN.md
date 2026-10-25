# WRDS pipeline plan

The plan for stages s01 to s08. All stages are implemented (phases 2a to 2e).
[Final outputs](#final-outputs) is what the pipeline must deliver; every stage builds toward it. The review
answers are in [Decisions](#decisions) and are written into the stage rules below.

## Final outputs

s08_export writes these files to `processed/final/` (Parquet, pyarrow, zstd; its summary.json goes to
`processed/s08_export/`). The columns as built are in `processed/final/DATA_DICTIONARY.md`; the table below is the plan. Every factor
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

Decisions 1-10 are from the first review, 11-15 from the second, 16-20 from the fourth, 21-23 and the s07 choices (24) from the fifth, 25-27 from the sixth (all 2026-10-25).

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
| 21 | f_sue is NaN when its absolute value is above 50 (counted). | s03 rule 8 |
| 22 | The IBES 100x rule compares each total-measure estimate with its neighbours: flagged only when more than 100x both the previous and the next estimate of the same ticker, measure and FPI (the one neighbour at the ends). | s05 rule 6 |
| 23 | The Fama-French monthly vs compounded daily check is strict (50 bp) for mktrf and rf, informational for smb, hml, rmw, cma and umd. | s06 checks |
| 24 | s07: the share-class gap compares Compustat cshoq (millions) with CRSP shrout (thousands) on the quarter-end date of cshoq; the analyst surprise uses the CRSP price and share basis of the last trading day before anndats_act; the neighbour rule runs on IBES tickers before linking; late-filed older quarters are dropped before the as-of merge (counted). | s07 rules 5, 9, checks; s05 rule 6 |
| 25 | Company market cap (sixth review): where Compustat cshoq, put on the CRSP share basis at t with cfacshr, exceeds the summed CRSP shares of our permnos with the same permco by more than 5%, company_mkt_cap_musd uses the Compustat count at the listed price (flag company_cap_from_compustat). mkt_cap_musd stays the listed security, for portfolio weights; company_mkt_cap_musd is for valuation. | s07 rule 3 |
| 26 | RV multiples: TTM and forward total denominators (ib_ttm, ebitda_ttm, ebit_ttm, revenue_ttm, EBT FY1, SAL FY1) must be >= $1M, forward EPS on the CRSP basis >= $0.01 for fwd_pe; otherwise NaN. | s07 rule 10 |
| 27 | Panel yield caps: f_forward_ep, f_forward_ep_fy2, f_forward_cfp NaN beyond +-5, f_analyst_surprise NaN beyond +-1 (counted). The median-multiples check is informational. | s07 rules 8-9, checks |

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
delreasontype, dlyprc, dlyret, dlyvol, shrout, dlyopen, dlyhigh, dlylow, dlycumfacpr, dlycumfacshr),
`raw/delistings.csv`.

**Outputs** (`processed/s02_prices/`):

| file | one row per | columns |
|---|---|---|
| daily/year=YYYY/data.parquet | permno, trading day | date, permno, permco, ticker, siccd, prc, prc_adj, cfacpr, cfacshr, shrout, mkt_cap, mkt_cap_musd, ret, vol, is_delisting_row, delist_ret_filled, mom_1m, mom_6m, mom_12m, vol_gk_{1m,3m,6m,12m}, vol_close_{1m,3m,6m,12m} |
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
8. **SUE cap** (decision 21): |f_sue| > 50 is NaN; 134 rows today. SUE mean -1,096 -> 0.018, p1 -7.09, p99 4.65.

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
6. **Neighbour rule** (decision 22): a SAL, EBT, EBI or CPX mean estimate (FPI 1-3) more than 100 times both the
   previous and the next estimate of the same ticker, measure and FPI (only the one neighbour for the first and
   last; neighbours that are missing or 0 are skipped) is set to NaN (all 5 values) and flagged `suspect_total`.
   It runs on IBES tickers before linking, so two share classes of one company never neighbour each other. Today
   193 rows (CPX 101, EBI 58, EBT 18, SAL 16), about 1,000x their neighbours (totals entered in thousands); the
   summary lists the 20 largest. The first rule (100x the ticker's median) flagged 6,494 rows, mostly real
   pre-revenue companies.
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
0.03-0.30) (FAIL); suspect totals under 0.01% (WARN; 0.003% today); CRSP price on statpers for 90%+ of EPS FY1 rows (WARN);
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
Russell 2000 price return 0.5-2.0% a year (WARN; 1.34% today); published monthly mktrf and rf within 50 bp of the
compounded daily factors (FAIL; decision 23); the same comparison for smb, hml, rmw, cma, umd reported as
information (see the phase 2c results);
no yield filled for more than 5 days in a row (WARN); every month present (FAIL). The summary has the end date of
every source, coverage, and count / missing % / mean / p1 / p50 / p99 of every column, daily and monthly.

**Tests:** Sandy days dropped and the next return spans them; percent to decimal; a holiday forward-filled and
flagged; spreads; first month dropped; ETF and Fama-French monthly compounding; the index monthly return from
month-end levels; yields at the month end.

## s07_panel

**Inputs:** s01 snapshots.parquet and permno_list.txt; s02 monthly.parquet, calendar.parquet and daily/ (date,
permno, prc, ret, cfacpr, cfacshr, is_delisting_row); s03 fundamentals_quarterly.parquet; s05 estimates_consensus.parquet
(EPS, CPS, SAL, EBT; FPI 0-2) and link.parquet; s06 market_daily and market_monthly (mktrf, rf). Not s04: the annual
statements stay their own table (decision 6).

**Outputs** (`processed/s07_panel/`; s08 exports them under the same names):

| file | one row per | columns |
|---|---|---|
| factor_panel_monthly.parquet | universe stock, month end t | identity (permno, permco, gvkey, ticker, security_name, gsector, gsector_name, ggroup, gind, gind_name, gsubind, sector_name, gics_from_sic, siccd), index_weight, snapshot_report_dt, snapshot_eff_dt, new_entrant, price_date, prc, shrout, mkt_cap_musd, company_mkt_cap_musd, company_mkt_cap_crsp_musd, company_cap_from_compustat, n_share_classes, company_shares_crsp_m, company_shares_compustat_m, cfacpr_t, cfacpr_last (s08 drops these two), mom_1m, mom_6m, mom_12m, mom_12_1, vol_gk_* and vol_close_*, beta_252d, beta_60m, fund_period_end, fund_report_date, fund_is_usd, fund_stale, the building blocks used (book_equity, ib_ttm, revenue_ttm, dv_ttm, fcf_ttm, ev_claims, ebitda_ttm, ebit_ttm, cshoq), ev, the 12 statement factors, the 8 price-scaled factors, the 5 quarter-end versions (`_qe`), ibes_statpers, numest_fy1, the 9 analyst factors, surprise_anndats, surprise_fpedats, ret_next_1m, delisted_next_month, and the statpers of each IBES series used |
| rv_multiples_monthly.parquet | same rows | date, permno, gsector, gind, prc, company_mkt_cap_musd, company_cap_from_compustat, ev, the 15 multiples, and `<multiple>_pct_gind`, `<multiple>_pct_gsector` |
| rv_peer_stats_monthly.parquet | gind, month end | date, gind, gind_name, n_peers, and per multiple `_n`, `_p25`, `_p50`, `_p75` |
| security_master.parquet | permno in permno_list.txt | permco, gvkey, ibes_ticker (all linked tickers, `;`-separated), ticker, security_name, GICS codes and names, siccd, first and last price date, first and last universe date, excluded_non_member |

The monthly returns table is s02's monthly.parquet; s08 exports it as returns_monthly.

**Rules:**
1. **Calendar:** every month end t (last CRSP trading day) from the first month with an effective snapshot (2008-04)
   to the last month with prices (2026-09): 222 months.
2. **Universe at t:** the latest s01 snapshot whose every line has eff_dt <= t, lines with in_universe = True
   (decision 11). 2009-04 uses the 2009-02 snapshot, 2009-05 and 2009-06 use 2009-03. new_entrant = not in the
   universe at the previous t.
3. **Market at t:** prc, shrout and mkt_cap_musd from the last real trading day of month t (s02 monthly; never a
   delisting row); mkt_cap_musd is the listed security, the one for portfolio weights. company_mkt_cap_musd (for
   valuation: B/P, EV, multiples) = sum over our permnos with the same permco at t (decision 3; kept as
   company_mkt_cap_crsp_musd), except where Compustat's common shares exceed ours by more than 5% (decision 25):
   company_shares_compustat_m = cshoq at the fundamentals' quarter end x cfacshr at the quarter end / cfacshr at t;
   then company_mkt_cap_musd = the CRSP company cap + (company_shares_compustat_m - company_shares_crsp_m) x the price
   of the company's largest listed class (with one listed class: Compustat shares x its price), and
   company_cap_from_compustat = True (28,202 company-months, 1,154 companies; nearly all real dual-class companies,
   e.g. GAMCO, UniFirst, Moog, AMC 2014-2016; two ADRs where Compustat counts ordinary shares, CSR 2011 and Nexxen
   2025, are wrongly included). Stale fundamentals give no Compustat count. mom_12_1 = (1 + mom_12m) / (1 + mom_1m) - 1. ret_next_1m = the stock's ret_m of month t+1, delisting returns included.
4. **Betas:** excess return = return - rf. beta_252d = cov(stock daily excess return, mktrf) / var(mktrf) over the
   stock's last 252 trading rows ending t, with at least 80% of them (202); beta_60m the same with monthly returns over
   the last 60 calendar months (at least 48).
5. **Fundamentals:** merge_asof by permno on report_date <= t. Before the merge, a quarter reported after a newer
   quarter of the same permno is dropped (a late-filed older quarter never replaces a newer one; 124 rows). After a
   fiscal-year change Compustat can list one quarter twice (old and new numbering, same period end and report date);
   the new numbering is used. If t - fund_period_end > 365 days the fundamentals are stale: every fundamental factor
   and building block is NaN (fund_stale; 378 rows).
6. **Sector:** GICS from the fundamentals row used at t, else the permno's latest row with GICS (decision 4);
   sector_name is the GICS sector name. Where GICS is missing (425 rows), the SIC code as of t is mapped to a GICS
   sector name with `config.SIC_TO_GICS_SECTOR` (the table below) and gics_from_sic = True.
7. **Price-scaled factors at t** (all $ millions, company market cap; NaN for non-USD fundamentals and prices below
   $0.01): f_book_to_price = book_equity / company mkt cap (NaN if book_equity <= 0); f_earnings_to_price = ib_ttm /
   company mkt cap; f_sales_to_price = revenue_ttm / company mkt cap; f_dividend_yield = dv_ttm / company mkt cap;
   f_fcf_yield = fcf_ttm / company mkt cap; ev = company mkt cap + ev_claims (debt + preferred + minority interest -
   cash), NaN if ev <= 0; f_ebitda_to_ev, f_ebit_to_ev, f_sales_to_ev = ebitda_ttm, ebit_ttm, revenue_ttm / ev. The
   quarter-end versions from s03 are kept with `_qe`. (Earnings and dividend yields now use company totals, ib_ttm
   and dv_ttm, instead of the notebook's per-share EPS and DPS over the price.)
8. **Analyst factors at t** (decision 5): for each series (EPS FY1, EPS FY2, CPS FY1, SAL FY1, EBT FY1, EPS FPI 0) the
   latest statpers <= t, at most 60 days old. Per-share values are put on the CRSP basis at t, estimate x cfacpr(t) /
   cfacpr_last (s05 rule 7), then divided by prc(t): f_forward_ep (EPS FY1), f_forward_ep_fy2, f_forward_cfp (CPS
   FY1). f_forward_sales_to_ev = SAL FY1 / ev, f_forward_ebitda_to_ev = EBT FY1 / ev, f_ltg = EPS FPI 0 (decimal),
   f_eps_revision = (numup - numdown) / numest, f_eps_dispersion = stdev / |meanest| (NaN if |meanest| < 0.01 or
   numest < 2), all EPS FY1. f_forward_ep, f_forward_ep_fy2 and f_forward_cfp are NaN beyond +-5 (decision 27;
   1,319, 1,012 and 709 values). f_forward_ep's 0.1% and 99.9% values are in the summary (decision 15; -3.34 and
   1.27, were -21.0 and 2.92 before the cap).
9. **Analyst surprise** (fifth review): for each EPS FY1 announcement, (actual - the last consensus before
   anndats_act), both converted to the CRSP share basis of the last trading day before anndats_act (cfacpr that day /
   cfacpr_last), divided by that day's CRSP price (at most 10 days before; NaN below $0.01). Two fiscal years
   announced the same day: the later one wins. At t: the latest announcement on or before t, at most 120 days old.
   NaN beyond +-1 (decision 27; 880 values).
10. **RV multiples** (same rows): pe = company mkt cap / ib_ttm, pb = / book_equity, ps = / revenue_ttm, ev_ebitda,
    ev_ebit, ev_sales; NaN when the denominator is below $1M (pb: book_equity <= 0) (decision 26); yields (may be
    negative): fcf_yield, dividend_yield, earnings_yield; forward: fwd_pe_fy1, fwd_pe_fy2 = price / EPS estimate on
    the CRSP basis (NaN below $0.01), fwd_ev_ebitda = ev / EBT FY1, fwd_ev_sales = ev / SAL FY1 (NaN below $1M); ltg; peg = fwd_pe_fy1 / (ltg x 100), only
    when ltg > 0. `<multiple>_pct_gind` and `_pct_gsector` = percentile rank (0-1) within the GICS industry and sector
    at t, NaN when the group has fewer than 5 stocks with a value.
11. **Peer stats:** per gind and t: n_peers, and per multiple the count and the 25th, 50th, 75th percentile (NaN
    with fewer than 5 values).
12. **Security master:** one row per permno in permno_list.txt (5,918): the latest CRSP permco, ticker and SIC, the
    latest Compustat gvkey and GICS, all linked IBES tickers, the latest IWM name, first and last CRSP price date,
    first and last date in the panel, and whether any of its snapshots was excluded (decision 11).

SIC fallback (rule 6), checked in order, the first match wins:

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

The app needs updating (separate PR): `app/streamlit_config.py` `SECTOR_OPTIONS` must become the 11 GICS sector
names. is_fossil_fuel (SIC 1200-1399, 2900-2999, 4920-4929, 5170-5179) is not in the panel yet.

**Checks:** no duplicate (t, permno) (FAIL); 1,800-2,150 stocks per month (WARN; 1,831-2,043); look-ahead asserts:
no fund_report_date, IBES statpers, snapshot eff_dt, surprise anndats or price date after t, every IBES series within
60 days (FAIL); excluded non-members not in the panel (FAIL); company mkt cap >= the stock's (FAIL); no price-scaled
factor on non-USD or sub-$0.01 rows (FAIL); stale rows have no factors (FAIL); every row has a sector (FAIL);
**share-class gap** (decision 12, fifth review): on the quarter-end date of cshoq, Compustat cshoq (millions) vs the
summed CRSP shrout (thousands / 1000) of our permnos with the same permco; a company counts when the gap is above 5%
of its Compustat shares and its market cap at t does not already use the Compustat count, and its missing market
cap at t is the missing share fraction applied to its CRSP company market cap; by year, the sum vs the universe's
company market cap, WARN above 1% (max 0.38% in 2022; 1.5-2.2% a year before decision 25); median pe, pb, ev_ebitda
by year vs 15-25, 1-3, 8-15 (informational, decision 27). The summary has coverage by year (prc, roa, book-to-price, forward E/P, beta_252d, ret_next_1m and
every multiple) and count / missing % / mean / p1 / p50 / p99 of every factor and multiple.

**Tests:** the universe uses the latest snapshot whose every line is effective, without excluded lines; the as-of
merge waits for the report date, ignores a late-filed older quarter, takes the new numbering after a fiscal-year
change, and goes stale after 365 days; the SIC fallback at boundary codes; beta on a series with a known beta, NaN
below 80% of days; EV and the <= 0 rules, sub-$0.01 and non-USD rows; the Compustat company cap across a split
and below 5%; the $1M and $0.01 denominator floors; the yield caps; percentile ranks and the 5-peer minimum; peer
percentiles; forward E/P across a 4-for-1 split; the surprise uses the price and share basis of the day before the
announcement.

## s08_export

**Inputs:** the outputs of s01-s07, and `processed/s00_inventory/inventory.json` (raw checksums and date ranges).

**Outputs:** `processed/final/`, exactly these files (s08's summary.json goes to `processed/s08_export/`):

| file | from |
|---|---|
| factor_panel_monthly.parquet | s07 (without the share-basis helpers cfacpr_t, cfacpr_last) |
| universe_periods.parquet | s01 periods.parquet (excluded_non_member flag included) |
| returns_monthly.parquet | s02 monthly.parquet without siccd, momentum and volatility (those are in the panel) |
| prices_daily/year=YYYY/data.parquet, prices_daily/_manifest.json | s02 daily: permno, ticker, date, prc, prc_adj, cfacpr, shrout, mkt_cap_musd, ret, vol, is_delisting_row |
| market_daily.parquet, market_monthly.parquet | s06 |
| rv_multiples_monthly.parquet, rv_peer_stats_monthly.parquet, security_master.parquet | s07 |
| dcf_inputs_annual.parquet | s04 |
| estimates_consensus.parquet | s05 |
| fundamentals_quarterly.parquet | s03 |
| manifest.json | data_version (build date), created_at, pipeline git commit (and whether the tree was dirty), sha256 of every raw file (from s00), per output file path, rows, MB, date range, permnos, column types; units, sectors (GICS and the SIC fallback), delisting fill rule, month-end convention, which market cap to use for what, point-in-time note, known gaps (the 72 unmatched IWM holdings, unlisted share classes, where each source ends) |
| DATA_DICTIONARY.md | rendered from `pipeline/dictionary.py`: per file what it is, its key and rows, and every column with type, meaning, unit, source and stage |
| UPLOAD.md | dry run, upload and listing check against `s3://sysen-5900-factor-lake/factor-lake-data/wrds_YYYY_MM/` for a teammate with the repo's .env AWS keys. **Nothing is uploaded by the pipeline.** |

**Rules:** permno int32, permco nullable int32, dates datetime64[ns] without time zone, floats float64, market cap in
$ millions, returns and yields as decimals. Helper columns are dropped by name (`SOURCES` in s08_export.py), so a
new column in an earlier stage stops s08 until it is documented in `pipeline/dictionary.py`. The daily files keep
the layout of the repo's `scripts/publish_daily_data_to_s3.py` (year=YYYY/data.parquet, sorted by permno and
date, row groups of 250,000, a `_manifest.json` with years, rows, bytes and columns) under `prices_daily/`, with
mkt_cap_musd instead of the old mkt_cap in $ thousands.

**Checks:** every file reads back with its source's row count (FAIL); no duplicate keys in any file (FAIL);
permno int32 and dates datetime64[ns] (FAIL); every output column is in DATA_DICTIONARY.md and every column there
is in the outputs, read from the written files (FAIL); final/ holds exactly the planned files (FAIL); the manifest
sha256 of every raw file equals the raw file re-hashed now (FAIL); total size and size per file (reported).

**Tests:** types are standardized; the dictionary renders every column and parses back; an undocumented or a
stale column FAILs; the daily files keep the publish layout and sort; the upload script targets the new folder and
compiles.

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

## Phase 2d results (s03, s05, s06 fixes; s07)

s03, s05, s06 PASS. s07 WARN (80 s): 435,846 rows, 222 months, 1,831-2,043 stocks a month; the two WARNs are
the share-class gap and the median multiples (open questions 1 and 2).

**s07 vs the notebook's MASTER MONTHLY PANEL cell** (rerun unchanged on the same raw files and the reruns of its
earlier cells):

- **Rows:** 435,860 in the notebook, 435,846 here. The 14 missing rows are exactly the excluded non-member
  stock-snapshots (decision 11) in the months they were used; no row is only here.
- **mom_12m:** identical on 99.55%; all 1,966 differing rows are months with the stock's delisting row, where the
  month-end value now comes from the last real trading day (decision 9).
- **f_roa:** identical on 99.92%; 331 rows are stale fundamentals (now NaN) and 14 use the newer quarter instead
  of a late-filed older one; no other difference.
- **ret_next_1m:** identical on every row.
- **f_forward_ep:** rank correlation 0.997, correlation 0.916, median difference 0.0023. The notebook divides by the
  price at statpers, this panel by the price at t; rebuilding the notebook's value from s05 (EPS / price at
  statpers on the IBES basis) matches it on 100% of 405,870 rows. 677 rows have no value here because prc(t) is
  missing or below $0.01; 62 have one only here (the notebook found no price within 7 days of statpers).

## Phase 2e results (s07 fixes; s08)

s02 rerun (keeps cfacshr), s07 PASS (114 s). Before -> after the s07 fixes (decisions 25-27):

- **Company market cap:** 28,202 company-months (1,154 companies) use the Compustat share count; the universe's
  company market cap rises 1.4-1.8% a year. The share-class gap falls from 1.5-2.2% to 0.06-0.38% a year.
- **Multiples** (mean; p1 / p99): pe 67.8 -> 36.3 (575.7 -> 343.4 at p99); ps 340 -> 8.0; ev_ebitda 2.9e13 -> 18.4
  (p99 203 -> 158); ev_ebit 48.8 -> 28.8; ev_sales 319 -> 8.0; fwd_pe_fy1 41.5 -> 35.0; fwd_ev_ebitda 22.4 -> 17.4;
  fwd_ev_sales 46.0 -> 8.6; the yields, pb and ltg move only through the company market cap.
- **Yield caps:** f_forward_ep mean -2.17 -> -0.007 (1,319 values removed), f_forward_ep_fy2 -1.06 -> 0.021 (1,012),
  f_forward_cfp -1.61 -> 0.087 (709), f_analyst_surprise -5.81 -> -0.004 (880).

**s08 PASS (35 s):** processed/final/ holds 36 files, 831 MB (the notebook's last run: 1,987 MB): factor_panel_monthly
435,846 rows (155.8 MB), universe_periods 7,576 (0.2), returns_monthly 733,698 (17.0), prices_daily 15,309,394 in
21 yearly files (333.5), market_daily 5,189 (0.4), market_monthly 237 (0.0), rv_multiples_monthly 435,846 (67.8),
rv_peer_stats_monthly 17,083 (5.4), dcf_inputs_annual 65,359 (11.1), estimates_consensus 10,061,304 (122.3),
fundamentals_quarterly 257,447 (116.8), security_master 5,918 (0.3); 506 documented columns. Every file reads back
with its source's rows, no duplicate keys, the 15 raw files re-hash to the s00 checksums.

## Open questions

None.

## Phase 2 order

s01 -> s02 -> s06 (needs only the s02 calendar) -> s03 -> s04 -> s05 -> s07 -> s08, one PR per one or two stages,
each with its tests and a run on the real data whose summary.json is compared with the last-run numbers above.
