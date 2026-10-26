# Factor-Lake WRDS data: data dictionary (schema)

Written by stage `s08_export` from `pipeline/dictionary.py`. It describes every file the pipeline exports to `processed/final/` and every column in it. It holds no numbers from the data: the full version with statistics and coverage by year (`processed/DATA_DICTIONARY.md`) stays next to the data, because the data is licensed from WRDS. All files are Parquet; read them with `pandas.read_parquet`.

## Conventions

- **Keys:** `permno` is CRSP's security id (one share class, int32); `permco` the company; `gvkey` Compustat's company id (text). Panels are keyed by a date plus permno.
- **Dates** are `datetime64[ns]` without a time zone. A *month end* is the last CRSP trading day of the month, not the calendar month end.
- **Units:** returns, yields and most factors are decimals (0.05 = 5%). Money is $ millions unless the column says per share. Compustat shares are millions, CRSP `shrout` thousands.
- **Point in time:** in factor_panel_monthly every value at date t was public by t (IWM snapshot effective date, Compustat report date, IBES statistics date, announcement date). The other tables are not lagged: filter on their report_date / statpers yourself.
- **Which market cap:** `mkt_cap_musd` (the listed security) for portfolio weights; `company_mkt_cap_musd` (all share classes, unlisted ones from Compustat) for valuation ratios.
- **Missing values** are NaN / null, never 0.

## Files

| file | one row per | key | built by |
|---|---|---|---|
| [factor_panel_monthly.parquet](#factor_panel_monthlyparquet) | Russell 2000 member (IWM holding), month end t | date, permno | `s07_panel` |
| [universe_periods.parquet](#universe_periodsparquet) | stock, membership period | permno, index_effective | `s01_membership` |
| [returns_monthly.parquet](#returns_monthlyparquet) | stock, month (every permno ever in the universe, also after it leaves) | month_end, permno | `s02_prices` |
| [prices_daily/year=YYYY/data.parquet](#prices_dailyyearyyyydataparquet) | stock, trading day (one file per calendar year) | permno, date | `s02_prices` |
| [market_daily.parquet](#market_dailyparquet) | trading day | date | `s06_market` |
| [market_monthly.parquet](#market_monthlyparquet) | month | month | `s06_market` |
| [rv_multiples_monthly.parquet](#rv_multiples_monthlyparquet) | Russell 2000 member, month end t (the panel's rows) | date, permno | `s07_panel` |
| [rv_peer_stats_monthly.parquet](#rv_peer_stats_monthlyparquet) | GICS industry, month end t | date, gind | `s07_panel` |
| [dcf_inputs_annual.parquet](#dcf_inputs_annualparquet) | stock, fiscal year | permno, fyear | `s04_fundamentals_a` |
| [estimates_consensus.parquet](#estimates_consensusparquet) | stock, IBES statistics date, measure, forecast period | permno, statpers, measure, fpi | `s05_ibes` |
| [fundamentals_quarterly.parquet](#fundamentals_quarterlyparquet) | stock, fiscal quarter | permno, fyearq, fqtr | `s03_fundamentals_q` |
| [security_master.parquet](#security_masterparquet) | permno ever in the universe (permno_list.txt) | permno | `s07_panel` |
| [column_stats.parquet](#column_statsparquet) | final table, column | file, column | `s08_export` |

## factor_panel_monthly.parquet

The point-in-time factor panel for backtests: at each month end t, every stock in the latest IWM snapshot effective by t, with factors built only from data public by t. Factors are raw values; rank or z-score them per date yourself.

One row per Russell 2000 member (IWM holding), month end t; key date, permno. Built by stage `s07_panel`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `date` | month end t: the last CRSP trading day of the month | date | CRSP daily stock file (CIZ) | s07_panel |
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s07_panel |
| `permco` | CRSP company id; share classes of one company share it | code | CRSP daily stock file (CIZ) | s07_panel |
| `gvkey` | Compustat company id of the fundamentals row used at t | text | Compustat quarterly (CCM) | s07_panel |
| `ticker` | CRSP ticker at t (IWM ticker when CRSP has none) | text | CRSP daily stock file (CIZ) | s07_panel |
| `security_name` | security name in the IWM snapshot | text | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `gsector` | GICS sector code (2 digits); current classification, not historical | code | Compustat quarterly (CCM) | s07_panel |
| `gsector_name` | GICS sector name | text | config.GICS_SECTORS | s07_panel |
| `ggroup` | GICS industry group code (4 digits) | code | Compustat quarterly (CCM) | s07_panel |
| `gind` | GICS industry code (6 digits); the RV peer group | code | Compustat quarterly (CCM) | s07_panel |
| `gind_name` | GICS industry name | text | config.GICS_INDUSTRIES | s07_panel |
| `gsubind` | GICS sub-industry code (8 digits) | code | Compustat quarterly (CCM) | s07_panel |
| `sector_name` | GICS sector name, or where GICS is missing the SIC code mapped to a GICS sector name (see the SIC fallback below) | text | derived | s07_panel |
| `gics_from_sic` | sector_name comes from the SIC fallback | true/false | derived | s07_panel |
| `siccd` | CRSP SIC code at t (last non-zero) | code | CRSP daily stock file (CIZ) | s07_panel |
| `index_weight` | weight in the IWM fund on the snapshot report date | percent of the fund | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `snapshot_report_dt` | report date of the IWM snapshot used at t | date | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `snapshot_eff_dt` | date the whole snapshot was public (latest eff_dt of its lines); always <= t | date | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `new_entrant` | not in the universe at the previous month end | true/false | derived | s07_panel |
| `price_date` | last real trading day of the stock in month t (never a delisting row) | date | CRSP daily stock file (CIZ) | s02_prices |
| `prc` | closing price on price_date (|DlyPrc|; bid/ask midpoints made positive) | $ per share | CRSP daily stock file (CIZ) | s02_prices |
| `shrout` | shares outstanding on price_date | thousands of shares | CRSP daily stock file (CIZ) | s02_prices |
| `mkt_cap_musd` | market cap of this listed security: prc x shrout. Use for portfolio weights | $ millions | CRSP daily stock file (CIZ) | s02_prices |
| `company_mkt_cap_musd` | company market cap, for valuation (B/P, EV, multiples): all our permnos with the same permco, plus unlisted shares at the largest class's price when company_cap_from_compustat | $ millions | CRSP daily stock file (CIZ); Compustat quarterly (CCM) | s07_panel |
| `company_mkt_cap_crsp_musd` | company market cap from CRSP only: sum of mkt_cap_musd over our permnos with the same permco | $ millions | CRSP daily stock file (CIZ) | s07_panel |
| `company_cap_from_compustat` | Compustat common shares exceed our CRSP shares by more than 5% (an unlisted share class), so company_mkt_cap_musd uses the Compustat count. Never for ADRs; not for episodes of at most 3 months within 15% of CRSP's count (share issuance between quarter ends) | true/false | derived | s07_panel |
| `is_adr` | CRSP share type is an ADR (security name '...; COM NONE ADR; ...'); a company with an ADR never uses the Compustat share count | true/false | CRSP daily stock file (CIZ) | s02_prices |
| `n_share_classes` | number of our permnos with this permco and a market cap at t | count | CRSP daily stock file (CIZ) | s07_panel |
| `company_shares_crsp_m` | shares of all our permnos with this permco at t | millions of shares | CRSP daily stock file (CIZ) | s07_panel |
| `company_shares_compustat_m` | Compustat cshoq at the quarter end put on the CRSP share basis at t (x cfacshr at the quarter end / cfacshr at t) | millions of shares | Compustat quarterly (CCM); CRSP daily stock file (CIZ) | s07_panel |
| `mom_1m` | price momentum: compounded return over the last 21 trading days to price_date (80% of days needed) | decimal | CRSP daily stock file (CIZ) | s02_prices |
| `mom_6m` | compounded return over the last 126 trading days | decimal | CRSP daily stock file (CIZ) | s02_prices |
| `mom_12m` | compounded return over the last 252 trading days | decimal | CRSP daily stock file (CIZ) | s02_prices |
| `mom_12_1` | 12-month momentum skipping the last month: (1 + mom_12m) / (1 + mom_1m) - 1 | decimal | derived | s07_panel |
| `vol_gk_1m` | Garman-Klass (open, high, low, close) volatility over the last 21 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol_close_1m` | close-to-close volatility over the last 21 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol_gk_3m` | Garman-Klass (open, high, low, close) volatility over the last 63 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol_close_3m` | close-to-close volatility over the last 63 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol_gk_6m` | Garman-Klass (open, high, low, close) volatility over the last 126 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol_close_6m` | close-to-close volatility over the last 126 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol_gk_12m` | Garman-Klass (open, high, low, close) volatility over the last 252 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol_close_12m` | close-to-close volatility over the last 252 trading days, annualized | annualized decimal | CRSP daily stock file (CIZ) | s02_prices |
| `beta_252d` | market beta: cov / var of daily excess returns (stock - rf vs Fama-French mktrf) over the last 252 trading rows, at least 202 | ratio (times) | CRSP daily stock file (CIZ); Fama-French | s07_panel |
| `beta_60m` | market beta from monthly excess returns over the last 60 months, at least 48 | ratio (times) | CRSP daily stock file (CIZ); Fama-French | s07_panel |
| `fund_period_end` | fiscal quarter end of the fundamentals used at t | date | Compustat quarterly (CCM) | s03_fundamentals_q |
| `fund_report_date` | date those fundamentals were public (rdq, else period end + 45 days, Q4 + 90); always <= t | date | Compustat quarterly (CCM) | s03_fundamentals_q |
| `fund_is_usd` | the statements are in USD (price-based factors are NaN otherwise) | true/false | Compustat quarterly (CCM) | s03_fundamentals_q |
| `fund_stale` | the fundamentals are more than 365 days old at t: every fundamental column is NaN | true/false | derived | s07_panel |
| `book_equity` | book equity: stockholders' equity (seqq, else ceqq + pstkq, else atq - ltq) + deferred taxes (txditcq) - preferred (pstkq), Fama-French style | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ib_ttm` | income before extraordinary items, sum of the last 4 fiscal quarters (ibq) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `revenue_ttm` | net sales, last 4 fiscal quarters (saleq) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dv_ttm` | cash dividends paid, last 4 fiscal quarters (from year-to-date dvy) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `fcf_ttm` | free cash flow, last 4 fiscal quarters: operating cash flow - capex | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ev_claims` | claims added to market cap for EV: total debt - cash + preferred + minority interest | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ebitda_ttm` | operating income before depreciation, last 4 fiscal quarters (oibdpq) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ebit_ttm` | operating income after depreciation, last 4 fiscal quarters (oiadpq) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `cshoq` | common shares outstanding at the fiscal quarter end, all classes | millions of shares | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ev` | enterprise value at t: company_mkt_cap_musd + ev_claims; NaN when <= 0 | $ millions | derived | s07_panel |
| `f_roe` | return on equity: TTM net income / book equity (book equity > 0) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_roa` | return on assets: TTM net income / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_gross_profitability` | (TTM revenue - TTM cost of goods sold) / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_accruals_to_assets` | (TTM net income - TTM operating cash flow) / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_leverage` | total debt / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_current_ratio` | current assets / current liabilities (liabilities > 0) | ratio (times) | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_rd_to_sales` | TTM R&D (0 if not reported) / TTM revenue; NaN when TTM revenue < $1M | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_asset_growth_1yr` | total assets / total assets 4 fiscal quarters earlier - 1 | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_sales_growth_1yr` | TTM revenue / TTM revenue 4 quarters earlier - 1; NaN when the earlier one < $1M | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_capex_growth_1yr` | TTM capex / TTM capex 4 quarters earlier - 1; NaN when the earlier one < $1M | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_net_share_issuance` | log(split-adjusted shares / split-adjusted shares 4 quarters earlier) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_sue` | standardized unexpected earnings: change in split-adjusted EPS vs 4 quarters earlier / std of the previous 8 such changes (at least 6, std >= $0.01); NaN beyond +-50 | std units | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_book_to_price` | book equity / company market cap at t (book equity > 0) | decimal | derived | s07_panel |
| `f_earnings_to_price` | TTM income before extraordinary items / company market cap at t | decimal | derived | s07_panel |
| `f_sales_to_price` | TTM revenue / company market cap at t | decimal | derived | s07_panel |
| `f_dividend_yield` | TTM dividends / company market cap at t | decimal | derived | s07_panel |
| `f_fcf_yield` | TTM free cash flow / company market cap at t | decimal | derived | s07_panel |
| `f_ebitda_to_ev` | TTM EBITDA / EV at t | decimal | derived | s07_panel |
| `f_ebit_to_ev` | TTM EBIT / EV at t | decimal | derived | s07_panel |
| `f_sales_to_ev` | TTM revenue / EV at t | decimal | derived | s07_panel |
| `f_book_to_price_qe` | book equity / market cap at the quarter end (Compustat price x shares): the notebook's quarter-end version, from Compustat's price (NaN for non-USD and prices below $0.01) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_earnings_to_price_qe` | TTM split-adjusted diluted EPS / Compustat price at the quarter end: the notebook's quarter-end version, from Compustat's price (NaN for non-USD and prices below $0.01) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_sales_to_price_qe` | TTM revenue / market cap at the quarter end: the notebook's quarter-end version, from Compustat's price (NaN for non-USD and prices below $0.01) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_ebitda_to_ev_qe` | TTM EBITDA / (market cap at the quarter end + net debt): the notebook's quarter-end version, from Compustat's price (NaN for non-USD and prices below $0.01) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_dividend_yield_qe` | TTM split-adjusted dividends per share / Compustat price at the quarter end: the notebook's quarter-end version, from Compustat's price (NaN for non-USD and prices below $0.01) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ibes_statpers` | IBES statistics date of the EPS FY1 consensus used at t (<= t, at most 60 days old) | date | IBES summary statistics | s05_ibes |
| `numest_fy1` | number of EPS FY1 estimates in that consensus | count | IBES summary statistics | s05_ibes |
| `f_forward_ep` | forward E/P: EPS FY1 consensus on the CRSP share basis at t / prc; NaN beyond +-5 | decimal | IBES summary statistics; CRSP daily stock file (CIZ) | s07_panel |
| `f_forward_ep_fy2` | EPS FY2 consensus on the CRSP share basis at t / prc; NaN beyond +-5 | decimal | IBES summary statistics; CRSP daily stock file (CIZ) | s07_panel |
| `f_forward_cfp` | cash flow per share FY1 consensus on the CRSP share basis at t / prc; NaN beyond +-5 | decimal | IBES summary statistics; CRSP daily stock file (CIZ) | s07_panel |
| `f_forward_sales_to_ev` | sales FY1 consensus / EV at t | decimal | IBES summary statistics | s07_panel |
| `f_forward_ebitda_to_ev` | EBITDA FY1 consensus (IBES measure EBT) / EV at t | decimal | IBES summary statistics | s07_panel |
| `f_ltg` | long-term EPS growth consensus (IBES FPI 0) | decimal per year | IBES summary statistics | s07_panel |
| `f_eps_revision` | (estimates revised up - revised down) / number of estimates, EPS FY1 | decimal | IBES summary statistics | s07_panel |
| `f_eps_dispersion` | std of EPS FY1 estimates / |mean|; NaN when |mean| < $0.01 or fewer than 2 estimates | ratio (times) | IBES summary statistics | s07_panel |
| `f_analyst_surprise` | latest EPS surprise announced on or before t (at most 120 days old): (actual - last consensus before the announcement) / price on the last trading day before it, both on that day's CRSP share basis; NaN beyond +-1 | decimal | IBES summary statistics; CRSP daily stock file (CIZ) | s07_panel |
| `surprise_anndats` | announcement date of that surprise | date | IBES summary statistics | s05_ibes |
| `surprise_fpedats` | fiscal period end of that surprise | date | IBES summary statistics | s05_ibes |
| `ret_next_1m` | the stock's return in month t+1, delisting return included (the backtest's holding return; never use it as a factor) | decimal | CRSP daily stock file (CIZ) | s02_prices |
| `delisted_next_month` | the stock delists in month t+1 | true/false | CRSP daily stock file (CIZ) | s02_prices |
| `eps2_statpers` | statistics date of the EPS FY2 consensus used at t | date | IBES summary statistics | s05_ibes |
| `cps1_statpers` | statistics date of the CPS FY1 consensus used at t | date | IBES summary statistics | s05_ibes |
| `sal1_statpers` | statistics date of the SAL FY1 consensus used at t | date | IBES summary statistics | s05_ibes |
| `ebt1_statpers` | statistics date of the EBT (EBITDA) FY1 consensus used at t | date | IBES summary statistics | s05_ibes |
| `ltg_statpers` | statistics date of the long-term growth consensus used at t | date | IBES summary statistics | s05_ibes |

## universe_periods.parquet

Periods each stock was in the IWM snapshots (consecutive snapshots, one missing snapshot bridged).

One row per stock, membership period; key permno, index_effective. Built by stage `s01_membership`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s01_membership |
| `index_effective` | report date of the period's first snapshot | date | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `index_through` | report date of the period's last snapshot | date | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `first_known` | first date the period was public (earliest eff_dt) | date | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `n_snapshots` | snapshots in the period | count | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `n_excluded` | snapshots of the period excluded as non-members (decision 11) | count | derived | s01_membership |
| `excluded_non_member` | every snapshot of the period is excluded: a merger stock that was never really a member, not in the panel | true/false | derived | s01_membership |
| `ticker` | IWM ticker in the period's last snapshot | text | IWM holdings (CRSP mutual fund holdings) | s01_membership |
| `security_name` | IWM security name in the period's last snapshot | text | IWM holdings (CRSP mutual fund holdings) | s01_membership |

## returns_monthly.parquet

Monthly returns for backtest holding periods, including the months after a stock leaves the index and the delisting month.

One row per stock, month (every permno ever in the universe, also after it leaves); key month_end, permno. Built by stage `s02_prices`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `month_end` | last CRSP trading day of the month | date | CRSP daily stock file (CIZ) | s02_prices |
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s02_prices |
| `permco` | CRSP company id; share classes of one company share it | code | CRSP daily stock file (CIZ) | s02_prices |
| `ticker` | CRSP ticker, last in the month | text | CRSP daily stock file (CIZ) | s02_prices |
| `last_trade_date` | last real trading day of the stock in the month | date | CRSP daily stock file (CIZ) | s02_prices |
| `prc` | closing price on last_trade_date | $ per share | CRSP daily stock file (CIZ) | s02_prices |
| `shrout` | shares outstanding on last_trade_date | thousands of shares | CRSP daily stock file (CIZ) | s02_prices |
| `mkt_cap_musd` | prc x shrout on last_trade_date | $ millions | CRSP daily stock file (CIZ) | s02_prices |
| `ret_m` | monthly return: every daily return of the month compounded, the delisting return included | decimal | CRSP daily stock file (CIZ) | s02_prices |
| `trading_days` | real trading days of the stock in the month | count | CRSP daily stock file (CIZ) | s02_prices |
| `delisted_in_month` | the month has the stock's delisting row | true/false | CRSP daily stock file (CIZ) | s02_prices |
| `delist_ret_filled` | the delisting return was missing and filled (-30% for GDR, else 0%) | true/false | derived | s02_prices |
| `is_adr` | CRSP share type is an ADR (from the security name) | true/false | CRSP daily stock file (CIZ) | s02_prices |

## prices_daily/year=YYYY/data.parquet

Daily CRSP prices of every permno ever in the universe, in the layout of the repo's scripts/publish_daily_data_to_s3.py (one file per year, sorted by permno and date) plus is_delisting_row. Momentum and volatility are in the monthly panel, not here.

One row per stock, trading day (one file per calendar year); key permno, date. Built by stage `s02_prices`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s02_prices |
| `ticker` | CRSP ticker | text | CRSP daily stock file (CIZ) | s02_prices |
| `date` | trading day | date | CRSP daily stock file (CIZ) | s02_prices |
| `prc` | closing price, |DlyPrc|; NaN on delisting rows and when CRSP has no price | $ per share | CRSP daily stock file (CIZ) | s02_prices |
| `prc_adj` | split-adjusted price: prc / cfacpr (prc when cfacpr is missing or 0) | $ per share | CRSP daily stock file (CIZ) | s02_prices |
| `cfacpr` | CRSP cumulative price adjustment factor (DlyCumFacPr); falls at splits, 1 at the end of the data | ratio (times) | CRSP daily stock file (CIZ) | s02_prices |
| `shrout` | shares outstanding | thousands of shares | CRSP daily stock file (CIZ) | s02_prices |
| `mkt_cap_musd` | prc x shrout | $ millions | CRSP daily stock file (CIZ) | s02_prices |
| `ret` | daily return (DlyRet); on the delisting row the delisting return (filled -30% for GDR, else 0% when missing) | decimal | CRSP daily stock file (CIZ) | s02_prices |
| `vol` | shares traded (DlyVol) | shares | CRSP daily stock file (CIZ) | s02_prices |
| `is_delisting_row` | CRSP's delisting row (DlyDelFlg = Y): carries only the delisting return, no price | true/false | CRSP daily stock file (CIZ) | s02_prices |

## market_daily.parquet

Benchmarks, Fama-French factors and interest rates by day. Returns are daily; yields are annual rates on that day (holidays forward-filled).

One row per trading day; key date. Built by stage `s06_market`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `date` | trading day | date | derived | s06_market |
| `r2000_price_ret` | Russell 2000 index price return (no dividends) | decimal | Compustat index prices (RUT) | s06_market |
| `r2000_growth_price_ret` | Russell 2000 Growth index price return | decimal | Compustat index prices | s06_market |
| `r2000_value_price_ret` | Russell 2000 Value index price return | decimal | Compustat index prices | s06_market |
| `iwm_total_ret` | IWM ETF total return (dividends included) | decimal | CRSP daily stock file (CIZ) | s06_market |
| `iwn_total_ret` | IWN (Russell 2000 Value ETF) total return | decimal | CRSP daily stock file (CIZ) | s06_market |
| `iwo_total_ret` | IWO (Russell 2000 Growth ETF) total return | decimal | CRSP daily stock file (CIZ) | s06_market |
| `mktrf` | Fama-French market excess return | decimal | Fama-French (WRDS) | s06_market |
| `smb` | Fama-French size factor (small minus big) | decimal | Fama-French (WRDS) | s06_market |
| `hml` | Fama-French value factor (high minus low book-to-market) | decimal | Fama-French (WRDS) | s06_market |
| `rmw` | Fama-French profitability factor (robust minus weak) | decimal | Fama-French (WRDS) | s06_market |
| `cma` | Fama-French investment factor (conservative minus aggressive) | decimal | Fama-French (WRDS) | s06_market |
| `umd` | momentum factor (up minus down) | decimal | Fama-French (WRDS) | s06_market |
| `rf` | risk-free rate for the period (one-month T-bill) | decimal | Fama-French (WRDS) | s06_market |
| `ust_10y` | 10-year Treasury yield | decimal per year | FRED DGS10 | s06_market |
| `tbill_3m` | 3-month Treasury bill yield | decimal per year | FRED DGS3MO | s06_market |
| `moodys_baa` | Moody's Baa corporate bond yield (the DCF cost-of-debt fallback) | decimal per year | FRED DBAA | s06_market |
| `moodys_aaa` | Moody's Aaa corporate bond yield | decimal per year | FRED DAAA | s06_market |
| `term_spread` | ust_10y - tbill_3m | decimal per year | derived | s06_market |
| `credit_spread` | moodys_baa - moodys_aaa | decimal per year | derived | s06_market |
| `was_filled` | at least one yield was forward-filled from an earlier day | true/false | derived | s06_market |

## market_monthly.parquet

The same by month: index returns from month-end levels, ETF returns compounded from daily, Fama-French as published monthly, yields and spreads as of the month end.

One row per month; key month. Built by stage `s06_market`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `month` | month as YYYY-MM | text | derived | s06_market |
| `month_end` | last CRSP trading day of the month | date | CRSP daily stock file (CIZ) | s06_market |
| `r2000_price_ret` | Russell 2000 index price monthly return (no dividends) | decimal | Compustat index prices (RUT) | s06_market |
| `r2000_growth_price_ret` | Russell 2000 Growth index price monthly return | decimal | Compustat index prices | s06_market |
| `r2000_value_price_ret` | Russell 2000 Value index price monthly return | decimal | Compustat index prices | s06_market |
| `iwm_total_ret` | IWM ETF total monthly return (dividends included) | decimal | CRSP daily stock file (CIZ) | s06_market |
| `iwn_total_ret` | IWN (Russell 2000 Value ETF) total monthly return | decimal | CRSP daily stock file (CIZ) | s06_market |
| `iwo_total_ret` | IWO (Russell 2000 Growth ETF) total monthly return | decimal | CRSP daily stock file (CIZ) | s06_market |
| `mktrf` | Fama-French market excess monthly return | decimal | Fama-French (WRDS) | s06_market |
| `smb` | Fama-French size factor (small minus big) | decimal | Fama-French (WRDS) | s06_market |
| `hml` | Fama-French value factor (high minus low book-to-market) | decimal | Fama-French (WRDS) | s06_market |
| `rmw` | Fama-French profitability factor (robust minus weak) | decimal | Fama-French (WRDS) | s06_market |
| `cma` | Fama-French investment factor (conservative minus aggressive) | decimal | Fama-French (WRDS) | s06_market |
| `umd` | momentum factor (up minus down) | decimal | Fama-French (WRDS) | s06_market |
| `rf` | risk-free rate for the period (one-month T-bill) | decimal | Fama-French (WRDS) | s06_market |
| `ust_10y` | 10-year Treasury yield at the month end | decimal per year | FRED DGS10 | s06_market |
| `tbill_3m` | 3-month Treasury bill yield at the month end | decimal per year | FRED DGS3MO | s06_market |
| `moodys_baa` | Moody's Baa corporate bond yield (the DCF cost-of-debt fallback) at the month end | decimal per year | FRED DBAA | s06_market |
| `moodys_aaa` | Moody's Aaa corporate bond yield at the month end | decimal per year | FRED DAAA | s06_market |
| `term_spread` | ust_10y - tbill_3m at the month end | decimal per year | derived | s06_market |
| `credit_spread` | moodys_baa - moodys_aaa at the month end | decimal per year | derived | s06_market |

## rv_multiples_monthly.parquet

Valuation multiples for the relative-value screen, with each stock's percentile rank within its GICS industry and sector at t. Multiples are NaN when the denominator is below its floor; ranks are NaN when fewer than 5 stocks of the group have a value. NaN for non-USD statements and prices below $0.01.

One row per Russell 2000 member, month end t (the panel's rows); key date, permno. Built by stage `s07_panel`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `date` | month end t | date | CRSP daily stock file (CIZ) | s07_panel |
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s07_panel |
| `gsector` | GICS sector code (2 digits); current classification, not historical | code | Compustat quarterly (CCM) | s07_panel |
| `gind` | GICS industry code (6 digits); the RV peer group | code | Compustat quarterly (CCM) | s07_panel |
| `prc` | closing price at t | $ per share | CRSP daily stock file (CIZ) | s02_prices |
| `company_mkt_cap_musd` | company market cap at t (see factor_panel_monthly) | $ millions | CRSP daily stock file (CIZ); Compustat quarterly (CCM) | s07_panel |
| `company_cap_from_compustat` | company_mkt_cap_musd uses the Compustat share count | true/false | derived | s07_panel |
| `ev` | enterprise value at t | $ millions | derived | s07_panel |
| `pe` | price / earnings: company market cap / TTM income before extraordinary items (>= $1M) | ratio (times) | derived | s07_panel |
| `pb` | price / book: company market cap / book equity (> 0) | ratio (times) | derived | s07_panel |
| `ps` | price / sales: company market cap / TTM revenue (>= $1M) | ratio (times) | derived | s07_panel |
| `ev_ebitda` | EV / TTM EBITDA (>= $1M) | ratio (times) | derived | s07_panel |
| `ev_ebit` | EV / TTM EBIT (>= $1M) | ratio (times) | derived | s07_panel |
| `ev_sales` | EV / TTM revenue (>= $1M) | ratio (times) | derived | s07_panel |
| `fcf_yield` | TTM free cash flow / company market cap (may be negative) | decimal | derived | s07_panel |
| `dividend_yield` | TTM dividends / company market cap | decimal | derived | s07_panel |
| `earnings_yield` | TTM income before extraordinary items / company market cap (may be negative) | decimal | derived | s07_panel |
| `fwd_pe_fy1` | prc / EPS FY1 consensus on the CRSP share basis at t (EPS >= $0.01) | ratio (times) | derived | s07_panel |
| `fwd_pe_fy2` | prc / EPS FY2 consensus on the CRSP share basis at t (EPS >= $0.01) | ratio (times) | derived | s07_panel |
| `fwd_ev_ebitda` | EV / EBITDA FY1 consensus (>= $1M) | ratio (times) | derived | s07_panel |
| `fwd_ev_sales` | EV / sales FY1 consensus (>= $1M) | ratio (times) | derived | s07_panel |
| `ltg` | long-term EPS growth consensus (IBES FPI 0) | decimal per year | derived | s07_panel |
| `peg` | fwd_pe_fy1 / (ltg x 100), only when ltg > 0 | ratio (times) | derived | s07_panel |
| `pe_pct_gind` | percentile rank (0-1, highest = 1) of pe within the GICS industry at t | 0-1 | derived | s07_panel |
| `pb_pct_gind` | percentile rank (0-1, highest = 1) of pb within the GICS industry at t | 0-1 | derived | s07_panel |
| `ps_pct_gind` | percentile rank (0-1, highest = 1) of ps within the GICS industry at t | 0-1 | derived | s07_panel |
| `ev_ebitda_pct_gind` | percentile rank (0-1, highest = 1) of ev_ebitda within the GICS industry at t | 0-1 | derived | s07_panel |
| `ev_ebit_pct_gind` | percentile rank (0-1, highest = 1) of ev_ebit within the GICS industry at t | 0-1 | derived | s07_panel |
| `ev_sales_pct_gind` | percentile rank (0-1, highest = 1) of ev_sales within the GICS industry at t | 0-1 | derived | s07_panel |
| `fcf_yield_pct_gind` | percentile rank (0-1, highest = 1) of fcf_yield within the GICS industry at t | 0-1 | derived | s07_panel |
| `dividend_yield_pct_gind` | percentile rank (0-1, highest = 1) of dividend_yield within the GICS industry at t | 0-1 | derived | s07_panel |
| `earnings_yield_pct_gind` | percentile rank (0-1, highest = 1) of earnings_yield within the GICS industry at t | 0-1 | derived | s07_panel |
| `fwd_pe_fy1_pct_gind` | percentile rank (0-1, highest = 1) of fwd_pe_fy1 within the GICS industry at t | 0-1 | derived | s07_panel |
| `fwd_pe_fy2_pct_gind` | percentile rank (0-1, highest = 1) of fwd_pe_fy2 within the GICS industry at t | 0-1 | derived | s07_panel |
| `fwd_ev_ebitda_pct_gind` | percentile rank (0-1, highest = 1) of fwd_ev_ebitda within the GICS industry at t | 0-1 | derived | s07_panel |
| `fwd_ev_sales_pct_gind` | percentile rank (0-1, highest = 1) of fwd_ev_sales within the GICS industry at t | 0-1 | derived | s07_panel |
| `ltg_pct_gind` | percentile rank (0-1, highest = 1) of ltg within the GICS industry at t | 0-1 | derived | s07_panel |
| `peg_pct_gind` | percentile rank (0-1, highest = 1) of peg within the GICS industry at t | 0-1 | derived | s07_panel |
| `pe_pct_gsector` | percentile rank (0-1, highest = 1) of pe within the GICS sector at t | 0-1 | derived | s07_panel |
| `pb_pct_gsector` | percentile rank (0-1, highest = 1) of pb within the GICS sector at t | 0-1 | derived | s07_panel |
| `ps_pct_gsector` | percentile rank (0-1, highest = 1) of ps within the GICS sector at t | 0-1 | derived | s07_panel |
| `ev_ebitda_pct_gsector` | percentile rank (0-1, highest = 1) of ev_ebitda within the GICS sector at t | 0-1 | derived | s07_panel |
| `ev_ebit_pct_gsector` | percentile rank (0-1, highest = 1) of ev_ebit within the GICS sector at t | 0-1 | derived | s07_panel |
| `ev_sales_pct_gsector` | percentile rank (0-1, highest = 1) of ev_sales within the GICS sector at t | 0-1 | derived | s07_panel |
| `fcf_yield_pct_gsector` | percentile rank (0-1, highest = 1) of fcf_yield within the GICS sector at t | 0-1 | derived | s07_panel |
| `dividend_yield_pct_gsector` | percentile rank (0-1, highest = 1) of dividend_yield within the GICS sector at t | 0-1 | derived | s07_panel |
| `earnings_yield_pct_gsector` | percentile rank (0-1, highest = 1) of earnings_yield within the GICS sector at t | 0-1 | derived | s07_panel |
| `fwd_pe_fy1_pct_gsector` | percentile rank (0-1, highest = 1) of fwd_pe_fy1 within the GICS sector at t | 0-1 | derived | s07_panel |
| `fwd_pe_fy2_pct_gsector` | percentile rank (0-1, highest = 1) of fwd_pe_fy2 within the GICS sector at t | 0-1 | derived | s07_panel |
| `fwd_ev_ebitda_pct_gsector` | percentile rank (0-1, highest = 1) of fwd_ev_ebitda within the GICS sector at t | 0-1 | derived | s07_panel |
| `fwd_ev_sales_pct_gsector` | percentile rank (0-1, highest = 1) of fwd_ev_sales within the GICS sector at t | 0-1 | derived | s07_panel |
| `ltg_pct_gsector` | percentile rank (0-1, highest = 1) of ltg within the GICS sector at t | 0-1 | derived | s07_panel |
| `peg_pct_gsector` | percentile rank (0-1, highest = 1) of peg within the GICS sector at t | 0-1 | derived | s07_panel |

## rv_peer_stats_monthly.parquet

Peer distribution of every multiple within each GICS industry at t (percentiles NaN with fewer than 5 values).

One row per GICS industry, month end t; key date, gind. Built by stage `s07_panel`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `date` | month end t | date | CRSP daily stock file (CIZ) | s07_panel |
| `gind` | GICS industry code (6 digits); the RV peer group | code | Compustat quarterly (CCM) | s07_panel |
| `gind_name` | GICS industry name | text | config.GICS_INDUSTRIES | s07_panel |
| `n_peers` | universe stocks in the industry at t | count | derived | s07_panel |
| `pe_n` | stocks with a pe value | count | derived | s07_panel |
| `pe_p25` | 25th percentile of pe | ratio (times) | derived | s07_panel |
| `pe_p50` | 50th (median) percentile of pe | ratio (times) | derived | s07_panel |
| `pe_p75` | 75th percentile of pe | ratio (times) | derived | s07_panel |
| `pb_n` | stocks with a pb value | count | derived | s07_panel |
| `pb_p25` | 25th percentile of pb | ratio (times) | derived | s07_panel |
| `pb_p50` | 50th (median) percentile of pb | ratio (times) | derived | s07_panel |
| `pb_p75` | 75th percentile of pb | ratio (times) | derived | s07_panel |
| `ps_n` | stocks with a ps value | count | derived | s07_panel |
| `ps_p25` | 25th percentile of ps | ratio (times) | derived | s07_panel |
| `ps_p50` | 50th (median) percentile of ps | ratio (times) | derived | s07_panel |
| `ps_p75` | 75th percentile of ps | ratio (times) | derived | s07_panel |
| `ev_ebitda_n` | stocks with a ev_ebitda value | count | derived | s07_panel |
| `ev_ebitda_p25` | 25th percentile of ev_ebitda | ratio (times) | derived | s07_panel |
| `ev_ebitda_p50` | 50th (median) percentile of ev_ebitda | ratio (times) | derived | s07_panel |
| `ev_ebitda_p75` | 75th percentile of ev_ebitda | ratio (times) | derived | s07_panel |
| `ev_ebit_n` | stocks with a ev_ebit value | count | derived | s07_panel |
| `ev_ebit_p25` | 25th percentile of ev_ebit | ratio (times) | derived | s07_panel |
| `ev_ebit_p50` | 50th (median) percentile of ev_ebit | ratio (times) | derived | s07_panel |
| `ev_ebit_p75` | 75th percentile of ev_ebit | ratio (times) | derived | s07_panel |
| `ev_sales_n` | stocks with a ev_sales value | count | derived | s07_panel |
| `ev_sales_p25` | 25th percentile of ev_sales | ratio (times) | derived | s07_panel |
| `ev_sales_p50` | 50th (median) percentile of ev_sales | ratio (times) | derived | s07_panel |
| `ev_sales_p75` | 75th percentile of ev_sales | ratio (times) | derived | s07_panel |
| `fcf_yield_n` | stocks with a fcf_yield value | count | derived | s07_panel |
| `fcf_yield_p25` | 25th percentile of fcf_yield | decimal | derived | s07_panel |
| `fcf_yield_p50` | 50th (median) percentile of fcf_yield | decimal | derived | s07_panel |
| `fcf_yield_p75` | 75th percentile of fcf_yield | decimal | derived | s07_panel |
| `dividend_yield_n` | stocks with a dividend_yield value | count | derived | s07_panel |
| `dividend_yield_p25` | 25th percentile of dividend_yield | decimal | derived | s07_panel |
| `dividend_yield_p50` | 50th (median) percentile of dividend_yield | decimal | derived | s07_panel |
| `dividend_yield_p75` | 75th percentile of dividend_yield | decimal | derived | s07_panel |
| `earnings_yield_n` | stocks with a earnings_yield value | count | derived | s07_panel |
| `earnings_yield_p25` | 25th percentile of earnings_yield | decimal | derived | s07_panel |
| `earnings_yield_p50` | 50th (median) percentile of earnings_yield | decimal | derived | s07_panel |
| `earnings_yield_p75` | 75th percentile of earnings_yield | decimal | derived | s07_panel |
| `fwd_pe_fy1_n` | stocks with a fwd_pe_fy1 value | count | derived | s07_panel |
| `fwd_pe_fy1_p25` | 25th percentile of fwd_pe_fy1 | ratio (times) | derived | s07_panel |
| `fwd_pe_fy1_p50` | 50th (median) percentile of fwd_pe_fy1 | ratio (times) | derived | s07_panel |
| `fwd_pe_fy1_p75` | 75th percentile of fwd_pe_fy1 | ratio (times) | derived | s07_panel |
| `fwd_pe_fy2_n` | stocks with a fwd_pe_fy2 value | count | derived | s07_panel |
| `fwd_pe_fy2_p25` | 25th percentile of fwd_pe_fy2 | ratio (times) | derived | s07_panel |
| `fwd_pe_fy2_p50` | 50th (median) percentile of fwd_pe_fy2 | ratio (times) | derived | s07_panel |
| `fwd_pe_fy2_p75` | 75th percentile of fwd_pe_fy2 | ratio (times) | derived | s07_panel |
| `fwd_ev_ebitda_n` | stocks with a fwd_ev_ebitda value | count | derived | s07_panel |
| `fwd_ev_ebitda_p25` | 25th percentile of fwd_ev_ebitda | ratio (times) | derived | s07_panel |
| `fwd_ev_ebitda_p50` | 50th (median) percentile of fwd_ev_ebitda | ratio (times) | derived | s07_panel |
| `fwd_ev_ebitda_p75` | 75th percentile of fwd_ev_ebitda | ratio (times) | derived | s07_panel |
| `fwd_ev_sales_n` | stocks with a fwd_ev_sales value | count | derived | s07_panel |
| `fwd_ev_sales_p25` | 25th percentile of fwd_ev_sales | ratio (times) | derived | s07_panel |
| `fwd_ev_sales_p50` | 50th (median) percentile of fwd_ev_sales | ratio (times) | derived | s07_panel |
| `fwd_ev_sales_p75` | 75th percentile of fwd_ev_sales | ratio (times) | derived | s07_panel |
| `ltg_n` | stocks with a ltg value | count | derived | s07_panel |
| `ltg_p25` | 25th percentile of ltg | decimal per year | derived | s07_panel |
| `ltg_p50` | 50th (median) percentile of ltg | decimal per year | derived | s07_panel |
| `ltg_p75` | 75th percentile of ltg | decimal per year | derived | s07_panel |
| `peg_n` | stocks with a peg value | count | derived | s07_panel |
| `peg_p25` | 25th percentile of peg | ratio (times) | derived | s07_panel |
| `peg_p50` | 50th (median) percentile of peg | ratio (times) | derived | s07_panel |
| `peg_p75` | 75th percentile of peg | ratio (times) | derived | s07_panel |

## dcf_inputs_annual.parquet

Annual statements for DCF valuation. Use a row only from report_date on (point in time). Financials and Real Estate are kept but marked dcf_applicable = False.

One row per stock, fiscal year; key permno, fyear. Built by stage `s04_fundamentals_a`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s04_fundamentals_a |
| `gvkey` | Compustat company id | text | Compustat annual (CCM) | s04_fundamentals_a |
| `tic` | Compustat ticker | text | Compustat annual (CCM) | s04_fundamentals_a |
| `conm` | Compustat company name | text | Compustat annual (CCM) | s04_fundamentals_a |
| `fyear` | fiscal year | year | Compustat annual (CCM) | s04_fundamentals_a |
| `fyr` | fiscal year-end month | month (1-12) | Compustat annual (CCM) | s04_fundamentals_a |
| `period_end` | fiscal year end (datadate) | date | Compustat annual (CCM) | s04_fundamentals_a |
| `report_date` | date the annual numbers were public: rdq of the quarterly Q4 row with the same period end, else period end + 90 days | date | Compustat annual (CCM); Compustat quarterly (CCM) | s04_fundamentals_a |
| `report_date_source` | q4_rdq or proxy (period end + 90 days) | text | derived | s04_fundamentals_a |
| `curcd` | reporting currency | text | Compustat annual (CCM) | s04_fundamentals_a |
| `is_usd` | reported in USD | true/false | Compustat annual (CCM) | s04_fundamentals_a |
| `ajex` | Compustat cumulative adjustment factor (divide per-share items, multiply shares) | ratio (times) | Compustat annual (CCM) | s04_fundamentals_a |
| `price_fye` | closing price at the fiscal year end (prcc_f) | $ per share | Compustat annual (CCM) | s04_fundamentals_a |
| `epsfx` | diluted EPS excluding extraordinary items | $ per share | Compustat annual (CCM) | s04_fundamentals_a |
| `emp` | employees | thousands | Compustat annual (CCM) | s04_fundamentals_a |
| `sich` | historical SIC code | code | Compustat annual (CCM) | s04_fundamentals_a |
| `gsector` | GICS sector code (2 digits); current classification, not historical | code | Compustat annual (CCM) | s04_fundamentals_a |
| `gsector_name` | GICS sector name | text | Compustat annual (CCM) | s04_fundamentals_a |
| `ggroup` | GICS industry group code (4 digits) | code | Compustat annual (CCM) | s04_fundamentals_a |
| `gind` | GICS industry code (6 digits); the RV peer group | code | Compustat annual (CCM) | s04_fundamentals_a |
| `gind_name` | GICS industry name | text | Compustat annual (CCM) | s04_fundamentals_a |
| `gsubind` | GICS sub-industry code (8 digits) | code | Compustat annual (CCM) | s04_fundamentals_a |
| `naics` | NAICS industry code | code | Compustat annual (CCM) | s04_fundamentals_a |
| `dcf_applicable` | False for GICS sectors 40 (Financials) and 60 (Real Estate), where a DCF on FCFF does not fit | true/false | derived | s04_fundamentals_a |
| `interest_source` | where interest_expense comes from: xint, xintd or missing (never 0-filled; the DCF then uses the Baa yield) | text | derived | s04_fundamentals_a |
| `revenue` | net sales (sale) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `ebitda` | operating income before depreciation (oibdp) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `ebit` | operating income after depreciation (oiadp) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `da` | depreciation and amortization (dp) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `capex` | capital expenditures (capx) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `oancf` | operating cash flow | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `pretax_income` | pretax income (pi) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `taxes` | income taxes (txt) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `tax_rate` | effective tax rate: taxes / pretax income when pretax > 0, clipped to 0-60% | decimal | derived | s04_fundamentals_a |
| `tax_rate_fcff` | tax rate for FCFF: tax_rate, or 0 when pretax income <= 0 | decimal | derived | s04_fundamentals_a |
| `statutory_tax_rate` | US federal statutory rate: 35% up to fiscal 2017, 21% from 2018 | decimal | config.STATUTORY_TAX_RATE | s04_fundamentals_a |
| `interest_expense` | interest expense: xint, else xintd | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `net_income` | net income (ni) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `nwc` | non-cash working capital: (current assets - cash) - (current liabilities - short-term debt) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `d_nwc` | change in nwc vs fiscal year - 1 (exactly the previous year) | $ millions | derived | s04_fundamentals_a |
| `total_debt` | long-term debt + debt in current liabilities | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `cash` | cash and short-term investments (che) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `net_debt` | total_debt - cash | $ millions | derived | s04_fundamentals_a |
| `preferred` | preferred stock (pstk) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `minority_interest` | noncontrolling interest (mib) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `diluted_shares` | shares for diluted EPS (cshfd) | millions of shares | Compustat annual (CCM) | s04_fundamentals_a |
| `basic_shares` | common shares outstanding (csho) | millions of shares | Compustat annual (CCM) | s04_fundamentals_a |
| `book_equity` | book equity, Fama-French style (as in fundamentals_quarterly) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `dividends` | common dividends (dvc) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `dividends_total` | total cash dividends (dv) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `buybacks` | purchase of common and preferred stock (prstkc) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `share_issuance` | sale of common and preferred stock (sstk) | $ millions | Compustat annual (CCM) | s04_fundamentals_a |
| `fcff` | free cash flow to the firm: EBIT x (1 - tax_rate_fcff) + D&A - capex - d_nwc; NaN if a part is missing | $ millions | derived | s04_fundamentals_a |
| `revenue_growth` | revenue / revenue of fiscal year - 1 - 1 (previous > 0) | decimal | derived | s04_fundamentals_a |
| `ebitda_growth` | EBITDA / EBITDA of fiscal year - 1 - 1 (previous > 0) | decimal | derived | s04_fundamentals_a |

## estimates_consensus.parquet

The full monthly IBES consensus in USD for 7 measures, linked to permno. IBES per-share values are on the latest share basis: multiply by crsp_basis_factor to compare with the CRSP price at statpers. Use a row only from statpers on.

One row per stock, IBES statistics date, measure, forecast period; key permno, statpers, measure, fpi. Built by stage `s05_ibes`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s05_ibes |
| `ibes_ticker` | IBES ticker | text | IBES summary statistics | s05_ibes |
| `statpers` | IBES statistics date (when the consensus was computed) | date | IBES summary statistics | s05_ibes |
| `measure` | EPS, CPS (cash flow per share), DPS (dividends per share): $ per share; SAL (sales), EBT (EBITDA), EBI (EBIT), CPX (capex): $ millions | text | IBES summary statistics | s05_ibes |
| `fpi` | forecast period: 0 long-term growth, 1 FY1, 2 FY2, 3 FY3 | code | IBES summary statistics | s05_ibes |
| `fpedats` | fiscal period end being forecast | date | IBES summary statistics | s05_ibes |
| `numest` | number of estimates | count | IBES summary statistics | s05_ibes |
| `numup` | estimates revised up in the last month | count | IBES summary statistics | s05_ibes |
| `numdown` | estimates revised down in the last month | count | IBES summary statistics | s05_ibes |
| `meanest` | mean estimate (per share, $ millions, or decimal growth for FPI 0) | see measure | IBES summary statistics | s05_ibes |
| `medest` | median estimate | see measure | IBES summary statistics | s05_ibes |
| `stdev` | std of the estimates | see measure | IBES summary statistics | s05_ibes |
| `highest` | highest estimate | see measure | IBES summary statistics | s05_ibes |
| `lowest` | lowest estimate | see measure | IBES summary statistics | s05_ibes |
| `actual` | reported actual for fpedats (NaN when IBES has two different actuals) | see measure | IBES summary statistics | s05_ibes |
| `anndats_act` | announcement date of the actual | date | IBES summary statistics | s05_ibes |
| `price_statpers` | CRSP price on statpers (last trading day, at most 7 days earlier) | $ per share | CRSP daily stock file (CIZ) | s05_ibes |
| `cfacpr_statpers` | CRSP cfacpr on that day | ratio (times) | CRSP daily stock file (CIZ) | s05_ibes |
| `crsp_basis_factor` | cfacpr_statpers / the permno's last cfacpr (1 in CIZ, so equal to cfacpr_statpers): per-share value x this = on the CRSP basis at statpers | ratio (times) | derived | s05_ibes |
| `suspect_total` | a total (SAL, EBT, EBI, CPX) more than 100x both its previous and next estimate; its values are NaN | true/false | derived | s05_ibes |
| `suspect_scale` | a total estimate or actual more than 100x the largest of the company's Compustat TTM revenue, total assets and market cap known at statpers (scale rule), or, with TTM revenue >= $50M, a positive EBIT/EBITDA or a capex estimate above 20x that revenue (margin rule); those values are NaN | true/false | derived | s05_ibes |
| `suspect_scale_reason` | which rule set suspect_scale: scale or margin (scale when both); empty otherwise | text | derived | s05_ibes |

## fundamentals_quarterly.parquet

Quarterly Compustat statements with TTM sums, building blocks and the quarter-end factors. TTM = exactly 4 consecutive fiscal quarters (NaN if one is missing). Use a row only from report_date on. After a fiscal-year change one period end can appear twice (old and new fiscal numbering).

One row per stock, fiscal quarter; key permno, fyearq, fqtr. Built by stage `s03_fundamentals_q`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s03_fundamentals_q |
| `gvkey` | Compustat company id | text | Compustat quarterly (CCM) | s03_fundamentals_q |
| `tic` | Compustat ticker | text | Compustat quarterly (CCM) | s03_fundamentals_q |
| `conm` | Compustat company name | text | Compustat quarterly (CCM) | s03_fundamentals_q |
| `period_end` | fiscal quarter end (datadate) | date | Compustat quarterly (CCM) | s03_fundamentals_q |
| `report_date` | date the quarter was public: rdq, else period end + 45 days (+ 90 for fiscal Q4) | date | Compustat quarterly (CCM) | s03_fundamentals_q |
| `report_date_source` | rdq or proxy | text | derived | s03_fundamentals_q |
| `rdq` | Compustat report date | date | Compustat quarterly (CCM) | s03_fundamentals_q |
| `fyearq` | fiscal year | year | Compustat quarterly (CCM) | s03_fundamentals_q |
| `fqtr` | fiscal quarter (1-4) | code | Compustat quarterly (CCM) | s03_fundamentals_q |
| `curcdq` | reporting currency | text | Compustat quarterly (CCM) | s03_fundamentals_q |
| `is_usd` | reported in USD | true/false | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ajexq` | Compustat cumulative adjustment factor (divide per-share items, multiply shares) | ratio (times) | Compustat quarterly (CCM) | s03_fundamentals_q |
| `gsector` | GICS sector code (2 digits); current classification, not historical | code | Compustat quarterly (CCM) | s03_fundamentals_q |
| `gsector_name` | GICS sector name | text | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ggroup` | GICS industry group code (4 digits) | code | Compustat quarterly (CCM) | s03_fundamentals_q |
| `gind` | GICS industry code (6 digits); the RV peer group | code | Compustat quarterly (CCM) | s03_fundamentals_q |
| `gind_name` | GICS industry name | text | Compustat quarterly (CCM) | s03_fundamentals_q |
| `gsubind` | GICS sub-industry code (8 digits) | code | Compustat quarterly (CCM) | s03_fundamentals_q |
| `naics` | NAICS industry code | code | Compustat quarterly (CCM) | s03_fundamentals_q |
| `price_qe` | closing price at the quarter end (prccq) | $ per share | Compustat quarterly (CCM) | s03_fundamentals_q |
| `cshoq` | common shares outstanding, all classes | millions of shares | Compustat quarterly (CCM) | s03_fundamentals_q |
| `atq` | total assets | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ltq` | total liabilities | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `actq` | current assets | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `lctq` | current liabilities | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `cheq` | cash and short-term investments | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `invtq` | inventories | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `rectq` | receivables | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `apq` | accounts payable | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ppentq` | net property, plant and equipment | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `gdwlq` | goodwill | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `intanq` | intangible assets | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dlcq` | debt in current liabilities | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dlttq` | long-term debt | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `pstkq` | preferred stock | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `mibtq` | noncontrolling interest | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `seqq` | stockholders' equity | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ceqq` | common equity | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `txditcq` | deferred taxes and investment tax credit | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `saleq` | net sales | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `revtq` | total revenue | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `cogsq` | cost of goods sold | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `xsgaq` | selling, general and administrative expense | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `xrdq` | R&D expense | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `oibdpq` | operating income before depreciation (EBITDA) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dpq` | depreciation and amortization | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `oiadpq` | operating income after depreciation (EBIT) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `xintq` | interest expense | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `piq` | pretax income | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `txtq` | income taxes | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `niq` | net income | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ibq` | income before extraordinary items | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `epsfxq` | diluted EPS excluding extraordinary items | $ per share | Compustat quarterly (CCM) | s03_fundamentals_q |
| `epspxq` | basic EPS excluding extraordinary items | $ per share | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dvpsxq` | dividends per share by ex-date | $ per share | Compustat quarterly (CCM) | s03_fundamentals_q |
| `oancf_q` | operating cash flow in the quarter (from Compustat's year-to-date oancfy: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `capx_q` | capital expenditures in the quarter (from Compustat's year-to-date capxy: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dv_q` | cash dividends in the quarter (from Compustat's year-to-date dvy: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `sstk_q` | sale of common and preferred stock in the quarter (from Compustat's year-to-date sstky: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `prstkc_q` | purchase of common and preferred stock in the quarter (from Compustat's year-to-date prstkcy: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dltis_q` | long-term debt issued in the quarter (from Compustat's year-to-date dltisy: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `dltr_q` | long-term debt repaid in the quarter (from Compustat's year-to-date dltry: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `aqc_q` | acquisitions in the quarter (from Compustat's year-to-date aqcy: Q1 = YTD, later = YTD - previous quarter's YTD) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `book_equity` | book equity: stockholders' equity (seqq, else ceqq + pstkq, else atq - ltq) + deferred taxes (txditcq) - preferred (pstkq), Fama-French style | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `total_debt` | long-term debt + debt in current liabilities | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `cash` | cash and short-term investments (cheq) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `net_debt` | total_debt - cash | $ millions | derived | s03_fundamentals_q |
| `preferred` | preferred stock (pstkq) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `minority_interest` | noncontrolling interest (mibtq) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `ev_claims` | claims added to market cap for EV: total debt - cash + preferred + minority interest | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `nwc` | non-cash working capital: (current assets - cash) - (current liabilities - short-term debt) | $ millions | Compustat quarterly (CCM) | s03_fundamentals_q |
| `d_nwc_4q` | nwc - nwc 4 fiscal quarters earlier | $ millions | derived | s03_fundamentals_q |
| `revenue_ttm` | net sales, sum of the last 4 fiscal quarters (saleq) | $ millions | derived | s03_fundamentals_q |
| `revt_ttm` | total revenue, sum of the last 4 fiscal quarters (revtq) | $ millions | derived | s03_fundamentals_q |
| `cogs_ttm` | cost of goods sold, sum of the last 4 fiscal quarters (cogsq) | $ millions | derived | s03_fundamentals_q |
| `xsga_ttm` | selling, general and administrative expense, sum of the last 4 fiscal quarters (xsgaq) | $ millions | derived | s03_fundamentals_q |
| `xrd_ttm` | R&D expense, sum of the last 4 fiscal quarters (xrdq) | $ millions | derived | s03_fundamentals_q |
| `ebitda_ttm` | operating income before depreciation (EBITDA), sum of the last 4 fiscal quarters (oibdpq) | $ millions | derived | s03_fundamentals_q |
| `ebit_ttm` | operating income after depreciation (EBIT), sum of the last 4 fiscal quarters (oiadpq) | $ millions | derived | s03_fundamentals_q |
| `da_ttm` | depreciation and amortization, sum of the last 4 fiscal quarters (dpq) | $ millions | derived | s03_fundamentals_q |
| `interest_ttm` | interest expense, sum of the last 4 fiscal quarters (xintq) | $ millions | derived | s03_fundamentals_q |
| `pretax_ttm` | pretax income, sum of the last 4 fiscal quarters (piq) | $ millions | derived | s03_fundamentals_q |
| `tax_ttm` | income taxes, sum of the last 4 fiscal quarters (txtq) | $ millions | derived | s03_fundamentals_q |
| `ni_ttm` | net income, sum of the last 4 fiscal quarters (niq) | $ millions | derived | s03_fundamentals_q |
| `ib_ttm` | income before extraordinary items, sum of the last 4 fiscal quarters (ibq) | $ millions | derived | s03_fundamentals_q |
| `oancf_ttm` | operating cash flow, sum of the last 4 fiscal quarters (oancf_q) | $ millions | derived | s03_fundamentals_q |
| `capx_ttm` | capital expenditures, sum of the last 4 fiscal quarters (capx_q) | $ millions | derived | s03_fundamentals_q |
| `dv_ttm` | cash dividends, sum of the last 4 fiscal quarters (dv_q) | $ millions | derived | s03_fundamentals_q |
| `sstk_ttm` | sale of common and preferred stock, sum of the last 4 fiscal quarters (sstk_q) | $ millions | derived | s03_fundamentals_q |
| `prstkc_ttm` | purchase of common and preferred stock, sum of the last 4 fiscal quarters (prstkc_q) | $ millions | derived | s03_fundamentals_q |
| `dltis_ttm` | long-term debt issued, sum of the last 4 fiscal quarters (dltis_q) | $ millions | derived | s03_fundamentals_q |
| `dltr_ttm` | long-term debt repaid, sum of the last 4 fiscal quarters (dltr_q) | $ millions | derived | s03_fundamentals_q |
| `aqc_ttm` | acquisitions, sum of the last 4 fiscal quarters (aqc_q) | $ millions | derived | s03_fundamentals_q |
| `fcf_ttm` | oancf_ttm - capx_ttm | $ millions | derived | s03_fundamentals_q |
| `tax_rate_ttm` | tax_ttm / pretax_ttm when pretax > 0, clipped to 0-60% | decimal | derived | s03_fundamentals_q |
| `epsfxq_ttm_adj` | TTM diluted EPS, each quarter split-adjusted with ajexq, in this quarter's shares | $ per share | derived | s03_fundamentals_q |
| `dvpsxq_ttm_adj` | TTM dividends per share, split-adjusted the same way | $ per share | derived | s03_fundamentals_q |
| `diluted_shares` | shares for diluted EPS (cshfdq) | millions of shares | Compustat quarterly (CCM) | s03_fundamentals_q |
| `csho_adj` | split-adjusted shares outstanding: cshoq x ajexq (comparable over time) | millions of shares (adjusted) | derived | s03_fundamentals_q |
| `cshfd_adj` | split-adjusted diluted shares: cshfdq x ajexq | millions of shares (adjusted) | derived | s03_fundamentals_q |
| `mkt_cap_qe_musd` | market cap at the quarter end: price_qe x cshoq | $ millions | derived | s03_fundamentals_q |
| `f_book_to_price` | book equity / market cap at the quarter end (Compustat price x shares) (NaN for non-USD and prices below $0.01) | decimal | derived | s03_fundamentals_q |
| `f_earnings_to_price` | TTM split-adjusted diluted EPS / Compustat price at the quarter end (NaN for non-USD and prices below $0.01) | decimal | derived | s03_fundamentals_q |
| `f_sales_to_price` | TTM revenue / market cap at the quarter end (NaN for non-USD and prices below $0.01) | decimal | derived | s03_fundamentals_q |
| `f_ebitda_to_ev` | TTM EBITDA / (market cap at the quarter end + net debt) (NaN for non-USD and prices below $0.01) | decimal | derived | s03_fundamentals_q |
| `f_dividend_yield` | TTM split-adjusted dividends per share / Compustat price at the quarter end (NaN for non-USD and prices below $0.01) | decimal | derived | s03_fundamentals_q |
| `f_roe` | return on equity: TTM net income / book equity (book equity > 0) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_roa` | return on assets: TTM net income / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_gross_profitability` | (TTM revenue - TTM cost of goods sold) / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_accruals_to_assets` | (TTM net income - TTM operating cash flow) / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_leverage` | total debt / total assets | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_current_ratio` | current assets / current liabilities (liabilities > 0) | ratio (times) | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_rd_to_sales` | TTM R&D (0 if not reported) / TTM revenue; NaN when TTM revenue < $1M | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_asset_growth_1yr` | total assets / total assets 4 fiscal quarters earlier - 1 | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_sales_growth_1yr` | TTM revenue / TTM revenue 4 quarters earlier - 1; NaN when the earlier one < $1M | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_capex_growth_1yr` | TTM capex / TTM capex 4 quarters earlier - 1; NaN when the earlier one < $1M | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_net_share_issuance` | log(split-adjusted shares / split-adjusted shares 4 quarters earlier) | decimal | Compustat quarterly (CCM) | s03_fundamentals_q |
| `f_sue` | standardized unexpected earnings: change in split-adjusted EPS vs 4 quarters earlier / std of the previous 8 such changes (at least 6, std >= $0.01); NaN beyond +-50 | std units | Compustat quarterly (CCM) | s03_fundamentals_q |

## security_master.parquet

Lookup table: the latest ids, names and classification of every permno.

One row per permno ever in the universe (permno_list.txt); key permno. Built by stage `s07_panel`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `permno` | CRSP security id (one share class) | code | CRSP daily stock file (CIZ) | s07_panel |
| `permco` | latest CRSP company id | code | CRSP daily stock file (CIZ) | s07_panel |
| `gvkey` | latest Compustat company id | text | Compustat quarterly (CCM) | s07_panel |
| `ibes_ticker` | every linked IBES ticker, ';'-separated, oldest first | text | CRSP-IBES link | s07_panel |
| `ticker` | latest CRSP ticker | text | CRSP daily stock file (CIZ) | s07_panel |
| `security_name` | latest IWM security name | text | IWM holdings (CRSP mutual fund holdings) | s07_panel |
| `gsector` | GICS sector code (2 digits); latest | code | Compustat quarterly (CCM) | s07_panel |
| `gsector_name` | GICS sector name | text | config.GICS_SECTORS | s07_panel |
| `ggroup` | GICS industry group code (4 digits) | code | Compustat quarterly (CCM) | s07_panel |
| `gind` | GICS industry code (6 digits); the RV peer group | code | Compustat quarterly (CCM) | s07_panel |
| `gind_name` | GICS industry name | text | config.GICS_INDUSTRIES | s07_panel |
| `gsubind` | GICS sub-industry code (8 digits) | code | Compustat quarterly (CCM) | s07_panel |
| `siccd` | latest CRSP SIC code | code | CRSP daily stock file (CIZ) | s07_panel |
| `first_price_date` | first CRSP trading day in our data | date | CRSP daily stock file (CIZ) | s07_panel |
| `last_price_date` | last CRSP trading day in our data | date | CRSP daily stock file (CIZ) | s07_panel |
| `first_universe_date` | first month end in factor_panel_monthly | date | derived | s07_panel |
| `last_universe_date` | last month end in factor_panel_monthly | date | derived | s07_panel |
| `excluded_non_member` | at least one of its IWM snapshots was excluded as a non-member (decision 11) | true/false | derived | s07_panel |

## column_stats.parquet

Statistics of every column of every other final table (prices_daily's years combined), rebuilt on every export; the Stats tables in this dictionary are printed from it.

One row per final table, column; key file, column. Built by stage `s08_export`.

| column | meaning | unit | source | stage |
|---|---|---|---|---|
| `file` | final table (prices_daily/year=YYYY/data.parquet for all daily years) | text | derived | s08_export |
| `column` | column name | text | derived | s08_export |
| `kind` | numeric, date, text or bool | text | derived | s08_export |
| `dtype` | Parquet (Arrow) type | text | derived | s08_export |
| `rows` | rows of the table | count | derived | s08_export |
| `non_null_pct` | share of rows with a value | percent | derived | s08_export |
| `min` | minimum of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `p1` | 1st percentile of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `p25` | 25th percentile of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `median` | median of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `mean` | mean of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `p75` | 75th percentile of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `p99` | 99th percentile of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `max` | maximum of the non-missing values (numeric columns) | the column's unit | derived | s08_export |
| `date_min` | earliest date (date columns) | date | derived | s08_export |
| `date_max` | latest date (date columns) | date | derived | s08_export |
| `unique` | distinct values (text and date columns) | count | derived | s08_export |
| `top_values` | the 5 most common values with their share of rows, as JSON [[value, percent], ...] (text columns) | text | derived | s08_export |
| `true_pct` | share True of the non-missing values (boolean columns) | percent | derived | s08_export |

## Sectors

`gsector` and `gsector_name` are Compustat GICS (the current classification, applied to all history). Where GICS is missing, `sector_name` maps the CRSP SIC code to a GICS sector name, first match wins:

| GICS sector | SIC ranges |
|---|---|
| Unclassified | missing, 0, 9900-9999 |
| Health Care | 2830-2836, 3841-3851, 8000-8099, 8731 |
| Consumer Staples | 100-999, 2000-2199, 2840-2844, 5140-5149, 5400-5499, 5912 |
| Consumer Discretionary | 2200-2399, 2500-2599, 3630-3639, 3651, 3710-3716, 3750-3751, 3940-3949, 5200-5999, 7000-7099, 7200-7299, 7500-7599, 7900-7999, 8200-8299 |
| Energy | 1200-1399, 2900-2999 |
| Utilities | 4900-4949, 4960-4999 |
| Real Estate | 6500-6599, 6798 |
| Financials | 6000-6799 |
| Communication Services | 2710-2799, 4800-4899, 7810-7899 |
| Information Technology | 3570-3579, 3600-3699, 3820-3829, 7370-7379 |
| Materials | 1000-1199, 1400-1499, 2400-2499, 2600-2699, 2800-2899, 3200-3399 |
| Industrials | every other SIC code |

GICS sector codes: 10 Energy, 15 Materials, 20 Industrials, 25 Consumer Discretionary, 30 Consumer Staples, 35 Health Care, 40 Financials, 45 Information Technology, 50 Communication Services, 55 Utilities, 60 Real Estate.
