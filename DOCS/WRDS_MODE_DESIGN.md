# WRDS mode: design

Status: design approved (stage 1 of 6), branch `feature/wrds-mode`. The review's decisions are in [Decisions](#decisions) and are worked into the sections below.

WRDS mode is a second data source next to today's data (called **Legacy** below). It backtests on the point-in-time WRDS panel (`pipeline/DATA_DICTIONARY.md`) with semi-annual, quarterly or monthly rebalancing, and it runs **only in AWS Lambda**: the app sends the job and shows the result. Legacy keeps working exactly as it does today.

Contents:

1. [Where Legacy and WRDS split](#1-where-legacy-and-wrds-split)
2. [WRDS factors and sectors](#2-wrds-factors-and-sectors)
3. [Rebalance dates and look-ahead rules](#3-rebalance-dates-and-look-ahead-rules)
4. [Benchmarks and metrics](#4-benchmarks-and-metrics)
5. [Jobs, cache keys, saved IDs and holdings files](#5-jobs-cache-keys-saved-ids-and-holdings-files)
6. [How Lambda loads the WRDS data, with measurements](#6-how-lambda-loads-the-wrds-data-with-measurements)
7. [The vectorized engine and the Legacy match test](#7-the-vectorized-engine-and-the-legacy-match-test)
8. [Each feature in WRDS mode, and the app without Lambda](#8-each-feature-in-wrds-mode-and-the-app-without-lambda)
9. [Test plan, open questions and risks](#9-test-plan-open-questions-and-risks)
10. [Stages and files](#10-stages-and-files)

A rule for this document and the code that follows: the WRDS data is licensed, so **no statistics of the data** (row counts, coverage, returns) go into the repo (see `pipeline/README.md`). The measurements in section 6 are file sizes, memory and run times only.

## Decisions

Decided in the stage 1 review:

| # | Question | Decision | Where |
|---|---|---|---|
| 1 | Delisting during a quarter or half year | The money is held as cash earning the risk-free rate until the next rebalance | 3.3 |
| 2 | Ranked stocks | A new `ranked` job that ranks every stock on one rebalance date at a time | 5.1, 8 |
| 3 | Fossil fuel filter | Keep the Legacy mapping (the GICS and SIC codes in 2.3; Oil & Gas Storage & Transportation is not excluded) | 2.3 |
| 4 | Default directions where research is mixed | Lower is better for all of them. The 1-month factor is named "1-Month Return (Reversal)". Every tooltip that defaults to Lower says why in one short sentence | 2.1 |
| 5 | Unclassified stocks | Left out when the sector filter is on | 2.2 |
| 6 | Today's GICS codes for the sector filter | Accepted, documented as a limitation | 2.2, R3 |
| 7 | Defaults | Quarterly, 2008-2026 | 8 |
| 8 | Missing return without a delisting | 0% for that month; the result reports how many stock-months that was | 3.3 |
| 9 | ETF total-return benchmarks | Accepted, labelled for example "IWM (ETF, total return)" | 4.1 |
| 10 | Missing risk-free month | Filled with `tbill_3m / 12`, and the filled months are flagged in the result | 4.1 |
| 11 | Top % | Fixed at 10%, as in Legacy | 5.1 |
| - | Legacy invariant (correction) | Legacy **results** must not change; its code version and cache keys may. Routing WRDS jobs may edit a file in `ENGINE_SOURCES` (for example `backtest_jobs.py`) | 1.2 |

---

## 1. Where Legacy and WRDS split

### 1.1 The split points

| Layer | Legacy (unchanged) | WRDS (new) | Where the choice is made |
|---|---|---|---|
| Sidebar | today's controls | Data source, frequency, GICS sectors, no delisting option, no Lambda switch | `settings['data_mode']`, new key; Legacy controls are drawn by the same code with the same widget keys |
| Factor list | `FACTOR_METADATA` | `WRDS_FACTOR_METADATA` (section 2) | `render_factor_selection(data_mode)` |
| Run | `run_backtest_logic` (local or Lambda) | `run_wrds_backtest` (Lambda only) | one `if` at the top of `run_backtest_logic`; the Legacy body below it is not edited |
| Job | `build_job` (`src/backtest_jobs.py`) | `build_wrds_job` (`src/wrds_jobs.py`) | the job's `data_mode` key: absent means Legacy |
| Lambda | `BacktestService.run` | `WrdsBacktestService.run` | one dispatch on `job.get('data_mode')` in `BacktestService.handle_event` |
| Engine | `src/backtest_engine.py` | `src/wrds_engine.py` (vectorized) | separate modules |
| Data | `full_precision_test.parquet`, `last_price_mapping.parquet` | `wrds_2026_10/factor_panel_monthly.parquet`, `returns_monthly.parquet`, `market_monthly.parquet` | the service that runs the job |
| Benchmarks, metrics | `src/benchmarks.py`, `src/performance_metrics.py` | `src/wrds_metrics.py` (IWM, IWN, IWO, Fama-French rf) | separate modules |
| Results tab | `render_results_tab` | `render_wrds_results_tab` (new `app/components/wrds_results.py`, reusing Legacy's render helpers read-only) | `run['data_mode']` |
| Saved IDs | `backtest-results/<hash>.json` | the same file layout, plus `<hash>.holdings.parquet` | the saved job's `data_mode` |
| Data Viewer | Legacy data | stays Legacy | not split |

### 1.2 How Legacy results stay unchanged

The invariant is that **Legacy results do not change**. Legacy's code version and cache keys may change: editing a file in `ENGINE_SOURCES` (for example `backtest_jobs.py`, to route WRDS jobs) changes the code version, so Legacy results are computed once more under new keys, with the same numbers. Old Legacy analysis IDs still open, because opening reads the saved file by its ID and never recomputes it (5.4).

1. **Engine.** The Legacy engine (`backtest_engine.py`, `benchmarks.py`, `performance_metrics.py`, `universe_selection.py`, ...) is not edited. WRDS code lives in new `src/wrds_*.py` modules that may import Legacy files but do not change them. Files that only route jobs, such as `backtest_jobs.py`, may get a WRDS branch.
2. **Job.** Legacy jobs are still built by `build_job`, which reads only the keys it knows. The sidebar's Legacy settings are exactly as on `main` (no `data_mode` key), and a Legacy job never has a `data_mode` key.
3. **Data.** Legacy reads the same two files. The WRDS data sits in its own folder (`factor-lake-data/wrds_2026_10/`), so publishing it changes nothing Legacy reads.
4. **Local Legacy runs.** `run_backtest_logic`'s Legacy body, `load_and_process_data` and `select_universe` are not edited. The WRDS branch is decided before the Legacy code runs.
5. **Pinned results.** `tests/unit/test_legacy_pinned.py` runs fixed Legacy jobs (Mixing and Integrating, all three delisting strategies, equal and cap weight, a sector filter, a cohort comparison) through `build_job` and `run_job`. It compares the whole result, holdings included, with `tests/fixtures/legacy_pinned_results.json`, which was captured from `main` (c063100) before any WRDS code existed.
6. **Tests.** Every existing test passes without being edited. New behaviour gets new test files. If an existing app test has to change because the sidebar gained a widget, that is a sign the Legacy path changed, and the change is reviewed as such.

---

## 2. WRDS factors and sectors

### 2.1 Factor list

Every factor is a column of `factor_panel_monthly.parquet` whose value at month end t was public by t (data dictionary, "Point in time"). The display names describe the WRDS definition, so they are not Legacy's names (no "9/30 Data"). **Direction** is the default when the factor is added: Higher = buy the highest values (`'top'`), Lower = buy the lowest (`'bottom'`).

| Group | Column | Display name | Default | Tooltip |
|---|---|---|---|---|
| Momentum | `mom_12_1` | 12-1 Mo Momentum | Higher | Return over the past 12 months, skipping the most recent month. |
| Momentum | `mom_12m` | 12-Mo Momentum | Higher | Return over the past 252 trading days. |
| Momentum | `mom_6m` | 6-Mo Momentum | Higher | Return over the past 126 trading days. |
| Momentum | `mom_1m` | 1-Month Return (Reversal) | Lower | Return over the past 21 trading days. Lower is better: last month's losers tend to rebound the next month. |
| Value | `f_book_to_price` | Book/Price | Higher | Book equity divided by the company's market cap today. |
| Value | `f_earnings_to_price` | Earnings/Price | Higher | Trailing 12-month earnings divided by market cap. |
| Value | `f_sales_to_price` | Sales/Price | Higher | Trailing 12-month revenue divided by market cap. |
| Value | `f_fcf_yield` | FCF Yield | Higher | Trailing 12-month free cash flow divided by market cap. |
| Value | `f_dividend_yield` | Dividend Yield | Higher | Trailing 12-month dividends divided by market cap. |
| Value | `f_ebitda_to_ev` | EBITDA/EV | Higher | Trailing 12-month EBITDA divided by enterprise value. |
| Value | `f_ebit_to_ev` | EBIT/EV | Higher | Trailing 12-month EBIT divided by enterprise value. |
| Value | `f_sales_to_ev` | Sales/EV | Higher | Trailing 12-month revenue divided by enterprise value. |
| Value | `f_forward_ep` | Forward E/P (FY1) | Higher | Analyst consensus EPS for this fiscal year divided by price. |
| Value | `f_forward_ep_fy2` | Forward E/P (FY2) | Higher | Analyst consensus EPS for next fiscal year divided by price. |
| Value | `f_forward_cfp` | Forward Cash Flow/P | Higher | Analyst consensus cash flow per share divided by price. |
| Value | `f_forward_sales_to_ev` | Forward Sales/EV | Higher | Analyst consensus sales divided by enterprise value. |
| Value | `f_forward_ebitda_to_ev` | Forward EBITDA/EV | Higher | Analyst consensus EBITDA divided by enterprise value. |
| Profitability | `f_roe` | ROE | Higher | Trailing 12-month net income divided by book equity. |
| Profitability | `f_roa` | ROA | Higher | Trailing 12-month net income divided by total assets. |
| Profitability | `f_gross_profitability` | Gross Profitability | Higher | Revenue minus cost of goods sold (trailing 12 months), divided by total assets. |
| Quality | `f_accruals_to_assets` | Accruals/Assets | Lower | Net income minus operating cash flow, over total assets. Lower is better: earnings backed by cash tend to last. |
| Quality | `f_leverage` | Debt/Assets | Lower | Total debt divided by total assets. Lower is better: less debt means less risk of financial distress. |
| Quality | `f_current_ratio` | Current Ratio | Higher | Current assets divided by current liabilities. |
| Quality | `f_net_share_issuance` | Net Share Issuance | Lower | Change in split-adjusted shares outstanding over the past year. Lower is better: companies that issue shares tend to trail those that buy them back. |
| Growth | `f_asset_growth_1yr` | 1-Yr Asset Growth | Lower | Total assets vs four quarters earlier. Lower is better: fast-growing companies tend to underperform. |
| Growth | `f_sales_growth_1yr` | 1-Yr Sales Growth | Higher | Trailing 12-month revenue vs a year earlier. |
| Growth | `f_capex_growth_1yr` | 1-Yr CapEx Growth | Lower | Trailing 12-month capex vs a year earlier. Lower is better: companies that ramp up investment tend to underperform. |
| Growth | `f_rd_to_sales` | R&D/Sales | Higher | Trailing 12-month R&D divided by revenue. |
| Risk | `vol_close_12m` | 12-Mo Volatility | Lower | Annualized volatility of daily returns over the past year. Lower is better: calmer stocks have earned as much with less risk. |
| Risk | `vol_close_6m` | 6-Mo Volatility | Lower | Annualized volatility of daily returns over the past 6 months. Lower is better: calmer stocks have earned as much with less risk. |
| Risk | `vol_close_3m` | 3-Mo Volatility | Lower | Annualized volatility of daily returns over the past 3 months. Lower is better: calmer stocks have earned as much with less risk. |
| Risk | `vol_gk_1m` | 1-Mo Volatility (High-Low) | Lower | Garman-Klass volatility from daily open, high, low and close over the past month, annualized. Lower is better: calmer stocks have earned as much with less risk. |
| Risk | `beta_252d` | Beta (1-Yr Daily) | Lower | Market beta from daily excess returns over the past year. Lower is better: low-beta stocks have earned more than their risk suggests. |
| Risk | `beta_60m` | Beta (5-Yr Monthly) | Lower | Market beta from monthly excess returns over the past 5 years. Lower is better: low-beta stocks have earned more than their risk suggests. |
| Analyst | `f_sue` | Earnings Surprise (SUE) | Higher | Change in quarterly EPS vs a year earlier, scaled by its past volatility. |
| Analyst | `f_analyst_surprise` | Analyst EPS Surprise | Higher | Latest reported EPS minus the consensus before it, over price. |
| Analyst | `f_eps_revision` | EPS Revisions | Higher | Analysts revising this year's EPS up minus those revising down, over the number of analysts. |
| Analyst | `f_eps_dispersion` | Analyst Disagreement | Lower | Spread of analysts' EPS estimates relative to their mean. Lower is better: stocks analysts disagree on tend to underperform. |
| Analyst | `f_ltg` | Long-Term Growth Forecast | Lower | Analysts' long-term EPS growth forecast. Lower is better: the most optimistic forecasts tend to disappoint. |
| Size | `mkt_cap_musd` | Market Cap | Lower | Market value of the listed shares. Lower is better: smaller companies have historically earned more. |

Left out on purpose:

- `ret_next_1m` and `delisted_next_month` describe the month after t. They are the holding return, never a factor (section 3.3).
- The `_qe` columns (`f_book_to_price_qe`, ...) use Compustat's quarter-end price, which is up to a quarter old at t. The table above uses today's price instead.
- `vol_close_1m`, `vol_gk_3m`, `vol_gk_6m`, `vol_gk_12m` repeat factors already listed.
- `rv_multiples_monthly.parquet` (within-industry percentiles) is a separate file. It can be added later without changing the design.

Where the list lives:

- `app/streamlit_config.py` gets `WRDS_FACTOR_METADATA` and `WRDS_FACTOR_GROUPS`, in the same shape as `FACTOR_METADATA` (`key`, `group`, `column`, `tooltip`, `higher_is_better`), keyed by display name. Widget keys get a `w_` prefix so they never clash with Legacy keys.
- `src/wrds_factors.py` holds `WRDS_FACTOR_COLUMNS`, the allowlist the Lambda checks jobs against. A test checks that both lists name the same columns.

The factor picker today lays out one column per group, so eight groups need two rows (stage 2).

### 2.2 Sectors

The sector filter offers the 11 GICS sectors, using the panel's `sector_name` column (GICS, with the SIC fallback from the data dictionary): Communication Services, Consumer Discretionary, Consumer Staples, Energy, Financials, Health Care, Industrials, Information Technology, Materials, Real Estate, Utilities.

`sector_name` also has the value `Unclassified`, for stocks with neither GICS nor a mapped SIC code. These stocks are in the universe when the sector filter is off. They are not offered in the list, so turning the filter on leaves them out (decision 5).

**Limitation (decision 6):** the panel's GICS code is today's classification, not the historical one (data dictionary), so the sector filter is the one input that is not strictly point in time. A stock that moved sectors is filtered by its current sector for its whole history. The sector list's tooltip says so (risk R3).

### 2.3 Fossil fuel filter

Legacy excludes five FactSet industries. WRDS mode excludes the matching GICS sub-industries (`gsubind`): 10101010 Oil & Gas Drilling, 10101020 Oil & Gas Equipment & Services, 10102010 Integrated Oil & Gas, 10102020 Oil & Gas Exploration & Production, 10102030 Oil & Gas Refining & Marketing, 10102050 Coal & Consumable Fuels. Where GICS is missing, it excludes SIC 1221, 1222, 1231 (coal), 1311 (crude oil and gas), 1381, 1382, 1389 (oil and gas field services) and 2911 (refining). This keeps Legacy's mapping (decision 3): 10102040 Oil & Gas Storage & Transportation is not in Legacy's list, so it is kept.

---

## 3. Rebalance dates and look-ahead rules

### 3.1 Rebalance months

| Frequency | Rebalance months | Periods per year (k) |
|---|---|---|
| Semi-annual | Jun, Dec | 2 |
| Quarterly | Mar, Jun, Sep, Dec | 4 |
| Monthly | every month | 12 |

A rebalance date is the panel's `date` in that month: the last CRSP trading day of the month, not the calendar month end. The UI shows the real dates.

### 3.2 Start and end years

The sidebar still asks for a start year and an end year, from 2008 to 2026. **The backtest covers the calendar years from start to end**, within the data's bounds (first panel month 2008-04, last month with returns 2026-09):

- The **first rebalance** is the December month end of `start_year - 1`. When that is before the data starts, it is the first rebalance month in the data instead.
- The **last holding period ends** at the December month end of `end_year`. When that is after the data ends, it ends at the last rebalance month whose whole holding period is in the data. There are no partial periods.
- Rebalances are every rebalance month between the two. Each holding period runs from one rebalance date to the next.
- `start_year == end_year` is allowed (one calendar year). `start_year > end_year` is fixed by the sidebar as today. A range with no complete period is rejected with a message.

| Run | Rebalance dates | Holding periods cover |
|---|---|---|
| Monthly, 2008-2026 | 2008-04 ... 2026-08 (221) | May 2008 - Sep 2026 |
| Quarterly, 2008-2026 | 2008-06 ... 2026-06 (73) | Jul 2008 - Sep 2026 |
| Semi-annual, 2008-2026 | 2008-06 ... 2025-12 (36) | Jul 2008 - Jun 2026 |
| Quarterly, 2010-2012 | 2009-12 ... 2012-09 (12) | Jan 2010 - Dec 2012 |
| Monthly, 2015-2015 | 2014-12 ... 2015-11 (12) | Jan 2015 - Dec 2015 |

`src/wrds_calendar.py` holds this rule as a pure function that works on year-month values, so the sidebar can show "Rebalances: 2009-12 to 2012-09 (12 quarterly)" without any data. The Lambda then maps each month to the panel's trading date and rejects a job whose months are not in the data. The data bounds (`2008-04`, `2026-09`) are constants in `src/wrds_calendar.py`, and a data test checks them against the files.

### 3.3 Look-ahead rules

At each rebalance date t:

1. **Universe.** The stocks are the panel rows at date t: the Russell 2000 members of the latest IWM snapshot that was public by t (`snapshot_eff_dt <= t`). The engine never filters on anything dated after t. In particular, a stock is not dropped because it later has no return or delists.
2. **Buyable.** A stock needs `prc > 0` and `mkt_cap_musd > 0` at t. Both are known at t.
3. **Factors.** Only allowlisted columns (section 2.1), and only their values at t. The pipeline guarantees these were public by t (Compustat report date, IBES statistics date, announcement date). Z-scores use only the stocks at t.
4. **Weights.** Market-cap weights use `mkt_cap_musd` at t.
5. **Returns start after t.** A position bought at the close on t earns the stock's monthly returns for the months after t, up to and including the next rebalance date t'. These are `ret_m` from `returns_monthly.parquet` for each month in (t, t'], for every frequency. The panel's `ret_next_1m` holds the same values for one month, but the loader never reads it, so no column about the month after t is ever in memory next to the factors. `returns_monthly` keeps a stock's rows after it leaves the index, so a stock that drops out mid-period is still held to t'.
6. **Delisting.** CRSP's delisting return is included in the delisting month's `ret_m`. A stock has no rows after it delists. For the rest of the period, its money is held as cash earning the risk-free rate until the next rebalance (decision 1), and the position is marked Delisted.
7. **Missing return.** If a held stock has no return for a month and has not delisted (it stopped trading, or CRSP has no price), that month counts as 0% and the position is still held. The result reports how many stock-months that was (`missing_return_stock_months`; decision 8).
8. **Benchmarks** compound over the same months (t, t'] (section 4).

Section 9.1 lists the tests for these rules.

---

## 4. Benchmarks and metrics

### 4.1 Benchmarks

All benchmarks come from `market_monthly.parquet`, compounded over each holding period's months (t, t']:

| Role | Column | Shown as |
|---|---|---|
| Benchmark | `iwm_total_ret` | IWM (ETF, total return) |
| Value | `iwn_total_ret` | IWN (ETF, total return) |
| Growth | `iwo_total_ret` | IWO (ETF, total return) |
| Risk-free | `rf` | Fama-French one-month T-bill |

- These are ETF total returns (dividends in, fund fees out). Legacy uses index returns, so the two modes' benchmarks are not the same numbers.
- The Fama-French `rf` is published a few weeks late, so the newest month can be missing. A missing month is filled with that month end's 3-month T-bill yield divided by 12, `tbill_3m / 12`, and the result flags the filled months (`rf_filled_months`; decision 10).
- `benchmark_index` is not part of a WRDS job: the benchmark is always IWM.

### 4.2 Metrics

`src/wrds_metrics.py` returns the same metric keys as `compute_comprehensive_metrics`, so the Results tab's grids work unchanged. It works on period returns, with k periods per year (section 3.1):

| Metric | WRDS definition | Legacy (k = 1) |
|---|---|---|
| Period returns | portfolio, benchmarks and rf per holding period | yearly |
| CAGR | `(final / initial) ** (k / n) - 1` for n periods | `n` years |
| Volatility (raw, excess) | `std(ddof=1) * sqrt(k)` | the same with k = 1 |
| Sharpe | `mean(excess) / std(excess) * sqrt(k)` | the same with k = 1 |
| Information ratio | `mean(active) / std(active) * sqrt(k)` | the same with k = 1 |
| Beta | slope of period excess returns (not annualized) | the same |
| Max drawdown | on the portfolio value after each period | the same |
| Win rate | share of periods beating the benchmark ("period win rate") | yearly |
| Calendar-year returns | each calendar year's periods compounded (new) | none |

With k = 1 the formulas are Legacy's. A test feeds the same returns to both functions and checks that every shared key matches (section 9.1). The result also says `frequency`, `periods_per_year` and `risk_free_rate_source: "Fama-French one-month T-bill"`.

---

## 5. Jobs, cache keys, saved IDs and holdings files

### 5.1 WRDS jobs

A WRDS job is plain JSON like a Legacy job, built by `build_wrds_job` and checked by `validate_wrds_job` (`src/wrds_jobs.py`):

| Field | Values | Notes |
|---|---|---|
| `kind` | `backtest`, `cohort`, `ranked` | `ranked` is new (section 8) |
| `data_mode` | `wrds` | absent in Legacy jobs |
| `wrds_job_version` | `1` | for future changes to the schema |
| `factors` | allowlisted columns, in the order added | |
| `factor_directions` | `{column: 'top' or 'bottom'}` | |
| `factor_labels` | `{column: display name}` | shown when an ID is opened |
| `frequency` | `semiannual`, `quarterly`, `monthly` | |
| `start_year`, `end_year` | 2008-2026 | section 3.2 |
| `initial_aum` | > 0 | |
| `top_pct` | 10.0 | fixed, as in Legacy (decision 11) |
| `use_market_cap_weight` | bool | |
| `factor_combination` | `mixing`, `integrating` | |
| `restrict_fossil_fuels` | bool | section 2.3 |
| `selected_sectors` | sorted GICS sector names | |
| `cohort_pct` | number for `cohort`, else null | |
| `rebalance_date` | `YYYY-MM` for `ranked`, else null | must be one of the run's rebalance months |

There is no `delisting_strategy` (returns include delisting returns) and no `benchmark_index`.

### 5.2 Cache key

The key is computed by the unchanged `job_hash(job, data_version, code_version)`, so it is the same kind of 64-character SHA-256 as Legacy's:

- `job`: the WRDS job. Its `data_mode` key means it can never equal a Legacy job.
- `data_version`: the WRDS folder name plus the S3 ETags of the three files the engine reads, for example `wrds_2026_10|factor_panel_monthly.parquet:<etag>|returns_monthly.parquet:<etag>|market_monthly.parquet:<etag>`. A new WRDS pull goes into a new folder, so it gives new keys.
- `code_version`: a hash of `WRDS_ENGINE_SOURCES` (the `src/wrds_*.py` modules plus the shared files they import, such as `factor_utils.py`), computed the same way as `code_version()`.

### 5.3 Where results are stored

| Object | Contents |
|---|---|
| `backtest-results/<hash>.json` | the same envelope as Legacy, `{'format': 2, 'job': ..., 'result': ...}`. A WRDS backtest result holds everything except the holdings, plus `holdings: {'file', 'dates', 'rows', 'bytes'}` |
| `backtest-results/<hash>.holdings.parquet` | WRDS backtest holdings, one row group per rebalance date (section 5.5) |

- Both are under `backtest-results/`, so the existing 90-day lifecycle rule and the Lambda role's write permission already cover them.
- The Parquet file is written **before** the JSON, so a cache hit (the JSON exists) normally has its holdings. A cached backtest whose holdings file is missing anyway (deleted by hand, or expired first) is computed again rather than shown without holdings. With the Lambda role's s3:ListBucket missing, S3 answers 403 for a missing file, which counts as missing too.
- The result JSON's `holdings` entry is `{'file': 'backtest-results/<hash>.holdings.parquet', 'bytes', 'dates', 'rows'}`: the row groups' dates (rebalance dates on which something was bought) in order, and their row counts.
- The WRDS code version hashes `WRDS_ENGINE_SOURCES` (`src/wrds_jobs.py`): the `wrds_*.py` modules plus the shared files they use (`backtest_jobs.py`, `performance_metrics.py`, `s3_client.py`). Legacy's `ENGINE_SOURCES` list is unchanged, so editing WRDS code never recomputes Legacy results.
- Without holdings, a monthly result is a few hundred kilobytes, well inside the inline limit, so the S3-link path is rarely needed.

### 5.4 Saved IDs, old and new

- **Old IDs keep opening.** Legacy files, their path, `decode_saved` and `SAVED_FORMAT` do not change. The app treats a saved job without `data_mode` as Legacy, exactly as today. Analyses saved before sharing still get today's message.
- **WRDS IDs** open through the same "Open a saved analysis" box. `load_saved_analysis` reads them unchanged (their `kind` is `backtest`). `open_saved_analysis` builds a WRDS run from the job when `job.get('data_mode') == 'wrds'` (today it reads Legacy-only keys such as `delisting_strategy`).
- `cohort` and `ranked` IDs are refused with today's message ("This ID belongs to a cohort comparison..."), with the wording widened to cover ranked lists.

### 5.5 Holdings as Parquet, loaded per date

The holdings file has one row per stock held per rebalance date, sorted by date and rank, with one row group per date.

Columns: `rebalance_date`, `rank`, `permno`, `ticker`, `name`, `sector`, `picked_by` (Mixing), `score`, one column per factor (raw value at t), `weight`, `buy_price`, `dollar_value`, `period_return`, `contribution`, `delisted`.

- A stock picked by several Mixing sleeves is one row, merged the way `_build_holdings_rows` merges them today.
- The app renames the columns to Legacy's display headers (Rank, Ticker, Name, Sector, Picked By, Score, factor labels, Weight %, Shares, Buy Price, Dollar Value, Period Return %, Contribution %, Delisted) and adds PERMNO, because two securities can share a ticker on one date.

When the user picks a date, the app reads the file's footer and that date's row group with ranged S3 GETs (`src/wrds_holdings.read_holdings_date`): one GET of the last 64 KB (the footer, plus a second GET if the footer is bigger), then one GET of exactly that row group's bytes. It caches the table per (file, date) with `st.cache_data`.

**The app reads with its own S3 keys, not a presigned link from Lambda.** The app already reads `backtest-results/` with its own keys to open saved analyses by ID. A presigned link expires (an hour today), so a shared ID opened later would need a new Lambda call just to get a fresh link. With its own keys, opening an ID and reading any date's holdings never calls Lambda, and nothing expires before the 90-day rule deletes the files. Opening a result therefore downloads only the small JSON, and each date costs one small read. For a 3-factor monthly Mixing run, the prototype's holdings file was 9.0 MB as Parquet (the same rows are about 44 MB as JSON), and reading one date back took about 0.02 s locally.

---

## 6. How Lambda loads the WRDS data, with measurements

### 6.1 Files and columns

| File (under `factor-lake-data/wrds_2026_10/`) | Columns read | Use |
|---|---|---|
| `factor_panel_monthly.parquet` | `date`, `permno`, `ticker`, `security_name`, `sector_name`, `gsubind`, `siccd`, `prc`, `mkt_cap_musd`, `ret_next_1m`, and the 39 allowlisted factor columns (49 of 103) | universe, factors, weights, one-month returns |
| `returns_monthly.parquet` | `month_end`, `permno`, `ret_m`, `delisted_in_month` | returns for later months of a period, delisting months |
| `market_monthly.parquet` | `month_end`, `iwm_total_ret`, `iwn_total_ret`, `iwo_total_ret`, `rf`, `tbill_3m` | benchmarks, risk-free rate |

- The panel is one zstd row group, so dates cannot be skipped inside the file. The Lambda downloads each file with one GET and decodes only the listed columns. Text columns become pandas categoricals.
- The Lambda builds a month by permno matrix of `ret_m` once per container.
- It keeps the frames for the life of the container, next to the Legacy universe, and reloads them only when the WRDS data version changes, as `_load_universe` does for Legacy.
- The folder name is `WRDS_DATA_DIR = 'wrds_2026_10'` in `src/wrds_data.py`, overridable with `FACTOR_LAKE_WRDS_DIR`, which the deploy script sets on the function.
- `prices_daily/` is not needed: every rebalance is at a month end, so monthly returns are enough.

### 6.2 Measured on the real files

These numbers come from a throwaway prototype of the vectorized engine (section 7), run locally on the files in `processed/final/`: Windows desktop, Python 3.10, pandas 2.3.3, pyarrow 24.0.0. "Peak" is the process's peak working set.

| Step | Time | Memory |
|---|---|---|
| Panel file (148.6 MB on disk) read into memory as bytes | 0.05 s | 220 MB process |
| Decode the 49 needed columns | 0.14 s | 631 MB peak; the frame itself is 155 MB |
| (for comparison) decode all 103 columns | 0.23 s | 862 MB process |
| `returns_monthly` (16.2 MB on disk) and `market_monthly`, plus the return matrix | 0.08 s | about 500 MB after the bytes are freed |
| Monthly, Integrating, 2 factors, equal weight (221 periods) | 0.73 s | |
| Monthly, Mixing, 3 factors, cap weight | 1.09 s | |
| Monthly, Mixing, 10 factors, equal weight | 2.86 s | 723 MB peak |
| Quarterly, Integrating, 2 factors / Semi-annual, Mixing, 2 factors | 0.25 s / 0.15 s | |
| Write a monthly 3-factor Mixing run's holdings as Parquet | 0.12 s | 9.0 MB file |
| Whole session (load and 7 runs) | 8.0 s | 728 MB peak |

### 6.3 Estimates for Lambda (monthly run)

Lambda at 3008 MB gets about 2 vCPUs, and pandas mostly uses one core. We assume a core about 1.5 to 2 times slower than the desktop above, and S3 reads in the same region at about 80 to 100 MB/s.

| Case | Estimate |
|---|---|
| Cold container, first WRDS job: init (as today) + download 165 MB + decode + 3-factor monthly run + holdings write and PUT | about 8 to 12 s |
| Warm container, new 3-factor monthly job | about 2 to 4 s |
| Warm container, 10-factor monthly Mixing | about 5 to 7 s |
| Cohort comparison (two backtests, no holdings) | about 2 times a backtest, without the holdings write |
| Ranked list for one date | under 1 s warm |
| Cached repeat | under 0.5 s, as today |
| Peak memory | about 0.8 GB for WRDS, about 1.0 to 1.2 GB with the Legacy universe also warm |

### 6.4 Measured with the stage 3 engine

`scripts/check_wrds_engine.py <processed/final folder>` runs each case in a fresh process, as a Lambda invocation would, on the same Windows desktop:

| Case (monthly, 2008-2026) | Load | Run | Peak memory | Holdings table in memory |
|---|---|---|---|---|
| Whole universe, cap weighted (the IWM tracking check) | 0.4 s | 1.5 s | 0.95 GB | 41 MB |
| Mixing, 3 factors, equal weight | 0.3 s | 1.4 s | 0.84 GB | 21 MB |
| Mixing, 10 factors, equal weight | 0.4 s | 4.4 s | 1.05 GB | 57 MB |

The tracking check passed (correlation at least 0.99, tracking error at most 3% a year; the values are printed, not stored). Peak memory is above the 0.8 GB estimate of 6.3, because the holdings table and its merges are built in memory, and still about a third of the function's 3008 MB.

On Windows, repeated runs in one process get slower: after one engine run, even a plain numpy `argsort` benchmark in the same process is about 4 times slower, so the cause is the process's memory allocator, not the engine. Lambda runs Linux, and stage 6 measures warm repeats there.

### 6.5 Measured in the dev Lambda (stage 4)

The stage 4 code, deployed to the dev function `sysen5900-factor-lake-backtest-dev` (3008 MB, the course role), driven from the local app. The times are the Lambda REPORT lines, taken from the invocation log tail.

| Job | Lambda duration | Max memory used | Result JSON | Holdings Parquet |
|---|---|---|---|---|
| Monthly Mixing, 3 factors, 2008-2026, cold container (init 2.6 s, plus the first WRDS data load) | 6.8 s | 1013 MB | 87 KB | 23.2 MB |
| The same job again (cache hit) | 0.09 s | 1013 MB | 87 KB | |
| Quarterly Mixing, 3 factors, warm | 1.1 s | 1013 MB | 30 KB | 7.3 MB |
| Quarterly cohort, 10% (two backtests) | 1.4 s | 1013 MB | 4 KB | |
| Ranked list, one date | 0.3 s | 1013 MB | 0.9 MB | |
| Legacy backtest, warm | 2.3 s | 1013 MB | | |

The package is 61.7 MB zipped and 199 MB unzipped (Lambda allows 250 MB). The holdings file is bigger than the 9 MB prototype in 6.2: it has more text columns (name, sector, picked_by), and 221 small row groups compress less well than one large one. That only affects storage: a date is still read with two small ranged GETs. The Legacy result from the dev function matched `main`'s code run locally on the same S3 data, within the 1e-9 relative tolerance of `scripts/check_backtest_lambda.py` (the largest difference was rounding in the last binary digit, Linux vs Windows).

**Recommendation:** keep the function at 3008 MB and a 600 s timeout. The memory leaves plenty of headroom, and Lambda's CPU share grows with memory. Stage 6 replaces these estimates with real numbers from CloudWatch `REPORT` lines (Duration, Max Memory Used, Init Duration) for cold, warm and cached monthly runs, and records them in `DOCS/AWS_LAMBDA_BACKTESTS.md`.

---

## 7. The vectorized engine and the Legacy match test

### 7.1 Engine

`src/wrds_engine.py` works on whole frames, with no per-stock Python loops. It reproduces Legacy's math exactly.

1. **Rebalance rows.** The panel rows at the rebalance dates that are buyable and pass the sector and fossil fuel filters.
2. **Scores**, for each factor and date, as `normalize_series` computes them: values to numbers with inf as NaN, winsorized at the 1% and 99% quantiles (`groupby('date')[col].transform('quantile', q)`, the same linear interpolation as `Series.quantile`), negated for Lower, then z-scored with the date's mean and `std(ddof=1)`, with the same `sigma > 1e-12` guard.
3. **Selection.** Integrating: the mean of the factors' z-scores for stocks that have every factor. Mixing: each factor's own z-score in its own sleeve. Within each date (and sleeve), stocks are sorted by score, highest first, with **ties broken by `permno`**, so every machine picks the same stocks. The top `max(1, floor(n * top_pct / 100))` are kept, where n is the number of stocks with a score.
4. **Weights.** Equal, or `mkt_cap_musd` share, within the portfolio (Integrating) or within each sleeve, with each sleeve given `1 / number of factors` of the money (Mixing).
5. **Returns.** Each position's growth is the product of (1 + monthly return) over the period's months (section 3.3), with delisted money earning rf after the delisting month. The period return is `sum(weight * growth) - 1`: buy and hold within the period, rebalanced at each rebalance date.
6. **Outputs.** Period returns, values, holdings rows (merged per stock as in Legacy), counts of delisted positions and missing-return months, and the metrics (section 4).

As built in stage 3:

- **Result.** It has the Legacy result keys (per-period returns in percent under `yearly_returns`, metrics annualized by frequency). `years` holds the period boundary dates as `YYYY-MM-DD` strings, and each `yearly_comparisons` entry also has `period_start`, `period_end` and the year of its end. The extras are `rebalance_dates`, `period_end_dates`, `cagr`, `rf_returns`, `benchmark_labels`, `rf_filled_months`, `total_delisted_positions`, `missing_return_stock_months` and `empty_rebalance_dates`. The Results tab still turns `years` into ints, so stage 5 renders WRDS runs with dates.
- **Holdings DataFrame.** One row per stock per rebalance date. Weights, returns and contributions are decimals (Legacy's rows are percent), and ticker, name and sector are categoricals. Factor values are under their column names; stage 5 maps them to display names.
- **A date with nothing to buy** (for example a narrow sector filter): the money earns the risk-free rate for that period, and the date is listed in `empty_rebalance_dates`.
- **A holding period with a month without benchmark data** is refused with an error.
- **Shared code.** `performance_metrics.compute_comprehensive_metrics` gained `periods_per_year` (default 1: Legacy unchanged, which the pinned Legacy test checks), and `S3DataManager` gained a public `read_bytes`. Both files are in `ENGINE_SOURCES`, so Legacy's code version changes, as the review allowed.

The core (steps 2 to 5) does not depend on the frequency: it takes rows of (date, id, factor values, cap, return for the period), so it can also be run on Legacy's annual data. That is what the match test does.

### 7.2 Deliberate differences from the Legacy engine

| Topic | Legacy | WRDS |
|---|---|---|
| Stock id | ticker (duplicate tickers are deduplicated) | `permno` (two securities can share a ticker) |
| Ties | sort order depends on the machine | broken by `permno` |
| Stocks without a price | ranked, then skipped when bought (n counts them) | not buyable, so not ranked |
| Cap weight with a missing cap | weight 0 | not buyable |
| Delisting | three strategies, `Delist_Price` and date | CRSP delisting return, then cash at rf until the next rebalance |

### 7.3 The match test

`tests/unit/test_wrds_engine_matches_legacy.py` builds a synthetic Legacy-style frame (`Year`, `Ticker`, `Ending_Price`, `Next-Years_Return`, `Market_Capitalization`, several factors with some NaN), with values chosen so that none of the differences in 7.2 apply: every stock has a price, a cap and a return, tickers are unique, and no cut falls inside a block of tied scores. It then:

- runs `rebalance_portfolio` (Legacy, `zero_return`), and
- runs the WRDS core on the same rows (date = `Year`, id = ticker code, return = `Next-Years_Return / 100`, cap = `Market_Capitalization`),

for Mixing and Integrating, equal and cap weight, mixed directions, and `top_pct` 2, 10 and 50. It asserts that every year's stocks are the same set, each stock's weight, return and contribution are equal to a relative 1e-12, and the period returns and portfolio values are equal to a relative 1e-12. A second test runs a tied cut and checks that WRDS picks by `permno`. The metrics test in 4.2 covers the rest of the shared math.

---

## 8. Each feature in WRDS mode, and the app without Lambda

| Feature | WRDS mode |
|---|---|
| Sidebar | **Data source**: Legacy (default) or WRDS. WRDS shows **Rebalancing**: Semi-annual, Quarterly (default), Monthly; ESG filter; weighting; factor combination; the 11 GICS sectors; the period (2008-2026, with a caption naming the first rebalance and the last period end); initial AUM. The Delisting Strategy and **Run backtests in AWS Lambda** controls are hidden. |
| Mode switch | Changing the data source clears the factor selection and its direction widgets (an `on_change` callback), because the two factor lists differ. Saved run tabs stay. |
| Analysis tab | The WRDS factor list (section 2). Load Market Data is hidden. The caption says the backtest runs in AWS Lambda on the WRDS data. Run Portfolio Analysis sends a WRDS job. |
| Results header | Data source WRDS, frequency, the real first rebalance date and last period end, number of rebalances, "Delisting returns included (CRSP)", how it was computed, and the Analysis ID. |
| Summary | Final value, total return, CAGR over calendar years (n / k), cumulative outperformance vs IWM. |
| Holdings and Ranks | A rebalance-date picker (newest first by default, remembered per run as today); only that date is read from the holdings Parquet (section 5.5). Legacy's columns and units (Rank, Ticker, Name, Sector, Picked By, Score, factor values, Weight %, Shares, Buy Price, Dollar Value, Period Return %, Contribution %, Delisted), plus PERMNO and Months Without Return. Factor values use the WRDS display names: rates and returns in percent with a " %" header (like Legacy's "... %" columns), market cap in $M, other ratios as stored (`app/wrds_tables.py`). Legacy's method / holdings / period return caption and a CSV download of the date. |
| Ranked stocks | In an expander under the holdings table, **Rank every stock on this date** sends a `ranked` job for the picked date (decision 2). The table uses the holdings table's names and units: Rank (by composite score), Ticker, PERMNO, Name, Sector, Held, Weight %, Score, then per factor its value, its z-score and (Mixing) its own rank, and Price. Cached by hash; CSV download. |
| Charts | On a date axis (`Visualizations/wrds_plots.py`; Legacy's chart files are unchanged): portfolio value with "IWM (ETF, total return)", "Value: IWN (ETF, total return)" and "Growth: IWO (ETF, total return)" in Legacy's colors; period returns (portfolio bars, IWM line); drawdown from peak. Monthly series are drawn without markers. The returns and drawdown charts put the legend above the plot, because their data fills the top-left corner where Legacy puts it. |
| Performance table | One row per period: dates, portfolio value and return, IWM, IWN and IWO, risk-free, and whether each benchmark was beaten (Legacy's separate win/loss ledger folded in). A **By calendar year** switch compounds each calendar year's periods. |
| Cohort | Generate Comparison sends a WRDS `cohort` job (same frequency, dates and filters). CAGR uses calendar years, not the number of periods (the old sub-annual branch had this bug). |
| Advanced statistics | One table: Portfolio, IWM, IWN (Value) and IWO (Growth) side by side (total return, CAGR, volatility, excess volatility, Sharpe, max drawdown), then the portfolio's beta, information ratio and period win rate against each benchmark. A caption gives the frequency and how the figures are annualized. Notes appear under the summary only for counts that are not zero: delisted positions, stock-months without a return, filled risk-free months, rebalance dates with nothing to buy. |
| Saved runs | Each run keeps `data_mode` and `frequency`; tabs read "Run 2: WRDS, Quarterly" ("(opened)" when opened by ID; the factors are in the header). Legacy labels are unchanged. Up to `MAX_SAVED_RUNS` tabs, Legacy and WRDS mixed. |
| Share by ID | As Legacy (section 5.4). An opened WRDS run loads its holdings and ranked lists the same way. |
| Session history | As today. |
| Data Viewer | Stays Legacy. In WRDS mode a one-line note says it shows the Legacy data, and to switch to Legacy and Load Market Data to use it. |

**Without Lambda** (`FACTOR_LAKE_BACKTEST_LAMBDA` not set): the Data source control is shown but disabled on Legacy, with the note "WRDS mode runs in AWS Lambda, which is not configured for this app." Everything else is today's app. A WRDS analysis ID can still be opened if the app has S3 read access (results and holdings are read straight from S3). Its Ranked stocks and Generate Comparison buttons are then disabled with the same note.

---

## 9. Test plan, open questions and risks

### 9.1 Tests (new files; existing tests are not edited)

| File | Checks |
|---|---|
| `test_legacy_pinned.py` | fixed Legacy jobs give exactly the results they gave on `main` (section 1.2) |
| `test_wrds_calendar.py` | the rebalance dates in the section 3.2 table for each frequency; data bounds; `start == end`; out-of-range years rejected; trading-day dates used, not calendar month ends (stage 3, when dates come from the panel) |
| `test_wrds_factors.py` | the app's factor list and the Lambda allowlist match; keys, groups and tooltips; every Lower default says why; the 11 GICS sectors |
| `test_wrds_lookahead.py` | changing any row dated after t leaves the holdings at t unchanged; changing returns changes returns only, never the selection; non-allowlisted columns (`ret_next_1m`, `delisted_next_month`, ...) are rejected as factors; a stock that joins later is never held earlier; a stock that leaves mid-period is held to t' with `returns_monthly` returns; a period's return uses exactly the months in (t, t'] |
| `test_wrds_engine.py` | the Legacy match of section 7.3 (stocks, ranks, scores, weights, returns, Picked By) and per-date z-scores equal to `normalize_series`; drift within a period, delisting then cash at rf, missing-return months at 0% and counted, rf fill from `tbill_3m / 12`, benchmarks compounded; sectors, fossil fuel GICS and SIC codes, buyable stocks, `permno` tie-break, at least one stock; rebalance dates and holding periods per frequency from the 2008 start to the 2026 end; the metrics default equals Legacy and k annualizes volatility, Sharpe and IR |
| `test_wrds_service.py` | the same data from a local folder and a fake S3, only the needed columns decoded; the runner equals the engine; data reused until a file is republished |
| `test_wrds_jobs.py` | build and validate; unknown factor or frequency rejected; `ranked` needs a scheduled date; WRDS and Legacy hashes never collide; the app and Lambda allowlists match |
| `test_wrds_service.py` | with a fake S3, as in `test_backtest_lambda.py`: dispatch on `data_mode`; data version from the WRDS ETags; holdings Parquet written before the JSON; cache hit returns the same result; one row group per date; the Legacy path still returns today's result |
| `test_wrds_mode_switch.py` | with Streamlit AppTest: Legacy by default; switching resets factors; WRDS disabled without Lambda; WRDS hides delisting, Load Market Data and the Lambda switch; Legacy shows no WRDS control; the factor groups and defaults; Legacy settings and Lambda jobs exactly as on `main`, also after a visit to WRDS |
| `test_wrds_saved.py` | a Legacy format-2 fixture, a pre-sharing fixture and a WRDS fixture open; WRDS cohort and ranked IDs are refused; holdings load per date |
| `scripts/check_backtest_lambda.py --wrds` | in AWS: each frequency end to end; a cached repeat equals the first run; the WRDS engine run locally on the S3 files equals the Lambda result |
| Stage 6 validation | a monthly cap-weighted run of the whole universe tracks IWM: correlation of monthly returns at least 0.99 and annualized tracking error at most 3%. The prototype passed; the measured numbers stay out of the repo |

Stages 3 to 6 run the full existing suite on every commit, so any change to a Legacy result shows up as a failing Legacy test.

### 9.2 Open questions

All eleven questions from the stage 1 review are decided; see [Decisions](#decisions).

### 9.3 Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | An edit to a shared file changes Legacy results | `test_legacy_pinned.py` compares whole Legacy results with `main`'s; WRDS code lives in new modules (a changed Legacy cache key alone is allowed) |
| R2 | Look-ahead creeps in through a new factor or filter | factor allowlist; look-ahead tests that change future rows |
| R3 | The sector filter uses today's GICS | documented in the UI tooltip and in this design; historical GICS would need a new pipeline pull |
| R4 | Licensed WRDS values are shown in the app (holdings, ranked lists, factor values) and in shared IDs | same private bucket and access model as Legacy; no WRDS data in the repo; the Data Viewer stays Legacy; check the WRDS terms before the class demo |
| R5 | Big holdings files for many-factor monthly Mixing runs | Parquet with one row group per date, read per date, so the app never downloads the whole file |
| R6 | Cold starts download about 165 MB | one GET per file per container; warm reuse; at most a few seconds in region |
| R7 | Rounding differs between the Mac and Lambda | ties broken by `permno`; comparisons use a relative tolerance of 1e-9, as the Legacy check does |
| R8 | The 2026-09 data end and the late Fama-French month | data bounds in one place with a data test; rf fill flagged in the result |
| R9 | Monthly plots and tables with 221 points | thinned date ticks; scrolling tables; calendar-year view |

---

## 10. Stages and files

| Stage | Commit | New files | Changed files |
|---|---|---|---|
| 1 | this design | `DOCS/WRDS_MODE_DESIGN.md` | |
| 2 | mode switch and WRDS factor list | `src/wrds_factors.py`, `src/wrds_calendar.py`, `tests/unit/test_legacy_pinned.py` with `tests/fixtures/legacy_pinned_results.json`, `test_wrds_mode_switch.py`, `test_wrds_calendar.py`, `test_wrds_factors.py` | `app/components/sidebar.py`, `app/components/factor_selection.py`, `app/streamlit_config.py`, `app/streamlit_app.py` (WRDS Run shows a "coming in the next stage" note until stage 3); this design (decisions) |
| 3 | WRDS engine (not deployed) | `src/wrds_data.py`, `src/wrds_engine.py` (metrics included; no separate `wrds_metrics.py`), `src/wrds_service.py` (runner), `scripts/check_wrds_engine.py`, `tests/unit/wrds_fixtures.py`, `test_wrds_engine.py`, `test_wrds_lookahead.py`, `test_wrds_service.py` | `src/performance_metrics.py` (`periods_per_year`), `src/s3_client.py` (`read_bytes`); the dispatch in `src/backtest_service.py` and `FACTOR_LAKE_WRDS_DIR` in the deploy script move to stage 4 |
| 4 | jobs, cache, holdings, Lambda wiring (dev function only) | `src/wrds_jobs.py`, `src/wrds_holdings.py`, `app/components/wrds_results.py` (a first results view), `tests/fixtures/saved_main_*.json` (written by `main`'s code), `test_wrds_jobs.py`, `test_wrds_saved.py` | `src/wrds_service.py` (the cached service), `src/wrds_engine.py` (`rank_stocks`), `src/backtest_service.py` (routing, shared `_respond`), `src/backtest_jobs.py` (`code_version(sources=...)`), `src/saved_analyses.py` (message names the job kind), `app/streamlit_utils.py`, `app/streamlit_app.py`, `app/components/results_tab.py` (one dispatch), `scripts/deploy_backtest_lambda.py` (`--leave-lifecycle-rules`, `--wrds-dir`) |
| 5 | feature parity (no Lambda code changed, so no redeploy) | `app/wrds_tables.py`, `Visualizations/wrds_plots.py`, `tests/unit/results_snapshot.py` with `tests/fixtures/legacy_results_render_main.json` (Legacy's Results tab rendered by `main`'s code), `test_wrds_results.py` | `app/components/wrds_results.py` (the full view), `app/streamlit_utils.py` (tab labels), `app/streamlit_config.py` (factor units), `app/streamlit_app.py` (Data Viewer note) |
| 6 | validation and docs | | `scripts/check_backtest_lambda.py` (`--wrds`), `DOCS/AWS_LAMBDA_BACKTESTS.md`, `DOCS/FACTOR_LAKE_USER_GUIDE.md`, `README.md` |

Ideas taken from the old `feature/sub-annual-rebalancing` branch: month-end rebalance dates, date labels on the plots with thinned ticks, and cohort CAGR in calendar years. Avoided from it: an annual risk-free rate applied to every period, delisted positions at 0%, and a universe that was not point in time.
