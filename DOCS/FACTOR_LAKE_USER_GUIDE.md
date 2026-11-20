# Factor-Lake User Guide

Factor-Lake builds factor-based stock portfolios and compares their historical performance. **WRDS is the only
source for new analyses.** Older Legacy analyses remain available by ID or share link in Results and Compare.

## 1. Configure an analysis

Choose the period, initial AUM, weighting and factor-combination method in the sidebar. Set the rebalancing
frequency, holdings percentage or count, GICS filters, fossil-fuel restriction, trading costs and optional
liquidity limit. Add factors and review their Higher/Lower directions and tooltip explanations.
Holding 100% of the filtered universe needs no factor. Other holdings rules require a factor.

## 2. Check the calculation preview

Expand **Calculation preview** below the factor controls. It reflects the current WRDS settings and requires no
market-data download. **Step-by-step calculation** explains selection, weighting and returns.

## 3. Run the analysis

Click **Run Portfolio Analysis**. The configured AWS Lambda reads the WRDS data, runs or retrieves the backtest,
and returns the result. There is no Load Market Data step. Without a configured Lambda, the Run button is disabled.
A completed run opens Results. The calculation used by each saved run remains attached to its original settings.

## 4. Review and share saved results

Results shows each run's settings, holdings, charts and performance measures. Use the Analysis ID or **Copy share
link** to open the stored result in another session. This works for both WRDS and saved Legacy results and does
not start a backtest or load the full market dataset. Invalid, missing or expired IDs show a message.
Saved analyses last **1 year**: a weekly cleanup deletes results saved more than a year ago, so an older ID
shows the expired message. Run the analysis again to get a new ID.
Localhost links work on the computer running the app; deployed links use the deployed app's address.

Use **Add to comparison** or enter IDs in Compare to compare two to four saved analyses. Legacy results retain
their original source labels and methodology; they are not converted to WRDS.

## 5. Save settings and reset inputs

**Export settings (JSON)** saves a run's inputs. Import a WRDS file or use a WRDS run's **Use these settings** to
prepare another analysis. Names and notes are included in settings files; renaming does not change calculations.
Legacy files can be retained/exported for reference, but cannot prepare a new run. Their saved analyses still open.
See [Strategy settings](STRATEGY_SETTINGS.md).

**Clear Parameters** resets factors and sidebar inputs to WRDS defaults and clears any preview table. Saved runs,
IDs, results and names remain available. Choose new inputs and run again.

## 6. Preview WRDS fields

When the team enables sample display, open **Data Viewer** and filter by Sector, Industry group, Industry or
Sub-industry. Click **Load preview** for up to 10 matching rows, each a different company (its most recent row).
Filters apply before sampling. Empty filters are explained. The preview uses native field units, is not
representative of a portfolio, and has no download control.

Preview filters are independent of the Analysis sidebar. Changing filters, pressing **Clear preview**, or leaving
the viewer releases its rows. The full WRDS dataset stays outside the app. Until the preview is enabled, the
viewer shows "WRDS sample rows aren't available yet." See [preview behavior and release process](WRDS_DATA_PREVIEW.md).

## 7. Moving from Legacy

Use the [Legacy-to-WRDS table](WRDS_FACTOR_REVIEW.md#legacy-to-wrds) to find each Legacy factor's WRDS
replacement. **Same** means the same definition; **Similar** rows list how the definitions differ, so values are
not guaranteed to match. Saved Legacy analyses remain accessible; new Legacy runs are unavailable.

## 15. Use WRDS mode

WRDS is the only data source for new analyses. It backtests on point-in-time data built from WRDS (CRSP, Compustat, IBES and the Russell 2000 holdings of the IWM fund), and rebalances **semi-annually** (June, December), **quarterly** (March, June, September, December) or **monthly**, at month ends. Saved Legacy results retain their original annual 9/30 methodology.

**Point in time.** At each rebalance date the universe is the Russell 2000 members known on that date, the factors use only data public by then, and returns start after the date. Each holding period is buy and hold until the next rebalance.

### Run a WRDS backtest

1. Open Analysis. WRDS is already selected; there is no data-source switch. A configured AWS Lambda is required to run.
2. Review the selected factors, or use Clear Parameters to restore WRDS defaults.
3. Pick the **rebalancing frequency** (Quarterly by default) and the start and end years (2002 to 2026). The backtest covers those calendar years, and the caption under the years shows the rebalances, for example "12 quarterly rebalances, 2009-12 to 2012-09; holds to 2012-12".
4. Weighting, factor combination, sector filter, fossil fuel filter and initial AUM work as in Legacy, and the sector filter can go finer (see **Filter by GICS industry** below). There is no delisting option (see below).
5. **Hold**, under the weighting, sets how many of the filtered stocks the portfolio holds at each rebalance, best scores first:
   - **Top X%**: 1 to 100, default 10 (as in Legacy). In Mixing, each factor's sleeve buys its own top X% with an equal share of the money. At least one stock is always held. **100%** holds every stock in the filter, with or without a score, so the factors do not matter: you can run with no factor, for example to hold a whole sector or sub-industry.
   - **Top N stocks**: N from 1, default 50. The portfolio holds **N different stocks in total**, weighted across all of them (equal or by market cap). In Mixing the factors take turns in the order they were added: each adds its next-best stock by its own score (ties go to the lower PERMNO), skipping stocks another factor already picked, until N are picked. If the factors run out first, every stock with a score is held. The holdings' **Picked By** names the factor whose turn picked each stock. Integrating holds the top N by the combined score. Top N only ever holds stocks with a score: on a date where fewer than N stocks have one, it holds all of those, even when N is at least the size of the filter (only **Top 100%** holds every stock in the filter). The filter's size changes by date, so a Top N run still needs a factor.
   The choice is shown in the run's **Universe** line ("Holds: Top 20 stocks"), in Compare's **Hold** row, in the calculation preview, and saved in exported settings files. Files and analyses saved before it hold the top 10%.
6. Add factors on the Analysis tab, grouped as **Momentum, Value, Profitability, Quality, Growth, Risk, Analyst and Size**; tinted cards with a link icon replace a Legacy factor, and each tooltip gives the default direction and why. The [factor review](WRDS_FACTOR_REVIEW.md) maps every Legacy factor to its WRDS replacement and explains the defaults.
7. Click **Run Portfolio Analysis**. There is no Load Market Data step: the backtest reads the WRDS data in AWS.

### Check the calculation preview

In WRDS mode, **Calculation preview** on the Analysis tab (collapsed, below Factor Selection) shows the WRDS engine's calculation for the backtest the Run button would send: the rebalance dates of your frequency and years, the month-end Russell 2000 universe after the GICS and fossil fuel filters, the z-scores (winsorized at 1% and 99% per date), the top X% or top N stocks with ties broken by PERMNO (or, at 100%, every stock in the filter), the weights, buy-and-hold periods on CRSP total returns with delisting returns included, the trading cost when set, and the IWM benchmark (IWO and IWN shown too). Each WRDS run's tab in Results keeps **Calculation used in this run** below Performance Summary, and a saved WRDS analysis you open shows the one from its stored settings. It never shows Legacy's formula.

### Filter by GICS industry

1. Tick **Enable Sector Filter**, then pick a **Filter level**: Sector, Industry group, Industry or Sub-industry.
2. One list appears per level down to the one you picked. Each list only offers what is inside your choices above it, so pick the sectors first (or leave all of them), then the industry groups, and so on. An empty list in the middle means all of it.
3. The portfolio can only hold stocks in your choices at the picked level. If that list is empty, the filter is off.
4. At the Sector level, a stock without GICS is placed by its SIC code, as before. At the finer levels, stocks without GICS are left out.

Where the codes and names come from, and how many stocks have no GICS: [GICS mapping](GICS_MAPPING.md).

The run's **Universe** caption and the **Sectors** row in Compare show the filter, e.g. "Industry: Banks, Insurance". The holdings table lists each stock's GICS sector, industry group, industry and sub-industry, and exported settings files keep the level and the choices.

### Limit positions by daily volume

Under **Liquidity Limit** in the sidebar (WRDS only), choose the most a position may be, as a share of its stock's daily volume, so the backtest only buys stocks your portfolio could realistically trade:

- **No limit** (the default): positions of any size, as before.
- **20%**, **10%** or **5%**: no position above that share of the stock's average daily dollar volume.

Liquidity is the stock's average daily dollar volume (price x shares traded) over its last 63 trading days, computed from the WRDS daily prices. Early Nasdaq volume is not adjusted for double counting, so before about 2004 a Nasdaq stock's volume can be overstated and the limit is looser for it.

At each rebalance, every position is sized at the portfolio's value on that date with the chosen weighting, and a position above the limit is cut to the limit: each stock holds the smaller of its target and the limit times its daily dollar volume. A stock with no volume data gets no position. The money that cannot be placed is held in cash until the next rebalance and earns the risk-free rate. Nothing is redistributed: every other position keeps the weight the strategy gave it, and no stock is left out for being too big.

The limit depends on the money invested: at the default $1,000 nothing is capped and the run is the same as with No limit, while at $100 million many small stocks are capped and much of the portfolio can sit in cash. Try the same strategy at a small and a large **Initial AUM**.

The run's **Liquidity Limit** caption, the **Liquidity Limit** row in Compare and the calculation preview show the limit, and exported settings files keep it. A run with a limit also shows, under Performance Summary, the average cash share, the most cash at one rebalance (with its date) and the average number of positions capped, and **Positions capped at each rebalance** lists every date with its capped positions, stocks without volume data and cash share. Its holdings table adds **Avg Daily $ Volume** and **% of Daily Volume**.

The dollar volume is computed from the daily prices already in the WRDS folder and saved next to them as one small file, so no new data pull is needed. It is loaded the first time a run with a limit needs it, and runs with No limit never load it.

### Read WRDS results

The results have Legacy's sections, tiles, tables and charts, on dates instead of years:

- The run's tab reads e.g. "Run 2: WRDS, Quarterly".
- **Holdings and Ranks**: pick a rebalance date to see what was held from that date, with weights, returns and factor values in percent as in Legacy, and a CSV download. **All ranked stocks on this date** ranks every stock the strategy could buy on that date.
- Legacy's growth chart against **IWM** (Russell 2000), **IWN** (Value) and **IWO** (Growth), and below it each period's return.
- **Period-by-Period Performance**: Legacy's Year-by-Year table with one row per period (switch to **By calendar year** for yearly returns). Then the cohort comparison, the statistics annualized by the frequency (the caption says how), **Factor Risk** (below), and a win/loss table per period, e.g. **Quarterly Win/Loss Summary**.
- **Factor Risk** shows how much of the return is exposure to common factors rather than alpha. The run's monthly returns after trading costs, minus the risk-free rate, are regressed on the Fama-French five factors (market, size, value, profitability, investment) and momentum, from the same WRDS data. It shows the annualized alpha (the intercept x 12) with its t-stat, R², each factor's exposure and t-stat (Newey-West standard errors), and one line such as "Mostly a value and momentum tilt; alpha not significant." (the two strongest significant tilts; the table has them all). Size is left out of that line: the factors are measured against the whole US market, so every Russell 2000 strategy shows a small-cap (positive Size) exposure, as the note under the table says. A |t| of 2 or more counts as significant. The section needs at least 36 months, so shorter runs do not show it, and neither do WRDS analyses saved before it existed.
- The Analysis ID shares a WRDS run exactly like a Legacy one.

### How WRDS mode treats delistings and gaps

- **Delistings**: CRSP delisting returns are included. After a stock delists, its money earns the risk-free rate until the next rebalance.
- **Missing returns**: a month without a return (not a delisting) counts as 0%. The results say how many stock-months that was.
- **Risk-free rate**: Fama-French one-month T-bill. A month it is not published yet is filled from the 3-month T-bill / 12, and the results list those months.

### Limitations

- **Sectors and industries use today's GICS codes**, not the codes a company had at the time, so the GICS filter is the one input that is not strictly point in time.
- **The benchmarks are ETF total returns** (IWM, IWN, IWO): they include dividends and are net of the funds' fees. They differ slightly from the index returns Legacy uses, so do not compare Legacy and WRDS numbers directly.
- **The data runs from July 2002 to September 2026.** A backtest ending in 2026 holds until September 2026 (semi-annual: June 2026), and one starting in 2002 starts with the first rebalance month in the data (July 2002 monthly, September 2002 quarterly, December 2002 semi-annual).
- **Before 2008 the index members come from annual Russell 2000 lists.** Each list is used from the June reconstitution until the next one; there are no mid-year additions (IPOs join at the next list), and a stock leaves when it stops trading. From 2008 the members are IWM's monthly holdings.
- **Sectors are each company's current GICS classification**, applied to all of its history: there is no historical GICS, so a company that changed sector shows its latest one throughout.
- **Analyst data is thin in the early years.** IBES capital expenditure estimates start in 2006, and EBIT, dividend and cash flow estimates are sparse in 2001-2004, so analyst factors cover fewer stocks then.
- Data Viewer offers a bounded WRDS preview only after sample-display approval. It has no data download control; see section 6.

## 16. Trading costs

Set **Trading cost (bps per trade)** in the sidebar (0 to 500, default 0) to see returns after trading costs and how much each strategy's turnover costs it. It applies to new WRDS analyses, including the cohort comparison. Saved Legacy results retain their original cost settings.

**What is charged.** Every buy and every sell at each rebalance pays the cost on its dollar value. That includes the first purchase (the backtest starts in cash, so it buys 100% of the portfolio) and putting a delisted stock's cash back into stocks at the next rebalance. Nothing is charged for the delisting itself, for holding cash, or for selling at the end of the backtest.

**The formula.** At each rebalance:

- The weights after drift are last period's weights grown by each stock's return, as shares of the portfolio's value now. A delisted stock's money counts as cash, not as a stock.
- Turnover = the sum over all stocks of |target weight − weight after drift|. Buys and sells both count, so selling one stock to buy another is 2 × its weight. A stock held by two Mixing sleeves is one position.
- Cost = bps / 10,000 × turnover × portfolio value. It comes out of the portfolio at the rebalance, before the next period's return: return after costs = (1 − cost / value) × (1 + return before costs) − 1.

For example, at 25 bps a rebalance that turns over 100% of a $10,000 portfolio costs $25 (0.25%).

**Typical range.** Russell 2000 stocks often cost roughly **20 to 60 bps per side**, and more for the smallest and least liquid names. 0 bps gives exactly the results the app gave before trading costs existed.

**What the results show** (only when the cost is above 0):

- The Performance Summary is after costs, with a caption giving the cost and the final value, total return and CAGR before costs. CAGR, Sharpe, drawdown and every other statistic use the returns after costs.
- The growth chart adds a dashed **Before costs** line.
- **Advanced Backtest Statistics** adds a Trading Costs row: average turnover per rebalance, total cost paid ($), annualized cost drag (CAGR before costs minus CAGR after) and CAGR before costs.
- The Year-by-Year (WRDS: Period-by-Period) table adds **Turnover %** and **Cost %** (percent of the portfolio's value at that rebalance). By calendar year, turnover is summed and cost compounded over the year's rebalances.
- **Holdings and Ranks** adds **Traded %** (the stock's |target − weight after drift|) to the table and its CSV, and the caption gives that rebalance's turnover and cost.
- The run tab, its header captions and the saved-analysis summaries show the cost, e.g. "Run 3: ROA % +1 | 25 bps".

Analyses saved before trading costs existed open at 0 bps.

**Limitation.** The cost is the same for every stock. Real costs are higher for small, illiquid stocks and for large trades; costs by stock size or liquidity are not modelled yet.

## 17. Compare analyses

The **Compare** tab, after Results, shows 2 to 4 saved analyses side by side, like comparing phones before buying one. Legacy and WRDS analyses can be compared together.

1. Open **Compare**. It starts with 2 boxes; **+ Add analysis** adds one, up to 4, and boxes 3 and 4 have a remove button (✕).
2. Paste an analysis ID in a box, or add one from the lists under the boxes: **This session's analyses**, **Earlier analyses on this browser** and **All saved analyses**, the same lists as in Results. Each entry's **Add to comparison** button puts its ID in the first empty box (adding a box if all are filled and there are fewer than 4); an entry already in the comparison shows **Added**. If all 4 boxes are full, a short message says so. As in Results, the lists hold analyses only, not cohort comparisons or ranked lists.
   In Results, each run tab with an analysis ID has **Add to comparison** beside its ID: it adds the run and opens the Compare tab. It shows **In comparison** once the run is there, and a short message instead of switching when all 4 boxes are full. Runs computed in the app have no ID, so they have no button.
3. The comparison appears as soon as 2 boxes hold valid IDs; there is nothing to click. The saved results are read from S3 once each and kept for the session, so adding, removing or sorting never reads one again. No backtest runs and no market data is loaded.

Only saved analysis IDs work, so only runs done in AWS Lambda can be compared. A box with a mistyped or expired ID, the same ID as another box, or the ID of a cohort comparison or ranked list shows a short error, and the other analyses still compare as long as at least 2 are valid. Empty boxes are skipped.

**The table** has one column per analysis, headed by its label and short ID; above the table each analysis has its color, its data source and frequency, and its full ID with a copy button. The rows are grouped:

- **Settings**: data source, frequency, period, initial AUM, factors with their directions, factor combination, weighting, sectors, fossil fuels, delisting strategy, benchmark and trading cost.
- **Performance**: final portfolio value, total return, CAGR, cumulative outperformance against the analysis's own benchmark (Russell 2000 for Legacy, IWM for WRDS), and its best and worst period.
- **Risk**: volatility (raw), Sharpe, max drawdown, beta and information ratio, with the same names and formats as Results, then a WRDS analysis's Factor Risk alpha, alpha t-stat and R² (n/a for Legacy and for WRDS analyses without the section). Sortino and tracking error are not in the backtest statistics, so they are not shown.
- **Costs**: CAGR before costs, average turnover per rebalance, total trading cost paid and annualized cost drag. Analyses run without a trading cost, or saved before trading costs existed, show 0 bps, n/a turnover, $0 cost and 0% drag.

**Reading it:**

- The best value of each row is green: higher CAGR, return, Sharpe, information ratio and factor alpha; the smallest drawdown; lower volatility, turnover, cost paid and cost drag. When analyses tie for the best value (they look the same), all of them are green; when every analysis ties, none is. Settings, beta, the alpha t-stat, R² and n/a are never highlighted.
- Volatility, Sharpe and information ratio are annualized, so they compare across frequencies as they are. Best and worst period and turnover per rebalance are each analysis's own periods (a year, half-year, quarter or month), so they are highlighted only when the analyses have the same frequency. Likewise, final portfolio value and total trading cost paid are in dollars, so they are highlighted only when every analysis has the same initial AUM; returns, CAGR, ratios and cost drag are always compared.
- A ≠ after a row's name means its values differ. **Show only differences** hides the rows that are the same.
- **Sort by** reorders the columns best first by CAGR, total return, Sharpe, max drawdown, volatility, information ratio or cost drag (n/a last, ties in the order entered). **Entered order** puts them back. Sorting never reloads the results.

**The chart** overlays every analysis's portfolio after costs, each in its own color, on one date axis, and draws each benchmark once (one Russell 2000 line, one IWM line), grey and dotted. A benchmark's line comes from the first analysis in the current order that uses it, so sorting can change which period it covers. When the analyses have different initial AUMs, every line (each portfolio and each benchmark) is scaled to start at $1,000 and the title says **Growth of $1,000**; when they match, the lines show dollar values. Each analysis keeps its color when you sort; the legend follows the sort. Legacy's yearly values are placed on September 30, since Legacy rebalances on the 9/30 data.

## 18. Backtest scoreboard

The **Scoreboard** tab, after Compare, ranks every saved analysis (everyone's, Legacy and WRDS) by the metric you choose, so you can see which factor combinations did best and open the winners.

1. Open **Scoreboard**. The table lists the analyses ranked by **Sharpe**, best first, 20 at a time; **Show all** lists the rest.
2. **Rank by** changes the metric: Sharpe, CAGR, cumulative outperformance, max drawdown, volatility, win rate, total return or final value. The ranked column's header has ▼ (highest first) or ▲ (lowest first, for volatility). Max drawdown is a loss, so the smallest loss ranks first. Analyses that tie (they look the same) share a rank and are listed newest first.
3. Narrow it down with the filters: **Data source** (Legacy, or WRDS with its frequency), **Factor combination** (Mixing or Integrating), **Uses every factor** (only analyses with all the factors you pick) and **Period within** (only analyses whose whole period lies within those years).
4. Pick an analysis under **Open an analysis** (listed by rank). Its full ID is shown with a copy button. **Open** opens it in Results as a new tab, as if you had pasted its ID; **Compare** adds it to the Compare tab and opens that.

The metrics are the ones the analysis's Results show (and the Compare tab's): the backtest Lambda saves them with each analysis, so the scoreboard reads them without downloading any result or rerunning anything. It reads them again at most once a minute; **Refresh** reads them now. Outperformance and win rate are against each analysis's own benchmark (Russell 2000 for Legacy, IWM for WRDS).

**Comparing fairly.** Analyses cover different periods and start with different amounts. Rank by Sharpe, CAGR or another rate rather than final value, which only compares at the same initial amount, and use **Period within** for a like-for-like ranking. A WRDS analysis's frequency is in its Source column.

Only saved analyses are ranked, so only runs done in AWS Lambda appear. Cohort comparisons and ranked lists are not analyses and are left out. Analyses saved before sharing existed have no settings to score, so the scoreboard leaves them out and says how many; they are still listed under **All saved analyses** in Results. Saved analyses are deleted after 1 year, and they leave the scoreboard then.

