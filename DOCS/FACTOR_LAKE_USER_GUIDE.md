# Factor Lake User Workflow

Use this guide when you want to run a factor experiment in the [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) and turn it into a class result.

## 1. Open the project links

| Tool | Link |
|---|---|
| Streamlit App | [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) |

## 2. Understand the app in one sentence

Factor Lake lets you choose a group of stock-selection factors, rank stocks by those factors, run a historical backtest, and compare the result to benchmarks.

## 3. Start each analysis with the same order

1. Open the Streamlit app.
2. Set sidebar controls.
3. Choose factors.
4. Load data.
5. Run the backtest.
6. Read the results.

## 4. Set the sidebar first

The steps below are for the default **Legacy** data source. For WRDS data with semi-annual, quarterly or monthly rebalancing, see [15. Use WRDS mode](#15-use-wrds-mode).

1. Decide whether to restrict fossil fuel companies.
2. Choose equal weight or market cap weight.
3. Choose how multiple factors are combined:
   - Mixing (default): each factor buys its own top stocks with an equal share of the money. A stock picked by two factors is held twice.
   - Integrating: stocks are ranked by their average score across all the factors and the top ones are bought once each. Stocks missing any selected factor are left out.
   With one factor, both give the same portfolio.
4. Turn on sector filtering only if you need it.
5. Set the start and end years.
6. Set the initial investment amount.
7. Pick the delisting strategy you want to test.
8. Set the trading cost in basis points per trade, or leave it at 0. See [16. Trading costs](#16-trading-costs).

## 5. Choose factors and directions

1. Open the Analysis tab.
2. Select one or more factors.
3. Decide whether each selected factor should rank high-to-low or low-to-high.
4. Keep the direction consistent with the financial meaning of the factor.
5. If you are unsure, start with one factor and add more later.

## 6. Load the market data

1. Click Load Market Data.
2. Wait for the app to fetch data.
3. Confirm the filters you chose are applied.
4. Make sure the app says the data loaded successfully.

## 7. Run the backtest

1. Click Run Portfolio Analysis.
2. Wait for the simulation to complete.
3. Confirm you see a success message. It names the saved run, e.g. "Run 2: ROA %".
4. Open the Results tab. The newest run's tab is selected automatically.

### Saved runs

1. Every successful run is saved as its own tab in Results, newest first.
2. The captions at the top of each tab show the factors and directions, period, initial AUM, universe, factor combination method, weighting, and delisting strategy that run used.
3. Saved tabs do not change when you edit the sidebar or factors afterwards. Run the analysis again to create a new tab.
4. Click Remove this run to delete a tab you no longer need.
5. Only the last 5 runs are kept, and all saved runs are cleared when you refresh the page, so record your numbers.

### Share an analysis or open it later

When backtests run in AWS Lambda, every analysis is saved for 90 days under an **analysis ID**.

1. The ID is shown in the captions at the top of each run's tab. Click the copy button on the right of the ID.
2. To open an analysis (yours or a colleague's), go to **Results → Open a saved analysis**, paste the ID and click **Open analysis**. It opens as a new tab, labelled "(opened)", with the original results, factors and settings. It loads from S3 and does not rerun the backtest.
3. **This session's analyses** in the Results tab lists every analysis you ran or opened since the page was loaded, with its ID, including runs no longer shown as tabs. It never shows other people's analyses.
4. Forgot to copy an ID? **Earlier analyses on this browser** lists the analyses you ran or opened in earlier sessions on the same browser, each with its ID and an **Open** button. They are kept in your browser only, so they don't follow you to another computer and disappear if you clear the site's data.
5. **All saved analyses** lists everyone's saved analyses (newest 20), with their factors, method, period and starting amount, and an **Open** button. It loads only when you expand it. WRDS analyses start with "WRDS" and their frequency, e.g. "WRDS, Quarterly | ROE (higher), ROA (higher) | ...", and the two lists above show the same line under a WRDS run.
6. Runs computed in the app (with the Lambda switch off) are not saved, so they have no ID.
7. An ID that is mistyped, or older than 90 days, shows an error saying no saved analysis has that ID.
8. To see 2 to 4 saved analyses side by side, use the **Compare** tab ([17. Compare analyses](#17-compare-analyses)).
9. To see which saved analyses did best, use the **Scoreboard** tab ([18. Backtest scoreboard](#18-backtest-scoreboard)).

## 8. Read the results in order

1. Pick the run tab you want and check its captions.
2. Start with Performance Summary.
3. Check the Holdings and Ranks table. Pick a rebalance year to see every stock the backtest bought that year: its rank, score, factor values, weight, shares, buy price, dollar value, period return, contribution and whether it was delisted. Mixing runs also show which factors picked each stock. The contributions add up to the portfolio return shown above the table. Download CSV exports the full table.
4. Review Portfolio Growth Over Time.
5. Review Year-by-Year Performance.
6. Review Top vs Bottom Cohort Analysis. It uses the settings of the run tab it is in.
7. Review Advanced Backtest Statistics.
8. Review Yearly Win/Loss Summary.

With a trading cost above 0, every number is after costs, and the views add what the costs did: see [16. Trading costs](#16-trading-costs).

### Check the raw data

1. Open the Data Viewer tab after loading market data in the Analysis tab. It uses that data and downloads nothing new.
2. Pick a date range, and optionally one or more sectors or industries. Leaving a list empty means all of them.
3. Click Load data to see the matching rows, then Download CSV to save exactly that table.
4. Changing a filter or clicking Clear removes the table. Click Load data again to see it.

## 9. Keep the factor logic straight

1. ROE and ROA are profitability signals.
2. Momentum signals look for continuation in price performance.
3. Price-to-book and earnings yield are valuation signals.
4. Volatility and accruals are quality/risk proxies.
5. Asset growth and CapEx growth are growth signals.

## 10. Know what the app is measuring

1. Returns come from the Next-Year Return % field.
2. Delisting handling depends on the strategy you choose.
3. The backtest is historical, not predictive.
4. One strong result is not enough on its own.

## 11. Avoid the common mistakes

1. Do not run the backtest before loading data.
2. Changing settings does not update saved tabs. Run again to create a new tab. If you change sectors, the fossil-fuel filter, or years, click Load Market Data first.
3. Do not compare experiments with different time windows without noting it.
4. Do not assume one factor is always best.

## 12. Troubleshoot when results look wrong

1. If you get no records, broaden the filters.
2. If results are empty, try a different factor or longer time window.
3. If data fails to load, check the Supabase-related guides.
4. If the UI changed, check the Streamlit guide for maintainers.

## 13. Write up your experiment the same way every time

1. State your research question.
2. Note the run label and list the factors and directions.
3. List your sidebar settings.
4. Report the main numbers.
5. Explain the result.
6. Note any follow-up test you would run.


## 14. Reference

Use this section if you want a quick understanding of what the app is doing behind the scenes.

1. Factor categories include momentum, value, profitability, quality, and growth.
2. The Results tab is where you verify the backtest output.
3. The app compares your strategy with benchmark behavior.
4. Returns come from the dataset's next-year return field.
5. Delisting behavior changes the outcome, so keep that setting recorded in every experiment.
6. The **Factor Lake Supabase Project** is the backing data source.

## 15. Use WRDS mode

WRDS mode is a second data source next to Legacy. It backtests on point-in-time data built from WRDS (CRSP, Compustat, IBES and the Russell 2000 holdings of the IWM fund), and rebalances **semi-annually** (June, December), **quarterly** (March, June, September, December) or **monthly**, at month ends. Legacy is unchanged: annual rebalancing on the 9/30 data, with the same results as before.

**Point in time.** At each rebalance date the universe is the Russell 2000 members known on that date, the factors use only data public by then, and returns start after the date. Each holding period is buy and hold until the next rebalance.

### Run a WRDS backtest

1. In the sidebar, set **Data Source** to **WRDS**. It is only available when the app is connected to AWS Lambda, because WRDS backtests run there; otherwise it is greyed out with a note.
2. Switching the data source clears the selected factors, because the two modes offer different factors.
3. Pick the **rebalancing frequency** (Quarterly by default) and the start and end years (2008 to 2026). The backtest covers those calendar years, and the caption under the years shows the rebalances, for example "12 quarterly rebalances, 2009-12 to 2012-09; holds to 2012-12".
4. Weighting, factor combination, sector filter, fossil fuel filter and initial AUM work as in Legacy. The portfolio buys the top 10%, as in Legacy. There is no delisting option (see below).
5. Add factors on the Analysis tab. They are grouped as **Momentum, Value, Profitability, Quality, Growth, Risk, Analyst and Size**, and each has a default direction and a tooltip. Factors that default to "Lower is better" say why.
6. Click **Run Portfolio Analysis**. There is no Load Market Data step: the backtest reads the WRDS data in AWS.

### Read WRDS results

The results have Legacy's sections, tiles, tables and charts, on dates instead of years:

- The run's tab reads e.g. "Run 2: WRDS, Quarterly".
- **Holdings and Ranks**: pick a rebalance date to see what was held from that date, with weights, returns and factor values in percent as in Legacy, and a CSV download. **All ranked stocks on this date** ranks every stock the strategy could buy on that date.
- Legacy's growth chart against **IWM** (Russell 2000), **IWN** (Value) and **IWO** (Growth), and below it each period's return.
- **Period-by-Period Performance**: Legacy's Year-by-Year table with one row per period (switch to **By calendar year** for yearly returns). Then the cohort comparison, the statistics annualized by the frequency (the caption says how), and a win/loss table per period, e.g. **Quarterly Win/Loss Summary**.
- The Analysis ID shares a WRDS run exactly like a Legacy one.

### How WRDS mode treats delistings and gaps

- **Delistings**: CRSP delisting returns are included. After a stock delists, its money earns the risk-free rate until the next rebalance.
- **Missing returns**: a month without a return (not a delisting) counts as 0%. The results say how many stock-months that was.
- **Risk-free rate**: Fama-French one-month T-bill. A month it is not published yet is filled from the 3-month T-bill / 12, and the results list those months.

### Limitations

- **Sectors use today's GICS codes**, not the codes a company had at the time, so the sector filter is the one input that is not strictly point in time.
- **The benchmarks are ETF total returns** (IWM, IWN, IWO): they include dividends and are net of the funds' fees. They differ slightly from the index returns Legacy uses, so do not compare Legacy and WRDS numbers directly.
- **The data runs from April 2008 to September 2026.** A backtest ending in 2026 holds until September 2026 (semi-annual: June 2026), and one starting in 2008 starts with the first rebalance month in the data.
- The Data Viewer shows Legacy data only.

## 16. Trading costs

Set **Trading cost (bps per trade)** in the sidebar (0 to 500, default 0) to see returns after trading costs and how much each strategy's turnover costs it. It works in Legacy and WRDS mode, including the cohort comparison.

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
- **Risk**: volatility (raw), Sharpe, max drawdown, beta and information ratio, with the same names and formats as Results. Sortino and tracking error are not in the backtest statistics, so they are not shown.
- **Costs**: CAGR before costs, average turnover per rebalance, total trading cost paid and annualized cost drag. Analyses run without a trading cost, or saved before trading costs existed, show 0 bps, n/a turnover, $0 cost and 0% drag.

**Reading it:**

- The best value of each row is green: higher CAGR, return, Sharpe and information ratio; the smallest drawdown; lower volatility, turnover, cost paid and cost drag. When analyses tie for the best value (they look the same), all of them are green; when every analysis ties, none is. Settings, beta and n/a are never highlighted.
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

Only saved analyses are ranked, so only runs done in AWS Lambda appear. Cohort comparisons and ranked lists are not analyses and are left out. Analyses saved before sharing existed have no settings to score, so the scoreboard leaves them out and says how many; they are still listed under **All saved analyses** in Results. Saved analyses are deleted after 90 days, and they leave the scoreboard then.

