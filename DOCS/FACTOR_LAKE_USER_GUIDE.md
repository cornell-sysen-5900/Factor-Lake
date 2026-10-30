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

## 8. Read the results in order

1. Pick the run tab you want and check its captions.
2. Start with Performance Summary.
3. Check the Holdings and Ranks table. Pick a rebalance year to see every stock the backtest bought that year: its rank, score, factor values, weight, shares, buy price, dollar value, period return, contribution and whether it was delisted. Mixing runs also show which factors picked each stock. The contributions add up to the portfolio return shown above the table. Download CSV exports the full table.
4. Review Portfolio Growth Over Time.
5. Review Year-by-Year Performance.
6. Review Top vs Bottom Cohort Analysis. It uses the settings of the run tab it is in.
7. Review Advanced Backtest Statistics.
8. Review Yearly Win/Loss Summary.

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
