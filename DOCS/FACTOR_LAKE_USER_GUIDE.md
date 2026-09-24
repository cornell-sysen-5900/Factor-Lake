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

1. Decide whether to restrict fossil fuel companies.
2. Choose equal weight or market cap weight.
3. Turn on sector filtering only if you need it.
4. Set the start and end years.
5. Set the initial investment amount.
6. Pick the delisting strategy you want to test.

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
2. The captions at the top of each tab show the factors and directions, period, initial AUM, universe, weighting, and delisting strategy that run used.
3. Saved tabs do not change when you edit the sidebar or factors afterwards. Run the analysis again to create a new tab.
4. Click Remove this run to delete a tab you no longer need.
5. Only the last 5 runs are kept, and all saved runs are cleared when you refresh the page, so record your numbers.

## 8. Read the results in order

1. Pick the run tab you want and check its captions.
2. Start with Performance Summary.
3. Check the Ranked Stocks table.
4. Review Portfolio Growth Over Time.
5. Review Year-by-Year Performance.
6. Review Top vs Bottom Cohort Analysis. It uses the settings of the run tab it is in.
7. Review Advanced Backtest Statistics.
8. Review Yearly Win/Loss Summary.

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
