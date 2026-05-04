# Factor Lake User Workflow

Use this guide when you want to run a factor experiment in the [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) and turn it into a class result.

## 1. Open the project links

| Tool | Link |
|---|---|
| Streamlit App | [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) |
| GitHub Repo | [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake) |
| Supabase Project | [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm) |

## 2. Understand the app in one sentence

Factor Lake lets you choose a group of stock-selection factors, rank stocks by those factors, run a historical backtest, and compare the result to benchmarks.

## 3. Start each analysis with the same order

1. Open the Streamlit app.
2. Set sidebar controls.
3. Choose factors.
4. Load data.
5. Run the backtest.
6. Read the results.
7. Write down the assumptions you used.

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
2. Wait for the app to fetch data from the [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm).
3. Confirm the filters you chose are applied.
4. Make sure the app says the data loaded successfully.

## 7. Run the backtest

1. Click Run Portfolio Analysis.
2. Wait for the simulation to complete.
3. Confirm you see a success message.
4. Open the Results tab if the app does not move there automatically.

## 8. Read the results in order

1. Start with Performance Summary.
2. Check the Ranked Stocks table.
3. Review Portfolio Growth Over Time.
4. Review Year-by-Year Performance.
5. Review Top vs Bottom Cohort Analysis.
6. Review Advanced Backtest Statistics.
7. Review Yearly Win/Loss Summary.

## 9. Use the result to answer the assignment question

1. Write down the factor set you used.
2. Write down the sidebar settings you used.
3. Write down the main performance result.
4. Write down how it compared with the benchmark.
5. Explain why the result may have happened.
6. Repeat the test with a second window or factor combination if you need more evidence.

## 10. Keep the factor logic straight

1. ROE and ROA are profitability signals.
2. Momentum signals look for continuation in price performance.
3. Price-to-book and earnings yield are valuation signals.
4. Volatility and accruals are quality/risk proxies.
5. Asset growth and CapEx growth are growth signals.

## 11. Know what the app is measuring

1. Returns come from the Next-Year Return % field.
2. Delisting handling depends on the strategy you choose.
3. The backtest is historical, not predictive.
4. One strong result is not enough on its own.

## 12. Avoid the common mistakes

1. Do not run the backtest before loading data.
2. Do not change settings and forget to rerun.
3. Do not compare experiments with different time windows without noting it.
4. Do not assume one factor is always best.

## 13. Troubleshoot when results look wrong

1. If you get no records, broaden the filters.
2. If results are empty, try a different factor or longer time window.
3. If data fails to load, check the Supabase-related guides.
4. If the UI changed, check the Streamlit guide for maintainers.

## 14. Write up your experiment the same way every time

1. State your research question.
2. List the factors and directions.
3. List your sidebar settings.
4. Report the main numbers.
5. Explain the result.
6. Note any follow-up test you would run.

## 15. Related guides

1. [Deployment](DEPLOYMENT.md)
2. [Supabase Setup](SUPABASE_SETUP.md)
3. [Streamlit Styling Guide](STREAMLIT_STYLING_GUIDE.md)

## 16. Reference

Use this section when you want a quick reminder of what the app is doing behind the scenes.

1. Factor categories include momentum, value, profitability, quality, and growth.
2. The Results tab is where you verify the backtest output.
3. The app compares your strategy with benchmark behavior.
4. Returns come from the dataset's next-year return field.
5. Delisting behavior changes the outcome, so keep that setting recorded in every experiment.
6. The [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm) is the backing data source.
