# Factor Lake User Guide

This guide is for users who want a practical walkthrough of what Factor Lake does and how to use it effectively for coursework.

## What Factor Lake does

Factor Lake is a portfolio backtesting tool. You pick a set of stock-selection factors (for example momentum, value, profitability, or growth), define how each factor should be ranked, and run a historical simulation.

At a high level, the app helps you:

- test a factor hypothesis on historical Russell 2000-style universes,
- compare your strategy against benchmark indices,
- evaluate return and risk metrics (CAGR, drawdown, Sharpe, win rate, etc.),
- inspect ranked stocks and top-vs-bottom cohort behavior.

## Before you start

Make sure you have:

- access to the Streamlit app URL.

## App workflow (recommended order)

Use this sequence each time so results are reproducible.

1. Configure global settings in the sidebar.
2. Select one or more factors in the Analysis tab.
3. Click Load Market Data.
4. Click Run Portfolio Analysis.
5. Review outputs in the Results tab.
6. Save your assumptions and findings for your class write-up.

## Step 1: Sidebar settings

The sidebar controls universe constraints and simulation settings.

### ESG Filters

- Restrict Fossil Fuel Companies: excludes oil/gas/coal-related industries.

### Portfolio Weighting

- Equal Weight: each selected stock gets the same weight.
- Market Cap Weight: larger companies get larger weights.

### Delisting Strategy

- Zero Return: assigns 0% return to delisted names.
- Hold Cash: moves delisted capital to risk-free cash.
- Reinvest: reallocates delisted capital across surviving positions.

### Sector Selection

- Enable Sector Filter to restrict analysis to selected sectors.

### Analysis Period

- Set Start Year and End Year for your backtest window.

### Initial Investment

- Set Initial AUM ($) to define portfolio starting value.

## Step 2: Choose factors and directions

In the Analysis tab, choose factors from categories such as:

- Momentum
- Value
- Profitability
- Quality
- Growth

For each selected factor, set direction:

- High to Low: favors high factor values.
- Low to High: favors low factor values.

Use directions intentionally. For example, some valuation metrics are often used with low-to-high ranking to target cheaper stocks.

## Step 3: Load market data

Click Load Market Data.

What this does:

- fetches market/factor data from Supabase,
- applies your sector and ESG filters,
- applies your date range.

If data loads successfully, the app confirms record count and enables the run step.

## Step 4: Run the backtest

Click Run Portfolio Analysis.

The engine then:

- maps your factor choices to internal data columns,
- builds yearly ranked cohorts,
- simulates rebalancing over your selected years,
- computes benchmark-relative and risk statistics.

When finished, a success message appears and the app routes you to Results.

## Step 5: Interpret the Results tab

Key sections you will use for assignments:

- Performance Summary: final value, total return, CAGR, cumulative outperformance.
- Ranked Stocks: selected cohort and composite scores for the ranking year.
- Portfolio Growth Over Time: strategy growth vs benchmarks.
- Year-by-Year Performance: annual returns across portfolio and benchmarks.
- Top vs Bottom Cohort Analysis: sanity-check whether your signal differentiates outcomes.
- Advanced Backtest Statistics: drawdown, Sharpe, volatility, beta, information ratios.
- Yearly Win/Loss Summary: annual head-to-head vs benchmarks.

## Understanding core factor ideas (quick reference)

This section mirrors the in-app About tab and adds class-friendly context.

- ROE / ROA: profitability and capital efficiency.
- 12M / 6M / 1M Momentum: trend persistence at different horizons.
- Price-to-Book, Book/Price, Next FY Earnings/P: valuation signals.
- 1Y Price Volatility: stability/risk proxy.
- Accruals/Assets: earnings quality proxy.
- 1Y Asset Growth / 1Y CapEx Growth: corporate growth and reinvestment behavior.

## Notes on methodology and assumptions

- Returns are based on Next-Year Return % in the dataset.
- Delisting handling depends on the selected Delisting Strategy.
- Results are historical simulations, not forecasts.
- Factor performance is regime-dependent, so test multiple periods.

## Suggested classwork template

For each experiment in your assignment, document:

1. Research question.
2. Factor set and directions.
3. Sidebar settings (weighting, ESG, sectors, years, AUM, delisting strategy).
4. Main outcomes (CAGR, total return, drawdown, benchmark outperformance).
5. Interpretation: why results may have occurred.
6. Robustness checks (different time windows, sectors, or factor combinations).

## Common mistakes to avoid

- Running analysis without loading data first.
- Changing settings after a run but forgetting to rerun.
- Comparing experiments with different date windows or delisting strategies.
- Over-interpreting one strong period without robustness checks.

## Troubleshooting

- No records after filtering: broaden sector/ESG/date constraints.
- Empty or weak results: try additional factors, different directions, or a longer period.

## Related guides

- Deployment: how the app is hosted and configured.
- Supabase Setup: data source and credentials details.
- Streamlit Styling Guide: UI customization for maintainers.
