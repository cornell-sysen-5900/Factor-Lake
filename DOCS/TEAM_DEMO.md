# Team demo: live calculation visibility

## Start locally

```powershell
uv run python scripts/run_demo.py
```

Open http://127.0.0.1:8502. The DEMO MODE banner identifies synthetic stock data and
returns; these are not real market performance. Benchmarks use the historical series
already bundled with the application. No AWS or Supabase credentials are required.
The regular app on port 8501 retains its normal data configuration.

## Two-minute walkthrough

1. Show Analysis before loading data: Calculation Preview displays the no-factor placeholder.
2. Add ROA % and 12-Mo Momentum %. Show both factor terms, then remove and re-add one.
3. Change a direction, switch Equal Weight to Market Cap Weight, toggle ESG exclusions,
   and select sectors. Show the matching formula updates.
4. Switch Mixing to Integrating and expand Step-by-step calculation. Explain normalization,
   stock selection, allocation, delisting and annual compounding.
5. Set Start Year 2010, End Year 2024 and initial AUM $1,000. Click Load Market Data,
   then Run Portfolio Analysis. Show Calculation Used in This Run in Results.
6. Change a sidebar setting while on Results. The saved formula remains unchanged.

Suggested narration: "Users can inspect the full strategy before loading data. Each choice
updates the calculation preview immediately. Results preserve the formula used for that
specific run, so later parameter changes cannot rewrite its explanation. This demonstration
uses synthetic stock data to show the workflow, not to claim investment performance."

This contribution implements calculation visibility only. The Clear and Rerun with New
Params feature is excluded because another teammate owns that implementation.

## Tests

```powershell
uv run pytest tests/unit/test_calculation_preview.py tests/unit/test_demo.py tests/unit/test_saved_runs.py
```
