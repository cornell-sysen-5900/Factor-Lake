# Supabase Maintenance Workflow

Use this guide when you need to add, clean, change, verify, or troubleshoot data in the [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm).

## 1. Know the moving parts

1. The [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm) stores the data.
2. The [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) reads that data.
3. `src/supabase_client.py` controls the data fetch path.
4. `app/streamlit_utils.py` calls the fetch path during app startup.
5. `app/streamlit_config.py` and `src/factor_registry.py` define column mappings used by the app.

## 2. Use the right access model

1. Use `SUPABASE_URL` and `SUPABASE_KEY` for the app.
2. Use the anon/public key for read-only app usage.
3. Keep service role keys out of the app and out of git.
4. Store production secrets in Streamlit Cloud and local secrets in `.env`.

## 3. Add data to an existing table

For a small edit:

1. Open the [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm).
2. Open Table Editor.
3. Select the target table.
4. Insert or edit the row.
5. Save the change.
6. Run a quick SQL check to confirm the change looks right.

For a bulk import:

1. Prepare a CSV with the exact column headers expected by the table.
2. Import it into a staging table first when possible.
3. Check row counts after import.
4. Check for nulls, duplicates, and bad numeric values.
5. Promote the cleaned data into the production table only after the checks pass.

## 4. Clean data before you insert it

1. Trim whitespace from IDs and ticker values.
2. Normalize tickers to a consistent case.
3. Replace sentinel strings like `N/A` or `--` with nulls.
4. Make sure `Date` is a valid date.
5. Derive `Year` consistently from `Date`.
6. Convert price, return, and factor columns to numeric types.
7. Remove duplicate business-key rows before promoting data.

Example staging flow:

```sql
update staging_prices
set "Ending_Price" = null
where "Ending_Price" in ('--', 'N/A', '#N/A', 'NULL', 'null', 'nan', '');

delete from staging_prices a
using staging_prices b
where a.ctid < b.ctid
   and a."Ticker-Region" = b."Ticker-Region"
   and a."Date" = b."Date";

insert into "Full Precision Test" ("Ticker-Region", "Date", "Ending_Price")
select "Ticker-Region", "Date", "Ending_Price"
from staging_prices;
```

## 5. Add a new table

1. Open Table Editor.
2. Create the new table.
3. Define columns and the primary key.
4. Add indexes for the columns you will filter or join on.
5. Configure RLS if the table needs it.
6. Load seed data.
7. Confirm the app does not depend on any missing columns.

## 6. Change which table the app uses

1. Open `src/supabase_client.py`.
2. Find `SupabaseManager.fetch_all_data(table_name='Full Precision Test')`.
3. Change the default table name if you want the app to read a different table.
4. Confirm that the new table still has the fields the app expects.
5. Run the app and load data.
6. Run the integration tests.

## 7. Keep schema compatibility intact

1. Preserve `Ticker` or `Ticker-Region` behavior.
2. Preserve `Year` behavior.
3. Preserve `Ending_Price` and `Next-Years_Return`.
4. Keep factor columns in sync with `app/streamlit_config.py` and `src/factor_registry.py`.
5. If you rename a column, update code and retest immediately.

## 8. Maintain the delisting mapping table

1. Keep `last_price_mapping` aligned with the code.
2. Preserve the source fields `ticker`, `last_date`, and `last_price` unless you are also updating `src/supabase_client.py`.
3. Confirm the app still merges `Delist_Date` and `Delist_Price` correctly.

## 9. Review policies and logs

1. Check RLS policies when access changes.
2. Keep read policies broad enough for the app to work.
3. Avoid broad write access.
4. Use Supabase logs when app data loads fail.

## 10. Troubleshoot a broken data load

1. Check the [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) logs.
2. Check Supabase logs for rejected queries.
3. Confirm the secrets are correct.
4. Confirm the table exists.
5. Confirm the required columns still exist.
6. Run the failing query manually in SQL Editor.

## 11. Make a safe schema change

1. Make the change in development or staging first.
2. Verify row counts and field types.
3. Update app mappings if needed.
4. Run integration tests.
5. Run the Streamlit smoke test.
6. Deploy only after the app still works.

## 12. Delete a table safely

1. Confirm no app code uses the table.
2. Confirm no tests depend on the table.
3. Export a backup.
4. Delete the table in non-production first.
5. Verify the app and tests still pass.
6. Delete in production only after that check.

## 13. Verify a Supabase change

1. Run `IntegrationTests/test_supabase_integration.py`.
2. Open the Streamlit app.
3. Click Load Market Data.
4. Run Portfolio Analysis.
5. Confirm the Results tab renders.
6. Check row counts and numeric ranges.

## 14. What success looks like

You are done when the app loads data cleanly, the tests pass, the schema matches the code, and the production app still runs after the change.

## 15. Related guides

1. [Supabase Setup](SUPABASE_SETUP.md)
2. [Streamlit Admin Guide](STREAMLIT_ADMIN_GUIDE.md)
3. [Deployment](DEPLOYMENT.md)

## 16. Reference

Use this section when you need the background details that support the workflow above.

1. The app reads from `src/supabase_client.py`.
2. The app startup path passes through `app/streamlit_utils.py`.
3. Factor labels and column names are mapped in `app/streamlit_config.py` and `src/factor_registry.py`.
4. The default table is `Full Precision Test` unless code changes it.
5. `last_price_mapping` supplies delisting fields that the client merges into the dataset.
6. Common required fields include `Ticker`, `Year`, `Ending_Price`, and `Next-Years_Return`.
7. Integration tests in `IntegrationTests/test_supabase_integration.py` should be kept aligned with any schema change.
