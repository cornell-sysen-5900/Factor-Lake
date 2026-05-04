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

## 3. Add or bulk-upload data

Use the dedicated upload workflow in [Supabase Upload Tool](SUPABASE_UPLAOD_TOOL.md).

Use this maintenance guide for operational checks after upload (schema compatibility, policies, logs, and app validation).

## 4. Change which table the app uses

1. Open `src/supabase_client.py`.
2. Find `SupabaseManager.fetch_all_data(table_name='Full Precision Test')`.
3. Change the default table name if you want the app to read a different table.
4. Confirm that the new table still has the fields the app expects.
5. Run the app and load data.
6. Run the integration tests.

## 5. Keep schema compatibility intact

1. Preserve `Ticker` or `Ticker-Region` behavior.
2. Preserve `Year` behavior.
3. Preserve `Ending_Price` and `Next-Years_Return`.
4. Keep factor columns in sync with `app/streamlit_config.py` and `src/factor_registry.py`.
5. If you rename a column, update code and retest immediately.

## 6. Maintain the delisting mapping table

1. Keep `last_price_mapping` aligned with the code.
2. Preserve the source fields `ticker`, `last_date`, and `last_price` unless you are also updating `src/supabase_client.py`.
3. Confirm the app still merges `Delist_Date` and `Delist_Price` correctly.

## 7. Review policies and logs

1. Check RLS policies when access changes.
2. Keep read policies broad enough for the app to work.
3. Avoid broad write access.
4. Use Supabase logs when app data loads fail.

## 8. Troubleshoot a broken data load

1. Check the [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) logs.
2. Check Supabase logs for rejected queries.
3. Confirm the secrets are correct.
4. Confirm the table exists.
5. Confirm the required columns still exist.
6. Run the failing query manually in SQL Editor.

## 9. Make a safe schema change

1. Make the change in development or staging first.
2. Verify row counts and field types.
3. Update app mappings if needed.
4. Run integration tests.
5. Run the Streamlit smoke test.
6. Deploy only after the app still works.

## 10. Delete a table safely

1. Confirm no app code uses the table.
2. Confirm no tests depend on the table.
3. Export a backup.
4. Delete the table in non-production first.
5. Verify the app and tests still pass.
6. Delete in production only after that check.

## 11. Verify a Supabase change

1. Run `tests/integration/test_supabase.py`.
2. Open the Streamlit app.
3. Click Load Market Data.
4. Run Portfolio Analysis.
5. Confirm the Results tab renders.
6. Check row counts and numeric ranges.

## 12. What success looks like

You are done when the app loads data cleanly, the tests pass, the schema matches the code, and the production app still runs after the change.

## 13. Related guides

1. [Supabase Setup](SUPABASE_SETUP.md)
2. [Supabase Upload Tool](SUPABASE_UPLAOD_TOOL.md)
3. [Streamlit Admin Guide](STREAMLIT_ADMIN_GUIDE.md)
4. [Deployment](DEPLOYMENT.md)

## 14. Reference

Use this section when you need the background details that support the workflow above.

1. The app reads from `src/supabase_client.py`.
2. The app startup path passes through `app/streamlit_utils.py`.
3. Factor labels and column names are mapped in `app/streamlit_config.py` and `src/factor_registry.py`.
4. The default table is `Full Precision Test` unless code changes it.
5. `last_price_mapping` supplies delisting fields that the client merges into the dataset.
6. Common required fields include `Ticker`, `Year`, `Ending_Price`, and `Next-Years_Return`.
7. Integration tests in `tests/integration/test_supabase.py` should be kept aligned with any schema change.
