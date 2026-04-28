# Supabase Maintenance and Operations Guide

This guide is for maintainers who operate Supabase for Factor Lake day-to-day.

It is intentionally practical and covers:

- using the Supabase UI to manage data and tables,
- data cleaning before insertion,
- safe schema changes,
- how code maps to Supabase tables/columns,
- how to switch the table used by the app,
- validation, testing, and incident response.

## 1. Architecture and ownership

Factor Lake depends on:

- Supabase project (Postgres database and API),
- Streamlit app that reads Supabase data,
- integration tests that verify data shape and load behavior.

Primary code touchpoints:

- src/supabase_client.py
- app/streamlit_utils.py
- app/streamlit_config.py
- src/factor_registry.py
- tests/integration/test_supabase.py

## 2. Access model and credentials

Production app secrets:

- SUPABASE_URL
- SUPABASE_KEY

Recommended key usage:

- Use anon/public key for Streamlit app read flows.
- Keep service_role keys out of the app and out of git.
- Store secrets in Streamlit Cloud settings for production and .env for local development.

## 3. Supabase UI basics for maintainers

Use Supabase dashboard sections:

- Table Editor: add/edit/delete rows and tables.
- SQL Editor: bulk updates, cleanup queries, schema migrations.
- Authentication and Policies: RLS and table policies.
- Logs: query and API error diagnosis.
- Backups: point-in-time recovery options by plan.

## 4. Add data to an existing table (UI workflow)

For small/manual updates:

1. Open Supabase dashboard.
2. Go to Table Editor.
3. Select target table.
4. Use Insert Row or Edit Cells.
5. Save changes.
6. Run a quick spot check query in SQL Editor.

For bulk updates (recommended):

1. Prepare CSV with exact column headers expected by table.
2. In Table Editor, use Import Data (CSV).
3. Validate row count added.
4. Run post-import quality checks.

Post-import checks (minimum):

- no duplicate primary keys or duplicate business keys,
- Date and Year fields parse correctly,
- required metric columns are numeric,
- no unexpected null spikes in required columns.

## 5. Data cleaning before insertion

Use a staging-first pattern:

1. Create staging table with raw columns.
2. Import raw data into staging table.
3. Clean/transform in SQL.
4. Insert into production table only after checks pass.

Recommended cleaning checklist:

- trim whitespace from text IDs,
- normalize ticker casing to uppercase,
- coerce sentinel values to null (for example --, N/A, null),
- ensure Date is valid date,
- derive Year consistently from Date,
- enforce numeric types for price/return/factor columns,
- deduplicate by business key such as (Ticker-Region, Date).

Example SQL snippets:

    -- Replace common sentinels with NULL in a staging column
    update staging_prices
    set "Ending_Price" = null
    where "Ending_Price" in ('--', 'N/A', '#N/A', 'NULL', 'null', 'nan', '');

    -- Remove exact duplicate business-key rows, keep newest by inserted_at
    delete from staging_prices a
    using staging_prices b
    where a.ctid < b.ctid
      and a."Ticker-Region" = b."Ticker-Region"
      and a."Date" = b."Date";

    -- Promote cleaned data into production table
    insert into "Full Precision Test" ("Ticker-Region", "Date", "Ending_Price")
    select "Ticker-Region", "Date", "Ending_Price"
    from staging_prices;

## 6. Add a new table

UI path:

1. Table Editor -> New Table.
2. Define table name and columns.
3. Set primary key.
4. Create indexes for common filters/joins.
5. Configure RLS policies (if RLS enabled).
6. Load seed data.

Design guidance for Factor Lake tables:

- Keep ticker identifier consistent with Ticker-Region convention when relevant.
- Keep Date column parseable and derive Year in code or SQL.
- Use numeric types for return and factor metrics.
- Avoid spaces in new column names when possible to reduce downstream friction.

## 7. Delete a table safely

Never drop a table before dependency checks.

Safe deletion runbook:

1. Confirm no app path currently reads the table.
2. Confirm no integration tests rely on the table.
3. Export backup snapshot.
4. Drop table in non-production first.
5. Validate app + tests.
6. Drop in production only after validation.

Dependency checks for this repo:

- src/supabase_client.py may read Full Precision Test and last_price_mapping.
- tests/integration/test_supabase.py references Full Precision Test and FR2000 Annual Quant Data.

## 8. How to change which table the app uses

Current default source table is set in:

- src/supabase_client.py -> SupabaseManager.fetch_all_data(table_name='Full Precision Test')

Current load path:

1. app/streamlit_utils.py calls SupabaseManager().fetch_all_data() without a table argument.
2. That means the default table_name in src/supabase_client.py is used.

To switch default table for the app:

1. Update the default table_name string in src/supabase_client.py.
2. Ensure required columns still exist after standardization.
3. Run integration tests.
4. Run Streamlit smoke test.

To load a specific table in test code:

- Integration tests already use fetch_all_data(table_name=...).
- Example appears in tests/integration/test_supabase.py.

## 9. Required columns and schema compatibility

The ingestion pipeline expects key fields.

Hard requirements checked by standardization:

- Ticker (derived from Ticker-Region when present)
- Year (derived from Date when present)
- Ending_Price
- Next-Years_Return

Additional project-critical columns:

- factor columns listed in app/streamlit_config.py and src/factor_registry.py,
- sector and industry columns used by filters,
- market cap column if market-cap weighting is used.

If you rename columns in Supabase:

1. Update app/streamlit_config.py factor mappings.
2. Update src/factor_registry.py if internal key mappings depend on name.
3. Re-run integration tests and unit tests.

## 10. Delisting support table maintenance

The app also attempts to merge delisting info from last_price_mapping.

Expected source columns in that table include:

- ticker
- last_date
- last_price

Code transforms those to:

- Ticker-Region
- Delist_Date
- Delist_Price

If you change this table schema, update src/supabase_client.py accordingly.

## 11. RLS and policy guidance

If Row Level Security is enabled:

- create read policy for anon/app role on tables needed by Streamlit,
- validate policy with a test query from app context,
- avoid over-broad write policies.

If RLS is disabled for specific tables, document why and review periodically.

## 12. Monitoring and troubleshooting

When app data load fails:

1. Check Streamlit app logs first.
2. Check Supabase logs for rejected queries/errors.
3. Confirm secrets are valid and project is not paused.
4. Confirm target table exists and has expected columns.
5. Test query directly in Supabase SQL Editor.

Common failure classes:

- invalid credentials,
- table not found,
- missing required columns,
- type conversion issues after import,
- policy/RLS blocks.

## 13. Change management process

For schema or major data changes:

1. Make change in development/staging project first.
2. Validate with test queries and row-count checks.
3. Update code mappings if needed.
4. Run integration tests.
5. Deploy code to main.
6. Verify production app smoke test.
7. Record the change in team notes/changelog.

## 14. Test plan for Supabase changes

Minimum verification after any table or schema update:

1. Integration: run tests/integration/test_supabase.py with valid SUPABASE_URL and SUPABASE_KEY.
2. App smoke test:
   - login,
   - Load Market Data,
   - Run Portfolio Analysis,
   - confirm Results render.
3. Data sanity checks:
   - row counts,
   - null rates in key columns,
   - plausible numeric ranges.

## 15. Operational best practices

- Keep a separate staging table for bulk imports.
- Prefer additive schema changes over destructive changes during semester.
- Use consistent naming and data types across yearly loads.
- Backup before drops or irreversible updates.
- Batch large updates during low-usage windows.
- Keep integration tests aligned with the active source tables.

## Related guides

- SUPABASE_SETUP.md
- STREAMLIT_ADMIN_GUIDE.md
- DEPLOYMENT.md
