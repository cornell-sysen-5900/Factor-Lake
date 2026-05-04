# Supabase Upload Tool Workflow

Use this guide when you need to create Supabase table(s) and upload local files with the wrapper script at `tools/run_data_upload_supabase.sh`.

## 1. Know what this tool does

1. Accepts a single file path or a folder path.
2. Supports `.csv`, `.xlsx`, `.xls`, and `.parquet`.
3. Creates one table per file.
4. Defaults table names to file names without extensions.
5. Enables RLS and adds a public read policy (`anon` and `authenticated`) for each created table.
6. Uploads data in chunks and validates row counts after upload.

## 2. Confirm prerequisites

1. You have `SUPABASE_URL` and a service role key.
2. You can run from repo root (`D:/Factor-Lake` on Windows or `/mnt/d/Factor-Lake` in WSL).
3. Sync the project environment first:
   `uv sync --group dev`
4. Upload format dependencies are included in `pyproject.toml` and installed by `uv sync`:
   `pandas`, `supabase`, `openpyxl` (`.xlsx`), `xlrd` (`.xls`), and `pyarrow` (`.parquet`).

## 3. Run the one-time Supabase SQL setup

This tool executes DDL via RPC. In Supabase SQL Editor, run:

```sql
create or replace function public.execute_sql(sql text)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  execute sql;
end;
$$;

grant execute on function public.execute_sql(text) to service_role;
```

## 4. Choose input and table naming behavior

1. Single file input:
   If `TABLE_NAME` is omitted, the table name is the file name without extension.
2. Folder input:
   The tool scans supported files and creates one table per file name.
3. Optional `TABLE_NAME`:
   Works as an override for single-file uploads only.

## 5. Run the upload command

General command form:

```bash
bash tools/run_data_upload_supabase.sh \
  [--supabase-url <SUPABASE_URL>] \
  [--supabase-service-role-key <SUPABASE_SERVICE_ROLE_KEY>] \
  <FILE_PATH_OR_FOLDER> \
  [TABLE_NAME] \
  [EXTRA_ARGS...]
```

WSL + Windows Python example:

```bash
cd /mnt/d/Factor-Lake && \
PYTHON_BIN="/mnt/c/Users/<you>/anaconda3/envs/<env>/python.exe" \
bash tools/run_data_upload_supabase.sh \
  --supabase-url "https://<project-ref>.supabase.co" \
  --supabase-service-role-key "<service-role-key>" \
  "C:/Users/<you>/Downloads/data.xlsx"
```

Folder upload example (common for parquet):

```bash
cd /mnt/d/Factor-Lake && \
PYTHON_BIN="/mnt/c/Users/<you>/anaconda3/envs/<env>/python.exe" \
bash tools/run_data_upload_supabase.sh \
  --supabase-url "https://<project-ref>.supabase.co" \
  --supabase-service-role-key "<service-role-key>" \
  "C:/Users/<you>/Downloads/parquets"
```

Environment-variable mode:

```bash
export SUPABASE_URL="https://<project-ref>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="<service-role-key>"
bash tools/run_data_upload_supabase.sh "C:/Users/<you>/Downloads/data.parquet"
```

## 6. Understand the execution flow

1. Resolve input path(s).
2. Derive table name(s).
3. Build and run create-table SQL.
4. Enable RLS and apply public read policy.
5. Reload PostgREST schema cache when needed.
6. Upload batches with retry for schema-cache lag.
7. Validate upload count and preview rows.

## 7. Troubleshoot common failures

1. `No such file or directory` for script:
   Run from repo root and call `tools/run_data_upload_supabase.sh`.
2. `python: command not found`:
   Set `PYTHON_BIN` explicitly.
3. `PGRST202` function not found:
   Run the SQL setup in Section 3.
4. `PGRST205` table not found in schema cache:
   Retry; schema reload and retry logic are built in.
5. Excel dependency error for `openpyxl`/`xlrd` or parquet engine error for `pyarrow`:
   Re-sync dependencies in the same environment: `uv sync --group dev`.

## 8. Verify the upload result

1. Confirm upload batch logs show success.
2. Confirm row count output is greater than zero.
3. In Supabase Table Editor, verify the table exists and has rows.
4. Confirm read access works for `anon` and `authenticated` roles.

## 9. What success looks like

You are done when the table(s) are created, data is uploaded, row counts look correct, and read access works with RLS enabled.

## 10. Related guides

1. [Supabase Setup](SUPABASE_SETUP.md)
2. [Supabase Maintenance](SUPABASE_MAINTENANCE_GUIDE.md)
3. [Deployment](DEPLOYMENT.md)

## 11. Reference

1. Wrapper script: `tools/run_data_upload_supabase.sh`
2. Upload engine: `tools/data_upload_supabase.py`
3. Supported input extensions: `.csv`, `.xlsx`, `.xls`, `.parquet`
4. Table naming default: file name without extension
