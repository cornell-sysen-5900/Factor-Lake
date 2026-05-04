import argparse
import datetime as dt
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from supabase import Client, create_client

# --- CONFIGURATION (edit these and run the script directly) ---
FILE_PATH = ""  # Example: "data/monthly_prices.csv"
TABLE_NAME = ""  # Example: "monthly_prices"
CHUNK_SIZE = 2500
DATE_COLUMN = "Date"
CREATE_SQL = ""  # Optional inline SQL for create mode
CREATE_SQL_FILE = ""  # Optional SQL file path for create mode

SUPABASE_URL = ""  # Optional fallback if SUPABASE_URL env var is not set
SUPABASE_KEY = ""  # Optional fallback if SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY env vars are not set
SUPPORTED_FILE_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
SCHEMA_CACHE_RETRY_ATTEMPTS = 6

SQL_EXECUTOR_SETUP_SQL = """
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
""".strip()


def get_supabase_client() -> Client:
    """Build a Supabase client from environment variables."""
    url = os.getenv("SUPABASE_URL", "").strip() or SUPABASE_URL.strip()
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_KEY", "").strip()
        or SUPABASE_KEY.strip()
    )

    if not url or not key:
        raise RuntimeError(
            "Missing Supabase credentials. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY)."
        )

    return create_client(url, key)


_SUPABASE_CLIENT: Optional[Client] = None


def get_supabase() -> Client:
    """Return a cached Supabase client, creating it on first use."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        _SUPABASE_CLIENT = get_supabase_client()
    return _SUPABASE_CLIENT


def _quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("SQL identifier must be a non-empty string")
    return '"' + identifier.replace('"', '""') + '"'


def run_sql_execute(sql: str) -> None:
    """Run SQL using Supabase RPC; requires a SQL-executor function in the project."""
    if not sql or not sql.strip():
        raise RuntimeError("SQL statement is empty.")

    rpc_attempts = [
        ("execute_sql", {"sql": sql}),
        ("execute_sql", {"query": sql}),
        ("exec_sql", {"sql": sql}),
        ("run_sql", {"sql": sql}),
    ]
    errors: List[str] = []

    for function_name, payload in rpc_attempts:
        try:
            get_supabase().rpc(function_name, payload).execute()
            return
        except Exception as exc:
            errors.append(f"{function_name}: {exc}")

    raise RuntimeError(
        "Could not execute SQL through Supabase RPC. "
        "Tried execute_sql, exec_sql, and run_sql. "
        "Create one of these RPC SQL executor functions in Supabase, or pre-create target tables. "
        f"Create this in Supabase SQL Editor:\n{SQL_EXECUTOR_SETUP_SQL}\n"
        f"Errors: {' | '.join(errors)}"
    )


def run_sql_script(sql_script: str) -> None:
    """Execute a SQL script one statement at a time via RPC."""
    statements = [statement.strip() for statement in sql_script.split(";") if statement.strip()]
    if not statements:
        raise RuntimeError("SQL script is empty.")

    for statement in statements:
        run_sql_execute(statement)


def request_postgrest_schema_reload() -> None:
    """Ask PostgREST to reload schema cache so new tables become visible quickly."""
    run_sql_execute("NOTIFY pgrst, 'reload schema'")


def is_schema_cache_table_miss(exc: Exception) -> bool:
    """Detect PostgREST schema-cache miss errors for newly created tables."""
    error_text = str(exc).lower()
    return "pgrst205" in error_text or "schema cache" in error_text


def load_create_sql(create_sql: str = "", create_sql_file: str = "") -> str:
    if create_sql.strip():
        return create_sql
    if create_sql_file.strip():
        with open(create_sql_file, "r", encoding="utf-8") as sql_file:
            return sql_file.read()
    raise RuntimeError("Provide create SQL via create_sql or create_sql_file.")


def resolve_input_files(file_path: str) -> List[Path]:
    """Resolve file_path into one or more supported input files."""
    input_path = Path(file_path).expanduser()

    if input_path.is_file():
        extension = input_path.suffix.lower()
        if extension not in SUPPORTED_FILE_EXTENSIONS:
            raise RuntimeError(
                f"Unsupported file extension '{extension}'. "
                "Use .csv, .xlsx, .xls, or .parquet files."
            )
        return [input_path]

    if input_path.is_dir():
        files = sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
        )
        if not files:
            raise RuntimeError(
                f"No supported files found in directory: {input_path}. "
                "Expected .csv, .xlsx, .xls, or .parquet files."
            )
        return files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def normalize_table_name(raw_name: str) -> str:
    """Convert an arbitrary string into a safe SQL table name."""
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip().lower()).strip("_")
    if not normalized:
        raise RuntimeError(f"Could not derive a valid table name from '{raw_name}'.")
    if normalized[0].isdigit():
        normalized = f"t_{normalized}"
    return normalized


def resolve_table_name(base_table_name: str, input_file: Path, multiple_files: bool) -> str:
    """Determine target table name for a file upload."""
    base_name = base_table_name.strip()
    file_stem = normalize_table_name(input_file.stem)

    if multiple_files:
        return file_stem

    if not base_name:
        return file_stem

    normalized_base = normalize_table_name(base_name)
    return normalized_base


def load_schema_sample(file_path: str, sample_rows: int = 1000) -> pd.DataFrame:
    """Load a sample DataFrame used to infer SQL schema for table creation."""
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(file_path, nrows=sample_rows)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=0, nrows=sample_rows)

    if extension == ".parquet":
        return pd.read_parquet(file_path).head(sample_rows).copy()

    raise RuntimeError(
        f"Unsupported file extension '{extension}'. Use .csv, .xlsx, .xls, or .parquet files."
    )


def map_dtype_to_postgres(dtype: Any) -> str:
    """Map inferred dtypes to a robust ingestion-first Postgres column type."""
    # Spreadsheets frequently contain mixed types within a single column.
    # Using TEXT prevents insert failures when values vary across rows.
    return "TEXT"


def build_create_table_sql(table_name: str, sample_df: pd.DataFrame) -> str:
    """Generate a DROP/CREATE TABLE statement from inferred DataFrame schema."""
    if sample_df.columns.empty:
        raise RuntimeError("Cannot create table from input with no columns.")

    column_lines = []
    for column_name in sample_df.columns:
        column = str(column_name).strip()
        if not column:
            raise RuntimeError("Input file contains an empty column name, cannot create SQL schema.")
        column_type = map_dtype_to_postgres(sample_df[column_name].dtype)
        column_lines.append(f"{_quote_identifier(column)} {column_type}")

    table_identifier = _quote_identifier(table_name)
    columns_sql = ",\n    ".join(column_lines)
    return (
        f"DROP TABLE IF EXISTS {table_identifier};\n"
        f"CREATE TABLE {table_identifier} (\n"
        f"    {columns_sql}\n"
        ");"
    )


def build_public_read_policy_sql(table_name: str) -> str:
    """Enable RLS and allow public read access for a table."""
    table_identifier = _quote_identifier(table_name)
    policy_identifier = _quote_identifier("public_read_all")
    return (
        f"ALTER TABLE {table_identifier} ENABLE ROW LEVEL SECURITY;\n"
        f"GRANT SELECT ON TABLE {table_identifier} TO anon, authenticated;\n"
        f"DROP POLICY IF EXISTS {policy_identifier} ON {table_identifier};\n"
        f"CREATE POLICY {policy_identifier} ON {table_identifier} "
        "FOR SELECT TO anon, authenticated USING (true);"
    )


def render_create_sql(sql_template: str, table_name: str) -> str:
    """Render optional placeholders in user-provided SQL templates."""
    rendered = sql_template
    rendered = rendered.replace("{table_name}", table_name)
    rendered = rendered.replace("{table_identifier}", _quote_identifier(table_name))
    return rendered


def clean_record(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure values are JSON-safe for Supabase inserts."""

    def _to_json_safe(value: Any) -> Any:
        if value is None:
            return None

        # Handle scalar missing values (including pd.NaT / np.nan) before type coercion.
        try:
            is_missing = pd.isna(value)
            if isinstance(is_missing, (bool, np.bool_)) and is_missing:
                return None
        except Exception:
            pass

        if isinstance(value, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
            return value.isoformat()

        if isinstance(value, pd.Timedelta):
            return str(value)

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, np.ndarray):
            return [_to_json_safe(item) for item in value.tolist()]

        if isinstance(value, (list, tuple, set)):
            return [_to_json_safe(item) for item in value]

        if isinstance(value, dict):
            return {str(key): _to_json_safe(item) for key, item in value.items()}

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"nat", "nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                return None

        if isinstance(value, float) and np.isinf(value):
            return None

        return value

    cleaned: Dict[str, Any] = {}
    for key, value in row_dict.items():
        cleaned[key] = _to_json_safe(value)
    return cleaned


def iter_data_chunks(file_path: str, chunk_size: int):
    """Yield DataFrame chunks from supported file types."""
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        yield from pd.read_csv(file_path, chunksize=chunk_size)
        return

    if extension in {".xlsx", ".xls"}:
        data_frame = pd.read_excel(file_path, sheet_name=0)
        for start in range(0, len(data_frame), chunk_size):
            yield data_frame.iloc[start : start + chunk_size].copy()
        return

    if extension == ".parquet":
        data_frame = pd.read_parquet(file_path)
        for start in range(0, len(data_frame), chunk_size):
            yield data_frame.iloc[start : start + chunk_size].copy()
        return

    raise RuntimeError(
        f"Unsupported file extension '{extension}'. Use .csv, .xlsx, .xls, or .parquet files."
    )


def get_last_table_date(table_name: str, date_column: str = DATE_COLUMN) -> Optional[pd.Timestamp]:
    """Return max date currently in table, or None if no rows exist."""
    try:
        response = (
            get_supabase().table(table_name)
            .select(date_column)
            .order(date_column, desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read latest date from {table_name}.{date_column}: {exc}"
        ) from exc

    rows = response.data if hasattr(response, "data") and response.data else []
    if not rows:
        return None

    raw_date = rows[0].get(date_column)
    if raw_date is None:
        return None

    parsed = pd.to_datetime(raw_date, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise RuntimeError(
            f"Could not parse existing max date value '{raw_date}' from {table_name}.{date_column}."
        )

    return parsed.tz_localize(None)


def upload_csv(
    file_path: str,
    table_name: str,
    chunk_size: int = CHUNK_SIZE,
    last_date: Optional[pd.Timestamp] = None,
    date_column: str = DATE_COLUMN,
) -> None:
    """Upload file rows in batches; if last_date is set, upload only newer rows."""
    if not file_path.strip():
        raise RuntimeError("file_path is required")
    if not table_name.strip():
        raise RuntimeError("table_name is required")
    if chunk_size <= 0:
        raise RuntimeError("chunk_size must be greater than 0")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    uploaded_rows = 0
    skipped_older_rows = 0
    skipped_invalid_date_rows = 0

    reader = iter_data_chunks(file_path=file_path, chunk_size=chunk_size)
    for batch_index, chunk in enumerate(reader, start=1):
        if last_date is not None:
            if date_column not in chunk.columns:
                raise RuntimeError(
                    f"Date column '{date_column}' not found in file. "
                    f"Available columns: {list(chunk.columns)}"
                )

            parsed_dates = pd.to_datetime(chunk[date_column], errors="coerce", utc=True).dt.tz_localize(None)
            valid_mask = parsed_dates.notna()
            new_rows_mask = valid_mask & (parsed_dates > last_date)

            skipped_invalid_date_rows += int((~valid_mask).sum())
            skipped_older_rows += int((valid_mask & ~new_rows_mask).sum())
            chunk = chunk.loc[new_rows_mask].copy()

        if chunk.empty:
            print(f"Skipped batch {batch_index}: no rows to upload")
            continue

        cleaned_batch = [clean_record(record) for record in chunk.to_dict(orient="records")]
        for attempt in range(1, SCHEMA_CACHE_RETRY_ATTEMPTS + 1):
            try:
                get_supabase().table(table_name).insert(cleaned_batch).execute()
                uploaded_rows += len(cleaned_batch)
                print(
                    f"Uploaded batch {batch_index}: {len(cleaned_batch)} rows "
                    f"(running total: {uploaded_rows})"
                )
                break
            except Exception as exc:
                if is_schema_cache_table_miss(exc) and attempt < SCHEMA_CACHE_RETRY_ATTEMPTS:
                    wait_seconds = 0.5 * attempt
                    print(
                        f"PostgREST schema cache not ready for table {table_name}; "
                        f"retrying in {wait_seconds:.1f}s (attempt {attempt}/{SCHEMA_CACHE_RETRY_ATTEMPTS - 1})."
                    )
                    try:
                        request_postgrest_schema_reload()
                    except Exception:
                        # Best-effort cache reload; retry still proceeds after delay.
                        pass
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(f"Upload failed at batch {batch_index}: {exc}") from exc

    print("Upload process finished.")
    print(f"Total uploaded rows: {uploaded_rows}")
    if last_date is not None:
        print(f"Rows skipped (older or same date): {skipped_older_rows}")
        print(f"Rows skipped (invalid date): {skipped_invalid_date_rows}")


def create_table_and_upload(
    file_path: str,
    table_name: str,
    create_sql: str = "",
    create_sql_file: str = "",
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Create table with SQL, then upload all rows from the input file."""
    if create_sql.strip() or create_sql_file.strip():
        sql_template = load_create_sql(create_sql=create_sql, create_sql_file=create_sql_file)
        sql = render_create_sql(sql_template, table_name)
    else:
        sample_df = load_schema_sample(file_path)
        sql = build_create_table_sql(table_name=table_name, sample_df=sample_df)

    run_sql_script(sql)
    print(f"Create table SQL executed for table: {table_name}")
    run_sql_script(build_public_read_policy_sql(table_name))
    print(f"RLS enabled with public read policy for table: {table_name}")
    try:
        request_postgrest_schema_reload()
    except Exception:
        # Retry logic in upload handles potential schema-cache lag if this fails.
        pass
    upload_csv(file_path=file_path, table_name=table_name, chunk_size=chunk_size)


def upload_new_data(
    file_path: str,
    table_name: str,
    date_column: str = DATE_COLUMN,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Upload only rows newer than the max existing date in table."""
    last_date = get_last_table_date(table_name=table_name, date_column=date_column)
    if last_date is None:
        print(
            f"No existing date found in {table_name}.{date_column}; "
            "performing full upload."
        )
    else:
        print(f"Latest existing date in {table_name}.{date_column}: {last_date.date()}")

    upload_csv(
        file_path=file_path,
        table_name=table_name,
        chunk_size=chunk_size,
        last_date=last_date,
        date_column=date_column,
    )


def test_table_upload(
    table_name: str,
    date_column: str = DATE_COLUMN,
    preview_rows: int = 5,
) -> Dict[str, Any]:
    """Validate uploaded table and show latest rows using Supabase queries."""
    response = get_supabase().table(table_name).select("*", count="exact").limit(1).execute()
    row_count = response.count if response.count is not None else 0
    status = "PASS" if row_count > 0 else "FAIL (EMPTY)"

    print(f"Table validation: {table_name}")
    print(f"Row count: {row_count}")
    print(f"Status: {status}")

    try:
        preview_response = (
            get_supabase()
            .table(table_name)
            .select("*")
            .order(date_column, desc=True)
            .limit(preview_rows)
            .execute()
        )
    except Exception:
        preview_response = get_supabase().table(table_name).select("*").limit(preview_rows).execute()

    preview_rows_data = preview_response.data if hasattr(preview_response, "data") and preview_response.data else []
    preview_df = pd.DataFrame(preview_rows_data)

    print(f"Last {preview_rows} rows preview:")
    if preview_df.empty:
        print("(no rows returned)")
    else:
        print(preview_df.to_string(index=False))

    return {
        "table_name": table_name,
        "row_count": row_count,
        "status": status,
        "preview": preview_df,
    }


def run_upload_flow(
    file_path: str,
    table_name: str = "",
    chunk_size: int = CHUNK_SIZE,
    date_column: str = DATE_COLUMN,
    create_sql: str = "",
    create_sql_file: str = "",
) -> Dict[str, Any]:
    """Create and upload one file, or loop a directory and process all supported files."""
    input_files = resolve_input_files(file_path)
    multiple_files = len(input_files) > 1 or Path(file_path).expanduser().is_dir()
    results: Dict[str, Any] = {}

    if multiple_files and table_name.strip():
        print("Ignoring provided table_name for folder upload; using each file name as table name.")

    for input_file in input_files:
        target_table = resolve_table_name(table_name, input_file=input_file, multiple_files=multiple_files)
        print(f"Processing {input_file} -> table {target_table}")

        create_table_and_upload(
            file_path=str(input_file),
            table_name=target_table,
            create_sql=create_sql,
            create_sql_file=create_sql_file,
            chunk_size=chunk_size,
        )

        results[target_table] = test_table_upload(
            table_name=target_table,
            date_column=date_column,
            preview_rows=5,
        )

    return {
        "processed_files": len(input_files),
        "tables": list(results.keys()),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create table(s) and upload file data into Supabase. Credentials can be provided via environment "
            "variables or via command-line args for temporary runs."
        )
    )
    parser.add_argument(
        "--file-path",
        default=FILE_PATH,
        help="Path to an input file (.csv, .xlsx, .xls, .parquet) or a folder of supported files",
    )
    parser.add_argument(
        "--table-name",
        default=TABLE_NAME,
        help="Optional table name override for a single file upload; folder uploads use each file name",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Rows per upload batch")
    parser.add_argument("--date-column", default=DATE_COLUMN, help="Date column used when ordering preview rows")
    parser.add_argument("--create-sql", default=CREATE_SQL, help="Inline SQL used to create the target table")
    parser.add_argument("--create-sql-file", default=CREATE_SQL_FILE, help="SQL file path used to create the target table")

    parser.add_argument("--supabase-url", default="", help="Temporary Supabase project URL")
    parser.add_argument(
        "--supabase-service-role-key",
        default="",
        help="Temporary Supabase service role key",
    )
    parser.add_argument("--supabase-key", default="", help="Fallback Supabase API key")

    return parser.parse_args()


def apply_runtime_credentials(args: argparse.Namespace) -> None:
    """Inject temporary credentials into environment variables for this run only."""
    global _SUPABASE_CLIENT

    if args.supabase_url:
        os.environ["SUPABASE_URL"] = args.supabase_url
    if args.supabase_service_role_key:
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = args.supabase_service_role_key
    if args.supabase_key:
        os.environ["SUPABASE_KEY"] = args.supabase_key

    _SUPABASE_CLIENT = None


if __name__ == "__main__":
    args = parse_args()
    apply_runtime_credentials(args)

    if not args.file_path:
        raise RuntimeError(
            "Provide --file-path (or set FILE_PATH constant), then run again."
        )

    run_upload_flow(
        file_path=args.file_path,
        table_name=args.table_name,
        chunk_size=args.chunk_size,
        date_column=args.date_column,
        create_sql=args.create_sql,
        create_sql_file=args.create_sql_file,
    )
