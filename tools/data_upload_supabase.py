import importlib
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from supabase import Client, create_client

# --- CONFIGURATION (edit these and run the script directly) ---
FILE_PATH = ""  # Example: "data/monthly_prices.csv"
TABLE_NAME = ""  # Example: "monthly_prices"
CHUNK_SIZE = 2500
DATE_COLUMN = "Date"
MODE = "upload-new-data"  # "new-table-upload" or "upload-new-data"
CREATE_SQL = ""  # Optional inline SQL for create mode
CREATE_SQL_FILE = ""  # Optional SQL file path for create mode

MODE_NEW_TABLE_UPLOAD = "new-table-upload"
MODE_UPLOAD_NEW_DATA = "upload-new-data"


def get_supabase_client() -> Client:
    """Build a Supabase client from environment variables."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_KEY", "").strip()
    )

    if not url or not key:
        raise RuntimeError(
            "Missing Supabase credentials. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY)."
        )

    return create_client(url, key)


supabase: Client = get_supabase_client()


def _load_psycopg():
    try:
        return importlib.import_module("psycopg")
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required for SQL operations. Install with pip install psycopg[binary]."
        ) from exc


def _get_db_url() -> str:
    db_url = os.getenv("SUPABASE_DB_URL", "").strip() or os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("Missing database URL. Set SUPABASE_DB_URL (or DATABASE_URL).")
    return db_url


def _quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("SQL identifier must be a non-empty string")
    return '"' + identifier.replace('"', '""') + '"'


def run_sql_execute(sql: str) -> None:
    """Run a SQL command that does not return rows (for example CREATE TABLE)."""
    if not sql or not sql.strip():
        raise RuntimeError("SQL statement is empty.")

    psycopg = _load_psycopg()
    with psycopg.connect(_get_db_url()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
        conn.commit()


def run_sql_fetch(sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
    """Run a SQL query and return results as a DataFrame."""
    psycopg = _load_psycopg()
    with psycopg.connect(_get_db_url()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or ())
            rows = cursor.fetchall() if cursor.description else []
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
    return pd.DataFrame(rows, columns=columns)


def load_create_sql(create_sql: str = "", create_sql_file: str = "") -> str:
    if create_sql.strip():
        return create_sql
    if create_sql_file.strip():
        with open(create_sql_file, "r", encoding="utf-8") as sql_file:
            return sql_file.read()
    raise RuntimeError("Provide create SQL via create_sql or create_sql_file.")


def clean_record(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure values are JSON-safe for Supabase inserts."""
    cleaned: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if pd.isna(value) or value is None or (isinstance(value, float) and np.isinf(value)):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def get_last_table_date(table_name: str, date_column: str = DATE_COLUMN) -> Optional[pd.Timestamp]:
    """Return max date currently in table, or None if no rows exist."""
    try:
        response = (
            supabase.table(table_name)
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
    """Upload CSV rows in batches; if last_date is set, upload only newer rows."""
    if not file_path.strip():
        raise RuntimeError("file_path is required")
    if not table_name.strip():
        raise RuntimeError("table_name is required")
    if chunk_size <= 0:
        raise RuntimeError("chunk_size must be greater than 0")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    uploaded_rows = 0
    skipped_older_rows = 0
    skipped_invalid_date_rows = 0

    reader = pd.read_csv(file_path, chunksize=chunk_size)
    for batch_index, chunk in enumerate(reader, start=1):
        if last_date is not None:
            if date_column not in chunk.columns:
                raise RuntimeError(
                    f"Date column '{date_column}' not found in CSV. "
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
        try:
            supabase.table(table_name).insert(cleaned_batch).execute()
            uploaded_rows += len(cleaned_batch)
            print(
                f"Uploaded batch {batch_index}: {len(cleaned_batch)} rows "
                f"(running total: {uploaded_rows})"
            )
        except Exception as exc:
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
    """Create table with SQL, then upload entire CSV file."""
    sql = load_create_sql(create_sql=create_sql, create_sql_file=create_sql_file)
    run_sql_execute(sql)
    print(f"Create table SQL executed for table: {table_name}")
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
    """Validate uploaded table and show latest rows using SQL."""
    response = supabase.table(table_name).select("*", count="exact").limit(1).execute()
    row_count = response.count if response.count is not None else 0
    status = "PASS" if row_count > 0 else "FAIL (EMPTY)"

    print(f"Table validation: {table_name}")
    print(f"Row count: {row_count}")
    print(f"Status: {status}")

    table_ident = _quote_identifier(table_name)
    date_ident = _quote_identifier(date_column)

    try:
        preview_sql = (
            f"SELECT * FROM {table_ident} "
            f"ORDER BY {date_ident} DESC NULLS LAST LIMIT %s"
        )
        preview_df = run_sql_fetch(preview_sql, (preview_rows,))
    except Exception:
        fallback_sql = f"SELECT * FROM {table_ident} LIMIT %s"
        preview_df = run_sql_fetch(fallback_sql, (preview_rows,))

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
    mode: str,
    file_path: str,
    table_name: str,
    chunk_size: int = CHUNK_SIZE,
    date_column: str = DATE_COLUMN,
    create_sql: str = "",
    create_sql_file: str = "",
) -> Dict[str, Any]:
    """Convenience wrapper: run mode-specific upload then validate table."""
    if mode == MODE_NEW_TABLE_UPLOAD:
        create_table_and_upload(
            file_path=file_path,
            table_name=table_name,
            create_sql=create_sql,
            create_sql_file=create_sql_file,
            chunk_size=chunk_size,
        )
    elif mode == MODE_UPLOAD_NEW_DATA:
        upload_new_data(
            file_path=file_path,
            table_name=table_name,
            date_column=date_column,
            chunk_size=chunk_size,
        )
    else:
        raise RuntimeError(
            f"Invalid mode '{mode}'. Use '{MODE_NEW_TABLE_UPLOAD}' or '{MODE_UPLOAD_NEW_DATA}'."
        )

    return test_table_upload(table_name=table_name, date_column=date_column, preview_rows=5)


if __name__ == "__main__":
    if not FILE_PATH or not TABLE_NAME:
        raise RuntimeError(
            "Set FILE_PATH and TABLE_NAME in this file, then run again. "
            "You can also import this module and call run_upload_flow(...)."
        )

    run_upload_flow(
        mode=MODE,
        file_path=FILE_PATH,
        table_name=TABLE_NAME,
        chunk_size=CHUNK_SIZE,
        date_column=DATE_COLUMN,
        create_sql=CREATE_SQL,
        create_sql_file=CREATE_SQL_FILE,
    )