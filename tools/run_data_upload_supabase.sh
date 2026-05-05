#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  tools/run_data_upload_supabase.sh \
    [--supabase-url <SUPABASE_URL>] \
    [--supabase-service-role-key <SUPABASE_SERVICE_ROLE_KEY>] \
    <FILE_PATH> \
    [TABLE_NAME] \
    [EXTRA_ARGS...]

Arguments:
  FILE_PATH                  Input file path (.csv, .xlsx, .xls, .parquet) or folder path
  TABLE_NAME                 Optional target table name override for single-file uploads

Options:
  --supabase-url             Optional Supabase project URL override
  --supabase-service-role-key Optional service role key override

Environment fallback:
  SUPABASE_URL               Used when --supabase-url is not provided
  SUPABASE_SERVICE_ROLE_KEY  Used when --supabase-service-role-key is not provided

Notes:
  Upload flow always creates a new table first.
  If FILE_PATH is a folder, all supported files in that folder are processed.
  For folder input, table names are derived from each file name (without extension).
  For single-file input, table name defaults to the file name unless TABLE_NAME is provided.
  Pass --create-sql or --create-sql-file in EXTRA_ARGS to override inferred schema.

Examples:
  tools/run_data_upload_supabase.sh \
    "C:/Users/you/data.csv" \
    "my_table" \
    --supabase-url "https://your-project.supabase.co" \
    --supabase-service-role-key "your-temp-service-role-key" \
    --create-sql-file "schema.sql"

  tools/run_data_upload_supabase.sh \
    "C:/Users/you/parquet_folder"
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SUPABASE_URL=""
SUPABASE_SERVICE_ROLE_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --supabase-url)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --supabase-url" >&2
        usage
        exit 1
      fi
      SUPABASE_URL="$2"
      shift 2
      ;;
    --supabase-service-role-key)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --supabase-service-role-key" >&2
        usage
        exit 1
      fi
      SUPABASE_SERVICE_ROLE_KEY="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

FILE_PATH="$1"
TABLE_NAME=""
shift

if [[ $# -gt 0 && "$1" != --* ]]; then
  TABLE_NAME="$1"
  shift
fi

PYTHON_SCRIPT="$SCRIPT_DIR/data_upload_supabase.py"

if [[ "$PYTHON_BIN" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
  # Windows Python invoked from WSL requires Windows-style paths.
  PYTHON_SCRIPT="$(wslpath -w "$PYTHON_SCRIPT")"
  if [[ "$FILE_PATH" == /mnt/* ]]; then
    FILE_PATH="$(wslpath -w "$FILE_PATH")"
  fi
fi

CMD=(
  "$PYTHON_BIN" "$PYTHON_SCRIPT"
  --file-path "$FILE_PATH"
)

if [[ -n "$SUPABASE_URL" ]]; then
  CMD+=(--supabase-url "$SUPABASE_URL")
fi

if [[ -n "$SUPABASE_SERVICE_ROLE_KEY" ]]; then
  CMD+=(--supabase-service-role-key "$SUPABASE_SERVICE_ROLE_KEY")
fi

if [[ -n "$TABLE_NAME" ]]; then
  CMD+=(--table-name "$TABLE_NAME")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

"${CMD[@]}"

