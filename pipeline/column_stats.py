"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/column_stats.py
PURPOSE: Per-column statistics of the final Parquet files (s08_export writes them to column_stats.parquet and into
         DATA_DICTIONARY.md), the factor coverage by year, and the Markdown tables built from them.

Numeric: dtype, rows, non-null %, min, p1, p25, median, mean, p75, p99, max. Dates: non-null %, min, max, unique
dates. Text: non-null %, unique count, top 5 values with their share of rows. Booleans: non-null %, share True.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NUMERIC_STATS = ("min", "p1", "p25", "median", "mean", "p75", "p99", "max")
COLUMNS = ("file", "column", "kind", "dtype", "rows", "non_null_pct", *NUMERIC_STATS, "date_min", "date_max",
           "unique", "top_values", "true_pct")
FACTOR_PREFIXES = ("f_", "mom_", "vol_", "beta_")
MULTIPLES = ("pe", "pb", "ps", "ev_ebitda", "ev_ebit", "ev_sales", "fcf_yield", "dividend_yield", "earnings_yield",
             "fwd_pe_fy1", "fwd_pe_fy2", "fwd_ev_ebitda", "fwd_ev_sales", "ltg", "peg")


def is_factor(column: str) -> bool:
    """Factors and multiples get p1 and p99 in the dictionary's stats table."""
    return column.startswith(FACTOR_PREFIXES) or column in MULTIPLES


def _kind(t: pa.DataType) -> str:
    if pa.types.is_dictionary(t):
        t = t.value_type
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        return "date"
    if pa.types.is_integer(t) or pa.types.is_floating(t):
        return "numeric"
    return "text"


def column_stat(name: str, column: str, arr: pa.ChunkedArray) -> dict:
    rows = len(arr)
    nulls = arr.null_count
    kind = _kind(arr.type)
    rec = {"file": name, "column": column, "kind": kind, "dtype": str(arr.type), "rows": rows,
           "non_null_pct": 100 * (rows - nulls) / rows if rows else np.nan}
    if kind == "numeric":
        x = arr.to_numpy(zero_copy_only=False).astype("float64")
        x = x[np.isfinite(x)]
        if len(x):
            q = np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])
            rec.update({"min": x.min(), "p1": q[0], "p25": q[1], "median": q[2], "mean": x.mean(), "p75": q[3],
                        "p99": q[4], "max": x.max()})
    elif kind == "date":
        a = arr.cast(pa.timestamp("ns")) if pa.types.is_date(arr.type) else arr
        if rows > nulls:
            rec.update({"date_min": pc.min(a).as_py(), "date_max": pc.max(a).as_py(),
                        "unique": len(pc.unique(a.drop_null()))})
    elif kind == "bool":
        if rows > nulls:
            rec["true_pct"] = 100 * pc.sum(arr).as_py() / (rows - nulls)
    else:
        a = arr.cast(arr.type.value_type) if pa.types.is_dictionary(arr.type) else arr
        vc = pc.value_counts(a.drop_null())
        counts = pd.Series(vc.field("counts").to_numpy(), index=vc.field("values").to_pylist())
        rec["unique"] = len(counts)
        top = counts.sort_values(ascending=False, kind="stable").head(5)
        rec["top_values"] = json.dumps([[str(v), round(100 * c / rows, 4)] for v, c in top.items()])
    return rec


def file_stats(name: str, files: list[Path]) -> list[dict]:
    """One row per column of a final table (several files, e.g. prices_daily's years, combined)."""
    schema = pq.read_schema(files[0])
    out = []
    for column in schema.names:
        arr = pa.chunked_array([c for f in files for c in pq.read_table(f, columns=[column]).column(0).chunks],
                               type=schema.field(column).type)
        out.append(column_stat(name, column, arr))
    return out


def stats_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records).reindex(columns=list(COLUMNS))
    for c in ("rows", "unique"):
        df[c] = df[c].astype("Int64")
    for c in ("non_null_pct", *NUMERIC_STATS, "true_pct"):
        df[c] = df[c].astype("float64")
    for c in ("date_min", "date_max"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    return df


def coverage_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """Non-null % by year of every factor column of factor_panel_monthly (rows: factors, columns: years)."""
    cols = [c for c in panel.columns if is_factor(c)]
    cov = panel[cols].notna().groupby(panel["date"].dt.year).mean().T * 100
    cov.columns = [int(y) for y in cov.columns]
    return cov


# ---------- Markdown ----------

def fmt(v) -> str:
    """4 significant figures; dates as YYYY-MM-DD; missing as blank."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return ""
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(v).date())
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}" if abs(int(v)) < 10_000 else f"{float(v):.4g}"
    return f"{float(v):.4g}"


def stats_cells(r: pd.Series) -> list[str]:
    """column, non-null %, min, p1, median, mean, p99, max, values: the dictionary's stats row of one column."""
    f = is_factor(r["column"])
    if r["kind"] == "numeric":
        cells = [fmt(r["min"]), fmt(r["p1"]) if f else "", fmt(r["median"]), fmt(r["mean"]),
                 fmt(r["p99"]) if f else "", fmt(r["max"]), ""]
    elif r["kind"] == "date":
        cells = [fmt(r["date_min"]), "", "", "", "", fmt(r["date_max"]),
                 "" if pd.isna(r["unique"]) else f"{int(r['unique']):,} dates"]
    elif r["kind"] == "bool":
        cells = ["", "", "", "", "", "", "" if pd.isna(r["true_pct"]) else f"True {fmt(r['true_pct'])}%"]
    else:
        top = json.loads(r["top_values"]) if isinstance(r["top_values"], str) else []
        cells = ["", "", "", "", "", "", (f"{int(r['unique']):,} unique; top: " if pd.notna(r["unique"]) else "")
                 + ", ".join(f"{_md(v)} {fmt(s)}%" for v, s in top[:3])]
    return [f"`{r['column']}`", fmt(r["non_null_pct"]), *cells]


def _md(v: str) -> str:
    return (v[:30] + "...") if len(v) > 33 else v.replace("|", "/")


STATS_HEADER = ("| column | non-null % | min | p1 | median | mean | p99 | max | values |",
                "|---|---|---|---|---|---|---|---|---|")


def stats_table(stats: pd.DataFrame, name: str) -> list[str]:
    rows = stats[stats["file"] == name]
    return ["### Stats", "", *STATS_HEADER, *("| " + " | ".join(stats_cells(r)) + " |" for _, r in rows.iterrows())]


def coverage_table(cov: pd.DataFrame) -> list[str]:
    years = list(cov.columns)
    return ["### Coverage by year (non-null %)", "",
            "| factor | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years),
            *(f"| `{c}` | " + " | ".join(f"{v:.0f}" for v in cov.loc[c]) + " |" for c in cov.index)]


def parse_stats(md: str) -> dict[str, list[str]]:
    """The stats table rows of each file section of a rendered DATA_DICTIONARY.md, as written."""
    out: dict[str, list[str]] = {}
    section, in_stats = None, False
    for line in md.splitlines():
        if line.startswith("## "):
            section, in_stats = line[3:].strip(), False
        elif line.startswith("### "):
            in_stats = line.strip() == "### Stats"
        elif in_stats and section and line.startswith("| `"):
            out.setdefault(section, []).append(line)
    return out


def stats_rows(stats: pd.DataFrame, name: str) -> list[str]:
    return [line for line in stats_table(stats, name) if line.startswith("| `")]
