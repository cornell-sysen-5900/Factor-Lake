"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s00_inventory.py
PURPOSE: Stage s00_inventory: what is in raw/ before anything is cleaned.

In:  raw/<file> for every file in config.RAW_FILES, raw/query_inputs/<list> for config.QUERY_INPUTS (read only)
Out: processed/s00_inventory/inventory.json  per file: size, sha256, rows, lowercase columns, date range of the
                                             main date column, distinct ids, row counts by group, missing values,
                                             coverage of the upload list, and the checks
     processed/s00_inventory/summary.json    the standard stage summary

Every CSV is streamed in 64 MB blocks and only the columns needed for the counts are read, as text, so the
2.9 GB price file never sits in memory and nothing is parsed into the wrong type.
"""

from __future__ import annotations

import csv
import hashlib
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from pipeline.checks import Check, columns_present, not_empty, overall_status, values_present
from pipeline.checks.core import FAIL, PASS, WARN
from pipeline.config import QUERY_INPUTS, RAW_FILES, DataPaths, RawFile
from pipeline.outputs import fresh_stage_dir, utc_now, write_json, write_summary

STAGE = "s00_inventory"
_ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"
_BLOCK_BYTES = 1 << 26
_MISSING = pa.array(["", "."])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return next(csv.reader(f), [])


def norm_id(value: str) -> str:
    """'76095.0' and ' 76095' both become '76095', so ids compare equal across files."""
    return re.sub(r"\.0+$", "", value.strip())


def scan_csv(path: Path, spec: RawFile) -> dict:
    """Stream the CSV once and collect rows, date range, distinct ids, group counts and missing values."""
    header = read_header(path)
    lower = [c.strip().lower() for c in header]
    original = dict(zip(lower, header))
    group_cols = {c for g in spec.groups for c in g}
    wanted = ({spec.date_col, *spec.unique, *spec.blanks, *spec.expected_values} | group_cols) - {None}
    wanted = sorted(wanted & set(lower)) or lower[:1]
    distinct_cols = (set(spec.unique) | set(spec.expected_values)) & set(wanted)
    groups = [g for g in spec.groups if set(g) <= set(wanted)]

    reader = pacsv.open_csv(
        path, read_options=pacsv.ReadOptions(block_size=_BLOCK_BYTES),
        convert_options=pacsv.ConvertOptions(include_columns=[original[c] for c in wanted],
                                             column_types={original[c]: pa.string() for c in wanted}))
    rows, blank_dates, bad_dates, first, last = 0, 0, 0, None, None
    distinct = {c: set() for c in distinct_cols}
    counts = {g: Counter() for g in groups}
    missing = Counter()
    for batch in reader:
        t = pa.Table.from_batches([batch]).rename_columns([c.strip().lower() for c in batch.schema.names])
        rows += t.num_rows
        if spec.date_col in t.column_names:
            d = pc.utf8_trim_whitespace(t[spec.date_col])
            ok = pc.match_substring_regex(d, _ISO_DATE)
            blank = pc.equal(d, "")
            blank_dates += pc.sum(blank).as_py() or 0
            bad_dates += pc.sum(pc.and_(pc.invert(ok), pc.invert(blank))).as_py() or 0
            valid = d.filter(ok)
            if len(valid):
                lo, hi = pc.min(valid).as_py(), pc.max(valid).as_py()
                first = lo if first is None else min(first, lo)
                last = hi if last is None else max(last, hi)
        for c in distinct:
            distinct[c].update(norm_id(v) for v in pc.unique(t[c]).to_pylist() if v and v.strip())
        for g in groups:
            sizes = t.select(list(g)).to_pandas().groupby(list(g), dropna=False).size()
            for key, n in sizes.items():
                counts[g][key if isinstance(key, tuple) else (key,)] += int(n)
        for c in spec.blanks:
            if c in t.column_names:
                missing[c] += pc.sum(pc.is_in(pc.utf8_trim_whitespace(t[c]), value_set=_MISSING)).as_py() or 0

    return {"columns": lower, "rows": rows, "date_range": [first, last] if first else None,
            "blank_dates": blank_dates, "bad_dates": bad_dates, "distinct": distinct,
            "counts": {", ".join(g): [dict(zip(g, k), rows=n) for k, n in sorted(c.items())]
                       for g, c in counts.items()},
            "missing_values": {c: missing[c] for c in spec.blanks if c in lower}}


def scan_list(path: Path) -> dict:
    """An upload list (one id per line): lines, distinct ids, duplicates."""
    if not path.exists():
        return {"path": path.name, "exists": False}
    lines = [norm_id(x) for x in path.read_text(encoding="utf-8-sig").splitlines()]
    ids = [x for x in lines if x]
    return {"path": path.name, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "lines": len(ids), "distinct": len(set(ids)), "duplicates": len(ids) - len(set(ids)),
            "blank_lines": len(lines) - len(ids), "ids": set(ids)}


def inventory_file(path: Path, spec: RawFile, lists: dict[str, dict]) -> tuple[dict, list[Check]]:
    """Inventory of one raw file and its checks."""
    name = spec.name
    if not path.exists():
        return {"path": f"raw/{name}", "exists": False}, [Check(f"{name}: exists", FAIL, "file not found")]

    scan = scan_csv(path, spec)
    checks = [Check(f"{name}: exists", PASS), columns_present(name, scan["columns"], spec.required),
              not_empty(name, scan["rows"])]
    dup_cols = sorted(c for c, n in Counter(scan["columns"]).items() if n > 1)
    if dup_cols:
        checks.append(Check(f"{name}: duplicate column names", FAIL, ", ".join(dup_cols)))
    if spec.date_col:
        if scan["bad_dates"] or scan["blank_dates"]:
            checks.append(Check(f"{name}: {spec.date_col} format", WARN,
                                f"{scan['bad_dates']:,} not YYYY-MM-DD, {scan['blank_dates']:,} blank"))
        if spec.first_date_by and scan["date_range"] and scan["date_range"][0] > spec.first_date_by:
            checks.append(Check(f"{name}: starts late", WARN,
                                f"first {spec.date_col} {scan['date_range'][0]}, expected by {spec.first_date_by}"))
    for col, expected in spec.expected_values.items():
        if col in scan["distinct"]:
            checks += values_present(name, col, scan["distinct"][col], expected)

    info = {"path": f"raw/{name}", "exists": True, "bytes": path.stat().st_size,
            "mb": round(path.stat().st_size / 1e6, 1), "sha256": sha256_file(path), "rows": scan["rows"],
            "n_columns": len(scan["columns"]), "columns": scan["columns"], "date_col": spec.date_col,
            "date_range": scan["date_range"], "bad_dates": scan["bad_dates"], "blank_dates": scan["blank_dates"],
            "distinct": {c: len(v) for c, v in scan["distinct"].items()}, "counts": scan["counts"],
            "missing_values": scan["missing_values"]}

    if spec.upload_list and lists.get(spec.upload_list[0], {}).get("exists"):
        list_name, col = spec.upload_list
        listed, found = lists[list_name]["ids"], scan["distinct"].get(col, set())
        absent = sorted(listed - found, key=lambda x: (len(x), x))
        info["upload_list"] = {"list": list_name, "column": col, "listed": len(listed),
                               "listed_not_in_file": len(absent), "in_file_not_listed": len(found - listed),
                               "examples_listed_not_in_file": absent[:20]}
    return info, checks


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    started = time.time()
    lists = {name: scan_list(paths.query_inputs / name) for name in QUERY_INPUTS}
    checks: list[Check] = []
    for name, lst in lists.items():
        if not lst["exists"]:
            checks.append(Check(f"query_inputs/{name}: exists", FAIL, "file not found"))
        elif lst["duplicates"] or lst["blank_lines"]:
            checks.append(Check(f"query_inputs/{name}: one id per line", WARN,
                                f"{lst['duplicates']} duplicates, {lst['blank_lines']} blank lines"))

    files = {}
    for spec in RAW_FILES:
        t0 = time.time()
        files[spec.name], file_checks = inventory_file(paths.raw / spec.name, spec, lists)
        checks += file_checks
        print(f"  scanned {spec.name} in {time.time() - t0:.0f}s", flush=True)

    known = {s.name for s in RAW_FILES}
    unknown = sorted(p.name for p in paths.raw.iterdir() if p.is_file() and p.name not in known)
    if unknown:
        checks.append(Check("raw/: files not in config.RAW_FILES", WARN, ", ".join(unknown)))

    inventory = {"stage": STAGE, "created_at": utc_now(), "data_dir": str(paths.root),
                 "status": overall_status(checks), "seconds": round(time.time() - started),
                 "files": files,
                 "query_inputs": {n: {k: v for k, v in lst.items() if k != "ids"} for n, lst in lists.items()},
                 "unknown_files": unknown, "checks": [c.to_dict() for c in checks]}
    write_json(inventory, out / "inventory.json")
    tables = {n: {"rows": f.get("rows"), "date_range": f.get("date_range"),
                  "permnos": f.get("distinct", {}).get("permno", f.get("distinct", {}).get("lpermno"))}
              for n, f in files.items()}
    summary = write_summary(out, STAGE, tables, checks)
    print_summary(inventory)
    return summary


def print_summary(inv: dict) -> None:
    print(f"\n{'file':<30} {'MB':>8} {'rows':>12} {'cols':>5}  date range")
    for name, f in inv["files"].items():
        if not f["exists"]:
            print(f"{name:<30} MISSING")
            continue
        rng = " to ".join(f["date_range"]) if f["date_range"] else "-"
        print(f"{name:<30} {f['mb']:>8,.1f} {f['rows']:>12,} {f['n_columns']:>5}  {rng}")
    ib = inv["files"].get("forward_estimates.csv", {})
    if ib.get("exists") and "measure, fpi" in ib["counts"]:
        by = {}
        for r in ib["counts"]["measure, fpi"]:
            by.setdefault(r["measure"], {})[r["fpi"]] = r["rows"]
        fpis = sorted({r["fpi"] for r in ib["counts"]["measure, fpi"]})
        print("\nIBES rows by measure and fpi:  " + "  ".join(f"{p:>9}" for p in fpis))
        for m, row in sorted(by.items()):
            print(f"  {m:<28}" + "  ".join(f"{row.get(p, 0):>9,}" for p in fpis))
    for name in ("fundamentals_raw.csv", "fundamentals_annual_raw.csv"):
        f = inv["files"].get(name, {})
        if f.get("exists"):
            print(f"{name}: {f['rows']:,} rows, {f['distinct'].get('lpermno', 0):,} permnos, "
                  f"{f['distinct'].get('gvkey', 0):,} gvkeys")
    problems = [c for c in inv["checks"] if c["status"] != PASS]
    print(f"\nChecks: {len(inv['checks'])} run, {len(problems)} not PASS -> status {inv['status']}")
    for c in problems:
        print(f"  {c['status']}  {c['name']}: {c['detail']}")
