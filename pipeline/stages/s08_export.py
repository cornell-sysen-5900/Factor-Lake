"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s08_export.py
PURPOSE: Stage s08_export: the final files in processed/final/ with standard types, manifest.json,
         DATA_DICTIONARY.md and UPLOAD.md. Nothing is uploaded.

In:  the outputs of s01-s07; processed/s00_inventory/inventory.json (raw checksums and date ranges)
Out: processed/final/factor_panel_monthly.parquet, universe_periods.parquet, returns_monthly.parquet,
     prices_daily/year=YYYY/data.parquet + prices_daily/_manifest.json, market_daily.parquet, market_monthly.parquet,
     rv_multiples_monthly.parquet, rv_peer_stats_monthly.parquet, dcf_inputs_annual.parquet,
     estimates_consensus.parquet, fundamentals_quarterly.parquet, security_master.parquet,
     manifest.json, DATA_DICTIONARY.md, UPLOAD.md (exactly these files);
     processed/s08_export/summary.json (kept out of final/)

Rules (pipeline/PLAN.md, s08_export): permno int32, dates datetime64[ns] without time zone, floats float64; only
the columns documented in pipeline/dictionary.py (undocumented helper columns are dropped by name, so a new column
FAILs until it is documented); the daily files keep the layout of scripts/publish_daily_data_to_s3.py.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline import dictionary
from pipeline.checks import PASS, Check, check
from pipeline.config import (DELISTING_FILL, DELISTING_FILL_DEFAULT, GICS_SECTORS, PARQUET_COMPRESSION,
                             SIC_TO_GICS_SECTOR, SIC_UNCLASSIFIED, UNITS, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, utc_now, write_json, write_parquet, write_summary

STAGE = "s08_export"
FINAL = "final"                     # processed/final/; the stage's own folder holds only summary.json
S3_BUCKET = "sysen-5900-factor-lake"
S3_PREFIX = "factor-lake-data"
DAILY = "prices_daily/year=YYYY/data.parquet"
DAILY_COLS = ("permno", "ticker", "date", "prc", "prc_adj", "cfacpr", "shrout", "mkt_cap_musd", "ret", "vol",
              "is_delisting_row")
# output -> (stage, file, date column, helper columns dropped)
SOURCES = {
    "factor_panel_monthly.parquet": ("s07_panel", "factor_panel_monthly.parquet", "date", ("cfacpr_t", "cfacpr_last")),
    "universe_periods.parquet": ("s01_membership", "periods.parquet", "index_effective", ()),
    "returns_monthly.parquet": ("s02_prices", "monthly.parquet", "month_end",
                                ("siccd", "mom_1m", "mom_6m", "mom_12m", "vol_gk_1m", "vol_close_1m", "vol_gk_3m",
                                 "vol_close_3m", "vol_gk_6m", "vol_close_6m", "vol_gk_12m", "vol_close_12m")),
    DAILY: ("s02_prices", "daily", "date", None),
    "market_daily.parquet": ("s06_market", "market_daily.parquet", "date", ()),
    "market_monthly.parquet": ("s06_market", "market_monthly.parquet", "month_end", ()),
    "rv_multiples_monthly.parquet": ("s07_panel", "rv_multiples_monthly.parquet", "date", ()),
    "rv_peer_stats_monthly.parquet": ("s07_panel", "rv_peer_stats_monthly.parquet", "date", ()),
    "dcf_inputs_annual.parquet": ("s04_fundamentals_a", "dcf_inputs_annual.parquet", "period_end", ()),
    "estimates_consensus.parquet": ("s05_ibes", "estimates_consensus.parquet", "statpers", ()),
    "fundamentals_quarterly.parquet": ("s03_fundamentals_q", "fundamentals_quarterly.parquet", "period_end", ()),
    "security_master.parquet": ("s07_panel", "security_master.parquet", None, ()),
}
DOCS = ("manifest.json", "DATA_DICTIONARY.md", "UPLOAD.md")


# ---------- Types ----------

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """permno int32 (never missing), permco nullable int32, dates datetime64[ns] without time zone, floats float64."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if c == "permno":
            if s.isna().any():
                raise ValueError("permno has missing values")
            out[c] = s.astype("int32")
        elif c == "permco":
            out[c] = s.astype("Float64").round().astype("Int32")
        elif pd.api.types.is_datetime64_any_dtype(s):
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert(None)
            out[c] = s.astype("datetime64[ns]")
        elif pd.api.types.is_float_dtype(s):
            out[c] = s.astype("float64")
    return out


def schema_types(path: Path) -> dict[str, str]:
    return {f.name: str(f.type) for f in pq.read_schema(path)}


# ---------- Writing ----------

def export_table(paths: DataPaths, out: Path, name: str) -> dict:
    stage, file, _, drop = SOURCES[name]
    src = paths.stage_dir(stage) / file
    df = pd.read_parquet(src)
    df = standardize(df.drop(columns=[c for c in drop if c in df.columns]))
    write_parquet(df, out / name)
    return {"source": f"processed/{stage}/{file}", "source_rows": pq.read_metadata(src).num_rows,
            "dropped_columns": [c for c in drop]}


def export_daily(paths: DataPaths, out: Path) -> dict:
    """One file per year, sorted by permno and date, like scripts/publish_daily_data_to_s3.py, plus its manifest."""
    src = paths.stage_dir("s02_prices") / "daily"
    years, source_rows = {}, {}
    for part in sorted(src.glob("year=*"), key=lambda p: int(p.name.split("=")[1])):
        year = int(part.name.split("=")[1])
        df = standardize(pd.read_parquet(part / "data.parquet", columns=list(DAILY_COLS)))
        df = df.assign(ticker=df["ticker"].astype(object)).sort_values(["permno", "date"], kind="stable")
        target = out / "prices_daily" / f"year={year}" / "data.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), target,
                       compression=PARQUET_COMPRESSION, row_group_size=250_000)
        years[str(year)] = {"rows": len(df), "bytes": target.stat().st_size, "securities": int(df["permno"].nunique()),
                            "first_date": str(df["date"].min().date()), "last_date": str(df["date"].max().date())}
        source_rows[str(year)] = pq.read_metadata(part / "data.parquet").num_rows
    first, last = min(years, key=int), max(years, key=int)
    manifest = {"created_at": utc_now(), "source": "processed/s02_prices/daily (pipeline stage s02_prices)",
                "security_key": "permno", "date_range": [years[first]["first_date"], years[last]["last_date"]],
                "rows_written": sum(y["rows"] for y in years.values()), "years": years,
                "columns": schema_types(out / "prices_daily" / f"year={first}" / "data.parquet"),
                "units": {"prc, prc_adj": "$ per share", "shrout": "thousands of shares",
                          "mkt_cap_musd": "$ millions (prc x shrout / 1000)", "ret": "daily simple return (decimal)",
                          "vol": "shares traded"},
                "notes": "Layout of scripts/publish_daily_data_to_s3.py (year=YYYY/data.parquet, sorted by permno and "
                         "date) plus is_delisting_row: CRSP's delisting row carries only the delisting return."}
    write_json(manifest, out / "prices_daily" / "_manifest.json")
    return {"source": "processed/s02_prices/daily", "source_rows": source_rows, "years": years}


# ---------- Manifest ----------

def git_commit() -> tuple[str, bool | None]:
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def output_info(out: Path, name: str, date_col: str | None) -> dict:
    """Path, rows, MB, date range, permnos and column types of one output (all years for the daily files)."""
    files = sorted((out / "prices_daily").glob("year=*/data.parquet")) if name == DAILY else [out / name]
    names = pq.read_schema(files[0]).names
    cols = [c for c in ("permno", date_col) if c and c in names]
    keys = pd.concat([pd.read_parquet(f, columns=cols) for f in files]) if cols else None
    info = {"path": "prices_daily/year=*/data.parquet" if name == DAILY else name,
            "files": len(files), "rows": sum(pq.read_metadata(f).num_rows for f in files),
            "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
            "date_range": [str(keys[date_col].min())[:10], str(keys[date_col].max())[:10]] if date_col else None,
            "permnos": int(keys["permno"].nunique()) if keys is not None and "permno" in keys else None,
            "columns": schema_types(files[0])}
    return info


def source_ends(inventory: dict) -> dict:
    rng = {k: v.get("date_range") for k, v in inventory["files"].items()}
    end = lambda k: rng[k][1] if rng.get(k) else None
    return {"prices (CRSP CIZ daily)": end("r2000_price_crsp_ciz.csv"),
            "holdings (IWM)": end("iwm_holdings_raw.csv"),
            "Fama-French": end("ff_daily.csv"),
            "IBES (statpers)": end("forward_estimates.csv"),
            "FRED": max(end(k) for k in rng if k.startswith("fred_")),
            "Compustat quarterly (datadate)": end("fundamentals_raw.csv"),
            "Compustat annual (datadate)": end("fundamentals_annual_raw.csv"),
            "index levels": end("index_close.csv"), "ETFs": end("benchmark_etfs.csv")}


def build_manifest(paths: DataPaths, out: Path, version: str, outputs: dict, inventory: dict) -> dict:
    commit, dirty = git_commit()
    gaps = pd.read_parquet(paths.stage_dir("s01_membership") / "known_gaps.parquet")
    panel = pd.read_parquet(out / "factor_panel_monthly.parquet",
                            columns=["date", "permco", "company_cap_from_compustat"]).drop_duplicates(["date", "permco"])
    return {
        "data_version": version, "created_at": utc_now(),
        "pipeline_git_commit": commit, "pipeline_git_dirty": dirty,
        "s3_target": f"s3://{S3_BUCKET}/{S3_PREFIX}/{s3_folder(version)}/",
        "raw_files": {k: {"path": v["path"], "bytes": v["bytes"], "sha256": v["sha256"], "rows": v.get("rows"),
                          "date_range": v.get("date_range")} for k, v in inventory["files"].items()},
        "query_inputs": {k: {"path": f"raw/query_inputs/{v['path']}", "sha256": v["sha256"], "lines": v.get("lines")}
                         for k, v in inventory.get("query_inputs", {}).items()},
        "outputs": outputs,
        "units": {**{k: v for k, v in UNITS.items() if not k.startswith("mkt_cap (daily")},
                  "prices_daily mkt_cap_musd": "$ millions (the old daily_prices mkt_cap was $ thousands)",
                  "shrout": "thousands of shares (CRSP)", "dates": "datetime64[ns], no time zone"},
        "sectors": {"source": "Compustat GICS, the current classification applied to all history (not historical)",
                    "gics_sectors": {str(k): v for k, v in GICS_SECTORS.items()},
                    "sic_fallback": {"used_when": "GICS is missing (factor_panel_monthly.gics_from_sic)",
                                     "first_match_wins": [{"sector": SIC_UNCLASSIFIED, "sic": "missing, 0, 9900-9999"}]
                                     + [{"sector": n, "sic": [list(r) for r in ranges]}
                                        for n, ranges in SIC_TO_GICS_SECTOR]
                                     + [{"sector": "Industrials", "sic": "every other code"}]}},
        "delisting_returns": {"rule": "CRSP's delisting row (DlyDelFlg = Y) carries the delisting return in ret; "
                                      "it is never added again and never gives a price or market cap. A missing "
                                      "delisting return is filled by delisting action type.",
                              "fill": {**DELISTING_FILL, "other": DELISTING_FILL_DEFAULT}},
        "month_end": "the last CRSP trading day of the month; month-end prices and market caps come from the stock's "
                     "last real trading day in the month (never the delisting row)",
        "market_cap": {"mkt_cap_musd": "the listed security (prc x shrout): use for portfolio weights",
                       "company_mkt_cap_musd": "the company, all share classes: use for valuation (B/P, EV, "
                                               "multiples). Sum of our permnos with the same permco, or with "
                                               "company_cap_from_compustat the Compustat share count at the listed "
                                               "price when it exceeds ours by more than 5%"},
        "point_in_time": "factor_panel_monthly at t uses only data public by t: IWM snapshot eff_dt, Compustat "
                         "report_date, IBES statpers (at most 60 days old), announcement dates. Other tables are not "
                         "lagged; filter on report_date / statpers.",
        "known_gaps": {
            "unmatched_holdings": {"securities": len(gaps), "note": "IWM holdings that look like common stock but "
                                   "have no permno on any report date; not in any table (no extra pull, audit A8)",
                                   "list": [{"cusip": r.cusip, "security_name": r.security_name, "ticker": r.ticker,
                                             "first": str(r.first)[:10], "last": str(r.last)[:10],
                                             "snapshots": int(r.snapshots), "max_weight_pct": float(r.max_weight)}
                                            for r in gaps.itertuples()]},
            "unlisted_share_classes": {"company_months_from_compustat": int(panel["company_cap_from_compustat"].sum()),
                                       "note": "share classes without a CRSP listing are added from Compustat cshoq "
                                               "(decision 25); ADRs where Compustat counts ordinary shares can be "
                                               "overstated (CSR 2011, Nexxen 2025)"},
            "sources_end": source_ends(inventory),
        },
    }


def s3_folder(version: str) -> str:
    return f"wrds_{version[:4]}_{version[5:7]}"


# ---------- UPLOAD.md ----------

def upload_md(version: str, outputs: dict) -> str:
    folder = s3_folder(version)
    total = sum(o["size_mb"] for o in outputs.values())
    n_files = sum(o["files"] for o in outputs.values()) + 1 + len(DOCS)
    return f"""# Uploading the WRDS data (version {version}) to S3

For a teammate with the Factor-Lake repo and its `.env` holding the AWS keys (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, optionally `AWS_DEFAULT_REGION`). The target is a new folder,

    s3://{S3_BUCKET}/{S3_PREFIX}/{folder}/

next to the files the live app reads (`{S3_PREFIX}/full_precision_test.parquet`, `{S3_PREFIX}/daily_prices/`, ...),
so nothing the app reads today is overwritten. The app does not read the new folder until someone points it there.

What goes up: everything in this `final/` folder, {n_files} files, about {total:,.0f} MB. `manifest.json` goes last, so
a reader that finds it knows every other file is complete.

## 1. Save the script

Save this as `upload_wrds.py` in the Factor-Lake repo root. It uses boto3 and python-dotenv, which the repo already
depends on. It never prints the AWS keys, only the S3 object keys (file paths in the bucket).

```python
\"\"\"Upload processed/final/ to s3://{S3_BUCKET}/{S3_PREFIX}/{folder}/. Dry run unless --upload.\"\"\"
import argparse
import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

BUCKET = "{S3_BUCKET}"
PREFIX = "{S3_PREFIX}/{folder}"
TYPES = {{".parquet": "application/vnd.apache.parquet", ".json": "application/json", ".md": "text/markdown"}}

parser = argparse.ArgumentParser()
parser.add_argument("final_dir", type=Path, help="the processed/final folder")
parser.add_argument("--upload", action="store_true", help="really upload (default: dry run)")
parser.add_argument("--check", action="store_true", help="list the S3 folder and compare with the local files")
parser.add_argument("--overwrite", action="store_true", help="allow replacing objects already in the folder")
parser.add_argument("--offline", action="store_true", help="dry run without credentials or network: only list keys")
args = parser.parse_args()
if args.offline and (args.upload or args.check):
    sys.exit("--offline is a dry run only")
if not PREFIX.startswith("{S3_PREFIX}/wrds_"):
    sys.exit("Refusing a prefix outside {S3_PREFIX}/wrds_*: the live app's files must not be touched")

final = args.final_dir.resolve()
if not (final / "manifest.json").is_file():
    sys.exit(f"{{final}} has no manifest.json; pass the processed/final folder")
files = sorted((p for p in final.rglob("*") if p.is_file()), key=lambda p: p.name == "manifest.json")
local = {{f"{{PREFIX}}/{{p.relative_to(final).as_posix()}}": p for p in files}}
if args.offline:
    for key, path in local.items():
        print(f"would upload s3://{{BUCKET}}/{{key}} ({{path.stat().st_size / 1e6:,.1f}} MB)")
    print(f"{{len(local)}} files, {{sum(p.stat().st_size for p in local.values()) / 1e6:,.0f}} MB (offline dry run)")
    sys.exit(0)

load_dotenv(Path(__file__).resolve().parent / ".env")          # the repo's .env
missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not os.environ.get(k)]
if missing:
    sys.exit(f"Not set in .env: {{', '.join(missing)}}")
s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def remote_objects():
    out = {{}}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX + "/"):
        out.update({{o["Key"]: o["Size"] for o in page.get("Contents", [])}})
    return out


if args.check:
    remote = remote_objects()
    bad = 0
    for key, path in local.items():
        size = path.stat().st_size
        status = "ok" if remote.get(key) == size else ("MISSING" if key not in remote else "SIZE DIFFERS")
        bad += status != "ok"
        print(f"{{status:12}} s3://{{BUCKET}}/{{key}} ({{size / 1e6:,.1f}} MB)")
    extra = sorted(set(remote) - set(local))
    for key in extra:
        print(f"{{'NOT LOCAL':12}} s3://{{BUCKET}}/{{key}}")
    print(f"{{len(local) - bad}} of {{len(local)}} files match; {{len(extra)}} objects in S3 not in the local folder")
    sys.exit(1 if bad else 0)

existing = remote_objects()
clash = sorted(set(existing) & set(local))
if clash and not args.overwrite:
    sys.exit(f"{{len(clash)}} objects already exist under s3://{{BUCKET}}/{{PREFIX}}/ (e.g. {{clash[0]}}); "
             "rerun with --overwrite to replace them")
for key, path in local.items():
    size = path.stat().st_size
    if not args.upload:
        print(f"would upload s3://{{BUCKET}}/{{key}} ({{size / 1e6:,.1f}} MB)")
        continue
    s3.upload_file(str(path), BUCKET, key, ExtraArgs={{"ContentType": TYPES.get(path.suffix, "binary/octet-stream")}})
    if s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] != size:
        sys.exit(f"Size mismatch after uploading {{key}}")
    print(f"uploaded s3://{{BUCKET}}/{{key}} ({{size / 1e6:,.1f}} MB)")
print(f"{{len(local)}} files, {{sum(p.stat().st_size for p in local.values()) / 1e6:,.0f}} MB"
      + ("" if args.upload else " (dry run: nothing uploaded)"))
```

## 2. Dry run (uploads nothing)

From the repo root, with the path of this `final/` folder:

    uv run python upload_wrds.py "C:/path/to/WRDS FA26/processed/final"

It loads the keys from the repo's `.env`, lists what is already in the target folder (this checks that the keys
work), and prints every S3 object key it would upload with its size. Add `--offline` to only print the keys, with
no credentials and no network. Check that every line starts with
`s3://{S3_BUCKET}/{S3_PREFIX}/{folder}/`.

## 3. Upload

    uv run python upload_wrds.py "C:/path/to/WRDS FA26/processed/final" --upload

Each file is uploaded and its size in S3 compared with the local file; the script stops on the first mismatch. If
the folder already has files from an earlier attempt it refuses to continue; add `--overwrite` only when you mean to
replace them.

## 4. Check afterwards

    uv run python upload_wrds.py "C:/path/to/WRDS FA26/processed/final" --check

Lists the S3 folder and compares every object with the local file (name and size). It should end with
`{n_files} of {n_files} files match; 0 objects in S3 not in the local folder` and exit with code 0.
The same listing with the AWS CLI, if you have it: `aws s3 ls s3://{S3_BUCKET}/{S3_PREFIX}/{folder}/ --recursive`.
"""


# ---------- Checks ----------

KEYS = {name: f["key"] for name, f in dictionary.FILES.items()}


def dictionary_checks(md: str, outputs: dict) -> list[Check]:
    """The written DATA_DICTIONARY.md against the columns of the written files, both ways."""
    documented = dictionary.parse_columns(md)
    undocumented = {n: sorted(set(o["columns"]) - set(documented.get(n, []))) for n, o in outputs.items()}
    stale = {n: sorted(set(documented.get(n, [])) - set(o["columns"])) for n, o in outputs.items()}
    undocumented, stale = {k: v for k, v in undocumented.items() if v}, {k: v for k, v in stale.items() if v}
    return [check("every output column is in DATA_DICTIONARY.md", not undocumented, str(undocumented)
                  if undocumented else f"{sum(len(o['columns']) for o in outputs.values()):,} columns"),
            check("every DATA_DICTIONARY.md column is in the outputs", not stale, str(stale) if stale else "")]


def run_checks(paths: DataPaths, out: Path, exported: dict, outputs: dict, manifest: dict) -> list[Check]:
    checks = []
    for name, e in exported.items():
        if name == DAILY:
            bad = {y: (e["source_rows"][y], pq.read_metadata(out / "prices_daily" / f"year={y}" / "data.parquet")
                       .num_rows) for y in e["source_rows"]}
            bad = {y: v for y, v in bad.items() if v[0] != v[1]}
            checks.append(check(f"{name}: every year reads back with s02's rows", not bad,
                                f"{sum(e['source_rows'].values()):,} rows in {len(e['source_rows'])} files"
                                + (f"; differ: {bad}" if bad else "")))
        else:
            n = sum(b.num_rows for b in pq.ParquetFile(out / name).iter_batches(batch_size=1_000_000))
            checks.append(check(f"{name}: reads back with the source's rows", n == e["source_rows"],
                                f"{n:,} vs {e['source_rows']:,} in {e['source']}"))
    for name, key in KEYS.items():
        files = sorted((out / "prices_daily").glob("year=*/data.parquet")) if name == DAILY else [out / name]
        dups = sum(int(pd.read_parquet(f, columns=key).duplicated().sum()) for f in files)
        checks.append(check(f"{name}: no duplicate ({', '.join(key)})", dups == 0, f"{dups:,} duplicates"))
    bad_types = {name: {c: t for c, t in o["columns"].items()
                        if (c == "permno" and t != "int32") or (t.startswith("timestamp") and t != "timestamp[ns]")}
                 for name, o in outputs.items()}
    bad_types = {k: v for k, v in bad_types.items() if v}
    checks.append(check("permno is int32 and every date datetime64[ns] without time zone", not bad_types,
                        str(bad_types) if bad_types else "all files"))
    checks += dictionary_checks((out / "DATA_DICTIONARY.md").read_text(encoding="utf-8"), outputs)
    expected = {o["path"] for n, o in outputs.items() if n != DAILY} | set(DOCS) | {"prices_daily/_manifest.json"}
    expected |= {f"prices_daily/year={y}/data.parquet" for y in exported[DAILY]["years"]}
    found = {f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()}
    checks.append(check("final/ holds exactly the planned files", found == expected and set(KEYS) == set(outputs),
                        f"{len(found)} files" + (f"; missing {sorted(expected - found)}" if expected - found else "")
                        + (f"; extra {sorted(found - expected)}" if found - expected else "")))
    differ = [k for k, v in manifest["raw_files"].items() if sha256(paths.root / v["path"]) != v["sha256"]]
    checks.append(check("manifest sha256 of every raw file matches the raw file now", not differ,
                        ", ".join(differ) or f"{len(manifest['raw_files'])} files re-hashed"))
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    checks.append(check("total size", True, f"{total / 1e6:,.0f} MB; " + ", ".join(
        f"{n} {o['size_mb']:,.1f}" for n, o in outputs.items())))
    return checks


# ---------- Stage ----------

def run(paths: DataPaths) -> dict:
    out, sdir = fresh_stage_dir(paths, FINAL), fresh_stage_dir(paths, STAGE)
    version = date.today().isoformat()
    inventory = json.loads((paths.stage_dir("s00_inventory") / "inventory.json").read_text(encoding="utf-8"))
    exported = {}
    for name in SOURCES:
        exported[name] = export_daily(paths, out) if name == DAILY else export_table(paths, out, name)
        print(f"  {name}")
    outputs = {name: output_info(out, name, SOURCES[name][2]) for name in dictionary.FILES}
    manifest = build_manifest(paths, out, version, outputs, inventory)
    write_json(manifest, out / "manifest.json")
    (out / "DATA_DICTIONARY.md").write_text(dictionary.render(manifest), encoding="utf-8")
    (out / "UPLOAD.md").write_text(upload_md(version, outputs), encoding="utf-8")

    checks = run_checks(paths, out, exported, outputs, manifest)
    tables = {name: {**{k: o[k] for k in ("rows", "date_range", "permnos", "size_mb", "files")},
                     "columns": len(o["columns"])} for name, o in outputs.items()}
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    extra = {"data_version": version, "pipeline_git_commit": manifest["pipeline_git_commit"],
             "pipeline_git_dirty": manifest["pipeline_git_dirty"], "total_size_mb": round(total / 1e6, 1),
             "s3_target": manifest["s3_target"],
             "dropped_columns": {n: e["dropped_columns"] for n, e in exported.items() if e.get("dropped_columns")},
             "daily_years": {y: v["rows"] for y, v in exported[DAILY]["years"].items()},
             "sources_end": manifest["known_gaps"]["sources_end"]}
    summary = write_summary(sdir, STAGE, tables, checks, extra)
    print(f"  {len(outputs)} tables, {total / 1e6:,.0f} MB, data version {version}, commit "
          f"{manifest['pipeline_git_commit'][:12]}{' (dirty)' if manifest['pipeline_git_dirty'] else ''}")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
