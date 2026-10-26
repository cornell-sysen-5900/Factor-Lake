#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/s3_inspect.py
PURPOSE: Read-only look at what is in the Factor-Lake S3 bucket: a tree with sizes and dates, and the head of each
         file without downloading whole files. Never writes: no put, copy or delete calls.

Usage (from the repo root; the AWS keys come from .streamlit/secrets.toml, which is local only and never on GitHub):
    uv run python scripts/s3_inspect.py                                   # factor-lake-data/, with heads
    uv run python scripts/s3_inspect.py --no-head                         # listing only
    uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_2026_10/ --rows 5
    uv run python scripts/s3_inspect.py --prefix factor-lake-data/wrds_2026_10/ --all-partitions --max-files 5

Heads:
    Parquet   the footer read with ranged GETs: rows, row groups, columns and types, and the first --rows rows of the
              first row group. pyarrow reads whole column chunks, so the head reads the leading columns of row group
              0 that fit --head-mb (default 8 MB) and says how many columns it shows; the footer covers them all
    text      CSV, JSON, MD, TXT: the first 2 KB with a ranged GET, first 10 lines
    other     size and content type only
Partitioned folders (.../year=YYYY/...): rows per partition from the footers, and the head of the newest partition
only (--all-partitions for every one).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow.parquet as pq

try:
    import tomllib                                   # Python 3.11+
except ImportError:
    import tomli as tomllib                          # Python 3.10: installed with the repo's dev group

BUCKET = "sysen-5900-factor-lake"
TEXT = {".csv", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".log"}
PARTITION = re.compile(r"^(?P<parent>.*/)?(?P<part>[^/=]+=[^/]+)/[^/]+$")
BLOCK = 1 << 20                                      # ranged reads in 1 MB blocks


def s3_client(secrets: Path):
    """boto3 S3 client from the Streamlit secrets (top-level AWS_* keys, or a section such as [aws]). The values
    are never printed."""
    import boto3
    with open(secrets, "rb") as f:
        s = tomllib.load(f)
    found = {k: v for k, v in s.items() if isinstance(v, str)}
    for table in (v for v in s.values() if isinstance(v, dict)):
        found.update({k.upper(): v for k, v in table.items() if isinstance(v, str)})
    missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not found.get(k)]
    if missing:
        sys.exit(f"Not in {secrets}: {', '.join(missing)}")
    return boto3.client("s3", aws_access_key_id=found["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=found["AWS_SECRET_ACCESS_KEY"],
                        region_name=found.get("AWS_DEFAULT_REGION") or "us-east-1")


def list_objects(s3, bucket: str, prefix: str) -> list[dict]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        out += [{"key": o["Key"], "size": o["Size"], "modified": o["LastModified"]} for o in page.get("Contents", [])]
    return out


class RangedFile:
    """A read-only file-like view of one S3 object: every read is a ranged GET, in cached 1 MB blocks, so pyarrow can
    read a Parquet footer or the first pages without downloading the file. Counts the bytes fetched."""

    def __init__(self, s3, bucket: str, key: str, size: int):
        self.s3, self.bucket, self.key, self.size = s3, bucket, key, size
        self.pos, self.fetched, self._blocks = 0, 0, {}
        self.closed = False

    def _block(self, i: int) -> bytes:
        if i not in self._blocks:
            lo, hi = i * BLOCK, min((i + 1) * BLOCK, self.size) - 1
            body = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={lo}-{hi}")["Body"].read()
            self.fetched += len(body)
            if len(self._blocks) > 64:                # keep memory low
                self._blocks.pop(next(iter(self._blocks)))
            self._blocks[i] = body
        return self._blocks[i]

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            i, off = divmod(self.pos, BLOCK)
            chunk = self._block(i)[off: off + end - self.pos]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self.closed, self._blocks = True, {}

    @property
    def mode(self) -> str:
        return "rb"


def head_columns(pf: pq.ParquetFile, budget_mb: float) -> list[str]:
    """The leading columns of row group 0 whose compressed chunks fit the budget (at least one). Columns are stored
    in order, so these are contiguous bytes near the start of the file."""
    names = pf.schema_arrow.names
    rg = pf.metadata.row_group(0)
    if rg.num_columns != len(names):                 # nested columns: no simple chunk-to-column map
        return names
    pick, used = [], 0
    for i, name in enumerate(names):
        size = rg.column(i).total_compressed_size
        if pick and used + size > budget_mb * 1e6:
            break
        pick.append(name)
        used += size
    return pick


def parquet_head(s3, bucket: str, obj: dict, rows: int, budget_mb: float = 8.0) -> tuple[dict, pd.DataFrame | None]:
    """Footer facts and the first rows of the first row group, read with ranged GETs."""
    f = RangedFile(s3, bucket, obj["key"], obj["size"])
    pf = pq.ParquetFile(f, pre_buffer=False)
    meta = {"rows": pf.metadata.num_rows, "row_groups": pf.metadata.num_row_groups,
            "columns": {fld.name: str(fld.type) for fld in pf.schema_arrow}}
    head = None
    if rows > 0 and pf.metadata.num_rows:
        cols = head_columns(pf, budget_mb)
        head = pf.read_row_group(0, columns=cols).slice(0, rows).to_pandas()
        meta["head_columns"] = len(cols)
    meta["fetched_mb"] = f.fetched / 1e6
    f.close()
    return meta, head


def parquet_rows(s3, bucket: str, obj: dict) -> int:
    f = RangedFile(s3, bucket, obj["key"], obj["size"])
    n = pq.ParquetFile(f).metadata.num_rows
    f.close()
    return n


def text_head(s3, bucket: str, key: str, lines: int = 10) -> list[str]:
    body = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-2047")["Body"].read()
    return body.decode("utf-8", errors="replace").splitlines()[:lines]


# ---------- printing ----------

def tree(objs: list[dict], prefix: str) -> list[str]:
    """Folders with their total objects and MB, files with size and LastModified, indented by depth."""
    folders = defaultdict(lambda: [0, 0])
    for o in objs:
        parts = PurePosixPath(o["key"][len(prefix):]).parts[:-1]
        for d in range(1, len(parts) + 1):
            folders["/".join(parts[:d]) + "/"][0] += 1
            folders["/".join(parts[:d]) + "/"][1] += o["size"]
    lines, shown = [], set()
    for o in sorted(objs, key=lambda o: o["key"]):
        parts = PurePosixPath(o["key"][len(prefix):]).parts
        for d in range(1, len(parts)):
            folder = "/".join(parts[:d]) + "/"
            if folder not in shown:
                shown.add(folder)
                n, b = folders[folder]
                lines.append(f"{'  ' * (d - 1)}{parts[d - 1]}/".ljust(60) + f"{n:6} objects {b / 1e6:10.1f} MB")
        lines.append(f"{'  ' * (len(parts) - 1)}{parts[-1]}".ljust(60)
                     + f"{o['size'] / 1e6:10.2f} MB  {o['modified']:%Y-%m-%d %H:%M}")
    return lines


def show_frame(df: pd.DataFrame) -> str:
    """Wide frames (more than 8 columns) are shown transposed: one line per column. Midnight timestamps as dates."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) and (df[c].dropna().dt.normalize() == df[c].dropna()).all():
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_colwidth", 30):
        return (df.T if df.shape[1] > 8 else df).to_string()


def main(argv: list[str] | None = None, s3=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only look at the Factor-Lake S3 bucket.")
    parser.add_argument("--prefix", default="factor-lake-data/", help="key prefix (default factor-lake-data/)")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--rows", type=int, default=5, help="rows of each Parquet head (default 5)")
    parser.add_argument("--max-files", type=int, help="show the head of at most this many files")
    parser.add_argument("--no-head", action="store_true", help="listing only")
    parser.add_argument("--all-partitions", action="store_true", help="head of every partition, not only the newest")
    parser.add_argument("--head-mb", type=float, default=8.0,
                        help="read at most about this many MB of column data per Parquet head (default 8)")
    parser.add_argument("--secrets", type=Path, default=Path(__file__).resolve().parents[1] / ".streamlit" /
                        "secrets.toml", help="Streamlit secrets with the AWS keys (default: the repo's)")
    args = parser.parse_args(argv)
    s3 = s3 or s3_client(args.secrets)

    objs = list_objects(s3, args.bucket, args.prefix)
    print(f"s3://{args.bucket}/{args.prefix}: {len(objs)} objects, {sum(o['size'] for o in objs) / 1e6:,.1f} MB\n")
    print("\n".join(tree(objs, args.prefix)))
    if args.no_head or not objs:
        return 0

    # partitioned folders: rows per partition, head of the newest only (unless --all-partitions)
    parts = defaultdict(list)
    for o in objs:
        m = PARTITION.match(o["key"])
        if m and o["key"].endswith(".parquet"):
            parts[m.group("parent") or ""].append((m.group("part"), o))
    skip = set()
    for parent, items in parts.items():
        items.sort(key=lambda po: po[0])
        print(f"\n== {parent} partitions: rows per partition (from the footers)")
        for part, o in items:
            print(f"  {part:20} {parquet_rows(s3, args.bucket, o):12,} rows  {o['size'] / 1e6:8.1f} MB")
        if not args.all_partitions:
            skip |= {o["key"] for _, o in items[:-1]}

    shown = 0
    for o in sorted(objs, key=lambda o: o["key"]):
        if o["key"] in skip or o["key"].endswith("/"):
            continue
        if args.max_files is not None and shown >= args.max_files:
            print(f"\n(heads stop after --max-files {args.max_files})")
            break
        shown += 1
        suffix = PurePosixPath(o["key"]).suffix.lower()
        print(f"\n== {o['key']}  ({o['size'] / 1e6:,.2f} MB, {o['modified']:%Y-%m-%d %H:%M})")
        if suffix == ".parquet":
            meta, head = parquet_head(s3, args.bucket, o, args.rows, args.head_mb)
            print(f"  {meta['rows']:,} rows, {meta['row_groups']} row groups, {len(meta['columns'])} columns "
                  f"({meta['fetched_mb']:.2f} MB read)")
            cols = list(meta["columns"].items())
            for i in range(0, len(cols), 4):
                print("  " + "  ".join(f"{c}: {t}"[:44].ljust(44) for c, t in cols[i:i + 4]))
            if head is not None:
                n, total = meta["head_columns"], len(meta["columns"])
                if n < total:
                    print(f"  head: the first {n} of {total} columns (within --head-mb {args.head_mb:g})")
                print(show_frame(head))
        elif suffix in TEXT:
            for line in text_head(s3, args.bucket, o["key"]):
                print(f"  {line[:200]}")
        else:
            head = s3.head_object(Bucket=args.bucket, Key=o["key"])
            print(f"  {head.get('ContentType', 'unknown type')}, {o['size']:,} bytes (no head for this type)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
