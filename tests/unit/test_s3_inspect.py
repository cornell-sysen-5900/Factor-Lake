"""
Tests for scripts/s3_inspect.py with a fake, read-only S3 client (no network, no credentials).
"""
import io
import runpy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "s3_inspect.py"
mod = runpy.run_path(str(SCRIPT))


class FakeS3:
    """Only list, get (with Range) and head exist; any other call (put, copy, delete, ...) fails the test."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects, self.gets = objects, []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Pager:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in fake.objects if k.startswith(Prefix))
                for i in range(0, len(keys), 2):                 # pages of 2 to exercise pagination
                    yield {"Contents": [{"Key": k, "Size": len(fake.objects[k]),
                                         "LastModified": datetime(2026, 10, 25, tzinfo=timezone.utc)}
                                        for k in keys[i:i + 2]]}
        return Pager()

    def get_object(self, Bucket, Key, Range=None):
        data = self.objects[Key]
        if Range:
            lo, hi = (int(x) for x in Range.removeprefix("bytes=").split("-"))
            data = data[lo:hi + 1]
        self.gets.append((Key, len(data)))
        return {"Body": io.BytesIO(data)}

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key]), "ContentType": "application/zip"}

    def __getattr__(self, name):
        raise AssertionError(f"s3_inspect called S3.{name}, which a read-only tool must not use")


def parquet_bytes(df: pd.DataFrame, row_group_size=None) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, row_group_size=row_group_size,
                   data_page_size=8 * 1024)
    return buf.getvalue()


@pytest.fixture
def bucket():
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"permno": np.arange(1_200_000, dtype="int32"), "ret": rng.normal(size=1_200_000)})
    return FakeS3({
        "p/wrds/big.parquet": parquet_bytes(big, row_group_size=400_000),       # ~14 MB, 3 row groups
        "p/wrds/prices_daily/year=2024/data.parquet": parquet_bytes(big.head(10)),
        "p/wrds/prices_daily/year=2025/data.parquet": parquet_bytes(big.head(20)),
        "p/wrds/prices_daily/_manifest.json": "\n".join(f'"line{i}": {i},' for i in range(30)).encode(),
        "p/_deploy/lambda.zip": b"PK" + b"0" * 5000,
    })


def run(capsys, bucket, *argv):
    assert mod["main"](list(argv), s3=bucket) == 0
    return capsys.readouterr().out


def test_listing_tree_with_folder_totals(capsys, bucket):
    out = run(capsys, bucket, "--prefix", "p/", "--no-head")
    assert "s3://sysen-5900-factor-lake/p/: 5 objects" in out
    assert "wrds/" in out and "prices_daily/" in out and "year=2025/" in out
    folder = next(line for line in out.splitlines() if line.strip().startswith("prices_daily/"))
    assert "3 objects" in folder
    assert "== " not in out                                              # --no-head: no heads


def test_parquet_head_reads_only_the_footer_and_first_pages(capsys, bucket):
    out = run(capsys, bucket, "--prefix", "p/wrds/big", "--rows", "3")
    assert "1,200,000 rows, 3 row groups, 2 columns" in out and "permno: int32" in out
    size = len(bucket.objects["p/wrds/big.parquet"])
    fetched = sum(n for k, n in bucket.gets if k == "p/wrds/big.parquet")
    assert fetched < size / 2                                             # ranged reads, not the whole file
    meta, head = mod["parquet_head"](bucket, "b", {"key": "p/wrds/big.parquet", "size": size}, 3)
    assert head["permno"].tolist() == [0, 1, 2]


def test_wide_heads_stay_within_the_budget(capsys):
    wide = pd.DataFrame({f"c{i:02d}": np.random.default_rng(i).normal(size=200_000) for i in range(30)})
    s3 = FakeS3({"w/wide.parquet": parquet_bytes(wide)})                    # one row group, ~48 MB
    out = run(capsys, s3, "--prefix", "w/", "--head-mb", "4", "--rows", "2")
    assert "head: the first 2 of 30 columns" in out
    size = len(s3.objects["w/wide.parquet"])
    assert sum(n for _, n in s3.gets) < 0.15 * size


def test_partitions_show_rows_each_and_the_newest_head(capsys, bucket):
    out = run(capsys, bucket, "--prefix", "p/wrds/prices_daily/")
    assert "year=2024" in out and "10 rows" in out and "20 rows" in out
    assert "== p/wrds/prices_daily/year=2025/data.parquet" in out
    assert "== p/wrds/prices_daily/year=2024/data.parquet" not in out    # only the newest head by default
    out = run(capsys, bucket, "--prefix", "p/wrds/prices_daily/", "--all-partitions")
    assert "== p/wrds/prices_daily/year=2024/data.parquet" in out


def test_text_head_and_other_types(capsys, bucket):
    out = run(capsys, bucket, "--prefix", "p/")
    manifest = out.split("== p/wrds/prices_daily/_manifest.json")[1].split("== ")[0]
    assert '"line9": 9' in manifest and '"line10"' not in manifest       # first 10 lines only
    assert ("p/wrds/prices_daily/_manifest.json", 2048) in bucket.gets or any(
        k.endswith("_manifest.json") and n <= 2048 for k, n in bucket.gets)
    assert "application/zip" in out and "no head for this type" in out


def test_max_files_limits_the_heads(capsys, bucket):
    out = run(capsys, bucket, "--prefix", "p/", "--max-files", "1")
    assert out.count("\n== p/") == 1 + 1                                  # the partition summary + one head
    assert "--max-files 1" in out


def test_the_script_never_writes():
    code = SCRIPT.read_text(encoding="utf-8")
    for call in ("put_object", "upload_file", "copy_object", "delete_object", "delete_objects", "upload_fileobj"):
        assert call not in code
