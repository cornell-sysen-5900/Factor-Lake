"""
Tests for scripts/upload_wrds.py with a fake S3 client (no network, no credentials).
"""
import hashlib
import io
import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "upload_wrds.py"
mod = runpy.run_path(str(SCRIPT))
TARGET = "s3://sysen-5900-factor-lake/factor-lake-data/wrds_2026_10/"
PREFIX = "factor-lake-data/wrds_2026_10"


class FakeS3:
    """List, get, head, download and upload only; delete, copy and anything else fail the test."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.uploads = []

    def get_paginator(self, name):
        fake = self

        class Pager:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v), "ETag": f'"{hashlib.md5(v).hexdigest()}"'}
                                    for k, v in sorted(fake.objects.items()) if k.startswith(Prefix)]}
        return Pager()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        v = self.objects[Key]
        return {"ContentLength": len(v), "ETag": f'"{hashlib.md5(v).hexdigest()}"'}

    def download_file(self, Bucket, Key, Filename):
        Path(Filename).write_bytes(self.objects[Key])

    def upload_file(self, Filename, Bucket, Key, Config=None, ExtraArgs=None):
        assert Key.startswith(PREFIX + "/")
        self.objects[Key] = Path(Filename).read_bytes()
        self.uploads.append(Key)

    def __getattr__(self, name):
        raise AssertionError(f"upload_wrds called S3.{name}")


@pytest.fixture
def final(tmp_path):
    f = tmp_path / "final"
    (f / "prices_daily" / "year=2025").mkdir(parents=True)
    (f / "prices_daily" / "year=2026").mkdir(parents=True)
    (f / "a.parquet").write_bytes(b"PAR1-a" * 10)
    (f / "prices_daily" / "year=2025" / "data.parquet").write_bytes(b"PAR1-2025")
    (f / "prices_daily" / "year=2026" / "data.parquet").write_bytes(b"PAR1-2026" * 3)
    (f / "prices_daily" / "_manifest.json").write_text(json.dumps({"created_at": "T1", "years": {"2025": 1}}))
    (f / "manifest.json").write_text(json.dumps({"s3_target": TARGET}))
    (f / "UPLOAD.md").write_text("instructions")
    return f


def test_upload_sends_the_parquet_files_then_the_daily_manifest_last(final, capsys):
    s3 = FakeS3()
    assert mod["main"]([str(final), "--upload"], s3=s3) == 0
    assert s3.uploads[-1] == f"{PREFIX}/prices_daily/_manifest.json"
    assert sorted(s3.uploads[:-1]) == sorted(f"{PREFIX}/{p}" for p in (
        "a.parquet", "prices_daily/year=2025/data.parquet", "prices_daily/year=2026/data.parquet"))
    assert not any(k.endswith((".md", "/manifest.json")) for k in s3.uploads)   # UPLOAD.md and manifest.json stay
    # a second run uploads nothing
    s3.uploads.clear()
    assert mod["main"]([str(final), "--upload"], s3=s3) == 0 and s3.uploads == []
    assert capsys.readouterr().out.count("skip") == 4


def test_dry_run_uploads_nothing_and_offline_needs_no_client(final, capsys):
    s3 = FakeS3()
    assert mod["main"]([str(final)], s3=s3) == 0 and s3.uploads == []
    assert mod["main"]([str(final), "--offline"], s3=None) == 0             # no client, no secrets
    out = capsys.readouterr().out
    assert out.count("would upload") == 8 and out.strip().endswith("(offline dry run)")


def test_check_ignores_created_at_in_json_and_compares_parquet_by_etag(final, capsys):
    s3 = FakeS3()
    mod["main"]([str(final), "--upload"], s3=s3)
    m = final / "prices_daily" / "_manifest.json"
    m.write_text(json.dumps({"created_at": "T2", "years": {"2025": 1}}))   # a rerun of s08: new created_at only
    assert mod["main"]([str(final), "--check"], s3=s3) == 0
    out = capsys.readouterr().out
    assert "4 of 4 files match" in out and "content matches (created_at ignored)" in out
    m.write_text(json.dumps({"created_at": "T2", "years": {"2025": 2}}))   # real content change
    (final / "a.parquet").write_bytes(b"changed")
    assert mod["main"]([str(final), "--check"], s3=s3) == 1
    assert "2 of 4 files match" in capsys.readouterr().out


def test_a_differing_object_stops_the_upload_unless_overwrite(final):
    s3 = FakeS3({f"{PREFIX}/a.parquet": b"old"})
    with pytest.raises(SystemExit, match="differ"):
        mod["main"]([str(final), "--upload"], s3=s3)
    assert s3.uploads == []
    assert mod["main"]([str(final), "--upload", "--overwrite"], s3=s3) == 0 and len(s3.uploads) == 4


@pytest.mark.parametrize("prefix", ["factor-lake-data", "factor-lake-data/daily_prices/x", "other/wrds_2026_10",
                                    "factor-lake-data/backtest-results"])
def test_prefix_guard(final, prefix):
    with pytest.raises(SystemExit, match="Refusing prefix"):
        mod["main"]([str(final), "--offline", "--prefix", prefix])


def test_the_script_never_deletes_or_copies():
    code = SCRIPT.read_text(encoding="utf-8")
    assert not any(c in code for c in ("delete_object", "delete_objects", "copy_object", "put_object("))
