"""Off-Railway DuckDB backup: EXPORT DATABASE -> tar -> upload to Cloudflare R2.

Runs as its own process on its own cron (POST /api/refresh?mode=backup,
driven by .github/workflows/backup.yml), scheduled after the weekly
ingest -- that's when the data structurally changes most.

It used to run inline as the last step of scripts/seed.py, wrapped in a
try/except so a backup failure couldn't fail the run. That wrapper turned
out to be no protection at all: on 2026-08-24 the backup step was killed
by a signal (exit=null), which no `except` can catch, taking down a
weekly run that had already finished ingesting, scoring and publishing.
Isolating it in its own process is the actual guarantee the try/except
was only pretending to give.

Uses DuckDB's own EXPORT DATABASE rather than copying the raw .duckdb
file: a raw file copy risks grabbing a torn snapshot if anything is
writing concurrently, and ties a restore to the exact DuckDB version that
wrote it. EXPORT DATABASE produces a consistent set of Parquet files
(already ZSTD-compressed, no benefit to a further compression pass) plus
a schema.sql/load.sql -- portable across DuckDB versions and noticeably
smaller than the source file on this kind of tabular data.
"""

from __future__ import annotations

import logging
import os
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path

import boto3
import duckdb

logger = logging.getLogger(__name__)

KEY_PREFIX = "duckdb-exports"
KEEP = 8  # weekly snapshots -> ~2 months of history


def _r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


# EXPORT DATABASE will happily buffer as much as DuckDB's default limit
# (a fraction of system RAM) allows. This process shares a container with
# the Next.js server, so cap it and let DuckDB spill instead of racing the
# container's memory ceiling. Overridable per-environment.
DUCKDB_MEMORY_LIMIT = os.environ.get("BACKUP_DUCKDB_MEMORY_LIMIT", "512MB")


def _export_and_tar(db_path: str, tar_path: Path) -> None:
    with tempfile.TemporaryDirectory() as export_dir:
        logger.info("backup: exporting %s -> %s", db_path, export_dir)
        con = duckdb.connect(db_path, read_only=True)
        try:
            con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
            con.execute(f"EXPORT DATABASE '{export_dir}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally:
            con.close()

        export_mb = sum(f.stat().st_size for f in Path(export_dir).rglob("*") if f.is_file())
        logger.info("backup: exported %.1f MB, tarring", export_mb / (1024 * 1024))
        with tarfile.open(tar_path, "w") as tar:
            tar.add(export_dir, arcname="export")


def _prune_old_backups(client, bucket: str) -> None:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=f"{KEY_PREFIX}/")
    objects = sorted(resp.get("Contents", []), key=lambda o: o["Key"])
    stale = objects[:-KEEP] if len(objects) > KEEP else []
    for obj in stale:
        client.delete_object(Bucket=bucket, Key=obj["Key"])
        logger.info("backup: pruned old snapshot %s", obj["Key"])


def run_backup(db_path: str, score_date: date | None = None) -> str:
    """Export `db_path`, upload to R2, prune anything past the last KEEP
    snapshots. Returns the uploaded object key."""
    score_date = score_date or date.today()
    bucket = os.environ["R2_BUCKET_NAME"]
    key = f"{KEY_PREFIX}/fraudhound-{score_date.isoformat()}.tar"

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "backup.tar"
        _export_and_tar(db_path, tar_path)
        size_mb = tar_path.stat().st_size / (1024 * 1024)

        client = _r2_client()
        logger.info("backup: uploading %s (%.1f MB)", key, size_mb)
        client.upload_file(str(tar_path), bucket, key)
        logger.info("backup: uploaded %s (%.1f MB)", key, size_mb)

    _prune_old_backups(client, bucket)
    return key


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from ingestion.load_db import resolve_db_path  # noqa: E402

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        run_backup(resolve_db_path())
    except Exception:
        logger.exception("backup: failed")
        sys.exit(1)
