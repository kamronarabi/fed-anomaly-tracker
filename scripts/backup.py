"""Off-Railway DuckDB backup: EXPORT DATABASE -> tar -> upload to Cloudflare R2.

Runs as the last step of the weekly orchestrator (scripts/seed.py), since
that's when the data structurally changes most (new awards from the
incremental ingest). Best-effort: a backup failure is logged loudly but
does not fail the pipeline run that already succeeded at the far more
important job of ingesting/scoring/publishing fresh data.

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


def _export_and_tar(db_path: str, tar_path: Path) -> None:
    with tempfile.TemporaryDirectory() as export_dir:
        con = duckdb.connect(db_path, read_only=True)
        try:
            con.execute(f"EXPORT DATABASE '{export_dir}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally:
            con.close()

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
