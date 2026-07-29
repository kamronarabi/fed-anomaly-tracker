"""Smoke test for Phase 1: small slice end-to-end (archive → awards → DuckDB).

Downloads one HHS FY archive ZIP via `pull_awards_archive`, parses it,
loads into a throwaway smoke DuckDB. Verifies each step's row counts so
we surface bugs in <2 minutes before committing to the multi-hour real
seed.

History: the SAM-enrichment leg was removed in the 2026-05-27 pivot
when SAM was dropped from the project entirely.

This script is intentionally short-lived and does not touch the real
data/anomaly_radar.duckdb.

Run: python scripts/smoke_phase1.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import polars as pl
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.load_db import effective_agency, load_all_parquet, load_config  # noqa: E402
from ingestion.pull_awards import pull_awards_archive  # noqa: E402


async def main():
    load_dotenv()
    config = load_config()

    hhs_raw = next(a for a in config["agencies"] if a["short"] == "HHS")
    hhs = effective_agency(hhs_raw, config)
    print(f"hhs effective config: seed={hhs['seed_strategy']} types={hhs['award_types']}")

    tmp_root = Path(tempfile.mkdtemp(prefix="smoke_phase1_"))
    parquet_dir = tmp_root / "parquet"
    parquet_dir.mkdir()
    db_path = str(tmp_root / "smoke.duckdb")
    print(f"smoke workspace: {tmp_root}")

    try:
        # ── Step 1: archive download for HHS FY2024 ───────────────────────
        # ~25 MB ZIP, downloads in 30-60s on a normal connection.
        fy = 2024
        print(f"\n[1/3] archive pulling HHS awards FY{fy}…")
        written = await asyncio.to_thread(
            pull_awards_archive, hhs, fy, config, parquet_dir,
        )
        if not written:
            print("FAIL: pull_awards_archive returned no Parquet files")
            sys.exit(1)
        total_rows = sum(pl.read_parquet(p).height for p in written)
        print(f"  → {len(written)} parquet file(s), {total_rows} rows")
        if total_rows == 0:
            print("FAIL: archive pulled zero rows from FY2024 HHS")
            sys.exit(1)
        # Sanity: HHS FY at award-summary granularity should be in the
        # 50K-300K range. Higher would suggest filtering misapplied.
        if total_rows > 500_000:
            print(
                f"WARN: row count {total_rows} is unexpectedly high — "
                "verify subaward exclusion and any agency-level filters"
            )

        # ── Step 2: load into smoke DuckDB ─────────────────────────────────
        print(f"\n[2/3] loading awards into {db_path}…")
        deltas = load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)
        print(f"  → deltas: {deltas}")
        if deltas["awards"] == 0:
            print("FAIL: zero rows landed in awards table")
            sys.exit(1)

        # ── Step 3: cross-table sanity ─────────────────────────────────────
        print("\n[3/3] sanity:")
        con = duckdb.connect(db_path, read_only=True)
        try:
            n = con.execute("SELECT COUNT(*) FROM awards").fetchone()[0]
            print(f"  awards: {n}")
            agencies = [
                r[0] for r in con.execute(
                    "SELECT DISTINCT awarding_agency FROM awards"
                ).fetchall()
            ]
            print(f"  distinct awarding_agency: {agencies}")
            sample = con.execute(
                "SELECT award_id, recipient_name, total_obligation "
                "FROM awards LIMIT 3"
            ).fetchall()
            print("  sample rows:")
            for s in sample:
                print(f"    {s}")
        finally:
            con.close()

        print("\nOK: phase 1 smoke passed")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
