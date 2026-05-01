"""Smoke test for Phase 1: small slice end-to-end (awards → entities → DuckDB).

Paginates a 1-week HHS window via `pull_window` (sync httpx), enriches
the first N UEIs from SAM, loads everything into a throwaway smoke DuckDB.
Verifies each step's row counts so we surface bugs in <2 minutes before
committing to the multi-hour real seed.

This script is intentionally short-lived and does not touch the real
data/anomaly_radar.duckdb.

Run: python scripts/smoke_phase1.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import polars as pl
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.load_db import effective_agency, load_all_parquet, load_config  # noqa: E402
from ingestion.pull_awards import pull_awards_archive  # noqa: E402
from ingestion.pull_entities import (  # noqa: E402
    _entities_schema,
    _pull_many,
    _snapshot_rows_from_pulled,
    _snapshot_schema,
)

SMOKE_UEI_LIMIT = 5


async def main():
    load_dotenv()
    config = load_config()
    sam_key = os.environ.get("SAM_API_KEY")
    if not sam_key:
        print("FAIL: SAM_API_KEY missing from .env")
        sys.exit(1)

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
        # ~25 MB ZIP, downloads in 30-60s on a normal connection. End-to-end
        # this exercises the new `seed_strategy: "archive"` path.
        fy = 2024
        print(f"\n[1/5] archive pulling HHS awards FY{fy}…")
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
        print(f"\n[2/5] loading awards into {db_path}…")
        deltas = load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)
        print(f"  → deltas: {deltas}")
        # Bulk may have duplicate rows (mod-level granularity vs award-level
        # PK); upsert keeps the schema intact, just reports a smaller delta.
        if deltas["awards"] == 0:
            print("FAIL: zero rows landed in awards table")
            sys.exit(1)

        # ── Step 3: pick a few UEIs to enrich ──────────────────────────────
        con = duckdb.connect(db_path, read_only=True)
        try:
            ueis = [
                r[0] for r in con.execute(
                    "SELECT DISTINCT recipient_uei FROM awards "
                    "WHERE recipient_uei IS NOT NULL AND recipient_uei != '' "
                    f"LIMIT {SMOKE_UEI_LIMIT}"
                ).fetchall()
            ]
        finally:
            con.close()
        print(f"\n[3/5] picked {len(ueis)} UEIs for SAM smoke: {ueis}")
        if not ueis:
            print("FAIL: no UEIs in pulled awards")
            sys.exit(1)

        # ── Step 4: hit SAM for those UEIs ─────────────────────────────────
        print(f"\n[4/5] pulling SAM entity records for {len(ueis)} UEIs…")
        rows = await _pull_many(ueis, sam_key, config["sam_api_base"])
        print(f"  → {len(rows)}/{len(ueis)} successfully enriched")
        if len(rows) == 0:
            print(
                "WARN: zero SAM enrichments — verify API key and field paths"
            )

        if rows:
            ent_path = parquet_dir / "entities.parquet"
            pl.DataFrame(rows, schema=_entities_schema()).write_parquet(ent_path)
            snap_path = parquet_dir / f"entity_snapshots_{date.today().isoformat()}.parquet"
            pl.DataFrame(
                _snapshot_rows_from_pulled(rows, date.today()),
                schema=_snapshot_schema(),
            ).write_parquet(snap_path)
            deltas2 = load_all_parquet(db_path=db_path, parquet_dir=parquet_dir)
            print(f"  → load deltas (post-entities): {deltas2}")

        # ── Step 5: cross-table sanity ─────────────────────────────────────
        print("\n[5/5] cross-table sanity:")
        con = duckdb.connect(db_path, read_only=True)
        try:
            for t in ["awards", "entities", "entity_snapshots"]:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  {t}: {n}")
            agencies = [
                r[0] for r in con.execute(
                    "SELECT DISTINCT awarding_agency FROM awards"
                ).fetchall()
            ]
            print(f"  distinct awarding_agency: {agencies}")
            sample = con.execute(
                "SELECT a.recipient_name, e.legal_business_name, "
                "       e.physical_state, e.cage_code "
                "FROM awards a LEFT JOIN entities e ON a.recipient_uei = e.uei "
                "WHERE e.uei IS NOT NULL LIMIT 3"
            ).fetchall()
            print("  joined sample (awards × entities):")
            for s in sample:
                print(f"    {s}")
        finally:
            con.close()

        print("\nOK: phase 1 smoke passed")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
