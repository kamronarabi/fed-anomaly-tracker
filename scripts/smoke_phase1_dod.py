"""DoD-only smoke: verifies the streaming archive path can handle a 1 GB
ZIP on 8 GB RAM without OOMing.

Pulls only the awards step for DoD FY2024 (no SAM enrichment, no DB
load) — the goal is to confirm `pl.scan_csv` + `sink_parquet` keeps
peak memory bounded. Watch with Activity Monitor / `top -pid <pid>`
to confirm RSS stays under ~2 GB.

Run: python scripts/smoke_phase1_dod.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.load_db import effective_agency, load_config  # noqa: E402
from ingestion.pull_awards import pull_awards_archive  # noqa: E402


async def main():
    config = load_config()
    dod_raw = next(a for a in config["agencies"] if a["short"] == "DoD")
    dod = effective_agency(dod_raw, config)
    print(
        f"DoD effective config: seed={dod['seed_strategy']} "
        f"types={dod['award_types']} min={dod['award_amount_min']}"
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="smoke_dod_"))
    parquet_dir = tmp_root / "parquet"
    parquet_dir.mkdir()
    print(f"workspace: {tmp_root}")
    print("WATCH MEMORY: top -pid $(pgrep -f smoke_phase1_dod) -stats pid,mem,rsize")

    try:
        fy = 2024
        t0 = time.monotonic()
        print(f"\nstreaming DoD FY{fy} (~1 GB ZIP, ~5 min download + parse)…")
        written = await asyncio.to_thread(
            pull_awards_archive, dod, fy, config, parquet_dir,
        )
        elapsed = time.monotonic() - t0

        if not written:
            print("FAIL: no parquet written")
            sys.exit(1)
        out_path = written[0]
        size_mb = out_path.stat().st_size / (1 << 20)
        rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
        print(f"\nDONE in {elapsed:.0f}s")
        print(f"  parquet: {size_mb:.1f} MB on disk")
        print(f"  rows after filters (C/D + $25K): {rows:,}")
        # Quick sanity: peek at award_type distribution
        types = (
            pl.scan_parquet(out_path)
            .group_by("award_type")
            .agg(pl.len().alias("n"))
            .collect()
            .sort("n", descending=True)
        )
        print("  award_type distribution:")
        for row in types.iter_rows(named=True):
            print(f"    {row['award_type']}: {row['n']:,}")
        print("\nOK: DoD streaming smoke passed")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
