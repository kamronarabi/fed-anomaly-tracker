"""Weekly orchestrator: incremental USAspending pull -> load -> composite score -> publish.

Usage: python scripts/seed.py

Triggered by POST /api/refresh?mode=weekly. Uses the incremental pull
(paginated delta since the last loaded award_date), not a full archive
re-seed -- a full seed takes 30-90+ minutes and is a one-time historical
backfill operation, not something safe to run synchronously inside an
HTTP request on a recurring cron.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from export.publish import publish  # noqa: E402
from ingestion.load_db import load_all_parquet, resolve_db_path  # noqa: E402
from ingestion.pull_awards import pull_awards  # noqa: E402
from scoring.composite import compute_composite_scores  # noqa: E402
from scripts.backup import run_backup  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    db_path = resolve_db_path()
    score_date = date.today()

    logger.info("seed: step 1/4 pull_awards (incremental)")
    asyncio.run(pull_awards(incremental=True, db_path=db_path))

    logger.info("seed: step 2/4 load_db")
    deltas = load_all_parquet(db_path=db_path)
    logger.info("seed:   loaded %s", deltas)

    logger.info("seed: step 3/4 composite scoring")
    scores = compute_composite_scores(db_path, score_date=score_date)
    logger.info("seed:   scored %d entities", scores.height)

    logger.info("seed: step 4/4 export.publish")
    counts = publish(db_path, score_date)
    logger.info("seed:   published %s", counts)

    # Best-effort: a backup failure shouldn't fail a run that already
    # succeeded at ingesting/scoring/publishing fresh data.
    try:
        key = run_backup(db_path, score_date=score_date)
        logger.info("seed:   backed up to %s", key)
    except Exception:
        logger.exception("seed:   backup failed (non-fatal, continuing)")

    logger.info("seed: complete for %s", score_date)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("seed: failed")
        sys.exit(1)
