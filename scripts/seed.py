"""Weekly orchestrator: incremental USAspending pull -> load -> composite score -> publish.

Usage: python scripts/seed.py

Triggered by POST /api/refresh?mode=weekly. Uses the incremental pull
(paginated delta since the recorded ingest watermark), not a full archive
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

from briefs.generator import generate_briefs  # noqa: E402
from export.publish import publish  # noqa: E402
from ingestion.load_db import (  # noqa: E402
    load_all_parquet,
    resolve_db_path,
    set_watermark,
)
from ingestion.pull_awards import pull_awards  # noqa: E402
from refresh_report import write_failure_report  # noqa: E402
from scoring.composite import compute_composite_scores  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    db_path = resolve_db_path()
    score_date = date.today()

    logger.info("seed: step 1/5 pull_awards (incremental)")
    pulled = asyncio.run(pull_awards(incremental=True, db_path=db_path))

    logger.info("seed: step 2/5 load_db")
    deltas = load_all_parquet(db_path=db_path)
    logger.info("seed:   loaded %s", deltas)

    # Only now that the Parquet is durably in the DB is it safe to say
    # we're pulled through that date -- the Parquet staging dir lives on
    # the container filesystem, not the volume, so advancing the
    # watermark any earlier could skip a window a redeploy then erased.
    for agency_name, through in pulled.pulled_through.items():
        set_watermark(db_path, agency_name, through)
        logger.info("seed:   watermark %s -> %s", agency_name, through)

    logger.info("seed: step 3/5 composite scoring")
    scores = compute_composite_scores(db_path, score_date=score_date)
    logger.info("seed:   scored %d entities", scores.height)

    # Brief here rather than leaning on the daily run to have already
    # covered this date. publish() carries forward the newest prior brief,
    # so skipping this wouldn't blank the site -- but a weekly ingest is
    # exactly when scores move most, and an unbriefed weekly would leave
    # the top-N narrated by pre-ingest text until the next daily run.
    # generate_briefs forward-carries by input hash, so entities whose
    # signal didn't actually change cost nothing.
    logger.info("seed: step 4/5 briefs")
    api_calls = generate_briefs(db_path, score_date=score_date)
    logger.info("seed:   %d fresh Anthropic calls", api_calls)

    logger.info("seed: step 5/5 export.publish")
    counts = publish(db_path, score_date)
    logger.info("seed:   published %s", counts)

    logger.info("seed: complete for %s", score_date)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("seed: failed")
        write_failure_report("weekly")
        sys.exit(1)
