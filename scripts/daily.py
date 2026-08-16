"""Daily orchestrator: rescore -> regenerate briefs -> publish.

Usage: python scripts/daily.py

Triggered by POST /api/refresh?mode=daily. Does not pull new USAspending
data -- rescores against whatever's already loaded, then briefs the
current top-N. Cheap on days where scores didn't move: generate_briefs
forward-carries cached briefs by content hash, so this only pays for
Anthropic calls on entities whose signal actually changed.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from briefs.generator import generate_briefs  # noqa: E402
from export.publish import publish  # noqa: E402
from ingestion.load_db import resolve_db_path  # noqa: E402
from scoring.composite import compute_composite_scores  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    db_path = resolve_db_path()
    score_date = date.today()

    logger.info("daily: step 1/3 composite scoring")
    scores = compute_composite_scores(db_path, score_date=score_date)
    logger.info("daily:   scored %d entities", scores.height)

    logger.info("daily: step 2/3 briefs")
    api_calls = generate_briefs(db_path, score_date=score_date)
    logger.info("daily:   %d fresh Anthropic calls", api_calls)

    logger.info("daily: step 3/3 export.publish")
    counts = publish(db_path, score_date)
    logger.info("daily:   published %s", counts)

    logger.info("daily: complete for %s", score_date)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("daily: failed")
        sys.exit(1)
