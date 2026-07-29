"""CLI entry: generate briefs for the top-N entities on a given score_date.

  python -m briefs.main                       # today, top 50
  python -m briefs.main --top-n 10            # today, top 10
  python -m briefs.main --score-date 2026-05-29 --top-n 50
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from briefs.generator import DEFAULT_MODEL, DEFAULT_TOP_N, generate_briefs
from ingestion.load_db import init_schema, resolve_db_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AI briefs for the top-N suspicion-scored entities."
    )
    parser.add_argument(
        "--score-date",
        type=date.fromisoformat,
        default=None,
        help="ISO date to brief (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"How many top-ranked entities to brief (default {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Anthropic model id (default {DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    db_path = resolve_db_path()
    init_schema(db_path)  # idempotent; ensures entity_briefs table exists

    score_date = args.score_date or date.today()
    n_calls = generate_briefs(
        db_path,
        score_date=score_date,
        top_n=args.top_n,
        model=args.model,
    )
    print(f"score_date={score_date} top_n={args.top_n} api_calls={n_calls}")


if __name__ == "__main__":
    main()
