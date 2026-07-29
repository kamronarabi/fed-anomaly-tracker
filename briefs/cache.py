"""Persistence layer for entity_briefs: forward-carry lookup and idempotent write.

The cache key is `(uei, input_hash, prompt_version)`. A hit means we've
already generated this exact brief for this entity at some prior point
and can reuse `brief_text` verbatim — write a new row stamped with
today's score_date but no API call.

Same-score_date re-runs are also handled here: a row keyed by
`(uei, score_date)` will be overwritten via DuckDB's INSERT OR REPLACE.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import duckdb


def find_cached_brief(
    db_path: str,
    uei: str,
    input_hash: str,
    prompt_version: str,
) -> Optional[dict]:
    """Look for any prior entity_briefs row matching (uei, input_hash, prompt_version).

    Returns the most recent matching row as a dict, or None if no match.
    Used by the generator to forward-carry briefs whose inputs haven't
    changed since they were last generated.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(
            """
            SELECT uei, score_date, input_hash, brief_text, model,
                   prompt_version, generated_at
            FROM entity_briefs
            WHERE uei = ? AND input_hash = ? AND prompt_version = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            [uei, input_hash, prompt_version],
        ).fetchone()
    finally:
        con.close()

    if row is None:
        return None
    return {
        "uei": row[0],
        "score_date": row[1],
        "input_hash": row[2],
        "brief_text": row[3],
        "model": row[4],
        "prompt_version": row[5],
        "generated_at": row[6],
    }


def write_brief(
    db_path: str,
    *,
    uei: str,
    score_date: date,
    input_hash: str,
    brief_text: str,
    model: str,
    prompt_version: str,
    generated_at: datetime,
) -> None:
    """Upsert a brief row keyed by (uei, score_date).

    INSERT OR REPLACE so repeated same-date runs don't error on the PK
    and the latest write wins.
    """
    con = duckdb.connect(db_path)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO entity_briefs (
                uei, score_date, input_hash, brief_text, model,
                prompt_version, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [uei, score_date, input_hash, brief_text, model,
             prompt_version, generated_at],
        )
    finally:
        con.close()
