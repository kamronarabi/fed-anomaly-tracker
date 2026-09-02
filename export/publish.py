"""Publish DuckDB scoring + brief data as JSON files for the Next.js frontend.

Run after `briefs.main` in the nightly cron:

    python -m export.publish --score-date 2026-05-29 --top-n 50

Writes (under `--out`, default $PUBLISH_DIR or `web/public/data`):
- `leaderboard.json` — one file describing the homepage's top-50 layer.
- `entities/<uei>.json` — one file per top-N entity for the detail page.

Atomic writes (tmp + os.replace) so a partially-rendered frontend never
sees a half-written JSON during a rebuild.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

from ingestion.load_db import resolve_db_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


DEFAULT_TOP_N = 50
# Where the frontend reads its JSON from. In production this MUST point
# at the persistent volume ($PUBLISH_DIR=/data/public/data): the repo's
# web/public/data lives in the Docker image, so anything published there
# is discarded the moment the container restarts, silently reverting the
# site to whatever JSON was committed at build time. That's exactly what
# happened between 2026-07-29 and 2026-08-24 -- every weekly refresh
# published fine and the live site kept serving the July snapshot.
# Mirrors the DB_PATH convention in ingestion/load_db.resolve_db_path.
DEFAULT_OUT_DIR = Path(os.environ.get("PUBLISH_DIR") or "web/public/data")
BRIEF_EXCERPT_MAX_CHARS = 240

# Same canonical order as scoring/composite.py and briefs/generator.py.
_DETECTOR_COLS: list[tuple[str, str]] = [
    ("benford", "benford_score"),
    ("new_entity", "new_entity_score"),
    ("mod_growth", "mod_growth_score"),
    ("isolation", "isolation_score"),
    ("sole_source_concentration", "sole_source_concentration_score"),
    ("award_velocity", "award_velocity_score"),
]

# Detector → field in details JSON that names a specific triggering award.
# Only detectors that point at a single contract are listed; the others
# (benford, isolation, sole_source_concentration, award_velocity) summarize
# the entity as a whole and don't surface one contract.
_DETECTOR_TRIGGER_AWARD_FIELDS: dict[str, str] = {
    "mod_growth": "worst_award_id",
    "new_entity": "first_award_id",
}


# ── Top-N selection ──────────────────────────────────────────────────────


def _select_top_n(
    con: duckdb.DuckDBPyConnection, score_date: date, top_n: int
) -> list[dict]:
    """Top-N entities by composite_score for `score_date`, enriched with
    awards aggregates. Mirrors briefs.generator.select_top_n's join shape.
    """
    score_cols_sql = ", ".join(c for _, c in _DETECTOR_COLS)
    rows = con.execute(
        f"""
        WITH agg AS (
            SELECT
                recipient_uei,
                ARG_MAX(recipient_name, award_date) AS recipient_name,
                ARG_MAX(awarding_agency, award_date) AS awarding_agency,
                ARG_MAX(naics_code, award_date) AS naics_code,
                ARG_MAX(naics_description, award_date) AS naics_description,
                SUM(total_obligation) AS total_obligated_lifetime,
                COUNT(*) AS award_count_lifetime
            FROM awards
            WHERE recipient_uei IS NOT NULL
            GROUP BY recipient_uei
        )
        SELECT
            s.uei,
            COALESCE(agg.recipient_name, s.uei) AS entity_name,
            agg.awarding_agency,
            agg.naics_code,
            agg.naics_description,
            agg.total_obligated_lifetime,
            agg.award_count_lifetime,
            s.composite_score,
            s.composite_percentile_rank,
            s.detector_details,
            {score_cols_sql}
        FROM suspicion_scores s
        LEFT JOIN agg ON s.uei = agg.recipient_uei
        WHERE s.score_date = ?
        ORDER BY s.composite_score DESC
        LIMIT ?
        """,
        [score_date, top_n],
    ).fetchall()

    out: list[dict] = []
    for rank, row in enumerate(rows, start=1):
        (uei, name, agency, naics_code, naics_desc, lifetime_total,
         lifetime_awards, composite_score, composite_pct, details_json,
         *score_values) = row
        try:
            details_map = json.loads(details_json) if details_json else {}
        except (json.JSONDecodeError, TypeError):
            details_map = {}

        detectors_fired: list[dict] = []
        for (det_name, _col), score in zip(_DETECTOR_COLS, score_values):
            if score is None or score <= 0:
                continue
            detectors_fired.append({
                "name": det_name,
                "score": float(score),
                "details": details_map.get(det_name, {}),
            })

        out.append({
            "rank": rank,
            "uei": uei,
            "name": name,
            "agency": agency,
            "naics_code": naics_code,
            "naics_description": naics_desc,
            "lifetime_total": float(lifetime_total) if lifetime_total is not None else None,
            "lifetime_awards": int(lifetime_awards) if lifetime_awards is not None else None,
            "composite_score": float(composite_score),
            "composite_percentile_rank": float(composite_pct),
            "detectors_fired": detectors_fired,
        })
    return out


def _total_scored_for_date(con: duckdb.DuckDBPyConnection, score_date: date) -> int:
    row = con.execute(
        "SELECT COUNT(*) FROM suspicion_scores WHERE score_date = ?", [score_date]
    ).fetchone()
    return int(row[0]) if row else 0


# ── Brief lookup ─────────────────────────────────────────────────────────


def _load_briefs(
    con: duckdb.DuckDBPyConnection, score_date: date, ueis: list[str]
) -> dict[str, str]:
    """Return {uei: brief_text}, carrying forward the most recent prior brief.

    Deliberately not an exact `score_date = ?` match. Only the daily
    pipeline generates briefs; the weekly one (scripts/seed.py) rescores
    and publishes without them, so on a date the daily run hasn't covered
    there are no rows for `score_date` at all. Matching the exact date
    published `brief_text: null` for every entity and blanked the briefs
    off the live site -- which is what happened on 2026-08-31, when a long
    weekly ingest held the pipeline lock through the daily run's window
    and then published over it. A slightly stale brief beats no brief.
    """
    if not ueis:
        return {}
    placeholders = ", ".join(["?"] * len(ueis))
    rows = con.execute(
        f"""
        SELECT uei, brief_text
        FROM entity_briefs
        WHERE score_date <= ? AND uei IN ({placeholders})
        QUALIFY row_number() OVER (
            PARTITION BY uei ORDER BY score_date DESC
        ) = 1
        """,
        [score_date, *ueis],
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _excerpt(brief_text: Optional[str]) -> Optional[str]:
    """First paragraph of a brief, truncated. Returns None for empty input."""
    if not brief_text:
        return None
    first_para = brief_text.split("\n\n", 1)[0].strip()
    if len(first_para) <= BRIEF_EXCERPT_MAX_CHARS:
        return first_para
    return first_para[: BRIEF_EXCERPT_MAX_CHARS - 1].rstrip() + "…"


# ── Score history + delta ───────────────────────────────────────────────


def _load_score_history(
    con: duckdb.DuckDBPyConnection, uei: str
) -> list[dict]:
    """All (date, composite_score, rank) tuples for this UEI across history,
    in chronological order. Rank is computed within each score_date across
    all entities scored that day.
    """
    rows = con.execute(
        """
        WITH ranked AS (
            SELECT
                score_date,
                uei,
                composite_score,
                ROW_NUMBER() OVER (
                    PARTITION BY score_date
                    ORDER BY composite_score DESC
                ) AS rank
            FROM suspicion_scores
        )
        SELECT score_date, composite_score, rank
        FROM ranked
        WHERE uei = ?
        ORDER BY score_date
        """,
        [uei],
    ).fetchall()
    return [
        {
            "date": r[0].isoformat(),
            "composite_score": float(r[1]),
            "rank": int(r[2]),
        }
        for r in rows
    ]


def _previous_day_score(
    con: duckdb.DuckDBPyConnection, uei: str, score_date: date
) -> Optional[float]:
    """Composite score for `uei` on the most recent score_date strictly
    before `score_date`. None if no prior record exists.
    """
    row = con.execute(
        """
        SELECT composite_score
        FROM suspicion_scores
        WHERE uei = ? AND score_date < ?
        ORDER BY score_date DESC
        LIMIT 1
        """,
        [uei, score_date],
    ).fetchone()
    return float(row[0]) if row else None


# ── Flagged contracts ────────────────────────────────────────────────────


def _extract_triggered_award_ids(detectors: list[dict]) -> list[tuple[str, str]]:
    """Walk fired detectors and return (award_id, detector_name) pairs for
    detectors that point at a specific triggering contract.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for d in detectors:
        field = _DETECTOR_TRIGGER_AWARD_FIELDS.get(d["name"])
        if not field:
            continue
        award_id = d.get("details", {}).get(field)
        if award_id and award_id not in seen:
            seen.add(award_id)
            out.append((award_id, d["name"]))
    return out


def _load_award_metadata(
    con: duckdb.DuckDBPyConnection, award_ids: list[str]
) -> dict[str, dict]:
    """Fetch amount/date/competition for a list of award IDs."""
    if not award_ids:
        return {}
    placeholders = ", ".join(["?"] * len(award_ids))
    rows = con.execute(
        f"""
        SELECT award_id, total_obligation, award_date, competition_type
        FROM awards
        WHERE award_id IN ({placeholders})
        """,
        award_ids,
    ).fetchall()
    return {
        r[0]: {
            "amount": float(r[1]) if r[1] is not None else None,
            "date": r[2].isoformat() if r[2] else None,
            "competition_type": r[3],
        }
        for r in rows
    }


def _usaspending_url(award_id: str) -> str:
    """Best-effort permalink. USAspending uses an internal id encoding we
    don't have at hand, so we point at the search page keyed by award_id —
    always resolves to a real result page, never 404s.
    """
    return f"https://www.usaspending.gov/search/?hash=&keyword={award_id}"


def _build_flagged_contracts(
    con: duckdb.DuckDBPyConnection, detectors: list[dict]
) -> list[dict]:
    pairs = _extract_triggered_award_ids(detectors)
    if not pairs:
        return []
    meta = _load_award_metadata(con, [p[0] for p in pairs])
    out: list[dict] = []
    for award_id, det_name in pairs:
        m = meta.get(award_id, {})
        out.append({
            "award_id": award_id,
            "amount": m.get("amount"),
            "date": m.get("date"),
            "competition_type": m.get("competition_type"),
            "triggered_detector": det_name,
            "usaspending_url": _usaspending_url(award_id),
        })
    return out


# ── Build JSON payloads ──────────────────────────────────────────────────


def _build_leaderboard(
    entries: list[dict],
    briefs: dict[str, str],
    score_date: date,
    total_scored: int,
) -> dict:
    """Produce the homepage leaderboard.json structure."""
    lead = None
    featured: list[dict] = []
    ranking: list[dict] = []

    for entry in entries:
        rank = entry["rank"]
        uei = entry["uei"]
        brief_text = briefs.get(uei)
        detector_names = [d["name"] for d in entry["detectors_fired"]]

        if rank == 1:
            lead = {
                "rank": 1,
                "uei": uei,
                "name": entry["name"],
                "agency": entry["agency"],
                "naics_description": entry["naics_description"],
                "lifetime_total": entry["lifetime_total"],
                "lifetime_awards": entry["lifetime_awards"],
                "composite_score": entry["composite_score"],
                "composite_percentile_rank": entry["composite_percentile_rank"],
                "detectors_fired": detector_names,
                "brief_text": brief_text,
            }
        elif rank <= 10:
            featured.append({
                "rank": rank,
                "uei": uei,
                "name": entry["name"],
                "agency": entry["agency"],
                "lifetime_total": entry["lifetime_total"],
                "composite_score": entry["composite_score"],
                "detectors_fired": detector_names,
                "brief_excerpt": _excerpt(brief_text),
            })
        else:
            ranking.append({
                "rank": rank,
                "uei": uei,
                "name": entry["name"],
                "agency": entry["agency"],
                "lifetime_total": entry["lifetime_total"],
                "composite_score": entry["composite_score"],
                "detectors_fired_count": len(detector_names),
            })

    return {
        "score_date": score_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "total_scored": total_scored,
        "total_flagged": len(entries),
        "lead": lead,
        "featured": featured,
        "ranking": ranking,
    }


def _build_entity_detail(
    con: duckdb.DuckDBPyConnection,
    entry: dict,
    brief_text: Optional[str],
    score_date: date,
) -> dict:
    history = _load_score_history(con, entry["uei"])
    prior = _previous_day_score(con, entry["uei"], score_date)
    delta = (
        round(entry["composite_score"] - prior, 6)
        if prior is not None else None
    )
    flagged_contracts = _build_flagged_contracts(con, entry["detectors_fired"])

    return {
        "uei": entry["uei"],
        "name": entry["name"],
        "agency": entry["agency"],
        "naics_code": entry["naics_code"],
        "naics_description": entry["naics_description"],
        "lifetime_total": entry["lifetime_total"],
        "lifetime_awards": entry["lifetime_awards"],
        "score_date": score_date.isoformat(),
        "composite_score": entry["composite_score"],
        "composite_score_delta": delta,
        "composite_percentile_rank": entry["composite_percentile_rank"],
        "brief_text": brief_text,
        "detectors": entry["detectors_fired"],
        "score_history": history,
        "flagged_contracts": flagged_contracts,
    }


# ── Atomic write ─────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


# ── Orchestrator ─────────────────────────────────────────────────────────


def publish(
    db_path: str,
    score_date: date,
    top_n: int = DEFAULT_TOP_N,
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> dict[str, int]:
    """Publish leaderboard.json + entities/<uei>.json into `out_dir`.

    Returns a small counts dict for logging.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path, read_only=True)
    try:
        entries = _select_top_n(con, score_date, top_n)
        total_scored = _total_scored_for_date(con, score_date)
        briefs = _load_briefs(con, score_date, [e["uei"] for e in entries])

        leaderboard = _build_leaderboard(entries, briefs, score_date, total_scored)
        _atomic_write_json(out_dir / "leaderboard.json", leaderboard)

        n_entities = 0
        for entry in entries:
            detail = _build_entity_detail(
                con, entry, briefs.get(entry["uei"]), score_date,
            )
            _atomic_write_json(out_dir / "entities" / f"{entry['uei']}.json", detail)
            n_entities += 1
    finally:
        con.close()

    logger.info(
        "published %d entity files + leaderboard.json into %s "
        "(score_date=%s, total_scored=%d)",
        n_entities, out_dir, score_date, total_scored,
    )
    return {"entities": n_entities, "total_scored": total_scored}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish DuckDB scoring + briefs as JSON for the frontend."
    )
    parser.add_argument(
        "--score-date",
        type=date.fromisoformat,
        default=None,
        help="ISO date to publish. Defaults to MAX(score_date) in DuckDB.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"How many top-ranked entities to publish (default {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    db_path = resolve_db_path()

    score_date = args.score_date
    if score_date is None:
        con = duckdb.connect(db_path, read_only=True)
        try:
            row = con.execute("SELECT MAX(score_date) FROM suspicion_scores").fetchone()
        finally:
            con.close()
        if not row or not row[0]:
            raise SystemExit(
                "No suspicion_scores rows in DB. Run scoring.composite first."
            )
        score_date = row[0]

    result = publish(db_path, score_date=score_date, top_n=args.top_n, out_dir=args.out)
    print(
        f"score_date={score_date}  "
        f"entities={result['entities']}  "
        f"total_scored={result['total_scored']}  "
        f"out={args.out}"
    )


if __name__ == "__main__":
    main()
