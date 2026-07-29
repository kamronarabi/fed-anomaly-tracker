"""New-to-federal fly-by-night sole-source anomaly detector.

Flags entities matching the **shell-company / fly-by-night pattern**:
- First-ever USAspending award is sole-source (non-competed)
- First award above the FAR simplified-acquisition threshold but under
  the IDV megaprime range (default $250K-$5M)
- Few lifetime federal awards (default <= 5)
- Modest lifetime federal contracting volume (default <= $10M)

Together these guardrails recover the "register, win one big no-bid
contract, disappear" pattern without SAM `registration_date`. They also
suppress two classes of noise:

1. **IDV megaprimes** — entities like Boeing, Lockheed, BAE whose entire
   federal relationship is one massive sole-source IDV. USAspending
   stores these as 1-2 rows for billions of dollars. The amount_max and
   lifetime_total_max filters exclude them.

2. **Established primes with sole-source first-in-window** — Raytheon's
   "first" award in our 2023-2026 data window is often sole-source,
   but the company has been a federal contractor for decades. The
   lifetime_awards_max filter excludes anyone with > 5 federal awards.

Score formula (unchanged from prior pivot):
  score = clamp(log10(first_award_obligation) / 7, 0, 1)

7 chosen so a $10M first award (log10=7) yields 1.0. A $250K first award
(the floor) yields log10(250000)/7 = 0.78. A $5M first award yields
log10(5_000_000)/7 = 0.96.

History:
- Pre-2026-05-17: SAM-anchored ("registered N days before sole-source").
- 2026-05-17: USAspending-only proxy, threshold $100K — flagged 6,365
  entities, ~90% noise from major primes and data-window artifacts.
- 2026-05-28 (post-evaluation): added amount_max, lifetime_awards_max,
  and lifetime_total_max guardrails. Flags reduced from 6,365 to ~2,585
  on live data, with sample qualitatively dominated by single-purpose
  LLCs with one-and-done sole-source contracts. See
  docs/superpowers/specs/2026-05-27-sam-removal-design.md and the
  evaluation in the 2026-05-28 session transcript.

Score interpretation: absolute, in [0.78, 0.96] given the dollar window.
Output rows are sparse: only entities matching ALL filter conditions
appear. Non-matching entities have implicit score=0.
"""

from __future__ import annotations

import json

import duckdb
import polars as pl

# Filter defaults — see module docstring for the reasoning behind each.
DEFAULT_AMOUNT_MIN = 250_000        # FAR simplified acquisition threshold + headroom
DEFAULT_AMOUNT_MAX = 5_000_000      # Excludes IDV megaprime first-awards
DEFAULT_LIFETIME_AWARDS_MAX = 5     # Excludes established federal contractors
DEFAULT_LIFETIME_TOTAL_MAX = 10_000_000  # Second guardrail against megaprimes

# DuckDB filter for non-competed awards.
_SOLE_SOURCE_FILTER = (
    "(UPPER(competition_type) LIKE '%NOT COMPETED%' "
    " OR UPPER(competition_type) LIKE '%NOT AVAILABLE%' "
    " OR UPPER(competition_type) LIKE '%SOLE SOURCE%')"
)


def detect_new_entity_sole_source(
    db_path: str,
    amount_min: float = DEFAULT_AMOUNT_MIN,
    amount_max: float = DEFAULT_AMOUNT_MAX,
    lifetime_awards_max: int = DEFAULT_LIFETIME_AWARDS_MAX,
    lifetime_total_max: float = DEFAULT_LIFETIME_TOTAL_MAX,
) -> pl.DataFrame:
    """Score entities matching the fly-by-night sole-source pattern.

    Returns one row per qualifying UEI with the four-column schema shared
    across all detectors.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Window function picks each entity's earliest award; aggregate
        # CTE computes lifetime stats for the post-filter conditions.
        rows = con.execute(
            f"""
            WITH first_awards AS (
              SELECT
                recipient_uei,
                award_id,
                award_date,
                total_obligation,
                competition_type,
                ROW_NUMBER() OVER (
                  PARTITION BY recipient_uei
                  ORDER BY award_date ASC NULLS LAST, award_id ASC
                ) AS rn
              FROM awards
              WHERE recipient_uei IS NOT NULL
                AND recipient_uei != ''
                AND total_obligation IS NOT NULL
                AND total_obligation > 0
            ),
            lifetimes AS (
              SELECT
                recipient_uei,
                COUNT(*)              AS lifetime_awards,
                SUM(total_obligation) AS lifetime_total
              FROM awards
              WHERE recipient_uei IS NOT NULL
                AND total_obligation > 0
              GROUP BY recipient_uei
            )
            SELECT
              fa.recipient_uei         AS uei,
              fa.award_id              AS first_award_id,
              fa.award_date            AS first_award_date,
              fa.total_obligation      AS first_award_obligation,
              fa.competition_type      AS competition_type,
              lt.lifetime_awards       AS lifetime_awards,
              lt.lifetime_total        AS lifetime_total
            FROM first_awards fa
            JOIN lifetimes lt USING (recipient_uei)
            WHERE fa.rn = 1
              AND fa.total_obligation >= ?
              AND fa.total_obligation <= ?
              AND lt.lifetime_awards   <= ?
              AND lt.lifetime_total    <= ?
              AND {_SOLE_SOURCE_FILTER.replace("competition_type", "fa.competition_type")}
            """,
            [amount_min, amount_max, lifetime_awards_max, lifetime_total_max],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return _empty()

    df = pl.DataFrame(
        rows,
        schema={
            "uei": pl.Utf8,
            "first_award_id": pl.Utf8,
            "first_award_date": pl.Date,
            "first_award_obligation": pl.Float64,
            "competition_type": pl.Utf8,
            "lifetime_awards": pl.Int64,
            "lifetime_total": pl.Float64,
        },
        orient="row",
    )

    df = df.with_columns(
        (pl.col("first_award_obligation").log(base=10.0) / 7.0)
        .clip(0.0, 1.0)
        .alias("score")
    )

    out_rows = []
    for r in df.iter_rows(named=True):
        details = {
            "first_award_id": r["first_award_id"],
            "first_award_date": (
                r["first_award_date"].isoformat()
                if r["first_award_date"] is not None
                else None
            ),
            "first_award_obligation": r["first_award_obligation"],
            "competition_type": r["competition_type"],
            "lifetime_awards": r["lifetime_awards"],
            "lifetime_total": r["lifetime_total"],
        }
        out_rows.append(
            {
                "uei": r["uei"],
                "detector": "new_entity",
                "score": r["score"],
                "details": json.dumps(details),
            }
        )

    return pl.DataFrame(out_rows, schema=_schema())


def _schema() -> dict:
    return {
        "uei": pl.Utf8,
        "detector": pl.Utf8,
        "score": pl.Float64,
        "details": pl.Utf8,
    }


def _empty() -> pl.DataFrame:
    return pl.DataFrame([], schema=_schema())


if __name__ == "__main__":
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    df = (
        detect_new_entity_sole_source(cfg["db_path"])
        .sort("score", descending=True)
        .head(10)
    )
    print(df)
