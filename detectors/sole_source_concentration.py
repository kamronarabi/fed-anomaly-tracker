"""Sole-source concentration vs NAICS peers detector.

Compares each entity's sole-source contracting fraction against the
distribution of sole-source fractions among peers in the same primary
NAICS code. The signal: some industries legitimately have high
sole-source rates (specialty research, classified work, sole suppliers
of unique products). Comparing each entity to peers in the SAME NAICS
normalizes for industry baseline. An entity with 95% sole-source in a
NAICS where the median is 20% is anomalous; the same 95% in a NAICS
where the median is 90% is not.

This is the most "forensic analyst" of the Wave 2 detectors — industry-
normalized comparison is how analysts actually think about contracting
anomalies. It also has the advantage of being entirely USAspending-
native: requires no schema extension or external data.

Method (three-pass SQL):
  1. Per entity: count total awards, count sole-source awards, compute
     ss_frac. Determine primary_naics = NAICS code with most awards
     (plurality). Filter to entities with >= min_awards.
  2. Per primary_naics: aggregate median + sample std of ss_frac across
     all entities in that NAICS. Filter to NAICS with >= min_naics_entities.
  3. Per entity: z = (entity_ss_frac - naics_median) / naics_std. Emit
     only entities where z >= z_threshold.

Score = sigmoid(z - 1) — gives a smooth [0, 1] mapping. Score at the
threshold (z=1) is 0.5; score at z=3 is ~0.88; score at z=5 is ~0.98.
Cross-detector comparisons use percentile rank downstream so absolute
score floor doesn't matter for ranking.

Gotchas + design choices:
- **Sparse NAICS exclusion:** NAICS codes with < min_naics_entities (default
  20) have unstable medians and std estimates. Entities in those NAICS
  are excluded from the output entirely.
- **Min-awards floor:** entities with < min_awards (default 10) have
  unstable ss_frac estimates. Excluded.
- **Std = 0 handling:** if every entity in a NAICS has identical
  ss_frac, std is 0 and z is undefined. Those NAICS are excluded by the
  `naics_std_ss > 0` filter (no entities from them are emitted).
- **Primary NAICS choice:** plurality of awards by count. Ties broken by
  lexically-smaller NAICS code for determinism. An alternative would be
  to weight by dollars or to score per (uei, naics) pair; plurality is
  the simplest reasonable choice for v1.
- **One-sided detection:** only entities with positive z (sole-source
  share ABOVE the peer median) are emitted. An entity well below the
  NAICS median isn't the signal we care about.
- **Mean + sample std rather than median + MAD:** statistical purists
  would pair median with MAD (median absolute deviation). Sample std
  is simpler and the sigmoid downstream squashes extremes, so the
  inconsistency doesn't bite. Document and move on.

Score interpretation: absolute, in (sigmoid(z_threshold - 1), 1.0).
With default z_threshold=1.0, scores live in (0.5, 1.0).
"""

from __future__ import annotations

import json
import math

import duckdb
import polars as pl

DEFAULT_MIN_AWARDS = 10
DEFAULT_MIN_NAICS_ENTITIES = 20
DEFAULT_Z_THRESHOLD = 1.0

# Matches the same filter used in detectors/new_entity.py.
_SOLE_SOURCE_FILTER = (
    "(UPPER(competition_type) LIKE '%NOT COMPETED%' "
    " OR UPPER(competition_type) LIKE '%NOT AVAILABLE%' "
    " OR UPPER(competition_type) LIKE '%SOLE SOURCE%')"
)


def detect_sole_source_concentration(
    db_path: str,
    min_awards: int = DEFAULT_MIN_AWARDS,
    min_naics_entities: int = DEFAULT_MIN_NAICS_ENTITIES,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> pl.DataFrame:
    """Score entities whose sole-source fraction is anomalously high vs
    same-NAICS peers.

    Returns the four-column detector schema. Score = sigmoid(z - 1).
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            f"""
            WITH uei_naics_counts AS (
              SELECT
                recipient_uei,
                naics_code,
                COUNT(*) AS naics_count
              FROM awards
              WHERE recipient_uei IS NOT NULL
                AND recipient_uei != ''
                AND naics_code IS NOT NULL
                AND total_obligation IS NOT NULL
                AND total_obligation > 0
              GROUP BY recipient_uei, naics_code
            ),
            primary_naics AS (
              SELECT recipient_uei, naics_code AS primary_naics
              FROM (
                SELECT
                  recipient_uei,
                  naics_code,
                  ROW_NUMBER() OVER (
                    PARTITION BY recipient_uei
                    ORDER BY naics_count DESC, naics_code ASC
                  ) AS rn
                FROM uei_naics_counts
              )
              WHERE rn = 1
            ),
            entity_stats AS (
              SELECT
                a.recipient_uei,
                pn.primary_naics,
                COUNT(*) AS total_awards,
                SUM(CASE WHEN {_SOLE_SOURCE_FILTER} THEN 1 ELSE 0 END)
                  AS sole_source_awards,
                SUM(CASE WHEN {_SOLE_SOURCE_FILTER} THEN 1.0 ELSE 0.0 END)
                  / COUNT(*) AS ss_frac
              FROM awards a
              JOIN primary_naics pn
                ON pn.recipient_uei = a.recipient_uei
              WHERE a.recipient_uei IS NOT NULL
                AND a.total_obligation IS NOT NULL
                AND a.total_obligation > 0
              GROUP BY a.recipient_uei, pn.primary_naics
              HAVING COUNT(*) >= ?
            ),
            naics_stats AS (
              SELECT
                primary_naics,
                COUNT(*) AS n_entities,
                MEDIAN(ss_frac) AS naics_median_ss,
                STDDEV_SAMP(ss_frac) AS naics_std_ss
              FROM entity_stats
              GROUP BY primary_naics
              HAVING COUNT(*) >= ?
            )
            SELECT
              es.recipient_uei              AS uei,
              es.total_awards,
              es.sole_source_awards,
              es.ss_frac,
              es.primary_naics,
              ns.n_entities                 AS naics_peer_count,
              ns.naics_median_ss,
              ns.naics_std_ss,
              (es.ss_frac - ns.naics_median_ss) / ns.naics_std_ss
                                            AS z_score
            FROM entity_stats es
            JOIN naics_stats ns
              ON ns.primary_naics = es.primary_naics
            WHERE ns.naics_std_ss > 0
              AND (es.ss_frac - ns.naics_median_ss) / ns.naics_std_ss >= ?
            """,
            [min_awards, min_naics_entities, z_threshold],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return _empty()

    out_rows = []
    for r in rows:
        (
            uei,
            total_awards,
            sole_source_awards,
            ss_frac,
            primary_naics,
            naics_peer_count,
            naics_median_ss,
            naics_std_ss,
            z_score,
        ) = r

        # sigmoid(z - 1) — see module docstring for the rationale.
        score = 1.0 / (1.0 + math.exp(-(z_score - 1.0)))
        score = max(0.0, min(1.0, score))

        details = {
            "total_awards": total_awards,
            "sole_source_awards": sole_source_awards,
            "ss_frac": ss_frac,
            "primary_naics": primary_naics,
            "naics_peer_count": naics_peer_count,
            "naics_median_ss": naics_median_ss,
            "naics_std_ss": naics_std_ss,
            "z_score": z_score,
        }
        out_rows.append(
            {
                "uei": uei,
                "detector": "sole_source_concentration",
                "score": score,
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
        detect_sole_source_concentration(cfg["db_path"])
        .sort("score", descending=True)
        .head(10)
    )
    print(df)
