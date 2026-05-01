"""New-entity sole-source anomaly detector.

Flags entities that received non-competed awards within `days_threshold`
days of their SAM.gov registration date. The signal: shell companies often
register, win a large no-bid contract, and disappear -- all within months.

Score = recency_factor * magnitude_factor, each in [0, 1]:
  - recency_factor  = 1 - (days_gap / days_threshold)  [closer to reg = higher]
  - magnitude_factor = log10(total_obligation) / 7      [larger award = higher]

7 is chosen so that a $10M award (log10=7) yields a full magnitude of 1.0.
"""

from __future__ import annotations

import json

import duckdb
import polars as pl

DEFAULT_DAYS_THRESHOLD = 180

# DuckDB filter for non-competed awards, matching USAspending's extent_competed
# field values that represent "no real competition."
_SOLE_SOURCE_FILTER = (
    "(UPPER(a.competition_type) LIKE '%NOT COMPETED%' "
    " OR UPPER(a.competition_type) LIKE '%NOT AVAILABLE%' "
    " OR UPPER(a.competition_type) LIKE '%SOLE SOURCE%')"
)


def detect_new_entity_sole_source(
    db_path: str, days_threshold: int = DEFAULT_DAYS_THRESHOLD
) -> pl.DataFrame:
    """Score entities that received sole-source awards shortly after SAM registration.

    Returns one row per UEI — the award with the highest anomaly score — with
    the four-column schema shared across all Wave 1 detectors.
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            f"""
            SELECT
              a.recipient_uei                                        AS uei,
              e.registration_date,
              a.award_date,
              a.total_obligation,
              a.competition_type,
              DATE_DIFF('day', e.registration_date, a.award_date)   AS days_gap
            FROM awards a
            JOIN entities e ON e.uei = a.recipient_uei
            WHERE e.registration_date IS NOT NULL
              AND a.award_date        IS NOT NULL
              AND a.total_obligation  IS NOT NULL
              AND a.total_obligation  > 0
              AND {_SOLE_SOURCE_FILTER}
              AND DATE_DIFF('day', e.registration_date, a.award_date) BETWEEN 0 AND ?
            """,
            [days_threshold],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return _empty()

    df = pl.DataFrame(
        rows,
        schema={
            "uei": pl.Utf8,
            "registration_date": pl.Date,
            "award_date": pl.Date,
            "total_obligation": pl.Float64,
            "competition_type": pl.Utf8,
            "days_gap": pl.Int64,
        },
        orient="row",
    )

    df = df.with_columns(
        (
            (1.0 - pl.col("days_gap") / days_threshold)
            * (pl.col("total_obligation").log(base=10.0) / 7.0)
        )
        .clip(0.0, 1.0)
        .alias("score")
    )

    # Keep the highest-scoring award per UEI.
    worst = (
        df.sort("score", descending=True)
        .group_by("uei", maintain_order=True)
        .agg(
            [
                pl.col("score").first(),
                pl.col("days_gap").first(),
                pl.col("registration_date").first(),
                pl.col("award_date").first(),
                pl.col("total_obligation").first(),
                pl.col("competition_type").first(),
            ]
        )
    )

    out_rows = []
    for r in worst.iter_rows(named=True):
        details = {
            "days_gap": r["days_gap"],
            "registration_date": r["registration_date"].isoformat(),
            "award_date": r["award_date"].isoformat(),
            "total_obligation": r["total_obligation"],
            "competition_type": r["competition_type"],
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
