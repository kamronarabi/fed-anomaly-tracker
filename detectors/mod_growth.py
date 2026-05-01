"""Modification growth anomaly detector.

For each parent_award_id, compute the growth ratio (final value / initial
value) where:
  - initial_value = obligation of the row with modification_number = '0',
                    falling back to the row with the smallest mod number;
  - final_value   = sum of obligations across all rows under that parent.

Z-score the growth ratio within each NAICS group. Entity score = sigmoid of
the max z-score across that entity's contracts.
"""

from __future__ import annotations

import json
import math

import duckdb
import polars as pl

Z_THRESHOLD = 2.0


def _sigmoid(z: float, threshold: float = Z_THRESHOLD) -> float:
    """Map z-score to [0, 1] with the inflection at `threshold`."""
    try:
        return 1.0 / (1.0 + math.exp(-(z - threshold)))
    except OverflowError:
        return 0.0 if z < threshold else 1.0


def detect_mod_growth(db_path: str, z_threshold: float = Z_THRESHOLD) -> pl.DataFrame:
    """Score entities by their worst mod-growth z-score across contracts.

    Only entities whose worst-contract z-score exceeds z_threshold are
    returned; their score is sigmoid(z) mapped to [0, 1].
    """
    con = duckdb.connect(db_path, read_only=True)
    try:
        # One row per (parent_award_id, uei, naics) with initial + final.
        # Initial = obligation where mod = '0', else smallest modification_number.
        per_parent = con.execute(
            """
            WITH ranked AS (
              SELECT
                parent_award_id,
                recipient_uei AS uei,
                naics_code,
                total_obligation,
                modification_number,
                ROW_NUMBER() OVER (
                  PARTITION BY parent_award_id
                  ORDER BY (modification_number = '0') DESC,
                           modification_number ASC
                ) AS rn
              FROM awards
              WHERE parent_award_id IS NOT NULL
                AND recipient_uei IS NOT NULL
                AND total_obligation IS NOT NULL
                AND total_obligation > 0
            ),
            initials AS (
              SELECT parent_award_id, uei, naics_code,
                     total_obligation AS initial_value
              FROM ranked WHERE rn = 1
            ),
            sums AS (
              SELECT parent_award_id,
                     SUM(total_obligation) AS final_value
              FROM ranked
              GROUP BY parent_award_id
            )
            SELECT i.parent_award_id, i.uei, i.naics_code,
                   i.initial_value, s.final_value,
                   s.final_value / i.initial_value AS growth_ratio
            FROM initials i
            JOIN sums s USING (parent_award_id)
            WHERE i.initial_value > 0
            """
        ).fetchall()
    finally:
        con.close()

    if not per_parent:
        return _empty()

    df = pl.DataFrame(
        per_parent,
        schema={
            "parent_award_id": pl.Utf8,
            "uei": pl.Utf8,
            "naics_code": pl.Utf8,
            "initial_value": pl.Float64,
            "final_value": pl.Float64,
            "growth_ratio": pl.Float64,
        },
        orient="row",
    )

    # Z-score within each NAICS bucket. NAICS groups with <2 rows can't
    # produce a stddev -> z=0 (no signal possible from that bucket).
    df = df.with_columns(
        [
            pl.col("growth_ratio").mean().over("naics_code").alias("naics_mean"),
            pl.col("growth_ratio").std().over("naics_code").alias("naics_std"),
        ]
    )
    df = df.with_columns(
        pl.when((pl.col("naics_std").is_not_null()) & (pl.col("naics_std") > 0))
        .then((pl.col("growth_ratio") - pl.col("naics_mean")) / pl.col("naics_std"))
        .otherwise(0.0)
        .alias("z_score")
    )

    # For each UEI, pick the worst contract (max z).
    worst = (
        df.sort("z_score", descending=True)
        .group_by("uei", maintain_order=True)
        .agg(
            [
                pl.col("parent_award_id").first().alias("worst_award_id"),
                pl.col("growth_ratio").first(),
                pl.col("naics_code").first(),
                pl.col("naics_mean").first().alias("naics_avg_ratio"),
                pl.col("z_score").first(),
            ]
        )
    )

    out_rows = []
    for r in worst.iter_rows(named=True):
        z = r["z_score"]
        # Only surface entities whose worst z-score clears the threshold.
        if z <= z_threshold:
            continue
        score = _sigmoid(z, z_threshold)
        details = {
            "worst_award_id": r["worst_award_id"],
            "growth_ratio": r["growth_ratio"],
            "naics_code": r["naics_code"],
            "naics_avg_ratio": r["naics_avg_ratio"],
            "z_score": z,
        }
        out_rows.append(
            {
                "uei": r["uei"],
                "detector": "mod_growth",
                "score": float(score),
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
    df = detect_mod_growth(cfg["db_path"]).sort("score", descending=True).head(10)
    print(df)
