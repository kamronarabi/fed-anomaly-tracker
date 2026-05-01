"""Isolation Forest multivariate outlier detector.

Build a feature vector per entity and run sklearn's IsolationForest with
contamination=0.05. Entities scoring as anomalous get a normalized [0, 1]
score and appear in the output; entities classified as normal are filtered.

Features:
  1. log_total_dollars      — log10 of total obligation (computed in SQL)
  2. award_count            — total award rows for this entity
  3. unique_agencies_count  — distinct awarding agencies
  4. naics_diversity        — distinct NAICS codes
  5. competition_ratio      — fraction of awards with full-and-open competition
  6. modification_frequency — fraction of award rows that are modifications
  7. entity_age_days        — days since SAM registration; median-imputed if missing

Score interpretation: BATCH-RELATIVE, not absolute. Scores are min-max
normalized within the flagged batch, so the most-anomalous entity in any
run is always exactly 1.0 and the least-anomalous emitted entity is always
exactly 0.0 -- regardless of how anomalous the population actually is.
Re-running on a different population shifts the scale. Cross-detector
comparisons must account for this; the composite scorer should be aware
that isolation scores rank within the batch but don't measure absolute
anomaly intensity the way Benford and new_entity do.
"""

from __future__ import annotations

import datetime
import json

import duckdb
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

CONTAMINATION = 0.05
RANDOM_STATE = 42

_FEATURE_COLS = [
    "log_total_dollars",
    "award_count",
    "unique_agencies_count",
    "naics_diversity",
    "competition_ratio",
    "modification_frequency",
    "entity_age_days",
]


def _build_features(db_path: str) -> pl.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT
              a.recipient_uei                                            AS uei,
              LOG10(GREATEST(SUM(a.total_obligation), 1.0))             AS log_total_dollars,
              COUNT(*)                                                   AS award_count,
              COUNT(DISTINCT a.awarding_agency)                         AS unique_agencies_count,
              COUNT(DISTINCT a.naics_code)                              AS naics_diversity,
              COALESCE(
                SUM(CASE
                      WHEN UPPER(a.competition_type) LIKE '%FULL AND OPEN%'
                      THEN 1.0 ELSE 0.0
                    END) / NULLIF(COUNT(*), 0),
                0.0
              )                                                         AS competition_ratio,
              COALESCE(
                SUM(CASE
                      WHEN a.modification_number IS NOT NULL
                       AND a.modification_number != '0'
                      THEN 1.0 ELSE 0.0
                    END) / NULLIF(COUNT(*), 0),
                0.0
              )                                                         AS modification_frequency,
              MAX(e.registration_date)                                  AS registration_date
            FROM awards a
            LEFT JOIN entities e ON e.uei = a.recipient_uei
            WHERE a.recipient_uei IS NOT NULL
              AND a.total_obligation IS NOT NULL
              AND a.total_obligation > 0
            GROUP BY a.recipient_uei
            """
        ).fetchall()
    finally:
        con.close()

    today = datetime.date.today()
    df = pl.DataFrame(
        rows,
        schema={
            "uei": pl.Utf8,
            "log_total_dollars": pl.Float64,
            "award_count": pl.Float64,
            "unique_agencies_count": pl.Float64,
            "naics_diversity": pl.Float64,
            "competition_ratio": pl.Float64,
            "modification_frequency": pl.Float64,
            "registration_date": pl.Date,
        },
        orient="row",
    )
    return df.with_columns(
        pl.when(pl.col("registration_date").is_not_null())
        .then(
            (pl.lit(today) - pl.col("registration_date"))
            .dt.total_days()
            .cast(pl.Float64)
        )
        .otherwise(float("nan"))
        .alias("entity_age_days")
    ).drop("registration_date")


def detect_isolation_outlier(db_path: str) -> pl.DataFrame:
    """Score each entity via IsolationForest; return only flagged outliers."""
    feats = _build_features(db_path)
    if feats.height == 0:
        return _empty()

    # Median-impute entity_age_days for entities lacking SAM enrichment.
    feats = feats.with_columns(
        pl.col("entity_age_days").fill_nan(
            pl.col("entity_age_days").median().fill_null(0.0)
        )
    )

    matrix = feats.select(_FEATURE_COLS).to_numpy().astype(float)
    matrix = MinMaxScaler().fit_transform(matrix)

    # For tiny datasets, ensure at least one entity is flagged.
    n = matrix.shape[0]
    contamination = CONTAMINATION if n >= 20 else max(1.0 / n, CONTAMINATION)

    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    model.fit(matrix)
    raw_scores = model.score_samples(matrix)  # higher = more normal
    predictions = model.predict(matrix)       # -1 = outlier, 1 = normal

    # Invert so that more anomalous → score closer to 1.0.
    inverted = -raw_scores
    rng = inverted.max() - inverted.min()
    normalized = (inverted - inverted.min()) / rng if rng > 0 else np.zeros_like(inverted)

    out_rows = []
    for i, row in enumerate(feats.iter_rows(named=True)):
        if predictions[i] != -1:
            continue
        details = {
            "feature_vector": {col: row[col] for col in _FEATURE_COLS},
            "raw_anomaly_score": float(raw_scores[i]),
            "contamination_param": contamination,
        }
        out_rows.append(
            {
                "uei": row["uei"],
                "detector": "isolation",
                "score": float(normalized[i]),
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
    df = detect_isolation_outlier(cfg["db_path"]).sort("score", descending=True).head(10)
    print(df)
